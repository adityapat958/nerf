#write your own data loader, parser, network and loss function for this phase.

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import argparse
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio
from skimage.metrics import structural_similarity

class NerfDataset(Dataset):
    def __init__(self, data_dir, transform_path, transform=None, cache_images=True):
        self.data_dir = data_dir
        self.transform_path = transform_path
        self.cache_images = cache_images
        
        with open(self.transform_path, 'r') as f:
            transform_data = json.load(f)
            self.frames = transform_data['frames']
            self.fov = transform_data['camera_angle_x']  # FOV in radians

        self.transform_map = {}
        for frame in self.frames:
  
            basename = os.path.basename(frame['file_path']) 
            self.transform_map[basename] = frame['transform_matrix']

        
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_paths = []
        for f in sorted(all_files):
            
            basename = os.path.splitext(f)[0]
            if basename in self.transform_map:
                self.image_paths.append(os.path.join(data_dir, f))

    
        sample_image = Image.open(self.image_paths[0])
        self.image_width, self.image_height = sample_image.size
        self.focal_length = self.focal_length_from_fov(self.fov, self.image_width)
        self.num_pixels = self.image_width * self.image_height

        grid_u, grid_v = torch.meshgrid(
            torch.arange(self.image_width, dtype=torch.float32),
            torch.arange(self.image_height, dtype=torch.float32),
            indexing='xy'
        )
        flat_u = grid_u.reshape(-1)
        flat_v = grid_v.reshape(-1)
        self.precomputed_cam_rays = self.image2cam(flat_u, flat_v).float().contiguous().to('cuda')  

        self.cached_images = None
        self.cached_transforms = None
        if self.cache_images:
            self.cached_images = []
            self.cached_transforms = []
            for img_path in tqdm(self.image_paths, desc=f"Caching {os.path.basename(data_dir)}", leave=False):
                image = Image.open(img_path).convert('RGB')
                basename = os.path.splitext(os.path.basename(img_path))[0]
                transform = self.transform_map[basename]

                image_tensor = torch.from_numpy(np.array(image) / 255.0).float().view(-1, 3).to('cuda')
                transform_tensor = torch.tensor(transform, dtype=torch.float32).to('cuda')

                self.cached_images.append(image_tensor)
                self.cached_transforms.append(transform_tensor)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.cache_images and self.cached_images is not None:
            return self.cached_images[idx], self.cached_transforms[idx]

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract "r_0" from the absolute path
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Instant O(1) lookup! No more for loops!
        transform = self.transform_map[basename]
        
        image = torch.from_numpy(np.array(image) / 255.0).float().to('cuda').view(-1, 3) 
        transform = torch.tensor(transform, dtype=torch.float32)
        
        return image, transform

    def get_camera_rays_from_flat_indices(self, flat_indices):
        
        return self.precomputed_cam_rays[flat_indices].to('cuda')
    
    
    # Fov present in transform JSON in radians, key is "camera_angle_x"
    def focal_length_from_fov(self,fov, image_width):
        # Calculate focal length from field of view (FOV) and image width
        focal_length = (image_width / 2) / np.tan((fov) / 2)
        return focal_length   


    def image2cam(self,u,v):
        # Convert image to camera coordinates (Intrinsic parameters)
        x_shifted= u - self.image_width/ 2
        y_shifted= v - self.image_height/ 2

        dir_x = x_shifted / self.focal_length
        dir_y = -y_shifted / self.focal_length
        dir_z = torch.full_like(dir_x, -1.0)  # Assuming the camera looks along the negative z-axis
        dir = torch.stack([dir_x, dir_y, dir_z], dim=-1)

        return dir

    def cam2world(self,cam_coords, extrinsic_matrix):
        # Convert camera coordinates to world coordinates (Extrinsic parameters)
        # TODO : Implement the conversion from camera coordinates to world coordinates using the extrinsic matrix
        rotation = extrinsic_matrix[:3, :3]
        translation = extrinsic_matrix[:3, 3]

        
        ray_d = torch.matmul(cam_coords, rotation.T )
        ray_o= translation.expand_as(ray_d)
        return ray_o, ray_d
        
class NeRFModel(nn.Module):
    def __init__(self, num_pts_freqs=10, num_view_freqs=4):
        super(NeRFModel, self).__init__()
        
        
        self.register_buffer('pts_freqs', 2.0 ** torch.arange(num_pts_freqs))
        self.register_buffer('view_freqs', 2.0 ** torch.arange(num_view_freqs))

        self.fc1 = nn.Linear(63, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(319, 256) 
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 257)
        self.fc9 = nn.Linear(283, 256)
        self.fc10 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(128, 3)    

    def positional_encoding(self, x, freqs):
        # x  flattened
        scaled = x.unsqueeze(-1) * freqs
        scaled = scaled.flatten(start_dim=-2)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return torch.cat([x, pe], dim=-1)
    
    def forward(self, pts, view_dir):
        batch_size, num_samples, _ = pts.shape
        
        
        pts_flat = pts.reshape(-1, 3) 
        
        rays_63 = self.positional_encoding(pts_flat, self.pts_freqs)  
        x = F.relu(self.fc1(rays_63))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.cat([rays_63, F.relu(self.fc4(x))], dim=-1) 
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        
        sigma = F.relu(x[..., 0]) 
        
        # Expand view_dir, flatten, and encode
        view_dir_expanded = view_dir.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)
        view_dir_encoded = self.positional_encoding(view_dir_expanded, self.view_freqs) 
        
        x = self.fc9(torch.cat([x[..., 1:257], view_dir_encoded], dim=-1)) 
        x = F.relu(self.fc10(x))
        output = torch.sigmoid(self.fc11(x))  
        
        # Reshape back to 3D for volume rendering
        sigma = sigma.view(batch_size, num_samples)
        output = output.view(batch_size, num_samples, 3)
        
        return output, sigma


def volume_rendering(model, ray_o, ray_d, near, far, num_samples=100):
    # Implement volume rendering to compute the final pixel color from the sampled points along the ray
    # TODO : Implement volume rendering using the NeRF model's output (e.g., using alpha compositing)
    current_device = ray_o.device
    # t = torch.linspace(near, far, steps=num_samples, device=current_device) # Number of samples along each ray
    # ray_d_math = ray_d.unsqueeze(1) # Expand ray_d to match the shape of t
    # ray_o_math = ray_o.unsqueeze(1) # Expand ray_o to match the shape of t
    # rays_t_math = ray_o_math + t * ray_d_math  # Sample points along the ray (assuming step size of 0.01)

    
    # ray_d_expanded= ray_d.unsqueeze(1).expand(-1, num_samples, -1) # Expand ray_d to match the batch size
    t = torch.linspace(near, far, steps=num_samples, device=current_device)
    rays_t_math = ray_o.unsqueeze(1) + t.view(1, -1, 1) * ray_d.unsqueeze(1)

    rgb,sigma = model(rays_t_math, ray_d)  # Get RGB and density from the model

    # Compute alpha values from density
    step_size = (far - near) / num_samples
    alpha = 1.0 - torch.exp(-sigma * step_size)

    # Compute weights for alpha compositing
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=1)[:, :-1]
    ones = torch.ones((alpha.shape[0], 1), device=current_device)
    weights = alpha * torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    # Compute final pixel color using alpha compositing
    pixel_color = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

    return pixel_color

def loss_function(predicted_color, true_color):
    # Implement a loss function to compare the predicted pixel color with the true pixel color
    # TODO : Implement a suitable loss function (e.g., mean squared error)
    error = predicted_color - true_color
    mse_loss = torch.mean(error ** 2)
    return mse_loss

def find_latest_epoch_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    pattern = re.compile(r"nerf_model_epoch_(\d{6})\.pth$")
    latest_epoch = 0
    latest_path = None

    checkpoint_files = os.listdir(checkpoint_dir)
    for filename in tqdm(checkpoint_files, desc="Scanning checkpoints", leave=False):
        match = pattern.match(filename)
        if not match:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path, latest_epoch

def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


@torch.no_grad()
def render_validation_preview(model, dataset, transform_matrix, near, far, chunk_size=32768, num_samples=100):
    current_device = dataset.precomputed_cam_rays.device
    transform_matrix = transform_matrix.to(current_device)
    total_pixels = dataset.num_pixels
    flat_indices = torch.arange(total_pixels, device=current_device)
    rays_cam = dataset.get_camera_rays_from_flat_indices(flat_indices)
    ray_o, ray_d = dataset.cam2world(rays_cam, transform_matrix)

    all_colors = []
    use_amp = torch.cuda.is_available()
    with torch.amp.autocast('cuda', enabled=use_amp):
        for i in range(0, ray_o.shape[0], chunk_size):
            chunk_ray_o = ray_o[i:i + chunk_size]
            chunk_ray_d = ray_d[i:i + chunk_size]
            chunk_color = volume_rendering(model, chunk_ray_o, chunk_ray_d, near, far, num_samples=num_samples)
            all_colors.append(chunk_color)

    img_flat = torch.cat(all_colors, dim=0)
    return img_flat.view(dataset.image_height, dataset.image_width, 3)


def compute_ssim(pred_img, gt_img):
    pred_np = pred_img.detach().float().cpu().numpy()
    gt_np = gt_img.detach().float().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)
    gt_np = np.clip(gt_np, 0.0, 1.0)
    return float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))

def train(model, train_dataloader, val_dataloader, nerfdata, optimizer, epochs, near, far, checkpoint_dir, val_dataset=None, start_epoch=0, rays_per_batch=4096, num_samples=100, val_interval=1, save_interval=10, render_interval=10, render_chunk_size=32768):
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    preview_dir = os.path.join(checkpoint_dir, 'renders_every_interval')
    os.makedirs(preview_dir, exist_ok=True)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")
        train_progress = tqdm(train_dataloader, desc=f"Train {epoch+1}/{epochs}", leave=False)
        for images, transforms in train_progress:
            optimizer.zero_grad(set_to_none=True)
            images= images.squeeze(0) # remove batch dimension
            transforms = transforms.squeeze(0) # remove batch dimension
            num_of_pixels = images.shape[0]

            batch_size = min(rays_per_batch, num_of_pixels)
            
            indices = torch.randint(0, num_of_pixels, (batch_size,))

        
            colors = images[indices] # Get the true colors for the sampled pixels

            ray = nerfdata.get_camera_rays_from_flat_indices(indices)
            ray_o, ray_d = nerfdata.cam2world(ray, transforms)
        
            ray_o = ray_o
            ray_d = ray_d
            colors = colors
            
            with torch.amp.autocast('cuda',enabled=use_amp):
                pixel_color = volume_rendering(model, ray_o, ray_d, near, far, num_samples=num_samples)
                loss = loss_function(pixel_color, colors)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.6f}")
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        run_validation = ((epoch + 1) % val_interval == 0)
        if run_validation:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                val_progress = tqdm(val_dataloader, desc=f"Val {epoch+1}/{epochs}", leave=False)
                for val_images, val_transforms in val_progress:
                    val_images = val_images.squeeze(0)  # remove batch dimension
                    val_transforms = val_transforms.squeeze(0)  # remove batch dimension
                    
                    num_val_pixels = val_images.shape[0]
                    val_indices = torch.randint(0, num_val_pixels, (batch_size,))
                    val_colors = val_images[val_indices]
                    
                    
                    v_ray = nerfdata.get_camera_rays_from_flat_indices(val_indices)
                    v_ray_o, v_ray_d = nerfdata.cam2world(v_ray, val_transforms)
                    
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        v_pixel_color = volume_rendering(model, v_ray_o, v_ray_d, near, far, num_samples=num_samples)
                        v_loss = loss_function(v_pixel_color, val_colors)
                    
                    total_val_loss += v_loss.item()
                    
                    # Calculate PSNR for the progress bar
                    v_psnr = -10.0 * torch.log10(v_loss).item()
                    val_progress.set_postfix(loss=f"{v_loss.item():.6f}", psnr=f"{v_psnr:.2f}dB")
                    
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_psnr = -10.0 * np.log10(avg_val_loss) # Overall validation PSNR
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val PSNR: {avg_val_psnr:.2f}dB")

        else:
            avg_val_loss = None
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val skipped")

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            periodic_model_path = os.path.join(checkpoint_dir, f'nerf_model_epoch_{epoch+1:06d}.pth')
            print(f"--> Saving periodic checkpoint to {periodic_model_path}")
            torch.save(model.state_dict(), periodic_model_path)

        if render_interval > 0 and val_dataset is not None and (epoch + 1) % render_interval == 0:
            model_was_training = model.training
            model.eval()
            with torch.no_grad():
                gt_image_flat, preview_transform = val_dataset[0]
                gt_image = gt_image_flat.view(val_dataset.image_height, val_dataset.image_width, 3)
                preview_image = render_validation_preview(
                    model=model,
                    dataset=val_dataset,
                    transform_matrix=preview_transform,
                    near=near,
                    far=far,
                    chunk_size=render_chunk_size,
                    num_samples=num_samples
                )
                gt_image = gt_image.to(preview_image.device)
                preview_mse = torch.mean((preview_image - gt_image) ** 2).item()
                preview_psnr = -10.0 * np.log10(max(preview_mse, 1e-12))
                preview_ssim = compute_ssim(preview_image, gt_image)

                preview_np = np.clip(preview_image.detach().cpu().numpy(), 0.0, 1.0)
                preview_uint8 = (preview_np * 255.0).astype(np.uint8)
                preview_path = os.path.join(preview_dir, f'epoch_{epoch+1:06d}.png')
                imageio.imwrite(preview_path, preview_uint8)
                print(f"--> Rendered preview @ epoch {epoch+1}: {preview_path} | PSNR: {preview_psnr:.2f}dB | SSIM: {preview_ssim:.4f}")
            if model_was_training:
                model.train()

        if avg_val_loss is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_nerf_model.pth')
            print(f"--> New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NeRF on lego or ship scene')
    parser.add_argument('--scene', choices=['ship', 'lego', 'drone1', 'drone2'], default=os.environ.get('NERF_SCENE', 'ship'))
    parser.add_argument(
        '--checkpoint-root',
        default=os.environ.get(
            'NERF_CKPT_DIR',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        ),
        help='Root directory for checkpoints (scene subfolder is appended automatically)'
    )
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('NERF_EPOCHS', 100000)))
    parser.add_argument('--rays-per-batch', type=int, default=int(os.environ.get('NERF_RAYS_PER_BATCH', 4096)))
    parser.add_argument('--num-samples', type=int, default=int(os.environ.get('NERF_NUM_SAMPLES', 100)))
    parser.add_argument('--val-interval', type=int, default=int(os.environ.get('NERF_VAL_INTERVAL', 5)))
    parser.add_argument('--save-interval', type=int, default=int(os.environ.get('NERF_SAVE_INTERVAL', 10)))
    parser.add_argument('--render-interval', type=int, default=int(os.environ.get('NERF_RENDER_INTERVAL', 10)))
    parser.add_argument('--render-chunk-size', type=int, default=int(os.environ.get('NERF_RENDER_CHUNK_SIZE', 32768)))
    parser.add_argument('--num-workers', type=int, default=int(os.environ.get('NERF_NUM_WORKERS', 4)))
    parser.add_argument('--cache-images', action='store_true', default=env_flag('NERF_CACHE_IMAGES', True))
    parser.add_argument('--no-cache-images', action='store_false', dest='cache_images')
    args = parser.parse_args()

    scene = args.scene

    base_scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nerf_lego_ship', scene)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Training scene: {scene}")

    # 1. Initialize Train Dataset and DataLoader
    train_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'train'),
        transform_path=os.path.join(base_scene_dir, 'transforms_train.json'),
        cache_images=args.cache_images
    )
    effective_num_workers = 0 if args.cache_images else args.num_workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=(torch.cuda.is_available() and not args.cache_images),
        persistent_workers=(effective_num_workers > 0)
    )

    # 2. Initialize Validation Dataset and DataLoader
    val_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'val'),
        transform_path=os.path.join(base_scene_dir, 'transforms_val.json'),
        cache_images=args.cache_images
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=(torch.cuda.is_available() and not args.cache_images),
        persistent_workers=(effective_num_workers > 0)
    ) # Validation doesn't need to be shuffled

    # 3. Initialize your model and optimizer
    nerf_model = NeRFModel().to(device)
    
    checkpoint_root = args.checkpoint_root
    checkpoint_dir = os.path.join(checkpoint_root, scene)
    start_epoch = 0

    # 4. Load Checkpoint BEFORE compiling
    latest_ckpt_path, latest_epoch = find_latest_epoch_checkpoint(checkpoint_dir)
    if latest_ckpt_path is not None:
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        with tqdm(total=1, desc="Loading checkpoint", leave=False) as pbar:
            # Load the state dict
            state_dict = torch.load(latest_ckpt_path, map_location=device)
            
            # Bulletproof fix: Strip the '_orig_mod.' prefix if it exists 
            # This allows you to load both compiled and uncompiled checkpoints safely
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            # Load into the UNCOMPILED model
            nerf_model.load_state_dict(clean_state_dict,strict=False)
            pbar.update(1)
            
        start_epoch = latest_epoch
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print(f"No periodic checkpoint found in {checkpoint_dir}. Starting from scratch.")

    # 5. NOW compile the model 
    nerf_model = torch.compile(nerf_model, mode="reduce-overhead")
    
    # 6. Initialize the optimizer AFTER compiling to ensure it tracks the right parameters
    optimizer = optim.Adam(nerf_model.parameters(), lr=5e-4) 

    # 7. Start training!
    train(
        model=nerf_model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        nerfdata=train_dataset, 
        optimizer=optimizer, 
        epochs=args.epochs,
        near=2.0,  
        far=6.0,
        checkpoint_dir=checkpoint_dir,
        val_dataset=val_dataset,
        start_epoch=start_epoch,
        rays_per_batch=args.rays_per_batch,
        num_samples=args.num_samples,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        render_interval=args.render_interval,
        render_chunk_size=args.render_chunk_size
    )