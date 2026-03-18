#write your own data loader, parser, network and loss function for this phase.

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class NerfDataset(Dataset):
    def __init__(self, data_dir, transform_path, transform=None):
        self.data_dir = data_dir
        self.transform_path = transform_path
        
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
        for f in all_files:
            
            basename = os.path.splitext(f)[0]
            if basename in self.transform_map:
                self.image_paths.append(os.path.join(data_dir, f))

    
        sample_image = Image.open(self.image_paths[0])
        self.image_width, self.image_height = sample_image.size
        self.focal_length = self.focal_length_from_fov(self.fov, self.image_width)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract "r_0" from the absolute path
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Instant O(1) lookup! No more for loops!
        transform = self.transform_map[basename]
        
        image = torch.from_numpy(np.array(image) / 255.0).float() 
        transform = torch.tensor(transform, dtype=torch.float32)
        
        return image, transform
    
    
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
    def __init__(self):
        super(NeRFModel, self).__init__()
        # TODO : Define the architecture of the NeRF model (e.g., MLP with positional encoding)
        self.fc1 = nn.Linear(63, 256)  # Input: ray direction (x, y, z)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(319, 256) # add skip connection here, add pose.
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 257)
        self.fc9 = nn.Linear(283, 256)# no relu , add 3D.
        self.fc10 = nn.Linear(256, 128)
        self.fc11 = nn.Linear(128, 3)    # Output: RGB color


    def positional_encoding(self, x, num_freqs=10):
        # TODO : Implement positional encoding for the input coordinates
        pe = [x]
        for i in range(num_freqs):
            for fn in [torch.sin, torch.cos]:
                pe.append(fn((2.0 ** i) * x))
        return torch.cat(pe, dim=-1)
    
    def forward(self, pts, view_dir):
        rays_63= self.positional_encoding(pts)  # Apply positional encoding to the input pts
        x = F.relu(self.fc1(rays_63))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.cat([rays_63, F.relu(self.fc4(x))], dim=-1) # skip connection, adds PE
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        sigma = F.relu(x[..., 0])  # Extract density value (first element of the output)
        view_dir_20 = self.positional_encoding(view_dir, num_freqs=4)  # Apply positional encoding to the input view_dir
        x = self.fc9((torch.cat([ x[..., 1:257], view_dir_20], dim=-1))) # skip connection, adds PE and 3D and density
        x = F.relu(self.fc10(x))
        
        output = torch.sigmoid(self.fc11(x))  # Output color in range [0, 1]
        return output,sigma



def volume_rendering(model, ray_o, ray_d, near, far):
    # Implement volume rendering to compute the final pixel color from the sampled points along the ray
    # TODO : Implement volume rendering using the NeRF model's output (e.g., using alpha compositing)
    t = torch.linspace(near, far, steps=100, device=device).unsqueeze(0).unsqueeze(-1) # Number of samples along each ray
    ray_d_math = ray_d.unsqueeze(1) # Expand ray_d to match the shape of t
    ray_o_math = ray_o.unsqueeze(1) # Expand ray_o to match the shape of t
    rays_t_math = ray_o_math + t * ray_d_math  # Sample points along the ray (assuming step size of 0.01)

    ray_d_expanded= ray_d.unsqueeze(1).expand(-1, 100, -1) # Expand ray_d to match the batch size

    rgb,sigma = model(rays_t_math, ray_d_expanded)  # Get RGB and density from the model

    # Compute alpha values from density
    step_size = (far - near) / 100
    alpha = 1.0 - torch.exp(-sigma * step_size)

    # Compute weights for alpha compositing
    ones = torch.ones((alpha.shape[0], 1), device=device)
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

def train(model, train_dataloader,val_dataloader,nerfdata,  optimizer,epochs,  near, far, checkpoint_dir):
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")
        for images, transforms in train_dataloader:
            optimizer.zero_grad()
            images= images.squeeze(0) # remove batch dimension
            transforms = transforms.squeeze(0) # remove batch dimension
            num_of_pixels = images.shape[0] * images.shape[1]

            batch_size = 4096 
            indices = torch.randperm(num_of_pixels)[:batch_size]

            image_flatten= images.view(-1,3)
            colors = image_flatten[indices] # Get the true colors for the sampled pixels

            r, c = torch.unravel_index(indices, images.shape[:2])

            
            ray = nerfdata.image2cam(u=c, v=r)  # Assuming focal length is 1.0 for simplicity
            ray_o, ray_d = nerfdata.cam2world(ray, transforms)
        
            ray_o = ray_o.to(device)
            ray_d = ray_d.to(device)
            colors = colors.to(device)
            
            
            pixel_color = volume_rendering(model, ray_o, ray_d, near, far)
            loss = loss_function(pixel_color, colors)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_images, val_transforms in val_dataloader:
                val_images = val_images.squeeze(0)  # remove batch dimension
                val_transforms = val_transforms.squeeze(0)  # remove batch dimension

                val_indices = torch.randperm(val_images.shape[0] * val_images.shape[1])[:batch_size]
                val_colors = val_images.view(-1, 3)[val_indices]
                v_r, v_c = torch.unravel_index(val_indices, val_images.shape[:2])
                
                v_ray = nerfdata.image2cam(u=v_c, v=v_r)
                v_ray_o, v_ray_d = nerfdata.cam2world(v_ray, val_transforms)
                
                v_ray_o = v_ray_o.to(device)
                v_ray_d = v_ray_d.to(device)
                val_colors = val_colors.to(device)
                
                v_pixel_color = volume_rendering(model, v_ray_o, v_ray_d, near, far)
                v_loss = loss_function(v_pixel_color, val_colors)
                total_val_loss += v_loss.item()
                
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if (epoch + 1) % 100 == 0:
            periodic_model_path = os.path.join(checkpoint_dir, f'nerf_model_epoch_{epoch+1:06d}.pth')
            print(f"--> Saving periodic checkpoint to {periodic_model_path}")
            torch.save(model.state_dict(), periodic_model_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_nerf_model.pth')
            print(f"--> New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NeRF on lego or ship scene')
    parser.add_argument('--scene', choices=['ship', 'lego'], default=os.environ.get('NERF_SCENE', 'ship'))
    parser.add_argument(
        '--checkpoint-root',
        default=os.environ.get(
            'NERF_CKPT_DIR',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        ),
        help='Root directory for checkpoints (scene subfolder is appended automatically)'
    )
    args = parser.parse_args()

    scene = args.scene

    base_scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nerf_lego_ship', scene)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Training scene: {scene}")

    # 1. Initialize Train Dataset and DataLoader
    train_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'train'),
        transform_path=os.path.join(base_scene_dir, 'transforms_train.json')
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 2. Initialize Validation Dataset and DataLoader
    val_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'val'),
        transform_path=os.path.join(base_scene_dir, 'transforms_val.json')
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Validation doesn't need to be shuffled

    # 3. Initialize your model and optimizer
    nerf_model = NeRFModel().to(device)
    optimizer = optim.Adam(nerf_model.parameters(), lr=5e-4) 
    checkpoint_root = args.checkpoint_root
    checkpoint_dir = os.path.join(checkpoint_root, scene)

    # 4. Start training!
    train(
        model=nerf_model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, # Pass the new validation dataloader
        nerfdata=train_dataset, 
        optimizer=optimizer, 
        epochs=100000,   # Bumped up epochs
        near=2.0,  
        far=6.0,
        checkpoint_dir=checkpoint_dir
    )