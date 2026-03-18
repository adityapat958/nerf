import torch
import numpy as np
import json
import os
import argparse
import imageio
from tqdm import tqdm

from Wrapper import NeRFModel, NerfDataset, volume_rendering

@torch.no_grad()
def render_image(model, dataset, transform_matrix, H, W, chunk_size, near, far, device):
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    u = u.flatten()
    v = v.flatten()

    rays_cam = dataset.image2cam(u=u, v=v)
    ray_o, ray_d = dataset.cam2world(rays_cam, transform_matrix)

    ray_o = ray_o.to(device)
    ray_d = ray_d.to(device)

    all_colors = []

    # Optimization 1: Use Automatic Mixed Precision for 2x faster rendering
    with torch.amp.autocast('cuda'):
        for i in range(0, ray_o.shape[0], chunk_size):
            chunk_ray_o = ray_o[i : i + chunk_size]
            chunk_ray_d = ray_d[i : i + chunk_size]
            
            chunk_color = volume_rendering(model, chunk_ray_o, chunk_ray_d, near, far)
            all_colors.append(chunk_color.cpu())

    img_flat = torch.cat(all_colors, dim=0)
    img_tensor = img_flat.view(H, W, 3) # Keep as tensor for PSNR calculation
    
    img_np = np.clip(img_tensor.numpy(), 0.0, 1.0)
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    
    return img_uint8, img_tensor

def main():
    parser = argparse.ArgumentParser(description='Render novel views from trained NeRF')
    parser.add_argument('--scene', choices=['ship', 'lego'], default='ship')
    # Optimization 2: Massive chunk size for RTX 6000
    parser.add_argument('--chunk-size', type=int, default=32768, help='Number of rays to process at once')
    args = parser.parse_args()

    scene = args.scene
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_scene_dir = os.path.join(base_dir, 'nerf_lego_ship', scene)
    checkpoint_path = os.path.join(base_dir, 'checkpoints', scene, 'best_nerf_model.pth')
    output_dir = os.path.join(base_dir, 'renders', scene)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rendering on device: {device}")

    # Load the Test Dataset (This gives us access to Ground Truth test images!)
    test_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'test'),
        transform_path=os.path.join(base_scene_dir, 'transforms_test.json'),
        cache_images=True # Load into RAM for fast PSNR comparison
    )
    H, W = test_dataset.image_height, test_dataset.image_width

    # Load the Model
    model = NeRFModel().to(device)
    
    # Safely load weights (stripping compile prefix if needed)
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    
    model.eval()
    
    # Optimization 3: Compile the model for inference speed!
    print("Compiling model for fast inference...")
    model = torch.compile(model, mode="reduce-overhead")

    with open(os.path.join(base_scene_dir, 'transforms_test.json'), 'r') as f:
        test_meta = json.load(f)

    frames = []
    total_test_mse = 0.0
    
    print(f"Rendering {len(test_meta['frames'])} frames and calculating Test PSNR...")
    
    for i, frame in enumerate(tqdm(test_meta['frames'])):
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
        # Render the image
        img_uint8, img_tensor = render_image(
            model=model, dataset=test_dataset, transform_matrix=transform_matrix, 
            H=H, W=W, chunk_size=args.chunk_size, near=2.0, far=6.0, device=device
        )
        
        # Calculate PSNR against the Ground Truth image from the test set
        # test_dataset[i] returns (image_tensor, transform_tensor)
        gt_image, _ = test_dataset[i] 
        gt_image = gt_image.cpu().view(H, W, 3)
        
        # Compute MSE for this image and add to total
        mse = torch.mean((img_tensor - gt_image) ** 2)
        total_test_mse += mse.item()
        
        # Save individual frame
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, img_uint8)
        frames.append(img_uint8)

    # Compile into a video
    video_path = os.path.join(output_dir, f'{scene}_render.mp4')
    print(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    
    # Output final metrics
    avg_test_mse = total_test_mse / len(test_meta['frames'])
    avg_test_psnr = -10.0 * np.log10(avg_test_mse)
    
    print("\n" + "="*40)
    print(f"FINAL EVALUATION METRICS ({scene})")
    print(f"Average Test MSE:  {avg_test_mse:.6f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print("="*40 + "\n")
    print("Done!")

if __name__ == "__main__":
    main()