import torch
import numpy as np
import json
import os
import argparse
import imageio
from tqdm import tqdm

from Wrapper import NeRFModel, NerfDataset, volume_rendering

def render_image(model, dataset, transform_matrix, H, W, chunk_size, near, far, device):
    """
    Renders a single image by processing rays in manageable chunks.
    """
    # 1. Create a grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    u = u.flatten()
    v = v.flatten()

    # 2. Get ray origins and directions for EVERY pixel
    rays_cam = dataset.image2cam(u=u, v=v)
    ray_o, ray_d = dataset.cam2world(rays_cam, transform_matrix)

    ray_o = ray_o.to(device)
    ray_d = ray_d.to(device)

    all_colors = []

    # 3. Process rays in chunks to avoid GPU OOM errors
    with torch.no_grad():
        for i in range(0, ray_o.shape[0], chunk_size):
            chunk_ray_o = ray_o[i : i + chunk_size]
            chunk_ray_d = ray_d[i : i + chunk_size]
            
            chunk_color = volume_rendering(model, chunk_ray_o, chunk_ray_d, near, far)
            all_colors.append(chunk_color.cpu())

    # 4. Reconstruct the 2D image from the flattened color chunks
    img_flat = torch.cat(all_colors, dim=0)
    img = img_flat.view(H, W, 3).numpy()
    
    # Ensure values are in [0, 1] range before converting to uint8
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Render novel views from trained NeRF')
    parser.add_argument('--scene', choices=['ship', 'lego'], default='ship')
    parser.add_argument('--chunk-size', type=int, default=4096, help='Number of rays to process at once')
    args = parser.parse_args()

    scene = args.scene
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_scene_dir = os.path.join(base_dir, 'nerf_lego_ship', scene)
    checkpoint_path = os.path.join(base_dir, 'checkpoints', scene, 'best_nerf_model.pth')
    output_dir = os.path.join(base_dir, 'renders', scene)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rendering on device: {device}")

    # 1. Load the Test Transforms
    # We use the dataset class to get the focal length and helper functions, 
    # pointing it at the test folder.
    test_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'test'),
        transform_path=os.path.join(base_scene_dir, 'transforms_test.json')
    )
    
    H, W = test_dataset.image_height, test_dataset.image_width

    # 2. Load the Model
    model = NeRFModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Read the Test Poses
    with open(os.path.join(base_scene_dir, 'transforms_test.json'), 'r') as f:
        test_meta = json.load(f)

    frames = []
    
    print(f"Rendering {len(test_meta['frames'])} frames...")
    
    # 4. Render loop
    for i, frame in enumerate(tqdm(test_meta['frames'])):
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
        img = render_image(
            model=model, 
            dataset=test_dataset, 
            transform_matrix=transform_matrix, 
            H=H, W=W, 
            chunk_size=args.chunk_size, 
            near=2.0, far=6.0, 
            device=device
        )
        
        # Save individual frame
        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, img)
        frames.append(img)

    # 5. Compile into a video
    video_path = os.path.join(output_dir, f'{scene}_render.mp4')
    print(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print("Done!")

if __name__ == "__main__":
    main()