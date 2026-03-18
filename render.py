import torch
import numpy as np
import json
import os
import argparse
import imageio
from tqdm import tqdm
from skimage.metrics import structural_similarity

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

    # Optimization 1: Use Automatic Mixed Precision for faster rendering on CUDA
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
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


def compute_ssim(pred_img, gt_img):
    pred_np = np.clip(pred_img.detach().float().cpu().numpy(), 0.0, 1.0)
    gt_np = np.clip(gt_img.detach().float().cpu().numpy(), 0.0, 1.0)
    return float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))


def reconstruct_scene(scene, checkpoint_root, chunk_size, output_root, near=2.0, far=6.0, compile_model=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_scene_dir = os.path.join(base_dir, 'nerf_lego_ship', scene)
    checkpoint_path = os.path.join(base_dir, checkpoint_root, scene, 'best_nerf_model.pth')
    output_dir = os.path.join(base_dir, output_root, scene)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rendering on device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    test_dataset = NerfDataset(
        data_dir=os.path.join(base_scene_dir, 'test'),
        transform_path=os.path.join(base_scene_dir, 'transforms_test.json'),
        cache_images=True
    )
    H, W = test_dataset.image_height, test_dataset.image_width

    model = NeRFModel().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    model.eval()

    if compile_model:
        print("Compiling model for fast inference...")
        model = torch.compile(model, mode="reduce-overhead")

    with open(os.path.join(base_scene_dir, 'transforms_test.json'), 'r') as f:
        test_meta = json.load(f)

    frames = []
    total_test_mse = 0.0
    total_test_ssim = 0.0

    print(f"Rendering {len(test_meta['frames'])} frames and calculating Test PSNR/SSIM...")

    for i, frame in enumerate(tqdm(test_meta['frames'])):
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

        img_uint8, img_tensor = render_image(
            model=model, dataset=test_dataset, transform_matrix=transform_matrix,
            H=H, W=W, chunk_size=chunk_size, near=near, far=far, device=device
        )

        gt_image, _ = test_dataset[i]
        gt_image = gt_image.cpu().view(H, W, 3)

        mse = torch.mean((img_tensor - gt_image) ** 2)
        ssim = compute_ssim(img_tensor, gt_image)
        total_test_mse += mse.item()
        total_test_ssim += ssim

        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, img_uint8)
        frames.append(img_uint8)

    video_path = os.path.join(output_dir, f'{scene}_render.mp4')
    print(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)

    avg_test_mse = total_test_mse / len(test_meta['frames'])
    avg_test_psnr = -10.0 * np.log10(max(avg_test_mse, 1e-12))
    avg_test_ssim = total_test_ssim / len(test_meta['frames'])

    print("\n" + "="*40)
    print(f"FINAL EVALUATION METRICS ({scene})")
    print(f"Average Test MSE:  {avg_test_mse:.6f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")
    print("="*40 + "\n")
    print("Done!")

    return {
        'scene': scene,
        'checkpoint_path': checkpoint_path,
        'output_dir': output_dir,
        'avg_test_mse': avg_test_mse,
        'avg_test_psnr': avg_test_psnr,
        'avg_test_ssim': avg_test_ssim,
        'video_path': video_path,
    }

def main():
    parser = argparse.ArgumentParser(description='Render novel views from trained NeRF')
    parser.add_argument('--scene', choices=['ship', 'lego', 'drone1', 'drone2'], default='ship')
    parser.add_argument('--checkpoint-root', default='checkpoints', help='Root checkpoint directory containing scene subfolders')
    parser.add_argument('--output-root', default='renders', help='Root output directory for reconstructed renders')
    parser.add_argument('--chunk-size', type=int, default=32768, help='Number of rays to process at once')
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile during rendering')
    args = parser.parse_args()
    reconstruct_scene(
        scene=args.scene,
        checkpoint_root=args.checkpoint_root,
        chunk_size=args.chunk_size,
        output_root=args.output_root,
        near=args.near,
        far=args.far,
        compile_model=(not args.no_compile)
    )

if __name__ == "__main__":
    main()