import torch
import numpy as np
import json
import os
import argparse
import csv
import imageio
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as skimage_structural_similarity
except ImportError:
    skimage_structural_similarity = None

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
    if skimage_structural_similarity is not None:
        pred_np = np.clip(pred_img.detach().float().cpu().numpy(), 0.0, 1.0)
        gt_np = np.clip(gt_img.detach().float().cpu().numpy(), 0.0, 1.0)
        return float(skimage_structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))

    pred = pred_img.detach().float().cpu().clamp(0.0, 1.0)
    gt = gt_img.detach().float().cpu().clamp(0.0, 1.0)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = pred.mean(dim=(0, 1))
    mu_gt = gt.mean(dim=(0, 1))

    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    sigma_pred = (pred_centered ** 2).mean(dim=(0, 1))
    sigma_gt = (gt_centered ** 2).mean(dim=(0, 1))
    sigma_cross = (pred_centered * gt_centered).mean(dim=(0, 1))

    numerator = (2 * mu_pred * mu_gt + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred + sigma_gt + c2)
    ssim_per_channel = numerator / (denominator + 1e-12)
    return float(ssim_per_channel.mean().item())


def reconstruct_scene(scene, checkpoint_root, chunk_size, output_root, near=2.0, far=6.0, compile_model=True, use_positional_encoding=True, run_tag='', zero_pe_features=False, checkpoint_name='best_nerf_model.pth'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_scene_dir = os.path.join(base_dir, 'nerf_lego_ship', scene)
    scene_root_dir = os.path.join(base_dir, checkpoint_root, scene)
    checkpoint_scene_dir = os.path.join(scene_root_dir, run_tag) if run_tag else scene_root_dir
    checkpoint_path = os.path.join(checkpoint_scene_dir, checkpoint_name)
    fallback_checkpoint_path = os.path.join(scene_root_dir, checkpoint_name)

    if not os.path.isfile(checkpoint_path):
        if run_tag and os.path.isfile(fallback_checkpoint_path):
            print(f"Tagged checkpoint not found at {checkpoint_path}")
            print(f"Falling back to scene checkpoint: {fallback_checkpoint_path}")
            checkpoint_path = fallback_checkpoint_path
        else:
            raise FileNotFoundError(
                f"Checkpoint not found. Tried: {checkpoint_path}"
                + (f" and {fallback_checkpoint_path}" if run_tag else "")
            )
    output_dir = os.path.join(base_dir, output_root, scene, run_tag) if run_tag else os.path.join(base_dir, output_root, scene)

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

    model = NeRFModel(
        use_positional_encoding=use_positional_encoding,
        zero_pe_features=zero_pe_features
    ).to(device)

    if zero_pe_features and not use_positional_encoding:
        raise ValueError("--zero-pe-features requires positional encoding architecture; do not combine with --no-positional-encoding")

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
    per_frame_metrics = []

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
        mse_value = mse.item()
        psnr_value = -10.0 * np.log10(max(mse_value, 1e-12))
        ssim = compute_ssim(img_tensor, gt_image)
        total_test_mse += mse_value
        total_test_ssim += ssim
        per_frame_metrics.append({
            'frame_index': i,
            'mse': mse_value,
            'psnr': psnr_value,
            'ssim': ssim,
        })

        frame_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, img_uint8)
        frames.append(img_uint8)

    video_path = os.path.join(output_dir, f'{scene}_render.mp4')
    print(f"Saving video to {video_path}")
    imageio.mimwrite(video_path, frames, fps=30, quality=8)

    avg_test_mse = total_test_mse / len(test_meta['frames'])
    avg_test_psnr = -10.0 * np.log10(max(avg_test_mse, 1e-12))
    avg_test_ssim = total_test_ssim / len(test_meta['frames'])

    metrics_csv_path = os.path.join(output_dir, 'metrics_per_frame.csv')
    with open(metrics_csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['frame_index', 'mse', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(per_frame_metrics)

    summary_path = os.path.join(output_dir, 'metrics_summary.json')
    summary = {
        'scene': scene,
        'run_tag': run_tag,
        'use_positional_encoding': use_positional_encoding,
        'zero_pe_features': zero_pe_features,
        'checkpoint_path': checkpoint_path,
        'output_dir': output_dir,
        'num_frames': len(test_meta['frames']),
        'avg_test_mse': avg_test_mse,
        'avg_test_psnr': avg_test_psnr,
        'avg_test_ssim': avg_test_ssim,
        'video_path': video_path,
        'metrics_per_frame_csv': metrics_csv_path,
    }
    with open(summary_path, 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\n" + "="*40)
    print(f"FINAL EVALUATION METRICS ({scene})")
    print(f"Average Test MSE:  {avg_test_mse:.6f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")
    print(f"Per-frame metrics: {metrics_csv_path}")
    print(f"Summary metrics:   {summary_path}")
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
        'metrics_per_frame_csv': metrics_csv_path,
        'metrics_summary_json': summary_path,
    }

def main():
    parser = argparse.ArgumentParser(description='Render novel views from trained NeRF')
    parser.add_argument('--scene', choices=['ship', 'lego', 'drone1', 'drone2'], default='ship')
    parser.add_argument('--checkpoint-root', default='checkpoints', help='Root checkpoint directory containing scene subfolders')
    parser.add_argument('--checkpoint-name', default='best_nerf_model.pth', help='Name of the checkpoint file to load')
    parser.add_argument('--output-root', default='renders', help='Root output directory for reconstructed renders')
    parser.add_argument('--chunk-size', type=int, default=32768, help='Number of rays to process at once')
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--run-tag', default='', help='Optional subfolder tag under scene for checkpoints/outputs')
    parser.add_argument('--no-positional-encoding', action='store_true', help='Use model variant without positional encoding')
    parser.add_argument('--zero-pe-features', action='store_true', help='Inference-only: zero sin/cos PE channels while keeping xyz/view base channels')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile during rendering')
    args = parser.parse_args()
    reconstruct_scene(
        scene=args.scene,
        checkpoint_root=args.checkpoint_root,
        chunk_size=args.chunk_size,
        output_root=args.output_root,
        near=args.near,
        far=args.far,
        compile_model=(not args.no_compile),
        use_positional_encoding=(not args.no_positional_encoding),
        run_tag=args.run_tag,
        zero_pe_features=args.zero_pe_features,
        checkpoint_name=args.checkpoint_name
    )

if __name__ == "__main__":
    main()