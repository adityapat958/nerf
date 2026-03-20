import argparse

from render import reconstruct_scene


def main():
    parser = argparse.ArgumentParser(description='Reconstruct test views using best NeRF checkpoint')
    parser.add_argument('--scene', choices=['ship', 'lego', 'drone1', 'drone2'], required=True)
    parser.add_argument('--checkpoint-root', default='checkpoints', help='Root checkpoint directory containing scene subfolders')
    parser.add_argument('--output-root', default='reconstructions_best', help='Root output directory for best-model reconstructions')
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
        zero_pe_features=args.zero_pe_features
    )


if __name__ == '__main__':
    main()
