"""CLI for unified SD image generation."""

import argparse
import json
import sys

from trees_sd.generation import generate_images_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from trained Trees SD LoRA checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--model_version", choices=["sd1.5", "sd3.5"], default=None)
    parser.add_argument("--dataset_type", choices=["inaturalist", "autoarborist"], default=None)
    parser.add_argument("--num_images", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--resolution", type=int, default=512)

    args = parser.parse_args()

    try:
        results = generate_images_from_config(
            config_path=args.config,
            num_images=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            resolution=args.resolution,
            model_version=args.model_version,
            dataset_type=args.dataset_type,
        )
    except Exception as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
