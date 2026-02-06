"""CLI for post-hoc generated image quality evaluation."""

import argparse
import json
import sys

from trees_sd.evaluation import evaluate_image_quality


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated image quality against real datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real_dir", required=True, help="Directory with reference dataset images")
    parser.add_argument("--generated_dir", required=True, help="Directory with generated images")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["fid", "basic"],
        default=["fid", "basic"],
        help="Metrics to compute",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for metric computation")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu, cuda, cuda:0)")
    parser.add_argument("--num_workers", type=int, default=0, help="Reserved for dataloader worker count")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON report")

    args = parser.parse_args()

    try:
        results = evaluate_image_quality(
            real_dir=args.real_dir,
            generated_dir=args.generated_dir,
            metrics=args.metrics,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        print(f"Error while evaluating images: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
