#!/usr/bin/env python3
"""
Download Stable Diffusion 1.5 and 3.5 model assets for offline use.
Run this on a machine with internet access, then copy the cache to HPC.
"""

import argparse
import os
from typing import Iterable, List

from huggingface_hub import snapshot_download


SD15_MODELS = ["sd-legacy/stable-diffusion-v1-5"]
SD35_MODELS = [
    "stabilityai/stable-diffusion-3.5-large",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "google/t5-v1_1-xxl",
]


def _download_models(model_ids: Iterable[str], cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for model_id in model_ids:
        print(f"\n{'=' * 60}")
        print(f"Downloading model: {model_id}")
        print(f"{'=' * 60}\n")
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, model_id.replace("/", "_")),
            local_dir_use_symlinks=False,
        )
        print(f"\n✓ {model_id} downloaded successfully!")


def download_all(cache_dir: str, model_versions: List[str]) -> None:
    print(f"Cache directory: {cache_dir}")
    if "sd1.5" in model_versions:
        _download_models(SD15_MODELS, cache_dir)
    if "sd3.5" in model_versions:
        _download_models(SD35_MODELS, cache_dir)

    print("\n" + "=" * 60)
    print("ALL REQUESTED MODELS DOWNLOADED!")
    print("=" * 60)
    print("\nTo use offline, copy the entire directory:")
    print(f"  {cache_dir}")
    print("\nThen set environment variables on HPC:")
    print(f"  export HF_HOME={cache_dir}")
    print(f"  export TRANSFORMERS_CACHE={cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Stable Diffusion 1.5 and 3.5 models for offline use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./model_cache",
        help="Cache directory for model files",
    )
    parser.add_argument(
        "--model_versions",
        nargs="+",
        choices=["sd1.5", "sd3.5"],
        default=["sd1.5", "sd3.5"],
        help="Which model families to download",
    )
    args = parser.parse_args()

    download_all(args.cache_dir, args.model_versions)


if __name__ == "__main__":
    main()
