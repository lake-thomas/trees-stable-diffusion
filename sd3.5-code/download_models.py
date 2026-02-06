#!/usr/bin/env python3
"""
Download Stable Diffusion 3.5 Large models for offline use on HPC.
Run this on a machine with internet access, then copy the cache to HPC.
"""

import os
import argparse
from huggingface_hub import snapshot_download

def download_sd3_models(cache_dir):
    """
    Download all components of Stable Diffusion 3.5 Large.

    Args:
        cache_dir: Directory to save model files
    """
    print(f"Cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    # All the model IDs for Stable Diffusion 3.5 Large (BEST QUALITY!)
    model_ids = [
        "stabilityai/stable-diffusion-3.5-large",
        "openai/clip-vit-large-patch14",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "google/t5-v1_1-xxl",
    ]

    for model_id in model_ids:
        print(f"\n{'='*60}")
        print(f"Downloading model: {model_id}")
        print(f"{'='*60}\n")
        
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, model_id.replace("/", "_")),
            local_dir_use_symlinks=False  # Recommended for easier transfer
        )
        
        print(f"\n✓ {model_id} downloaded successfully!")

    print("\n" + "="*60)
    print("ALL STABLE DIFFUSION 3.5 LARGE MODELS DOWNLOADED!")
    print("="*60)
    print(f"\nTo use offline, copy the entire directory:")
    print(f"  {cache_dir}")
    print(f"\nThen set environment variable on HPC:")
    print(f"  export HF_HOME={cache_dir}")
    print(f"  export TRANSFORMERS_CACHE={cache_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Stable Diffusion 3.5 Large models for offline use")
    parser.add_argument("--cache_dir", type=str, default="./model_cache",
                       help="Cache directory for model files")
    args = parser.parse_args()

    download_sd3_models(args.cache_dir)

if __name__ == "__main__":
    main()