#!/usr/bin/env python3
"""
Download Stable Diffusion 1.5 model for offline use on HPC.
Run this on a machine with internet access, then copy the cache to HPC.
"""

import os
import argparse
from huggingface_hub import snapshot_download

def download_sd1_5_models(cache_dir):
    """
    Download all components of Stable Diffusion 1.5.

    Args:
        cache_dir: Directory to save model files
    """
    print(f"Cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    # Model for Stable Diffusion 1.5.
    model_ids = ["sd-legacy/stable-diffusion-v1-5"]

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
    print("ALL STABLE DIFFUSION 1.5 MODELS DOWNLOADED!")
    print("="*60)
    print(f"\nTo use offline, copy the entire directory:")
    print(f"  {cache_dir}")
    print(f"\nThen set environment variable on HPC:")
    print(f"  export HF_HOME={cache_dir}")
    print(f"  export TRANSFORMERS_CACHE={cache_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Stable Diffusion 1.5 for offline use")
    parser.add_argument("--cache_dir", type=str, default="./model_cache",
                       help="Cache directory for model files")
    args = parser.parse_args()

    download_sd1_5_models(args.cache_dir)

if __name__ == "__main__":
    main()