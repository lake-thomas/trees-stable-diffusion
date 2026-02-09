#!/usr/bin/env python3
"""
Download everything needed for offline HPC use.
Run this once on a machine with internet, then copy to HPC.
"""

import os
from diffusers import StableDiffusionPipeline

print("="*60)
print("DOWNLOADING STABLE DIFFUSION MODEL FOR OFFLINE USE")
print("="*60)

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
output_dir = "./sd_model_offline"

print(f"\nModel: {model_id}")
print(f"Output: {output_dir}")
print("\nThis will download ~5-6GB of model files...")
print("="*60)

# Download entire pipeline
print("\nDownloading full pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir="./hf_cache",  # Also cache in HF format
)

# Save to directory
print(f"\nSaving to {output_dir}...")
pipe.save_pretrained(output_dir)

print("\n" + "="*60)
print("✓ DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nFiles saved to:")
print(f"  1. {output_dir}/          (main model)")
print(f"  2. ./hf_cache/             (HuggingFace cache)")
print(f"\nNext steps:")
print(f"  1. Copy {output_dir}/ to HPC:")
print(f"     scp -r {output_dir} btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/model/stable_diffusion_v1_5")
print(f"  2. Copy hf_cache/ to HPC:")
print(f"     scp -r hf_cache btfarre2@login.hpc.ncsu.edu:/share/rkmeente/btfarre2/hf_cache")
print("="*60)
