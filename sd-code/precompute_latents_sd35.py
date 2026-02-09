#!/usr/bin/env python3
"""
Pre-compute VAE latents for all images in a dataset.
This dramatically speeds up training by avoiding VAE encoding on every batch.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import AutoencoderKL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_image_paths(data_dir):
    """Get all image paths from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_paths = []

    data_path = Path(data_dir)
    for ext in image_extensions:
        image_paths.extend(data_path.glob(f'*{ext}'))
        image_paths.extend(data_path.glob(f'*{ext.upper()}'))

    return sorted(image_paths)


def precompute_latents(
    data_dir: str,
    output_dir: str,
    model_cache: str,
    resolution: int = 1024,
    batch_size: int = 8,
    mixed_precision: str = "fp16",
    center_crop: bool = True,
):
    """
    Pre-compute VAE latents for all images in data_dir.

    Args:
        data_dir: Directory containing images
        output_dir: Where to save the latents
        model_cache: Path to model cache directory
        resolution: Image resolution
        batch_size: Batch size for VAE encoding
        mixed_precision: fp16, bf16, or no
        center_crop: Whether to center crop images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set dtype
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load VAE
    logger.info("Loading VAE...")
    sd3_path = os.path.join(model_cache, "stabilityai_stable-diffusion-3.5-large")

    vae = AutoencoderKL.from_pretrained(
        sd3_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Get scaling factor
    scaling_factor = vae.config.scaling_factor
    logger.info(f"VAE scaling factor: {scaling_factor}")

    # Build transform
    transform_list = []
    if center_crop:
        transform_list.append(transforms.Resize(resolution))
        transform_list.append(transforms.CenterCrop(resolution))
    else:
        transform_list.append(transforms.Resize((resolution, resolution)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_fn = transforms.Compose(transform_list)

    # Get image paths
    image_paths = get_image_paths(data_dir)
    logger.info(f"Found {len(image_paths)} images in {data_dir}")

    if len(image_paths) == 0:
        logger.error(f"No images found in {data_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process in batches
    all_latents = []
    all_filenames = []

    logger.info(f"Encoding images to latents (batch_size={batch_size})...")

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_names = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform_fn(img)
                batch_images.append(img_tensor)
                batch_names.append(img_path.stem)  # filename without extension
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

        if len(batch_images) == 0:
            continue

        # Stack and encode
        pixel_values = torch.stack(batch_images).to(device, dtype=weight_dtype)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * scaling_factor
            # Store as fp32 for training stability
            latents = latents.cpu().float()

        all_latents.append(latents)
        all_filenames.extend(batch_names)

        # Clear cache periodically
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    # Concatenate all latents
    all_latents = torch.cat(all_latents, dim=0)
    logger.info(f"Encoded {len(all_latents)} images to latents")
    logger.info(f"Latent shape: {all_latents.shape}")

    # Save as single file for fast loading
    output_file = os.path.join(output_dir, "latents.pt")
    torch.save({
        "latents": all_latents,
        "filenames": all_filenames,
        "resolution": resolution,
        "scaling_factor": scaling_factor,
    }, output_file)

    logger.info(f"Saved latents to {output_file}")
    logger.info(f"File size: {os.path.getsize(output_file) / 1024**2:.2f} MB")

    # Clean up
    del vae
    torch.cuda.empty_cache()

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save latents")
    parser.add_argument("--model_cache", type=str,
                        default="/share/rkmeente/btfarre2/model/model_cache",
                        help="Path to model cache")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for VAE encoding (can be larger than training batch)")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--center_crop", action="store_true")

    args = parser.parse_args()

    precompute_latents(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_cache=args.model_cache,
        resolution=args.resolution,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
        center_crop=args.center_crop,
    )


if __name__ == "__main__":
    main()
