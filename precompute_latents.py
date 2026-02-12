#!/usr/bin/env python3
"""
Pre-compute VAE latents for Stable Diffusion datasets.

Supports:
- SD1.5 (runwayml/stable-diffusion-v1-5)
- SD3.5 (stabilityai/stable-diffusion-3.5-large / turbo)

Writes:
- latents-sd15.pt
- latents-sd35.pt

Latents are saved as float32 on CPU for stability and portability.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("precompute_latents")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------
# Helpers
# ----------------------------

def get_image_paths(data_dir: str) -> List[Path]:
    data_path = Path(data_dir)
    paths: List[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(data_path.glob(f"*{ext}"))
        paths.extend(data_path.glob(f"*{ext.upper()}"))
    return sorted(paths)


def resolve_model_path(model: str, model_cache: Optional[str]) -> Tuple[str, bool]:
    """
    Returns (path_or_id, local_files_only).
    - If model is a directory, use it.
    - Else if model_cache is provided, try to map known aliases to cache dirs.
    - Else treat as HF model id (download allowed).
    """
    # If user gave an explicit directory
    if os.path.isdir(model):
        return model, True

    # If a cache is provided, try common local folder names
    if model_cache:
        cache = Path(model_cache)

        # You can customize these to match your own cache folder names
        known = {
            "sd1.5": [
                cache / "sd-legacy_stable-diffusion-v1-5"
            ],
            "sd3.5": [
                cache / "stabilityai_stable-diffusion-3.5-large-turbo",
            ],
        }

        # If model matches one of our keys
        if model in known:
            for p in known[model]:
                if p.exists():
                    return str(p), True

        # Otherwise: if model looks like a folder name under cache, try it
        candidate = cache / model
        if candidate.exists():
            return str(candidate), True

    # Fall back to HF id
    return model, False


def load_vae(model_path_or_id: str, torch_dtype: torch.dtype, local_files_only: bool) -> AutoencoderKL:
    logger.info(f"Loading VAE from: {model_path_or_id} (local_files_only={local_files_only})")
    vae = AutoencoderKL.from_pretrained(
        model_path_or_id,
        subfolder="vae",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    vae.eval()
    vae.requires_grad_(False)
    return vae


def build_transform(resolution: int, center_crop: bool) -> transforms.Compose:
    t: List[transforms.Transform] = []
    if center_crop:
        t.append(transforms.Resize(resolution))
        t.append(transforms.CenterCrop(resolution))
    else:
        t.append(transforms.Resize((resolution, resolution)))

    t.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(t)


@torch.no_grad()
def encode_batches(
    vae: AutoencoderKL,
    image_paths: List[Path],
    transform_fn: transforms.Compose,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str], int, float]:
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    logger.info(f"VAE scaling_factor: {scaling_factor}")

    all_latents: List[torch.Tensor] = []
    all_names: List[str] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
        batch_paths = image_paths[i:i + batch_size]
        imgs: List[torch.Tensor] = []
        names: List[str] = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform_fn(img))
                names.append(p.stem)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

        if not imgs:
            continue

        pixel_values = torch.stack(imgs).to(device=device, dtype=dtype)

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * scaling_factor
        all_latents.append(latents.cpu().float())  # store fp32 on CPU
        all_names.extend(names)

        if (i // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    latents_cat = torch.cat(all_latents, dim=0)
    return latents_cat, all_names, len(all_names), scaling_factor


def save_latents(
    output_path: Path,
    latents: torch.Tensor,
    filenames: List[str],
    resolution: int,
    scaling_factor: float,
    model_id: str,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": latents,          # [N,C,H,W] float32 CPU
            "filenames": filenames,      # list[str] stems
            "resolution": resolution,
            "scaling_factor": scaling_factor,
            "model_id": model_id,
        },
        str(output_path),
    )
    size_mb = output_path.stat().st_size / 1024**2
    logger.info(f"Saved: {output_path} ({size_mb:.2f} MB)")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for SD1.5 and SD3.5")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save latents")

    parser.add_argument(
        "--model_cache",
        type=str,
        default=None,
        help="Optional local cache root containing downloaded HF models",
    )

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--center_crop", action="store_true")

    # Models to run
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sd1.5", "sd3.5"],
        help="Which models to precompute: sd1.5 sd3.5 (or explicit HF ids / paths)",
    )

    # Optional explicit model ids/paths
    parser.add_argument(
        "--sd15_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HF id or local path for SD1.5",
    )
    parser.add_argument(
        "--sd35_model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large-turbo",
        help="HF id or local path for SD3.5",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # dtype
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    image_paths = get_image_paths(args.data_dir)
    logger.info(f"Found {len(image_paths)} images in {args.data_dir}")
    if not image_paths:
        raise SystemExit(f"No images found in {args.data_dir}")

    transform_fn = build_transform(args.resolution, args.center_crop)
    out_dir = Path(args.output_dir)

    # Resolve model references
    model_map: Dict[str, str] = {
        "sd1.5": args.sd15_model,
        "sd3.5": args.sd35_model,
    }

    for m in args.models:
        if m not in {"sd1.5", "sd3.5"}:
            # allow explicit id/path via --models
            model_ref = m
            key = Path(m).name.replace("/", "_")
            out_name = f"latents-{key}.pt"
        else:
            model_ref = model_map[m]
            out_name = "latents-sd15.pt" if m == "sd1.5" else "latents-sd35.pt"

        model_path_or_id, local_only = resolve_model_path(model_ref if m in {"sd1.5", "sd3.5"} else model_ref, args.model_cache)

        vae = load_vae(model_path_or_id, torch_dtype=dtype, local_files_only=local_only).to(device)

        latents, names, n, scaling_factor = encode_batches(
            vae=vae,
            image_paths=image_paths,
            transform_fn=transform_fn,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )

        save_latents(
            output_path=out_dir / out_name,
            latents=latents,
            filenames=names,
            resolution=args.resolution,
            scaling_factor=scaling_factor,
            model_id=str(model_path_or_id),
        )

        del vae
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()