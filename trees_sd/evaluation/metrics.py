"""Post-hoc quality evaluation utilities for generated images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_image_paths(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Image directory does not exist or is not a directory: {image_dir}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported image files found in: {image_dir}")
    return paths


def _compute_basic_image_stats(real_paths: List[Path], generated_paths: List[Path]) -> Dict[str, float]:
    def summarize(paths: List[Path]) -> Dict[str, float]:
        widths = []
        heights = []
        means = []

        for path in paths:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                heights.append(arr.shape[0])
                widths.append(arr.shape[1])
                means.append(arr.mean(axis=(0, 1)))

        mean_rgb = np.stack(means, axis=0).mean(axis=0)
        return {
            "count": float(len(paths)),
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
        }

    real_stats = summarize(real_paths)
    gen_stats = summarize(generated_paths)

    return {
        "real_count": real_stats["count"],
        "generated_count": gen_stats["count"],
        "real_avg_width": real_stats["avg_width"],
        "real_avg_height": real_stats["avg_height"],
        "generated_avg_width": gen_stats["avg_width"],
        "generated_avg_height": gen_stats["avg_height"],
        "real_mean_r": real_stats["mean_r"],
        "real_mean_g": real_stats["mean_g"],
        "real_mean_b": real_stats["mean_b"],
        "generated_mean_r": gen_stats["mean_r"],
        "generated_mean_g": gen_stats["mean_g"],
        "generated_mean_b": gen_stats["mean_b"],
    }


def _compute_fid_score(
    real_paths: List[Path],
    generated_paths: List[Path],
    batch_size: int,
    device: str,
    num_workers: int,
) -> float:
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "FID requires `torchmetrics` and `torchvision` extras. Install them with "
            "`pip install torchmetrics torchvision`."
        ) from exc

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def load_batch(paths: List[Path], start: int, end: int) -> "torch.Tensor":
        tensors = []
        for path in paths[start:end]:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB").resize((299, 299), Image.Resampling.BILINEAR))
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.uint8))
        return torch.stack(tensors, dim=0)

    for i in range(0, len(real_paths), batch_size):
        batch = load_batch(real_paths, i, i + batch_size).to(device)
        metric.update(batch, real=True)

    for i in range(0, len(generated_paths), batch_size):
        batch = load_batch(generated_paths, i, i + batch_size).to(device)
        metric.update(batch, real=False)

    _ = num_workers  # reserved for future dataloader-based implementation
    return float(metric.compute().cpu().item())


def evaluate_image_quality(
    real_dir: str,
    generated_dir: str,
    metrics: List[str] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> Dict[str, float]:
    """Evaluate generated images against real dataset images."""
    requested_metrics = metrics or ["fid", "basic"]

    real_paths = _collect_image_paths(real_dir)
    generated_paths = _collect_image_paths(generated_dir)

    results: Dict[str, float] = {}

    if "basic" in requested_metrics:
        results.update(_compute_basic_image_stats(real_paths, generated_paths))

    if "fid" in requested_metrics:
        results["fid"] = _compute_fid_score(
            real_paths=real_paths,
            generated_paths=generated_paths,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
        )

    return results
