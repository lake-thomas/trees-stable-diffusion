"""Tests for image quality evaluation utilities."""

from pathlib import Path

import numpy as np
from PIL import Image

from trees_sd.evaluation import evaluate_image_quality


def _write_image(path: Path, color):
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :] = np.asarray(color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_basic_metrics_returns_expected_keys(tmp_path):
    real_dir = tmp_path / "real"
    gen_dir = tmp_path / "generated"
    real_dir.mkdir()
    gen_dir.mkdir()

    _write_image(real_dir / "real_1.png", (10, 30, 50))
    _write_image(real_dir / "real_2.png", (20, 40, 60))

    _write_image(gen_dir / "gen_1.png", (15, 35, 55))
    _write_image(gen_dir / "gen_2.png", (25, 45, 65))

    metrics = evaluate_image_quality(
        real_dir=str(real_dir),
        generated_dir=str(gen_dir),
        metrics=["basic"],
    )

    assert metrics["real_count"] == 2.0
    assert metrics["generated_count"] == 2.0
    assert metrics["real_avg_width"] == 16.0
    assert metrics["generated_avg_height"] == 16.0
    assert "fid" not in metrics
