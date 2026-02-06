"""
Python script to split Autoarborist and iNaturalist images for lora fine-tuning, cnn training, and cnn testing.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List

# ---------------- CONFIG ---------------- #

SEED = 42
N_TEST = 10
N_LORA = 10
N_CNN_TRAIN = 10

SD_ROOT = Path.home() / "Desktop" / "sd3.5"
OUTPUT_ROOT = SD_ROOT / "sd-genus-images"

INAT_ROOT = Path(r"Y:\inat_original_full")
AA_ROOT = Path(
    r"C:\Users\talake2\Desktop\auto_arborist_cvpr2022_v015"
    r"\jpegs_streetlevel_genus_idx_label\train"
)

GENERA = [
    "acer", "fraxinus", "quercus", "prunus", "pinus",
    "ailanthus", "malus", "pyrus", "picea", "ulmus",
    "fagus", "betula", "juniperus", "juglans", "citrus",
    "magnolia", "washingtonia", "catalpa", "populus", "thuja"
]

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------- #


def list_images(root: Path) -> List[Path]:
    return [
        p for p in root.iterdir()
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]


def copy_images(imgs: List[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in imgs:
        shutil.copy2(img, out_dir / img.name)


def split_and_copy(dataset: str, genus: str, src_root: Path):
    if dataset == "inat":
        genus_root = src_root / genus
    else:  # aa
        genus_root = src_root / genus / "images"

    if not genus_root.exists():
        print(f"[WARN] Missing {dataset}/{genus}")
        return

    images = list_images(genus_root)
    if len(images) == 0:
        print(f"[WARN] No images for {dataset}/{genus}")
        return

    random.shuffle(images)

    test_imgs = images[:N_TEST]
    lora_imgs = images[N_TEST:N_TEST + N_LORA]
    cnn_imgs = images[N_TEST + N_LORA:N_TEST + N_LORA + N_CNN_TRAIN]

    out_base = OUTPUT_ROOT / dataset / genus

    copy_images(test_imgs, out_base / "cnn_test")
    copy_images(lora_imgs, out_base / "lora")
    copy_images(cnn_imgs, out_base / "cnn_train")

    print(
        f"[OK] {dataset}/{genus}: "
        f"test={len(test_imgs)}, "
        f"lora={len(lora_imgs)}, "
        f"cnn_train={len(cnn_imgs)}"
    )


def main():
    random.seed(SEED)

    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Processing {len(GENERA)} genera...")
    for genus in GENERA:
        split_and_copy("inat", genus, INAT_ROOT)
        split_and_copy("aa", genus, AA_ROOT)


if __name__ == "__main__":
    main()

