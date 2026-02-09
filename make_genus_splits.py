#!/usr/bin/env python3
"""
Split Autoarborist and iNaturalist images into LoRA, CNN train, and CNN test sets.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_GENERA = [
    "acer",
    "fraxinus",
    "quercus",
    "prunus",
    "pinus",
    "ailanthus",
    "malus",
    "pyrus",
    "picea",
    "ulmus",
    "fagus",
    "betula",
    "juniperus",
    "juglans",
    "citrus",
    "magnolia",
    "washingtonia",
    "catalpa",
    "populus",
    "thuja",
]


def list_images(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]


def copy_images(imgs: Iterable[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in imgs:
        shutil.copy2(img, out_dir / img.name)


def split_and_copy(
    dataset: str,
    genus: str,
    src_root: Path,
    output_root: Path,
    n_test: int,
    n_lora: int,
    n_cnn_train: int,
) -> None:
    if dataset == "inat":
        genus_root = src_root / genus
    else:  # autoarborist
        genus_root = src_root / genus / "images"

    if not genus_root.exists():
        print(f"[WARN] Missing {dataset}/{genus} at {genus_root}")
        return

    images = list_images(genus_root)
    if len(images) == 0:
        print(f"[WARN] No images for {dataset}/{genus}")
        return

    random.shuffle(images)

    test_imgs = images[:n_test]
    lora_imgs = images[n_test : n_test + n_lora]
    cnn_imgs = images[n_test + n_lora : n_test + n_lora + n_cnn_train]

    out_base = output_root / dataset / genus
    copy_images(test_imgs, out_base / "cnn_test")
    copy_images(lora_imgs, out_base / "lora")
    copy_images(cnn_imgs, out_base / "cnn_train")

    print(
        f"[OK] {dataset}/{genus}: "
        f"test={len(test_imgs)}, "
        f"lora={len(lora_imgs)}, "
        f"cnn_train={len(cnn_imgs)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split iNaturalist and Autoarborist datasets into genus-level sets",
    )
    parser.add_argument(
        "--inat_root",
        type=Path,
        help="Path to iNaturalist genus folder root",
    )
    parser.add_argument(
        "--aa_root",
        type=Path,
        help="Path to Autoarborist genus folder root",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("./sd-genus-images"),
        help="Output directory for split datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["inat", "autoarborist"],
        default=["inat", "autoarborist"],
        help="Which datasets to prepare",
    )
    parser.add_argument(
        "--genera",
        nargs="+",
        default=DEFAULT_GENERA,
        help="Genus list to split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--n_lora", type=int, default=100)
    parser.add_argument("--n_cnn_train", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Output root: {args.output_root}")
    print(f"Processing {len(args.genera)} genera...")

    for dataset in args.datasets:
        if dataset == "inat":
            root = args.inat_root
            if root is None:
                raise ValueError("Missing --inat_root for dataset inat")
        else:
            root = args.aa_root
            if root is None:
                raise ValueError("Missing --aa_root for dataset autoarborist")
        if not root.exists():
            raise FileNotFoundError(f"{dataset} root does not exist: {root}")

        for genus in args.genera:
            split_and_copy(
                dataset,
                genus,
                root,
                args.output_root,
                args.n_test,
                args.n_lora,
                args.n_cnn_train,
            )


if __name__ == "__main__":
    main()
