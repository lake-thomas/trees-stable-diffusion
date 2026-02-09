#!/usr/bin/env python3
"""Unified entrypoint for SD1.5 and SD3.5 text-to-image LoRA training."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def infer_model_version(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    model_lower = model_path.lower()
    if "stable-diffusion-3" in model_lower or "sd3" in model_lower:
        return "sd3.5"
    if "v1-5" in model_lower or "sd1" in model_lower:
        return "sd1.5"
    return None


def strip_model_version(argv: List[str]) -> List[str]:
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--model_version":
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_version", choices=["sd1.5", "sd3.5"], default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    args, _ = parser.parse_known_args()

    model_version = (
        args.model_version
        or infer_model_version(args.pretrained_model_name_or_path)
        or "sd1.5"
    )
    script_name = (
        "train_text_to_image_lora_sd35.py"
        if model_version == "sd3.5"
        else "train_text_to_image_lora_sd15.py"
    )

    script_path = Path(__file__).with_name(script_name)
    forward_args = strip_model_version(sys.argv[1:])

    subprocess.run([sys.executable, str(script_path), *forward_args], check=True)


if __name__ == "__main__":
    main()
