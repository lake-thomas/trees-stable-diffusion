#!/usr/bin/env python3
"""Unified entrypoint for SD1.5 and SD3.5 image generation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def infer_model_version(config_path: Optional[str]) -> Optional[str]:
    if not config_path:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    model_version = str(config.get("model_version", "")).strip()
    if model_version in {"sd1.5", "sd3.5"}:
        return model_version

    model_path = str(config.get("model_path", "")).lower()
    if "stable-diffusion-3" in model_path or "sd3" in model_path:
        return "sd3.5"
    if "v1-5" in model_path or "sd1" in model_path:
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
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()

    model_version = args.model_version or infer_model_version(args.config) or "sd1.5"
    script_name = "generate_images_sd35.py" if model_version == "sd3.5" else "generate_images_sd15.py"

    script_path = Path(__file__).with_name(script_name)
    forward_args = strip_model_version(sys.argv[1:])

    subprocess.run([sys.executable, str(script_path), *forward_args], check=True)


if __name__ == "__main__":
    main()
