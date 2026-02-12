#!/usr/bin/env python3
"""Run or print a CNN-data grid search from config."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List


def _expand_range(spec: Dict[str, int]) -> List[int]:
    for key in ("start", "end", "step"):
        if key not in spec:
            raise ValueError(f"Range spec missing required key: {key}")
    return list(range(spec["start"], spec["end"] + 1, spec["step"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN grid-search ops runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cnn_grid_search.json"),
        help="Path to grid-search config JSON",
    )
    parser.add_argument(
        "--dataset",
        choices=["inat", "autoarborist"],
        default=None,
        help="If set, run only one dataset",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands instead of printing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())

    datasets = [args.dataset] if args.dataset else config["datasets"]
    cnn_sizes = _expand_range(config["cnn_train_images"])
    synthetic_pcts = _expand_range(config["synthetic_generated_proportion"])
    template = config["command_template"]

    for dataset in datasets:
        for n_cnn_train in cnn_sizes:
            for synthetic_proportion in synthetic_pcts:
                command = template.format(
                    dataset=dataset,
                    n_cnn_train=n_cnn_train,
                    synthetic_proportion=synthetic_proportion,
                )
                print(command)
                if args.execute:
                    if any(token in command for token in ("|", "&", ";", "`", "$(", ">", "<")):
                        raise ValueError(f"Unsafe command detected: {command}")
                    try:
                        subprocess.run(shlex.split(command), check=True)
                    except subprocess.CalledProcessError as exc:
                        raise RuntimeError(
                            f"Grid search command failed for dataset={dataset}, "
                            f"n_cnn_train={n_cnn_train}, "
                            f"synthetic_proportion={synthetic_proportion}"
                        ) from exc


if __name__ == "__main__":
    main()
