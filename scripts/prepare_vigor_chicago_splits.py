#!/usr/bin/env python3

"""
Prepare train/val/test scene lists for the VIGOR Chicago WAI dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def save_split(root: Path, split: str, scene_names: list[str]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        split_dir / f"vigor_chicago_scene_list_{split}.npy",
        np.array(scene_names, dtype=object),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/vigor_chicago_wai"),
    )
    parser.add_argument(
        "--metadata_root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/vigor_chicago"),
    )
    parser.add_argument("--train_scenes", type=int, default=40)
    parser.add_argument("--val_scenes", type=int, default=5)
    parser.add_argument("--test_scenes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_names = sorted(
        [path.name for path in args.dataset_root.iterdir() if path.is_dir()],
        key=natural_key,
    )

    required = args.train_scenes + args.val_scenes + args.test_scenes
    if len(scene_names) < required:
        raise ValueError(
            f"Found {len(scene_names)} scenes in {args.dataset_root}, but splits require {required}"
        )

    train_names = scene_names[: args.train_scenes]
    val_start = args.train_scenes
    val_end = val_start + args.val_scenes
    val_names = scene_names[val_start:val_end]
    test_names = scene_names[val_end : val_end + args.test_scenes]

    save_split(args.metadata_root, "train", train_names)
    save_split(args.metadata_root, "val", val_names)
    save_split(args.metadata_root, "test", test_names)

    summary = {
        "dataset_root": str(args.dataset_root),
        "metadata_root": str(args.metadata_root),
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }
    args.metadata_root.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_root / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Prepared splits at {args.metadata_root}: "
        f"{len(train_names)} train / {len(val_names)} val / {len(test_names)} test"
    )


if __name__ == "__main__":
    main()
