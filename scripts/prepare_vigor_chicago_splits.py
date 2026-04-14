#!/usr/bin/env python3

"""
Prepare train/val/test scene lists for a Crossview WAI dataset.

When multiple cities are requested, train/val/test quotas are applied per city and
then concatenated, so each city contributes the same number of scenes to each split.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def split_scene_id(scene_name: str) -> tuple[str | None, str]:
    scene_name = str(scene_name)
    if '__' in scene_name:
        city, local_name = scene_name.split('__', 1)
        return city, local_name
    return None, scene_name


def save_split(root: Path, split: str, scene_names: list[str]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = np.array(scene_names, dtype=object)
    np.save(split_dir / f"Crossview_scene_list_{split}.npy", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_wai"),
    )
    parser.add_argument(
        "--metadata_root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/Crossview"),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--train_scenes", type=int, default=40)
    parser.add_argument("--val_scenes", type=int, default=5)
    parser.add_argument("--test_scenes", type=int, default=5)
    return parser.parse_args()


def normalize_requested_cities(raw_cities, scene_names: list[str]) -> list[str]:
    if raw_cities:
        return [city.strip() for city in raw_cities if city.strip()]

    inferred = []
    seen = set()
    for scene_name in scene_names:
        city, _ = split_scene_id(scene_name)
        city = 'chicago' if city is None else city
        if city not in seen:
            seen.add(city)
            inferred.append(city)
    return inferred


def main() -> None:
    args = parse_args()
    all_scene_names = sorted(
        [path.name for path in args.dataset_root.iterdir() if path.is_dir()],
        key=natural_key,
    )
    requested_cities = normalize_requested_cities(args.cities, all_scene_names)

    grouped: dict[str, list[str]] = defaultdict(list)
    for scene_name in all_scene_names:
        city, _ = split_scene_id(scene_name)
        city = 'chicago' if city is None else city
        if city in requested_cities:
            grouped[city].append(scene_name)

    missing_cities = [city for city in requested_cities if city not in grouped]
    if missing_cities:
        raise ValueError(f'Missing requested cities in {args.dataset_root}: {missing_cities}')

    per_city_required = args.train_scenes + args.val_scenes + args.test_scenes
    per_city_summary = {}
    train_names: list[str] = []
    val_names: list[str] = []
    test_names: list[str] = []

    for city in requested_cities:
        city_scene_names = sorted(grouped[city], key=natural_key)
        if len(city_scene_names) < per_city_required:
            raise ValueError(
                f"City {city!r} has {len(city_scene_names)} scenes in {args.dataset_root}, "
                f"but requires at least {per_city_required} "
                f"({args.train_scenes} train + {args.val_scenes} val + {args.test_scenes} test)."
            )

        city_train = city_scene_names[: args.train_scenes]
        val_start = args.train_scenes
        val_end = val_start + args.val_scenes
        city_val = city_scene_names[val_start:val_end]
        city_test = city_scene_names[val_end : val_end + args.test_scenes]

        train_names.extend(city_train)
        val_names.extend(city_val)
        test_names.extend(city_test)
        per_city_summary[city] = {
            'num_available_scenes': len(city_scene_names),
            'train': city_train,
            'val': city_val,
            'test': city_test,
        }

    save_split(args.metadata_root, 'train', train_names)
    save_split(args.metadata_root, 'val', val_names)
    save_split(args.metadata_root, 'test', test_names)

    summary = {
        'dataset_root': str(args.dataset_root),
        'metadata_root': str(args.metadata_root),
        'cities': requested_cities,
        'per_city_quota': {
            'train': args.train_scenes,
            'val': args.val_scenes,
            'test': args.test_scenes,
        },
        'train': train_names,
        'val': val_names,
        'test': test_names,
        'per_city': per_city_summary,
    }
    args.metadata_root.mkdir(parents=True, exist_ok=True)
    with open(args.metadata_root / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(
        f"Prepared splits at {args.metadata_root}: "
        f"{len(train_names)} train / {len(val_names)} val / {len(test_names)} test "
        f"across {len(requested_cities)} city/cities"
    )


if __name__ == '__main__':
    main()
