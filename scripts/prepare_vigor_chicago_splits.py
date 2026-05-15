#!/usr/bin/env python3

"""
Prepare train/val/test scene lists for a Crossview WAI dataset.

When multiple cities are requested, train/val/test quotas are applied per city and
then concatenated, so each city contributes the same number of scenes to each split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROVIDER_COLUMN_PREFIX = "provider__"


def parse_bool_like(value) -> bool:
    if value is None:
        return False
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def provider_columns_from_fieldnames(fieldnames) -> list[str]:
    return [
        name for name in fieldnames
        if str(name).startswith(PROVIDER_COLUMN_PREFIX)
    ]


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


def load_manual_split_spec(path: Path) -> dict[str, list[str]]:
    """Load a manual split spec from one txt file.

    Format:
        [train]
        scene_a
        scene_b
        [val]
        scene_c
        [test]
        scene_d

    Empty lines and lines starting with # are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing split spec txt: {path}")

    sections = {"train": [], "val": [], "test": []}
    current = None
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].strip().lower()
            if current not in sections:
                raise ValueError(f"Unsupported split section {line!r} at line {line_no} in {path}")
            continue
        if current is None:
            raise ValueError(f"Scene entry before any [train]/[val]/[test] section at line {line_no} in {path}")
        sections[current].append(line)

    return sections


def load_manual_split_csv(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load a manual split/provider spec from CSV.

    Required columns:
        scene_name, split

    Optional columns:
        city, provider__<ProviderName>
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing split spec csv: {path}")

    sections = {"train": [], "val": [], "test": []}
    provider_map: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        required_columns = {"scene_name", "split"}
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"CSV file {path} is missing required columns: {missing_columns}"
            )
        provider_columns = provider_columns_from_fieldnames(reader.fieldnames)
        for row_idx, row in enumerate(reader, start=2):
            scene_name = str(row["scene_name"]).strip()
            split = str(row["split"]).strip().lower()
            if not scene_name:
                continue
            if split not in sections:
                raise ValueError(
                    f"Unsupported split {split!r} at line {row_idx} in {path}"
                )
            sections[split].append(scene_name)
            providers: list[str] = []
            for column in provider_columns:
                if parse_bool_like(row.get(column)):
                    providers.append(column[len(PROVIDER_COLUMN_PREFIX):])
            providers = list(dict.fromkeys(providers))
            if providers:
                provider_map[scene_name] = providers

    return sections, provider_map


def write_split_csv(
    path: Path,
    split_scene_names: dict[str, list[str]],
    default_remote_provider: str,
    provider_names: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provider_names = [] if provider_names is None else list(provider_names)
    provider_columns = [f"{PROVIDER_COLUMN_PREFIX}{name}" for name in provider_names]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_name", "city", "split", *provider_columns],
        )
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for scene_name in split_scene_names[split_name]:
                city, _ = split_scene_id(scene_name)
                row = {
                    "scene_name": scene_name,
                    "city": city or "chicago",
                    "split": split_name,
                }
                for column in provider_columns:
                    row[column] = "0"
                default_column = f"{PROVIDER_COLUMN_PREFIX}{default_remote_provider}"
                if default_column in row:
                    row[default_column] = "1"
                writer.writerow(row)


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
    parser.add_argument("--split_spec_txt", type=Path, default=None, help="Optional manual split spec txt with [train]/[val]/[test] sections. If provided, automatic sequential splitting is skipped.")
    parser.add_argument("--split_spec_csv", type=Path, default=None, help="Optional manual split/provider CSV with columns scene_name,split[,city,provider__<ProviderName>...]. If provided, automatic sequential splitting is skipped.")
    parser.add_argument("--draft_csv_out", type=Path, default=None, help="Optional output CSV path. Writes the resolved split draft with columns scene_name,city,split,provider__<ProviderName>....")
    parser.add_argument("--default_remote_provider", type=str, default="Google_Satellite")
    parser.add_argument(
        "--provider_names",
        nargs="*",
        default=["Google_Satellite", "Bing_Satellite", "ESRI_Satellite", "OSM_Standard", "Yandex_Satellite"],
        help="Provider columns to include when writing a draft CSV.",
    )
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
    if args.split_spec_txt is not None and args.split_spec_csv is not None:
        raise ValueError("Use at most one of --split_spec_txt or --split_spec_csv")

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

    per_city_summary = {}
    train_names: list[str] = []
    val_names: list[str] = []
    test_names: list[str] = []
    provider_map: dict[str, list[str]] = {}

    if args.split_spec_csv is not None:
        split_spec, provider_map = load_manual_split_csv(args.split_spec_csv)
        train_names = split_spec['train']
        val_names = split_spec['val']
        test_names = split_spec['test']
    elif args.split_spec_txt is not None:
        split_spec = load_manual_split_spec(args.split_spec_txt)
        train_names = split_spec['train']
        val_names = split_spec['val']
        test_names = split_spec['test']
    else:
        split_spec = None

    if split_spec is not None:

        known_scenes = set(all_scene_names)
        for split_name, split_scene_names in split_spec.items():
            unknown = [scene for scene in split_scene_names if scene not in known_scenes]
            if unknown:
                raise ValueError(f"Unknown scenes in manual {split_name} split: {unknown[:10]}")

        duplicates = (set(train_names) & set(val_names)) | (set(train_names) & set(test_names)) | (set(val_names) & set(test_names))
        if duplicates:
            raise ValueError(f"Manual split spec contains scenes assigned to multiple splits: {sorted(duplicates)[:10]}")

        for city in requested_cities:
            city_train = [scene for scene in train_names if (split_scene_id(scene)[0] or 'chicago') == city]
            city_val = [scene for scene in val_names if (split_scene_id(scene)[0] or 'chicago') == city]
            city_test = [scene for scene in test_names if (split_scene_id(scene)[0] or 'chicago') == city]
            per_city_summary[city] = {
                'num_available_scenes': len(grouped[city]),
                'train': city_train,
                'val': city_val,
                'test': city_test,
            }
    else:
        per_city_required = args.train_scenes + args.val_scenes + args.test_scenes
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

    split_scene_names = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }
    if args.draft_csv_out is not None:
        write_split_csv(
            path=args.draft_csv_out,
            split_scene_names=split_scene_names,
            default_remote_provider=args.default_remote_provider,
            provider_names=args.provider_names,
        )

    summary = {
        'dataset_root': str(args.dataset_root),
        'metadata_root': str(args.metadata_root),
        'cities': requested_cities,
        'split_source': (
            str(args.split_spec_csv)
            if args.split_spec_csv is not None
            else str(args.split_spec_txt)
            if args.split_spec_txt is not None
            else 'sequential'
        ),
        'draft_csv_out': str(args.draft_csv_out) if args.draft_csv_out is not None else None,
        'default_remote_provider': args.default_remote_provider,
        'per_city_quota': {
            'train': args.train_scenes,
            'val': args.val_scenes,
            'test': args.test_scenes,
        },
        'train': train_names,
        'val': val_names,
        'test': test_names,
        'provider_map_num_scenes': len(provider_map),
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
