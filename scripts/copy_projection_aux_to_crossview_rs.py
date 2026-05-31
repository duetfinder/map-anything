#!/usr/bin/env python3
"""Copy exp_005 projection_aux labels into traindata/Crossview_rs.

Source layout:
  outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_1/Google_Satellite/projection_aux.npz

Destination layout:
  traindata/Crossview_rs/chicago__location_1/Google_Satellite/projection_aux.npz

The script can also update traindata/Crossview_rs/location_manifest.json by
adding provider["projection_aux_path"] with a path relative to Crossview_rs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy projection_aux.npz files from exp_005 outputs into Crossview_rs."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("outputs/experiments/exp_005_map_points_generate/vigor"),
        help="Root containing city/location/provider/projection_aux.npz outputs.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("traindata/Crossview_rs"),
        help="Crossview_rs dataset root.",
    )
    parser.add_argument(
        "--city",
        default="chicago",
        help="City subdirectory to copy, e.g. chicago.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional provider names to include. Default copies every provider found.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path. Defaults to <dst-root>/location_manifest.json.",
    )
    parser.add_argument(
        "--no-update-manifest",
        action="store_true",
        help="Only copy files; do not update location_manifest.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination projection_aux.npz files.",
    )
    parser.add_argument(
        "--allow-missing-scenes",
        action="store_true",
        help="Create destination scene/provider directories even if the scene is absent.",
    )
    parser.add_argument(
        "--no-backup-manifest",
        action="store_true",
        help="Do not write location_manifest.json.bak before updating manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies/manifest updates without changing files.",
    )
    return parser.parse_args()


def iter_projection_aux_files(city_src: Path, providers: set[str] | None) -> Iterable[Path]:
    for path in sorted(city_src.glob("location_*/*/projection_aux.npz")):
        provider = path.parent.name
        if providers is not None and provider not in providers:
            continue
        yield path


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("locations"), list):
        raise ValueError(f"unsupported manifest format: {path}")
    return data


def build_manifest_index(manifest: dict) -> dict[tuple[str, str, str], dict]:
    index = {}
    for location_entry in manifest["locations"]:
        city = location_entry.get("city")
        location = location_entry.get("location")
        scene_name = location_entry.get("scene_name") or (
            f"{city}__{location}" if city and location else None
        )
        for provider_entry in location_entry.get("providers", []):
            provider = provider_entry.get("provider")
            if city and location and provider:
                index[(city, location, provider)] = provider_entry
            if scene_name and provider:
                index[(city, scene_name, provider)] = provider_entry
    return index


def rel_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    city_src = src_root / args.city
    manifest_path = (args.manifest or (dst_root / "location_manifest.json")).resolve()
    providers = set(args.providers) if args.providers else None

    if not city_src.exists():
        raise FileNotFoundError(f"city source directory not found: {city_src}")
    if not dst_root.exists():
        raise FileNotFoundError(f"destination root not found: {dst_root}")

    manifest = None
    manifest_index = {}
    if not args.no_update_manifest:
        manifest = load_manifest(manifest_path)
        manifest_index = build_manifest_index(manifest)

    copied = 0
    skipped_existing = 0
    skipped_missing_scene = 0
    manifest_updates = 0
    missing_manifest_entries = 0

    for src_path in iter_projection_aux_files(city_src, providers):
        location = src_path.parent.parent.name
        provider = src_path.parent.name
        scene_name = f"{args.city}__{location}"
        scene_dir = dst_root / scene_name
        dst_path = scene_dir / provider / "projection_aux.npz"

        if not args.allow_missing_scenes and not scene_dir.exists():
            print(f"skip missing scene: {scene_name}")
            skipped_missing_scene += 1
            continue

        if dst_path.exists() and not args.overwrite:
            skipped_existing += 1
        else:
            print(f"copy {src_path} -> {dst_path}")
            if not args.dry_run:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            copied += 1

        if manifest is not None:
            entry = manifest_index.get((args.city, location, provider)) or manifest_index.get(
                (args.city, scene_name, provider)
            )
            if entry is None:
                print(f"manifest missing entry: {scene_name}/{provider}")
                missing_manifest_entries += 1
            else:
                rel_path = rel_to_root(dst_path, dst_root)
                if entry.get("projection_aux_path") != rel_path:
                    entry["projection_aux_path"] = rel_path
                    manifest_updates += 1

    if manifest is not None and manifest_updates > 0:
        print(f"update manifest: {manifest_path} ({manifest_updates} entries)")
        if not args.dry_run:
            if not args.no_backup_manifest:
                backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
                shutil.copy2(manifest_path, backup_path)
                print(f"backup manifest: {backup_path}")
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
                f.write("\n")

    print(
        "summary: "
        f"copied={copied}, "
        f"skipped_existing={skipped_existing}, "
        f"skipped_missing_scene={skipped_missing_scene}, "
        f"manifest_updates={manifest_updates}, "
        f"missing_manifest_entries={missing_manifest_entries}"
    )
    if args.dry_run:
        print("dry-run: no files were changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
