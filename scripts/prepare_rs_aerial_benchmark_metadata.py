#!/usr/bin/env python3
"""
Prepare RS-Aerial benchmark metadata by linking aerial WAI scenes with remote-sensing image and
per-pixel pointmap labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_scene_list(split_dir: Path, split: str) -> list[str]:
    split_file = split_dir / split / f"vigor_chicago_scene_list_{split}.npy"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [str(x) for x in np.load(split_file, allow_pickle=True).tolist()]


def build_scene_manifest(
    scene_name: str,
    aerial_root: Path,
    remote_root: Path,
    satellite_maps_root: Path | None,
    provider: str,
) -> dict:
    aerial_scene_dir = aerial_root / scene_name
    remote_scene_dir = remote_root / scene_name / provider
    unified_remote_image_path = remote_scene_dir / "image.png"

    if not aerial_scene_dir.exists():
        raise FileNotFoundError(f"Missing aerial scene dir: {aerial_scene_dir}")
    if not (aerial_scene_dir / "scene_meta.json").exists():
        raise FileNotFoundError(f"Missing aerial scene meta: {aerial_scene_dir / 'scene_meta.json'}")
    if not remote_scene_dir.exists():
        raise FileNotFoundError(f"Missing remote scene dir: {remote_scene_dir}")

    if unified_remote_image_path.exists():
        remote_image_path = unified_remote_image_path
    else:
        if satellite_maps_root is None:
            raise FileNotFoundError(
                f"Missing unified remote image {unified_remote_image_path} and no legacy satellite_maps_root was provided"
            )
        remote_image_path = satellite_maps_root / scene_name / f"{provider}.png"
        if not remote_image_path.exists():
            raise FileNotFoundError(f"Missing remote image: {remote_image_path}")

    pointmap_path = remote_scene_dir / "pixel_to_point_map.npz"
    valid_mask_path = remote_scene_dir / "valid_mask.npy"
    height_map_path = remote_scene_dir / "height_map.npy"
    info_path = remote_scene_dir / "info.json"

    required_paths = [pointmap_path, valid_mask_path, height_map_path, info_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing remote modalities for {scene_name}: {missing_paths}"
        )

    pointmap_npz = np.load(pointmap_path)
    if "xyz" not in pointmap_npz.files:
        raise KeyError(f"{pointmap_path} does not contain key 'xyz'")
    xyz = pointmap_npz["xyz"]
    valid_mask = np.load(valid_mask_path)
    height_map = np.load(height_map_path)
    remote_image = np.array(Image.open(remote_image_path))

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Unexpected pointmap shape for {pointmap_path}: {xyz.shape}")
    if valid_mask.shape != xyz.shape[:2]:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match pointmap shape {xyz.shape[:2]} for {scene_name}"
        )
    if height_map.shape != xyz.shape[:2]:
        raise ValueError(
            f"height_map shape {height_map.shape} does not match pointmap shape {xyz.shape[:2]} for {scene_name}"
        )
    if remote_image.shape[:2] != xyz.shape[:2]:
        raise ValueError(
            f"remote image shape {remote_image.shape[:2]} does not match pointmap shape {xyz.shape[:2]} for {scene_name}"
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return {
        "scene_name": scene_name,
        "aerial_scene_dir": str(aerial_scene_dir),
        "aerial_scene_meta": str(aerial_scene_dir / "scene_meta.json"),
        "remote_provider": provider,
        "remote_scene_dir": str(remote_scene_dir),
        "remote_image_path": str(remote_image_path),
        "remote_pointmap_path": str(pointmap_path),
        "remote_valid_mask_path": str(valid_mask_path),
        "remote_height_map_path": str(height_map_path),
        "remote_info_path": str(info_path),
        "remote_image_hw": list(remote_image.shape[:2]),
        "remote_pointmap_hw": list(xyz.shape[:2]),
        "remote_valid_ratio": float(valid_mask.mean()),
        "remote_projection_type": "rs_global_projective",
        "remote_pointmap_key": "xyz",
        "meters_per_pixel": info.get("meters_per_pixel"),
        "coverage_meters": info.get("coverage_meters"),
        "ground_z_approx": info.get("ground_z_approx"),
        "map_name": info.get("map_name"),
        "map_key": info.get("map_key"),
        "angle": info.get("angle"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RS-Aerial benchmark metadata")
    parser.add_argument(
        "--aerial-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/vigor_chicago_wai"),
    )
    parser.add_argument(
        "--aerial-split-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/vigor_chicago"),
    )
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/vigor_chicago_rs"),
    )
    parser.add_argument(
        "--satellite-maps-root",
        type=Path,
        default=None,
        help="Optional legacy map root. Only needed when remote-root still uses the old split geometry/image layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/vigor_chicago_rs_aerial"),
    )
    parser.add_argument("--providers", nargs="+", default=["Google_Satellite"])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for split in args.splits:
        scene_names = load_scene_list(args.aerial_split_root, split)
        manifests = []
        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        missing_scenes = []
        per_scene_primary_manifest = {}
        for scene_name in scene_names:
            scene_manifests = []
            scene_errors = []
            for provider in args.providers:
                try:
                    manifest = build_scene_manifest(
                        scene_name=scene_name,
                        aerial_root=args.aerial_root,
                        remote_root=args.remote_root,
                        satellite_maps_root=args.satellite_maps_root,
                        provider=provider,
                    )
                except FileNotFoundError as exc:
                    scene_errors.append({"provider": provider, "reason": str(exc)})
                    continue
                scene_manifests.append(manifest)
                manifests.append(manifest)

            if not scene_manifests:
                missing_scenes.append({"scene_name": scene_name, "errors": scene_errors})
                continue

            per_scene_primary_manifest[scene_name] = scene_manifests[0]
            manifest_path = split_dir / f"{scene_name}.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(scene_manifests[0], f, indent=2)

        scene_list_path = split_dir / f"vigor_chicago_rs_aerial_scene_list_{split}.npy"
        np.save(scene_list_path, np.array(sorted(per_scene_primary_manifest.keys()), dtype=object))

        aggregate_path = split_dir / f"vigor_chicago_rs_aerial_{split}.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(manifests, f, indent=2)

        summary[split] = {
            "requested_num_scenes": len(scene_names),
            "num_scenes": len(manifests),
            "num_missing_scenes": len(missing_scenes),
            "providers": args.providers,
            "scene_list_path": str(scene_list_path),
            "aggregate_manifest_path": str(aggregate_path),
        }
        if missing_scenes:
            missing_path = split_dir / f"vigor_chicago_rs_aerial_missing_{split}.json"
            with open(missing_path, "w", encoding="utf-8") as f:
                json.dump(missing_scenes, f, indent=2)
            summary[split]["missing_manifest_path"] = str(missing_path)

    with open(args.output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
