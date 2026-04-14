#!/usr/bin/env python3

"""
Convert VIGOR reconstruction outputs into a minimal WAI dataset.

Each source scene is expected to look like:
    location_x/
      location_x_00.jpg
      location_x_00.exr
      location_x_00.npz
      ...

The npz file must contain:
    - intrinsics: 3x3 pinhole intrinsics
    - cam2world: 4x4 OpenCV camera-to-world transform
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def build_scene_meta(scene_name: str, frames: list[dict], scene_modalities: dict | None = None) -> dict:
    return {
        "scene_name": scene_name,
        "dataset_name": "vigor",
        "version": "0.1",
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "scene_modalities": scene_modalities or {},
        "frames": frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
        },
    }


def convert_scene(scene_dir: Path, target_root: Path, overwrite: bool, city: str | None = None) -> dict[str, int | str | None]:
    local_scene_name = scene_dir.name
    scene_name = f"{city}__{local_scene_name}" if city else local_scene_name
    target_scene_root = target_root / scene_name
    target_scene_root.mkdir(parents=True, exist_ok=True)

    frame_stems = sorted({path.stem for path in scene_dir.glob("*.jpg")}, key=natural_key)
    if not frame_stems:
        raise ValueError(f"No JPG frames found in {scene_dir}")

    wai_frames = []
    for frame_stem in frame_stems:
        image_src = scene_dir / f"{frame_stem}.jpg"
        depth_src = scene_dir / f"{frame_stem}.exr"
        camera_src = scene_dir / f"{frame_stem}.npz"

        if not image_src.exists() or not depth_src.exists() or not camera_src.exists():
            raise FileNotFoundError(
                f"Missing modality for frame {frame_stem} in scene {local_scene_name}"
            )

        camera_data = np.load(camera_src)
        intrinsics = camera_data["intrinsics"].astype(np.float32)
        cam2world = camera_data["cam2world"].astype(np.float32)
        width, height = Image.open(image_src).size

        image_rel = Path("images") / image_src.name
        depth_rel = Path("depth") / depth_src.name
        image_dst = target_scene_root / image_rel
        depth_dst = target_scene_root / depth_rel

        if overwrite or not image_dst.exists():
            copy_file(image_src, image_dst)
        if overwrite or not depth_dst.exists():
            copy_file(depth_src, depth_dst)

        wai_frames.append(
            {
                "frame_name": frame_stem,
                "image": str(image_rel),
                "file_path": str(image_rel),
                "depth": str(depth_rel),
                "transform_matrix": cam2world.tolist(),
                "h": int(height),
                "w": int(width),
                "fl_x": float(intrinsics[0, 0]),
                "fl_y": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
            }
        )

    existing_modalities = {}
    existing_meta_path = target_scene_root / "scene_meta.json"
    if existing_meta_path.exists():
        with open(existing_meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        existing_modalities = existing_meta.get("scene_modalities", {})

    scene_meta = build_scene_meta(scene_name, wai_frames, scene_modalities=existing_modalities)
    with open(target_scene_root / "scene_meta.json", "w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2)

    return {
        "scene_name": scene_name,
        "local_scene_name": local_scene_name,
        "city": city,
        "num_frames": len(wai_frames),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/experiments/exp_001_reconstrc/vigor_chicago_processed"),
    )
    parser.add_argument(
        "--source_parent",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/experiments/exp_001_reconstrc"),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument(
        "--target_root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_wai"),
    )
    parser.add_argument(
        "--max_locations",
        type=int,
        default=500,
        help="Only convert the first N locations for pilot experiments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files and metadata in the target dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cities:
        source_sets = []
        for city in args.cities:
            city = str(city).strip()
            if not city:
                continue
            source_sets.append((city, args.source_parent / f"vigor_{city}_processed"))
    else:
        source_sets = [(None, args.source_root)]

    args.target_root.mkdir(parents=True, exist_ok=True)
    stats = []
    for city, source_root in source_sets:
        scene_dirs = sorted(
            [path for path in source_root.iterdir() if path.is_dir()],
            key=lambda path: natural_key(path.name),
        )
        if args.max_locations is not None:
            scene_dirs = scene_dirs[: args.max_locations]
        for scene_dir in scene_dirs:
            stats.append(convert_scene(scene_dir, args.target_root, args.overwrite, city=city))

    summary = {
        "source_root": str(args.source_root),
        "source_parent": str(args.source_parent),
        "cities": args.cities,
        "target_root": str(args.target_root),
        "num_scenes": len(stats),
        "num_frames_total": int(sum(item["num_frames"] for item in stats)),
        "scenes": stats,
    }
    with open(args.target_root / "conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Converted {summary['num_scenes']} scenes with {summary['num_frames_total']} frames "
        f"into {args.target_root}"
    )


if __name__ == "__main__":
    main()
