#!/usr/bin/env python3
"""Summarize projection_aux labels under Crossview_rs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def load_scene_list(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_metadata_scene_list(metadata_dir: Path, kind: str, split: str) -> Path:
    candidates = [
        metadata_dir / kind / split / f"{kind}_scene_list_{split}.npy",
        metadata_dir / kind / split / f"Crossview_scene_list_{split}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find metadata scene list. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def summarize_file(path: Path) -> dict[str, object]:
    data = np.load(path)
    rel_height = data["rel_height"].astype(np.float32)
    offset_xy = data["offset_xy"].astype(np.float32)
    valid_mask = data.get("valid_mask")
    tilt_mask = data.get("tilt_projected_mask")
    building_mask = data.get("building_mask")

    if valid_mask is None:
        valid_mask = np.isfinite(rel_height)
    valid_mask = valid_mask.astype(bool)
    if tilt_mask is None:
        tilt_mask = valid_mask
    tilt_mask = tilt_mask.astype(bool) & valid_mask
    if building_mask is None:
        building_mask = tilt_mask
    building_mask = building_mask.astype(bool) & valid_mask

    rel_pos = np.maximum(rel_height, 0.0)
    offset_norm = np.linalg.norm(offset_xy, axis=-1)
    height_mask = valid_mask & (rel_pos > 0.0)
    offset_mask = valid_mask & (offset_norm > 0.0)

    scene_dir = path.parent
    scene = scene_dir.parent.name
    provider = scene_dir.name
    total = float(rel_height.size)

    row = {
        "scene": scene,
        "provider": provider,
        "path": str(path),
        "valid_ratio": float(valid_mask.mean()),
        "tilt_ratio": float(tilt_mask.mean()),
        "building_ratio": float(building_mask.mean()),
        "height_pos_ratio": float(height_mask.mean()),
        "offset_pos_ratio": float(offset_mask.mean()),
        "height_mean_valid": _safe_mean(rel_pos[valid_mask]),
        "height_mean_tilt": _safe_mean(rel_pos[tilt_mask]),
        "height_p50_tilt": _safe_percentile(rel_pos[tilt_mask], 50),
        "height_p90_tilt": _safe_percentile(rel_pos[tilt_mask], 90),
        "height_p95_tilt": _safe_percentile(rel_pos[tilt_mask], 95),
        "offset_mean_valid": _safe_mean(offset_norm[valid_mask]),
        "offset_mean_tilt": _safe_mean(offset_norm[tilt_mask]),
        "offset_p50_tilt": _safe_percentile(offset_norm[tilt_mask], 50),
        "offset_p90_tilt": _safe_percentile(offset_norm[tilt_mask], 90),
        "offset_p95_tilt": _safe_percentile(offset_norm[tilt_mask], 95),
        "score_height_offset": _safe_mean(rel_pos[tilt_mask]) * max(float(tilt_mask.mean()), 1e-6)
        + _safe_mean(offset_norm[tilt_mask]) * max(float(tilt_mask.mean()), 1e-6),
        "num_pixels": int(total),
    }
    if "global_slope" in data:
        row["global_slope"] = float(np.asarray(data["global_slope"]).reshape(-1)[0])
    if "azimuth_deg" in data:
        row["azimuth_deg"] = float(np.asarray(data["azimuth_deg"]).reshape(-1)[0])
    if "tilt_deg" in data:
        row["tilt_deg"] = float(np.asarray(data["tilt_deg"]).reshape(-1)[0])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--scene-prefix", default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--metadata-dir", default=None, type=Path)
    parser.add_argument("--metadata-kind", default="Crossview_rs_aerial")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-list-path", default=None, type=Path)
    parser.add_argument("--output-top-scenes-npy", default=None, type=Path)
    parser.add_argument("--output-top-scenes-txt", default=None, type=Path)
    args = parser.parse_args()

    allowed_scenes = None
    allowed_scene_source = None
    if args.scene_list_path is not None:
        allowed_scenes = set(load_scene_list(args.scene_list_path))
        allowed_scene_source = args.scene_list_path
    elif args.metadata_dir is not None:
        metadata_scene_list = resolve_metadata_scene_list(
            args.metadata_dir, args.metadata_kind, args.split
        )
        allowed_scenes = set(load_scene_list(metadata_scene_list))
        allowed_scene_source = metadata_scene_list

    paths = sorted(args.root.glob("**/projection_aux.npz"))
    if args.provider:
        paths = [p for p in paths if p.parent.name == args.provider]
    if args.scene_prefix:
        paths = [p for p in paths if p.parent.parent.name.startswith(args.scene_prefix)]
    before_allowed_filter = len(paths)
    if allowed_scenes is not None:
        paths = [p for p in paths if p.parent.parent.name in allowed_scenes]
    rows = [summarize_file(path) for path in paths]
    rows.sort(key=lambda row: float(row["score_height_offset"]), reverse=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    top_scene_names = [str(row["scene"]) for row in rows[: args.top_k]]
    if args.output_top_scenes_npy is not None:
        args.output_top_scenes_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_top_scenes_npy, np.asarray(top_scene_names, dtype=object))
    if args.output_top_scenes_txt is not None:
        args.output_top_scenes_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_top_scenes_txt.write_text(
            "\n".join(top_scene_names) + ("\n" if top_scene_names else ""),
            encoding="utf-8",
        )

    if allowed_scene_source is not None:
        print(
            f"metadata/scene-list filter: {allowed_scene_source} "
            f"kept {len(paths)}/{before_allowed_filter} projection_aux files"
        )
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    if args.output_top_scenes_npy is not None:
        print(f"wrote top scenes npy to {args.output_top_scenes_npy}")
    if args.output_top_scenes_txt is not None:
        print(f"wrote top scenes txt to {args.output_top_scenes_txt}")
    print("top scenes:")
    for row in rows[: args.top_k]:
        print(
            f"{row['scene']} {row['provider']} "
            f"tilt={row['tilt_ratio']:.4f} "
            f"h_mean={row['height_mean_tilt']:.2f} h_p95={row['height_p95_tilt']:.2f} "
            f"off_mean={row['offset_mean_tilt']:.2f} off_p95={row['offset_p95_tilt']:.2f} "
            f"score={row['score_height_offset']:.4f}"
        )


if __name__ == "__main__":
    main()
