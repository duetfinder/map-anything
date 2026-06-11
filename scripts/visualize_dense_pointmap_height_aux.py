#!/usr/bin/env python3
"""Visualize dense relative-height labels derived directly from remote pointmaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a denser projection relative-height target from "
            "pixel_to_point_map.npz and compare it with projection_aux.npz."
        )
    )
    parser.add_argument("--remote_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ground_mode",
        choices=["aux_median", "aux_p10", "pointmap_p10"],
        default="aux_median",
        help="How to estimate the reference ground z used by dense rel_height.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Percentile used for visualization clipping.",
    )
    return parser.parse_args()


def finite_xyz_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=-1)


def load_rgb(remote_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    image_path = remote_dir / "image.png"
    h, w = shape
    if not image_path.exists():
        return np.full((h, w, 3), 180, dtype=np.uint8)
    image = Image.open(image_path).convert("RGB")
    if image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def estimate_ground_z(
    xyz: np.ndarray,
    point_valid: np.ndarray,
    aux_rel_height: np.ndarray,
    aux_original_xyz: np.ndarray,
    aux_valid: np.ndarray,
    mode: str,
) -> float:
    common = aux_valid & finite_xyz_mask(aux_original_xyz) & np.isfinite(aux_rel_height)
    if mode == "aux_median":
        if not common.any():
            raise ValueError("aux_median ground mode requires valid projection_aux labels")
        return float(np.median(aux_original_xyz[..., 2][common] - aux_rel_height[common]))
    if mode == "aux_p10":
        if not common.any():
            raise ValueError("aux_p10 ground mode requires valid projection_aux labels")
        return float(np.percentile(aux_original_xyz[..., 2][common], 10))
    if mode == "pointmap_p10":
        if not point_valid.any():
            raise ValueError("pointmap_p10 ground mode requires valid pointmap labels")
        return float(np.percentile(xyz[..., 2][point_valid], 10))
    raise ValueError(mode)


def normalize_image(values: np.ndarray, mask: np.ndarray, vmax: float | None = None) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    if not mask.any():
        return out
    selected = values[mask]
    if vmax is None:
        vmax = float(np.percentile(np.abs(selected), 99.0))
    vmax = max(vmax, 1e-6)
    out[mask] = np.clip(values[mask] / vmax, 0.0, 1.0)
    return out


def colorize_height(values: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    norm = normalize_image(np.maximum(values, 0.0), mask, vmax=vmax)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    # Blue -> cyan -> yellow -> red, implemented without matplotlib.
    rgb[..., 0] = np.clip(255.0 * np.maximum(0.0, 2.0 * norm - 0.5), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * norm - 1.0)), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255.0 * np.maximum(0.0, 1.0 - 2.0 * norm), 0, 255).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def colorize_abs(values: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    norm = normalize_image(np.abs(values), mask, vmax=vmax)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255.0 * norm).astype(np.uint8)
    rgb[..., 1] = (255.0 * (1.0 - norm)).astype(np.uint8)
    rgb[..., 2] = 0
    rgb[~mask] = 0
    return rgb


def mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = color
    return rgb


def add_title(panel: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(panel)
    canvas = Image.new("RGB", (image.width, image.height + 28), (255, 255, 255))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0))
    return canvas


def save_grid(panels: list[tuple[str, np.ndarray]], path: Path, columns: int = 3) -> None:
    titled = [add_title(panel, title) for title, panel in panels]
    w, h = titled[0].size
    rows = int(np.ceil(len(titled) / columns))
    grid = Image.new("RGB", (columns * w, rows * h), (255, 255, 255))
    for idx, image in enumerate(titled):
        x = (idx % columns) * w
        y = (idx // columns) * h
        grid.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not mask.any():
        return {"count": 0}
    selected = values[mask]
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "p01": float(np.percentile(selected, 1)),
        "p50": float(np.percentile(selected, 50)),
        "p95": float(np.percentile(selected, 95)),
        "p99": float(np.percentile(selected, 99)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
    }


def main() -> None:
    args = parse_args()
    aux_path = args.remote_dir / "projection_aux.npz"
    pointmap_path = args.remote_dir / "pixel_to_point_map.npz"
    if not aux_path.exists():
        raise FileNotFoundError(aux_path)
    if not pointmap_path.exists():
        raise FileNotFoundError(pointmap_path)

    aux = np.load(aux_path)
    xyz = np.load(pointmap_path)["xyz"].astype(np.float32)
    point_valid = finite_xyz_mask(xyz)
    aux_valid = aux["valid_mask"].astype(bool)
    aux_rel_height = aux["rel_height"].astype(np.float32)
    aux_original = aux["original_xyz_world"].astype(np.float32)

    ground_z = estimate_ground_z(
        xyz,
        point_valid,
        aux_rel_height,
        aux_original,
        aux_valid,
        args.ground_mode,
    )
    dense_rel_height = xyz[..., 2] - np.float32(ground_z)
    dense_rel_height[~point_valid] = np.nan

    common = point_valid & aux_valid & np.isfinite(aux_rel_height)
    dense_only = point_valid & ~aux_valid
    height_vmax = float(
        np.percentile(
            np.maximum(dense_rel_height[point_valid], 0.0),
            float(args.clip_percentile),
        )
    )
    height_vmax = max(height_vmax, 1e-6)
    diff = dense_rel_height - aux_rel_height
    diff_vmax = float(np.percentile(np.abs(diff[common]), 95)) if common.any() else 1.0
    diff_vmax = max(diff_vmax, 1e-6)

    rgb = load_rgb(args.remote_dir, xyz.shape[:2])
    overlay = rgb.copy()
    overlay[dense_only] = (0.55 * overlay[dense_only] + np.array([255, 0, 255]) * 0.45).astype(
        np.uint8
    )

    panels = [
        ("rgb", rgb),
        ("pointmap valid", mask_to_rgb(point_valid, (255, 255, 255))),
        ("aux valid", mask_to_rgb(aux_valid, (255, 255, 255))),
        ("dense-only added", mask_to_rgb(dense_only, (255, 0, 255))),
        ("dense-only overlay", overlay),
        ("aux rel_height", colorize_height(aux_rel_height, aux_valid, height_vmax)),
        ("dense rel_height", colorize_height(dense_rel_height, point_valid, height_vmax)),
        ("abs diff on common", colorize_abs(diff, common, diff_vmax)),
        ("common mask", mask_to_rgb(common, (255, 255, 255))),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_grid(panels, args.output_dir / "dense_pointmap_height_grid.png")
    Image.fromarray(colorize_height(dense_rel_height, point_valid, height_vmax)).save(
        args.output_dir / "dense_rel_height.png"
    )
    Image.fromarray(colorize_height(aux_rel_height, aux_valid, height_vmax)).save(
        args.output_dir / "aux_rel_height.png"
    )
    Image.fromarray(mask_to_rgb(dense_only, (255, 0, 255))).save(
        args.output_dir / "dense_only_mask.png"
    )
    np.savez_compressed(
        args.output_dir / "dense_pointmap_height_aux.npz",
        dense_rel_height=dense_rel_height.astype(np.float32),
        dense_valid_mask=point_valid,
        aux_valid_mask=aux_valid,
        dense_only_mask=dense_only,
        ground_z=np.asarray([ground_z], dtype=np.float32),
    )

    summary = {
        "remote_dir": str(args.remote_dir),
        "ground_mode": args.ground_mode,
        "ground_z": ground_z,
        "pointmap_valid_ratio": float(point_valid.mean()),
        "aux_valid_ratio": float(aux_valid.mean()),
        "common_ratio": float(common.mean()),
        "dense_only_ratio": float(dense_only.mean()),
        "relative_coverage_gain": float(point_valid.sum() / max(int(aux_valid.sum()), 1)),
        "height_vmax": height_vmax,
        "diff_vmax": diff_vmax,
        "dense_rel_height_stats": stats(dense_rel_height, point_valid),
        "aux_rel_height_stats": stats(aux_rel_height, aux_valid),
        "dense_minus_aux_common_stats": stats(diff, common),
        "abs_dense_minus_aux_common_stats": stats(np.abs(diff), common),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {args.output_dir / 'dense_pointmap_height_grid.png'}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
