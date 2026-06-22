#!/usr/bin/env python3
"""Reconstruct remote point clouds from projection_aux labels for diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct point clouds from projection_aux.npz and compare them "
            "with the original pixel_to_point_map.npz remote pointmap label."
        )
    )
    parser.add_argument(
        "--remote_dir",
        type=Path,
        required=True,
        help=(
            "Remote provider directory containing projection_aux.npz, "
            "pixel_to_point_map.npz, and image.png."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for diagnostic PLY files and summary.json.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=600000,
        help="Maximum points per exported PLY. Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when max_points subsampling is enabled.",
    )
    parser.add_argument(
        "--export_dense_pointmap_height",
        action="store_true",
        help=(
            "Also derive a dense rel_height label from pixel_to_point_map.npz "
            "and export dense/global-projection diagnostic PLYs and PNGs."
        ),
    )
    parser.add_argument(
        "--ground_mode",
        choices=["aux_median", "aux_p10", "pointmap_p10"],
        default="aux_median",
        help="Ground-z estimate used by --export_dense_pointmap_height.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Height percentile used for PNG visualization clipping.",
    )
    return parser.parse_args()


def load_rgb(remote_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    image_path = remote_dir / "image.png"
    if not image_path.exists():
        h, w = shape
        return np.full((h, w, 3), 180, dtype=np.uint8)
    image = Image.open(image_path).convert("RGB")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def finite_xyz_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=-1)


def sample_mask(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0:
        return mask
    indices = np.flatnonzero(mask.reshape(-1))
    if indices.size <= max_points:
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(indices, size=max_points, replace=False)
    sampled = np.zeros(mask.size, dtype=bool)
    sampled[keep] = True
    return sampled.reshape(mask.shape)


def export_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray) -> int:
    points = xyz[mask].reshape(-1, 3)
    colors = rgb[mask].reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(path)
    return int(points.shape[0])


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, dx: float, dy: float, color: tuple[int, int, int]) -> None:
    x1 = x0 + dx
    y1 = y0 + dy
    draw.line((x0, y0, x1, y1), fill=color, width=2)
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    head = min(max(norm * 0.35, 3.0), 8.0)
    draw.line((x1, y1, x1 - head * ux + 0.5 * head * px, y1 - head * uy + 0.5 * head * py), fill=color, width=2)
    draw.line((x1, y1, x1 - head * ux - 0.5 * head * px, y1 - head * uy - 0.5 * head * py), fill=color, width=2)


def save_direction_overlay(
    path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    offset_xy: np.ndarray,
    global_offset_xy: np.ndarray,
    stride: int = 64,
) -> dict:
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sample = mask & (xx % stride == 0) & (yy % stride == 0) & np.isfinite(offset_xy).all(axis=-1)
    if not sample.any():
        return {"written": False, "reason": "no sampled valid vectors"}

    vectors = np.concatenate(
        [
            offset_xy[sample].reshape(-1, 2),
            global_offset_xy[sample].reshape(-1, 2),
        ],
        axis=0,
    )
    mag = np.linalg.norm(vectors, axis=-1)
    scale = 24.0 / max(float(np.percentile(mag[np.isfinite(mag)], 90)), 1e-6)

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y, off, goff in zip(xx[sample], yy[sample], offset_xy[sample], global_offset_xy[sample]):
        # Red: projection offset direction. Green: inverse correction direction used for reconstruction.
        # Blue: global rel_height*slope*dir approximation to the projection offset.
        draw_arrow(draw, float(x), float(y), float(off[0] * scale), float(off[1] * scale), (255, 40, 40))
        draw_arrow(draw, float(x), float(y), float(-off[0] * scale), float(-off[1] * scale), (40, 220, 80))
        draw_arrow(draw, float(x), float(y), float(goff[0] * scale), float(goff[1] * scale), (40, 120, 255))
    draw.rectangle((8, 8, 565, 76), fill=(255, 255, 255))
    draw.text((16, 16), "red: label projection offset = projected - original", fill=(255, 40, 40))
    draw.text((16, 36), "green: reconstruction correction = original - projected", fill=(40, 160, 60))
    draw.text((16, 56), "blue: rel_height * slope * global_dir", fill=(40, 80, 220))
    image.save(path)
    return {
        "written": True,
        "path": str(path),
        "stride": int(stride),
        "sampled_vectors": int(sample.sum()),
        "arrow_scale_pixels_per_world_unit": float(scale),
    }


def estimate_ground_z(
    pointmap: np.ndarray,
    point_valid: np.ndarray,
    aux_rel_height: np.ndarray,
    aux_original: np.ndarray,
    aux_valid: np.ndarray,
    mode: str,
) -> float:
    common = aux_valid & finite_xyz_mask(aux_original) & np.isfinite(aux_rel_height)
    if mode == "aux_median":
        if not common.any():
            raise ValueError("aux_median ground mode requires valid projection_aux labels")
        return float(np.median(aux_original[..., 2][common] - aux_rel_height[common]))
    if mode == "aux_p10":
        if not common.any():
            raise ValueError("aux_p10 ground mode requires valid projection_aux labels")
        return float(np.percentile(aux_original[..., 2][common], 10))
    if mode == "pointmap_p10":
        if not point_valid.any():
            raise ValueError("pointmap_p10 ground mode requires valid pointmap labels")
        return float(np.percentile(pointmap[..., 2][point_valid], 10))
    raise ValueError(mode)


def colorize_height(values: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not mask.any():
        return out
    norm = np.zeros(values.shape, dtype=np.float32)
    norm[mask] = np.clip(np.maximum(values[mask], 0.0) / max(vmax, 1e-6), 0.0, 1.0)
    out[..., 0] = np.clip(255.0 * np.maximum(0.0, 2.0 * norm - 0.5), 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * norm - 1.0)), 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(255.0 * np.maximum(0.0, 1.0 - 2.0 * norm), 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def colorize_abs(values: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not mask.any():
        return out
    norm = np.zeros(values.shape, dtype=np.float32)
    norm[mask] = np.clip(np.abs(values[mask]) / max(vmax, 1e-6), 0.0, 1.0)
    out[..., 0] = (255.0 * norm).astype(np.uint8)
    out[..., 1] = (255.0 * (1.0 - norm)).astype(np.uint8)
    out[~mask] = 0
    return out


def mask_to_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = color
    return out


def save_grid(panels: list[tuple[str, np.ndarray]], path: Path, columns: int = 3) -> None:
    titled = []
    for title, panel in panels:
        image = Image.fromarray(panel)
        canvas = Image.new("RGB", (image.width, image.height + 28), (255, 255, 255))
        canvas.paste(image, (0, 28))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=(0, 0, 0))
        titled.append(canvas)
    width, height = titled[0].size
    rows = int(np.ceil(len(titled) / columns))
    grid = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(titled):
        grid.paste(image, ((idx % columns) * width, (idx // columns) * height))
    grid.save(path)


def scalar_stats(values: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(values)
    if not valid.any():
        return {"count": 0}
    selected = values[valid]
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


def error_stats(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & finite_xyz_mask(pred) & finite_xyz_mask(gt)
    if not valid.any():
        return {"count": 0}
    err_xyz = pred[valid] - gt[valid]
    err = np.linalg.norm(err_xyz, axis=-1)
    abs_z = np.abs(err_xyz[:, 2])
    return {
        "count": int(valid.sum()),
        "mae_l2": float(err.mean()),
        "p50_l2": float(np.percentile(err, 50)),
        "p95_l2": float(np.percentile(err, 95)),
        "p99_l2": float(np.percentile(err, 99)),
        "max_l2": float(err.max()),
        "mae_xyz": [float(v) for v in np.abs(err_xyz).mean(axis=0)],
        "p95_abs_z": float(np.percentile(abs_z, 95)),
    }


def xyz_stats(xyz: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & finite_xyz_mask(xyz)
    if not valid.any():
        return {"count": 0}
    pts = xyz[valid]
    return {
        "count": int(valid.sum()),
        "min": [float(v) for v in pts.min(axis=0)],
        "max": [float(v) for v in pts.max(axis=0)],
        "mean": [float(v) for v in pts.mean(axis=0)],
        "std": [float(v) for v in pts.std(axis=0)],
        "z_p50": float(np.percentile(pts[:, 2], 50)),
        "z_p95": float(np.percentile(pts[:, 2], 95)),
        "z_p99": float(np.percentile(pts[:, 2], 99)),
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
    pointmap = np.load(pointmap_path)["xyz"].astype(np.float32)

    valid_mask = aux["valid_mask"].astype(bool)
    projected = aux["projected_xyz_centered"].astype(np.float32)
    aux_original = aux["original_xyz_world"].astype(np.float32)
    rel_height = aux["rel_height"].astype(np.float32)
    offset_xy = aux["offset_xy"].astype(np.float32)
    center_xy = aux["projection_center_xy"].astype(np.float32).reshape(1, 1, 2)
    global_dir = aux["global_dir_xy"].astype(np.float32).reshape(1, 1, 2)
    global_slope = float(aux["global_slope"].reshape(-1)[0])

    recon_from_offset = projected.copy()
    recon_from_offset[..., :2] = projected[..., :2] + center_xy - offset_xy

    offset_from_rel_global = rel_height[..., None] * global_slope * global_dir
    recon_from_rel_global = projected.copy()
    recon_from_rel_global[..., :2] = projected[..., :2] + center_xy - offset_from_rel_global

    recon_from_rel_global_plus = projected.copy()
    recon_from_rel_global_plus[..., :2] = projected[..., :2] + center_xy + offset_from_rel_global

    common_mask = (
        valid_mask
        & finite_xyz_mask(pointmap)
        & finite_xyz_mask(aux_original)
        & finite_xyz_mask(recon_from_offset)
    )
    rgb = load_rgb(args.remote_dir, valid_mask.shape)
    export_mask = sample_mask(common_mask, args.max_points, args.seed)
    projected_mask = sample_mask(valid_mask & finite_xyz_mask(projected), args.max_points, args.seed)
    pointmap_mask = sample_mask(finite_xyz_mask(pointmap), args.max_points, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = {
        "pixel_to_point_map": export_ply(
            args.output_dir / "pixel_to_point_map_xyz.ply", pointmap, rgb, pointmap_mask
        ),
        "aux_original_xyz_world": export_ply(
            args.output_dir / "aux_original_xyz_world_common.ply", aux_original, rgb, export_mask
        ),
        "aux_reconstructed_from_offset": export_ply(
            args.output_dir / "aux_reconstructed_from_offset_common.ply",
            recon_from_offset,
            rgb,
            export_mask,
        ),
        "aux_reconstructed_from_rel_global": export_ply(
            args.output_dir / "aux_reconstructed_from_rel_global_common.ply",
            recon_from_rel_global,
            rgb,
            export_mask,
        ),
        "aux_reconstructed_from_rel_global_plus": export_ply(
            args.output_dir / "aux_reconstructed_from_rel_global_plus_common.ply",
            recon_from_rel_global_plus,
            rgb,
            export_mask,
        ),
        "aux_projected_xyz_centered": export_ply(
            args.output_dir / "aux_projected_xyz_centered.ply", projected, rgb, projected_mask
        ),
    }
    direction_overlay = save_direction_overlay(
        args.output_dir / "projection_direction_overlay.png",
        rgb,
        common_mask,
        offset_xy,
        offset_from_rel_global,
    )

    dense_summary = None
    if args.export_dense_pointmap_height:
        point_valid = finite_xyz_mask(pointmap)
        dense_ground_z = estimate_ground_z(
            pointmap,
            point_valid,
            rel_height,
            aux_original,
            valid_mask,
            args.ground_mode,
        )
        dense_rel_height = pointmap[..., 2] - np.float32(dense_ground_z)
        dense_rel_height[~point_valid] = np.nan
        dense_offset_from_global = dense_rel_height[..., None] * global_slope * global_dir

        dense_projected_from_global = pointmap.copy()
        dense_projected_from_global[..., :2] = (
            pointmap[..., :2] - center_xy + dense_offset_from_global
        )
        dense_recon_from_global = dense_projected_from_global.copy()
        dense_recon_from_global[..., :2] = (
            dense_projected_from_global[..., :2] + center_xy - dense_offset_from_global
        )

        dense_mask = sample_mask(point_valid, args.max_points, args.seed)
        exported.update(
            {
                "dense_pointmap_original": export_ply(
                    args.output_dir / "dense_pointmap_original_xyz.ply",
                    pointmap,
                    rgb,
                    dense_mask,
                ),
                "dense_projected_from_global_centered": export_ply(
                    args.output_dir / "dense_projected_from_global_centered.ply",
                    dense_projected_from_global,
                    rgb,
                    dense_mask,
                ),
                "dense_reconstructed_from_rel_global": export_ply(
                    args.output_dir / "dense_reconstructed_from_rel_global.ply",
                    dense_recon_from_global,
                    rgb,
                    dense_mask,
                ),
            }
        )

        common_dense = point_valid & valid_mask & np.isfinite(rel_height)
        dense_only = point_valid & ~valid_mask
        aux_only = valid_mask & ~point_valid
        dense_diff = dense_rel_height - rel_height
        height_vmax = float(
            np.percentile(
                np.maximum(dense_rel_height[point_valid], 0.0),
                float(args.clip_percentile),
            )
        )
        diff_vmax = (
            float(np.percentile(np.abs(dense_diff[common_dense]), 95))
            if common_dense.any()
            else 1.0
        )
        overlay = rgb.copy()
        overlay[dense_only] = (
            0.55 * overlay[dense_only] + np.array([255, 0, 255]) * 0.45
        ).astype(np.uint8)
        overlay[aux_only] = (
            0.55 * overlay[aux_only] + np.array([0, 255, 255]) * 0.45
        ).astype(np.uint8)
        save_grid(
            [
                ("rgb", rgb),
                ("pointmap valid", mask_to_rgb(point_valid, (255, 255, 255))),
                ("aux valid", mask_to_rgb(valid_mask, (255, 255, 255))),
                ("dense-only added", mask_to_rgb(dense_only, (255, 0, 255))),
                ("aux-only old", mask_to_rgb(aux_only, (0, 255, 255))),
                ("mask overlay", overlay),
                ("aux rel_height", colorize_height(rel_height, valid_mask, height_vmax)),
                ("dense rel_height", colorize_height(dense_rel_height, point_valid, height_vmax)),
                ("abs diff common", colorize_abs(dense_diff, common_dense, diff_vmax)),
            ],
            args.output_dir / "dense_pointmap_height_grid.png",
        )
        Image.fromarray(colorize_height(dense_rel_height, point_valid, height_vmax)).save(
            args.output_dir / "dense_rel_height.png"
        )
        Image.fromarray(colorize_height(rel_height, valid_mask, height_vmax)).save(
            args.output_dir / "aux_rel_height.png"
        )
        Image.fromarray(mask_to_rgb(dense_only, (255, 0, 255))).save(
            args.output_dir / "dense_only_mask.png"
        )
        np.savez_compressed(
            args.output_dir / "dense_pointmap_projection_aux_labels.npz",
            dense_valid_mask=point_valid,
            dense_rel_height=dense_rel_height.astype(np.float32),
            dense_offset_from_global=dense_offset_from_global.astype(np.float32),
            dense_projected_from_global_centered=dense_projected_from_global.astype(np.float32),
            dense_reconstructed_from_rel_global=dense_recon_from_global.astype(np.float32),
            ground_z=np.asarray([dense_ground_z], dtype=np.float32),
            global_dir_xy=global_dir.reshape(2).astype(np.float32),
            global_slope=np.asarray([global_slope], dtype=np.float32),
            projection_center_xy=center_xy.reshape(2).astype(np.float32),
        )
        dense_summary = {
            "ground_mode": args.ground_mode,
            "ground_z": dense_ground_z,
            "pointmap_valid_ratio": float(point_valid.mean()),
            "aux_valid_ratio": float(valid_mask.mean()),
            "common_ratio": float(common_dense.mean()),
            "dense_only_ratio": float(dense_only.mean()),
            "aux_only_ratio": float(aux_only.mean()),
            "relative_coverage_gain": float(point_valid.sum() / max(int(valid_mask.sum()), 1)),
            "dense_rel_height_stats": scalar_stats(dense_rel_height, point_valid),
            "aux_rel_height_stats": scalar_stats(rel_height, valid_mask),
            "dense_minus_aux_common_stats": scalar_stats(dense_diff, common_dense),
            "abs_dense_minus_aux_common_stats": scalar_stats(np.abs(dense_diff), common_dense),
            "dense_recon_global_vs_pointmap": error_stats(
                dense_recon_from_global, pointmap, point_valid
            ),
            "dense_projected_from_global_stats": xyz_stats(
                dense_projected_from_global, point_valid
            ),
        }

    high_mask = common_mask & (rel_height >= np.percentile(rel_height[valid_mask], 80))
    low_mask = common_mask & (rel_height <= np.percentile(rel_height[valid_mask], 50))

    summary = {
        "remote_dir": str(args.remote_dir),
        "valid_ratio_aux": float(valid_mask.mean()),
        "valid_ratio_pointmap": float(finite_xyz_mask(pointmap).mean()),
        "common_ratio": float(common_mask.mean()),
        "global_dir_xy": [float(v) for v in global_dir.reshape(-1)],
        "global_slope": global_slope,
        "projection_center_xy": [float(v) for v in center_xy.reshape(-1)],
        "exported_points": exported,
        "stats": {
            "pixel_to_point_map": xyz_stats(pointmap, common_mask),
            "aux_original_xyz_world": xyz_stats(aux_original, common_mask),
            "aux_reconstructed_from_offset": xyz_stats(recon_from_offset, common_mask),
            "aux_reconstructed_from_rel_global": xyz_stats(recon_from_rel_global, common_mask),
            "aux_reconstructed_from_rel_global_plus": xyz_stats(recon_from_rel_global_plus, common_mask),
            "aux_projected_xyz_centered": xyz_stats(projected, valid_mask),
        },
        "errors_common": {
            "aux_original_vs_pixel_to_point_map": error_stats(aux_original, pointmap, common_mask),
            "recon_offset_vs_aux_original": error_stats(recon_from_offset, aux_original, common_mask),
            "recon_rel_global_vs_aux_original": error_stats(
                recon_from_rel_global, aux_original, common_mask
            ),
            "recon_rel_global_plus_vs_aux_original": error_stats(
                recon_from_rel_global_plus, aux_original, common_mask
            ),
            "projected_vs_aux_original": error_stats(projected, aux_original, common_mask),
        },
        "errors_high_rel_height": {
            "aux_original_vs_pixel_to_point_map": error_stats(aux_original, pointmap, high_mask),
            "recon_offset_vs_aux_original": error_stats(recon_from_offset, aux_original, high_mask),
            "recon_rel_global_vs_aux_original": error_stats(
                recon_from_rel_global, aux_original, high_mask
            ),
            "recon_rel_global_plus_vs_aux_original": error_stats(
                recon_from_rel_global_plus, aux_original, high_mask
            ),
        },
        "errors_low_rel_height": {
            "aux_original_vs_pixel_to_point_map": error_stats(aux_original, pointmap, low_mask),
            "recon_offset_vs_aux_original": error_stats(recon_from_offset, aux_original, low_mask),
            "recon_rel_global_vs_aux_original": error_stats(
                recon_from_rel_global, aux_original, low_mask
            ),
            "recon_rel_global_plus_vs_aux_original": error_stats(
                recon_from_rel_global_plus, aux_original, low_mask
            ),
        },
        "direction_overlay": direction_overlay,
        "dense_pointmap_height": dense_summary,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote diagnostics to {args.output_dir}")
    print(json.dumps(summary["errors_common"], indent=2))


if __name__ == "__main__":
    main()
