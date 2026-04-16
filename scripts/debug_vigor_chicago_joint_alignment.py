#!/usr/bin/env python3
"""
Debug the coordinate-frame alignment between VIGOR Chicago aerial multi-view point clouds
and remote-sensing global pointmaps for a single scene.

This script is intended to test the hypothesis:
    remote_global ~= aerial_world
and therefore:
    geotrf(inv(view0_pose), remote_global) ~= aerial_view0

PYTHONPATH=. python scripts/debug_vigor_chicago_joint_alignment.py --scene-name location_1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mapanything.datasets.wai.vigor_chicago_joint_rs_aerial import (
    VigorChicagoJointRSAerial,
)


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect aerial/remote point-cloud alignment for one VIGOR Chicago scene."
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument(
        "--aerial-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_wai"),
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/Crossview"),
    )
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_rs"),
    )
    parser.add_argument("--remote-provider", default="Google_Satellite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/Crossview_joint_alignment"
        ),
    )
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--nn-samples", type=int, default=2048)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan, dtype=np.float32)
    if valid.any():
        homo = np.concatenate(
            [flat[valid].astype(np.float32), np.ones((valid.sum(), 1), dtype=np.float32)],
            axis=1,
        )
        transformed = (transform.astype(np.float32) @ homo.T).T[:, :3]
        out[valid] = transformed
    return out.reshape(points.shape)


def subsample_valid_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    flat = flat[valid]
    if flat.shape[0] == 0:
        return flat.astype(np.float32)
    if flat.shape[0] > max_points:
        idx = rng.choice(flat.shape[0], size=max_points, replace=False)
        flat = flat[idx]
    return flat.astype(np.float32)


def cloud_stats(points: np.ndarray) -> dict:
    if points.shape[0] == 0:
        return {
            "num_points": 0,
            "centroid": [float("nan")] * 3,
            "bbox_min": [float("nan")] * 3,
            "bbox_max": [float("nan")] * 3,
            "mean_radius": float("nan"),
        }
    centroid = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    mean_radius = np.linalg.norm(points - centroid[None], axis=1).mean()
    return {
        "num_points": int(points.shape[0]),
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "mean_radius": float(mean_radius),
    }


def symmetric_nn_l2(src: np.ndarray, dst: np.ndarray, num_samples: int, rng: np.random.Generator) -> dict:
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return {
            "src_to_dst_mean_l2": float("nan"),
            "dst_to_src_mean_l2": float("nan"),
            "symmetric_mean_l2": float("nan"),
            "src_to_dst_median_l2": float("nan"),
            "dst_to_src_median_l2": float("nan"),
        }

    if src.shape[0] > num_samples:
        src = src[rng.choice(src.shape[0], size=num_samples, replace=False)]
    if dst.shape[0] > num_samples:
        dst = dst[rng.choice(dst.shape[0], size=num_samples, replace=False)]

    src_t = torch.from_numpy(src)
    dst_t = torch.from_numpy(dst)
    dists = torch.cdist(src_t, dst_t, p=2)
    src_to_dst = dists.min(dim=1).values.cpu().numpy()
    dst_to_src = dists.min(dim=0).values.cpu().numpy()
    return {
        "src_to_dst_mean_l2": float(src_to_dst.mean()),
        "dst_to_src_mean_l2": float(dst_to_src.mean()),
        "symmetric_mean_l2": float(0.5 * (src_to_dst.mean() + dst_to_src.mean())),
        "src_to_dst_median_l2": float(np.median(src_to_dst)),
        "dst_to_src_median_l2": float(np.median(dst_to_src)),
    }



def write_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors size mismatch: {points.shape[0]} vs {colors.shape[0]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = """ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".format(num_vertices=points.shape[0])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_colored_clouds(clouds: list[tuple[str, np.ndarray, tuple[int, int, int]]], out_path: Path) -> None:
    all_points = []
    all_colors = []
    for _, points, color in clouds:
        if points.shape[0] == 0:
            continue
        all_points.append(points.astype(np.float32))
        color_arr = np.tile(np.asarray(color, dtype=np.uint8)[None], (points.shape[0], 1))
        all_colors.append(color_arr)
    if not all_points:
        raise ValueError(f"No valid points to write: {out_path}")
    write_ply(np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), out_path)


def export_alignment_plys(output_dir: Path, scene_name: str, aerial_world: np.ndarray, aerial_view0: np.ndarray, remote_global: np.ndarray, remote_view0: np.ndarray) -> dict:
    paths = {
        "aerial_world_ply": output_dir / f"{scene_name}_aerial_world.ply",
        "aerial_view0_ply": output_dir / f"{scene_name}_aerial_view0.ply",
        "remote_global_ply": output_dir / f"{scene_name}_remote_global.ply",
        "remote_view0_ply": output_dir / f"{scene_name}_remote_view0.ply",
        "global_combo_ply": output_dir / f"{scene_name}_global_combo.ply",
        "view0_combo_ply": output_dir / f"{scene_name}_view0_combo.ply",
    }

    write_colored_clouds([
        ("aerial_world", aerial_world, (255, 80, 80)),
    ], paths["aerial_world_ply"])
    write_colored_clouds([
        ("aerial_view0", aerial_view0, (255, 80, 80)),
    ], paths["aerial_view0_ply"])
    write_colored_clouds([
        ("remote_global", remote_global, (80, 170, 255)),
    ], paths["remote_global_ply"])
    write_colored_clouds([
        ("remote_view0", remote_view0, (80, 170, 255)),
    ], paths["remote_view0_ply"])
    write_colored_clouds([
        ("aerial_world", aerial_world, (255, 80, 80)),
        ("remote_global", remote_global, (80, 170, 255)),
    ], paths["global_combo_ply"])
    write_colored_clouds([
        ("aerial_view0", aerial_view0, (255, 80, 80)),
        ("remote_view0", remote_view0, (80, 170, 255)),
    ], paths["view0_combo_ply"])
    return {k: str(v) for k, v in paths.items()}

def make_topdown_plot(clouds: list[tuple[str, np.ndarray]], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(clouds), figsize=(5 * len(clouds), 5), constrained_layout=True)
    if len(clouds) == 1:
        axes = [axes]

    for ax, (name, pts) in zip(axes, clouds):
        if pts.shape[0] > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.35)
            ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VigorChicagoJointRSAerial(
        split=args.split,
        resolution=(args.resolution, args.resolution),
        principal_point_centered=False,
        seed=777,
        transform="imgnorm",
        data_norm_type="croco",
        ROOT=str(args.aerial_root),
        dataset_metadata_dir=str(args.metadata_root),
        overfit_num_sets=None,
        variable_num_views=False,
        num_views=args.num_views,
        covisibility_thres=0.05,
        remote_ROOT=str(args.remote_root),
        remote_providers=[args.remote_provider],
        remote_resolution=(args.resolution, args.resolution),
        remote_transform="imgnorm",
        skip_missing_remote=False,
    )

    if args.scene_name is None:
        scene_index = int(args.scene_index)
    else:
        if args.scene_name not in dataset.scenes:
            raise ValueError(f"Scene {args.scene_name} not found in split {args.split}")
        scene_index = dataset.scenes.index(args.scene_name)

    scene_name = dataset.scenes[scene_index]
    sample = dataset[scene_index]

    rng = np.random.default_rng(0)
    view0_pose = to_numpy(sample[0]["camera_pose"]).astype(np.float32)
    in_view0 = np.linalg.inv(view0_pose).astype(np.float32)

    aerial_world_per_view = []
    aerial_view0_per_view = []
    for view in sample:
        world_pts = to_numpy(view["pts3d"]).astype(np.float32)
        aerial_world_per_view.append(world_pts)
        aerial_view0_per_view.append(homogeneous_transform(world_pts, in_view0))

    remote_global = to_numpy(sample[0]["remote_pointmap"]).astype(np.float32)
    remote_view0 = homogeneous_transform(remote_global, in_view0)

    aerial_world = np.concatenate(
        [subsample_valid_points(points, args.max_points, rng) for points in aerial_world_per_view],
        axis=0,
    )
    aerial_view0 = np.concatenate(
        [subsample_valid_points(points, args.max_points, rng) for points in aerial_view0_per_view],
        axis=0,
    )
    remote_global_pts = subsample_valid_points(remote_global, args.max_points, rng)
    remote_view0_pts = subsample_valid_points(remote_view0, args.max_points, rng)

    metrics = {
        "aerial_world_vs_remote_global": symmetric_nn_l2(
            aerial_world, remote_global_pts, args.nn_samples, rng
        ),
        "aerial_view0_vs_remote_global": symmetric_nn_l2(
            aerial_view0, remote_global_pts, args.nn_samples, rng
        ),
        "aerial_view0_vs_remote_view0": symmetric_nn_l2(
            aerial_view0, remote_view0_pts, args.nn_samples, rng
        ),
    }

    ply_paths = export_alignment_plys(
        args.output_dir,
        scene_name,
        aerial_world,
        aerial_view0,
        remote_global_pts,
        remote_view0_pts,
    )

    summary = {
        "scene_name": scene_name,
        "split": args.split,
        "num_views": len(sample),
        "view0_pose": view0_pose.tolist(),
        "cloud_stats": {
            "aerial_world": cloud_stats(aerial_world),
            "aerial_view0": cloud_stats(aerial_view0),
            "remote_global": cloud_stats(remote_global_pts),
            "remote_view0": cloud_stats(remote_view0_pts),
        },
        "nn_metrics": metrics,
        "ply_paths": ply_paths,
    }

    summary_path = args.output_dir / f"{scene_name}_alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_plots:
        make_topdown_plot(
            [
                ("aerial_world", aerial_world),
                ("remote_global", remote_global_pts),
            ],
            args.output_dir / f"{scene_name}_global_topdown.png",
            f"{scene_name}: aerial world vs remote global",
        )
        make_topdown_plot(
            [
                ("aerial_view0", aerial_view0),
                ("remote_global", remote_global_pts),
                ("remote_view0", remote_view0_pts),
            ],
            args.output_dir / f"{scene_name}_view0_topdown.png",
            f"{scene_name}: view0-frame test",
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
