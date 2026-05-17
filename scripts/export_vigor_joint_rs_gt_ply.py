#!/usr/bin/env python3

"""
Export GT point clouds from the VIGOR Chicago joint RS aerial dataset.

This is intended for debugging p5b/p5c alignment issues by visualizing:
1. aerial GT point clouds in the common world/view0 frame
2. remote GT point cloud transformed into view0 frame
3. optional raw remote GT point cloud without the view0 transform
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything.datasets.wai.vigor_chicago_joint_rs_aerial import (  # noqa: E402
    VigorChicagoJointRSAerial,
)
from mapanything.utils.inference import _transform_remote_pointmap_to_view0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GT aerial+remote point clouds from VIGOR Chicago joint RS."
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--remote_provider", type=str, default="Google_Satellite")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--root_data_dir", type=str, default="/root/autodl-tmp/traindata")
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="/root/autodl-tmp/traindata/mapanything_metadata",
    )
    parser.add_argument(
        "--remote_color_mode",
        type=str,
        default="solid",
        choices=["solid", "image"],
    )
    parser.add_argument(
        "--export_raw_remote",
        action="store_true",
        default=True,
        help="Also export a second PLY using raw remote GT without view0 transform.",
    )
    parser.add_argument(
        "--no_export_raw_remote",
        action="store_false",
        dest="export_raw_remote",
    )
    return parser.parse_args()


def tensor_image_to_uint8_chw(image_tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy()
    image = np.clip(image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return image


def gather_points_and_colors(pts3d, mask, colors_hw3):
    points = pts3d[mask]
    colors = colors_hw3[mask]
    return points, colors


def build_dataset(args: argparse.Namespace) -> VigorChicagoJointRSAerial:
    resolution = (518, 518)
    return VigorChicagoJointRSAerial(
        split=args.split,
        resolution=resolution,
        principal_point_centered=False,
        seed=args.seed,
        transform="imgnorm",
        data_norm_type="identity",
        ROOT=f"{args.root_data_dir}/Crossview_wai",
        dataset_metadata_dir=f"{args.metadata_dir}/Crossview",
        overfit_num_sets=None,
        variable_num_views=False,
        num_views=args.num_views,
        covisibility_thres=0.0,
        view_sampling_mode="connected",
        remote_ROOT=f"{args.root_data_dir}/Crossview_rs",
        remote_providers=[args.remote_provider],
        remote_provider_map_csv=None,
        remote_dataset_metadata_dir=f"{args.metadata_dir}/Crossview_rs_aerial",
        remote_provider_sampling_mode="first_available",
        remote_resolution=resolution,
        remote_transform="imgnorm",
        skip_missing_remote=False,
        remote_crop_mode="none",
        remote_crop_scale_range=(1.0, 1.0),
        remote_image_resize_mode="nearest",
        remote_label_resize_mode="nearest",
        cities=[],
        sample_specific_scene=args.scene_name is not None,
        specific_scene_name=args.scene_name,
    )


def resolve_output_paths(output_path_str: str, export_raw_remote: bool):
    output_path = Path(output_path_str)
    if output_path.suffix.lower() == ".ply":
        base_path = output_path
    else:
        base_path = output_path / "vigor_joint_rs_gt_view0.ply"

    raw_path = None
    if export_raw_remote:
        raw_path = base_path.with_name(base_path.stem + "_raw_remote.ply")
    return base_path, raw_path


def export_point_cloud(points: np.ndarray, colors: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path)


def main():
    args = parse_args()
    dataset = build_dataset(args)
    sample = dataset[args.sample_idx]

    scene_label = sample[0]["label"]
    print(f"Loaded split={args.split} sample_idx={args.sample_idx} scene={scene_label}")
    print(f"Remote provider: {sample[0].get('remote_provider')}")

    all_points = []
    all_colors = []
    for view_idx, view in enumerate(sample):
        image = tensor_image_to_uint8_chw(view["img"])
        pts, cols = gather_points_and_colors(view["pts3d"], view["valid_mask"], image)
        print(f"Aerial view {view_idx}: {pts.shape[0]} points")
        all_points.append(pts)
        all_colors.append(cols)

    remote_pointmap = sample[0]["remote_pointmap"]
    remote_pointmap_view0 = _transform_remote_pointmap_to_view0(
        remote_pointmap, sample[0]["camera_pose"]
    )
    remote_mask = sample[0]["remote_valid_mask"].astype(bool)

    if args.remote_color_mode == "image":
        remote_image = tensor_image_to_uint8_chw(sample[0]["remote_image"])
        remote_colors = remote_image[remote_mask]
    else:
        remote_colors = np.zeros((int(remote_mask.sum()), 3), dtype=np.uint8)
        remote_colors[:, 0] = 255
        remote_colors[:, 1] = 64
        remote_colors[:, 2] = 64

    remote_points_view0 = remote_pointmap_view0[remote_mask]
    print(f"Remote GT in view0: {remote_points_view0.shape[0]} points")
    all_points.append(remote_points_view0)
    all_colors.append(remote_colors)

    combined_points = np.concatenate(all_points, axis=0)
    combined_colors = np.concatenate(all_colors, axis=0)

    output_path, raw_remote_output_path = resolve_output_paths(
        args.output_path, args.export_raw_remote
    )
    export_point_cloud(combined_points, combined_colors, output_path)
    print(f"Saved GT view0-aligned PLY to: {output_path}")

    if raw_remote_output_path is not None:
        raw_remote_points = remote_pointmap[remote_mask]
        raw_points = np.concatenate(
            [np.concatenate(all_points[:-1], axis=0), raw_remote_points], axis=0
        )
        raw_colors = np.concatenate(
            [np.concatenate(all_colors[:-1], axis=0), remote_colors], axis=0
        )
        export_point_cloud(raw_points, raw_colors, raw_remote_output_path)
        print(f"Saved raw-remote comparison PLY to: {raw_remote_output_path}")


if __name__ == "__main__":
    main()
