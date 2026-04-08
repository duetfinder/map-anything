#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Offline inference from local weights directly to a unified PLY point cloud.

Example:
    python scripts/export_pointcloud_ply.py \
        --image_folder /path/to/images \
        --checkpoint_path /path/to/checkpoint.pth \
        --output_path /path/to/output.ply
"""

import argparse
import os
import sys
from pathlib import Path
from time import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import trimesh

from mapanything.utils.colmap_export import voxel_downsample_point_cloud
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_local
from mapanything.utils.image import load_images

DEFAULT_CONFIG_OVERRIDES = [
    "machine=aws",
    "model=pi3",
    "model/task=images_only",
    "model.encoder.uses_torch_hub=false",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MapAnything / PI3 local-weight inference on an image folder and "
            "export the unified world-space point cloud as PLY."
        )
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing input images.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the local checkpoint (.pth/.pt/.safetensors).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mapanything_pointcloud.ply",
        help="Output PLY path.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/train.yaml",
        help="Hydra config path used to build the model.",
    )
    parser.add_argument(
        "--config_json_path",
        type=str,
        default=None,
        help="Optional JSON containing model_str/model_config overrides.",
    )
    parser.add_argument(
        "--model_str",
        type=str,
        default=None,
        help="Optional model alias, e.g. pi3 or mapanything. Defaults to config/json.",
    )
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=DEFAULT_CONFIG_OVERRIDES,
        help=(
            "Hydra override list. Defaults are tuned for local PI3 images-only inference."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load checkpoint with strict=True. Default is False for compatibility.",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory-efficient inference.",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size used by model.infer in memory-efficient mode.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="fixed_mapping",
        choices=["fixed_mapping", "longest_side", "square", "fixed_size"],
        help="Resize mode passed to load_images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize size for longest_side/square modes.",
    )
    parser.add_argument(
        "--fixed_width",
        type=int,
        default=None,
        help="Resize width for fixed_size mode.",
    )
    parser.add_argument(
        "--fixed_height",
        type=int,
        default=None,
        help="Resize height for fixed_size mode.",
    )
    parser.add_argument(
        "--resolution_set",
        type=int,
        default=518,
        choices=[504, 512, 518],
        help="Resolution preset used by load_images when resize_mode=fixed_mapping.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Load every nth image from the folder.",
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        default=True,
        help="Apply non-ambiguous masks to output geometry.",
    )
    parser.add_argument(
        "--no_apply_mask",
        action="store_false",
        dest="apply_mask",
        help="Disable non-ambiguous masking.",
    )
    parser.add_argument(
        "--mask_edges",
        action="store_true",
        default=True,
        help="Filter depth discontinuity / normal edges.",
    )
    parser.add_argument(
        "--no_mask_edges",
        action="store_false",
        dest="mask_edges",
        help="Disable edge masking.",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        default=False,
        help="Apply confidence mask before exporting the point cloud.",
    )
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=50.0,
        help="Percentile threshold used when apply_confidence_mask is enabled.",
    )
    parser.add_argument(
        "--voxel_downsample",
        action="store_true",
        default=False,
        help="Apply voxel downsampling before exporting. Requires open3d.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Explicit voxel size in world units. If unset, use voxel_fraction.",
    )
    parser.add_argument(
        "--voxel_fraction",
        type=float,
        default=0.01,
        help="Adaptive voxel size fraction used when voxel_size is not set.",
    )
    return parser.parse_args()


def resolve_load_size(args: argparse.Namespace):
    if args.resize_mode == "fixed_size":
        if args.fixed_width is None or args.fixed_height is None:
            raise ValueError(
                "--fixed_width and --fixed_height are required when --resize_mode fixed_size"
            )
        return (args.fixed_width, args.fixed_height)
    if args.resize_mode in {"longest_side", "square"}:
        if args.size is None:
            raise ValueError(
                f"--size is required when --resize_mode {args.resize_mode}"
            )
        return args.size
    return None


def build_local_config(args: argparse.Namespace) -> dict:
    local_config = {
        "path": args.config_path,
        "checkpoint_path": args.checkpoint_path,
        "config_overrides": args.config_overrides,
        "strict": args.strict,
    }
    if args.config_json_path is not None:
        local_config["config_json_path"] = args.config_json_path
    if args.model_str is not None:
        local_config["model_str"] = args.model_str
    return local_config


def collect_world_space_point_cloud(outputs):
    all_points = []
    all_colors = []
    per_view_stats = []

    for view_idx, pred in enumerate(outputs):
        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]

        pts3d_world, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        valid_mask_np = valid_mask.cpu().numpy()
        if "mask" in pred:
            export_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
            export_mask &= valid_mask_np
        else:
            export_mask = valid_mask_np

        pts3d_np = pts3d_world.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()
        colors_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

        selected_points = pts3d_np[export_mask]
        selected_colors = colors_np[export_mask]

        per_view_stats.append(
            {
                "view_idx": view_idx,
                "points": int(selected_points.shape[0]),
            }
        )

        if selected_points.shape[0] == 0:
            continue

        all_points.append(selected_points)
        all_colors.append(selected_colors)

    if not all_points:
        raise RuntimeError("No valid points remained after masking; cannot export PLY.")

    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0),
        per_view_stats,
    )


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    local_config = build_local_config(args)
    print(f"Initializing model from local config: {local_config}")
    model = initialize_mapanything_local(local_config, device)
    print("Successfully loaded local checkpoint")

    load_size = resolve_load_size(args)
    print(f"Loading images from: {args.image_folder}")
    views = load_images(
        args.image_folder,
        resize_mode=args.resize_mode,
        size=load_size,
        resolution_set=args.resolution_set,
        stride=args.stride,
    )
    if len(views) == 0:
        raise ValueError(f"No images found in {args.image_folder}")
    print(f"Loaded {len(views)} views")

    print("Running inference...")
    start_time = time()
    outputs = model.infer(
        views,
        memory_efficient_inference=args.memory_efficient_inference,
        minibatch_size=args.minibatch_size,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=args.apply_mask,
        mask_edges=args.mask_edges,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
    )
    duration = time() - start_time
    print(f"Inference finished in {duration:.3f}s")

    print("Collecting unified world-space point cloud...")
    points, colors, per_view_stats = collect_world_space_point_cloud(outputs)
    for stat in per_view_stats:
        print(f"View {stat['view_idx']}: kept {stat['points']} points")
    print(f"Total points before downsampling: {points.shape[0]}")

    if args.voxel_downsample:
        points, colors = voxel_downsample_point_cloud(
            points,
            colors,
            voxel_fraction=args.voxel_fraction,
            voxel_size=args.voxel_size,
        )
        print(f"Total points after downsampling: {points.shape[0]}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path)
    print(f"Saved unified point cloud PLY to: {output_path}")


if __name__ == "__main__":
    main()
