#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Export a unified world-space point cloud from an image folder.

Supported benchmark models from bash_scripts/benchmark/rs_guided_dense_mv:
- pi3
- vggt
- da3
- mapanything

Examples:
    python scripts/export_pointcloud_ply.py \
        --model pi3 \
        --image_folder /path/to/images \
        --output_path /path/to/output_dir

    python scripts/export_pointcloud_ply.py \
        --model mapanything \
        --checkpoint_path /path/to/checkpoint.pth \
        --image_folder /path/to/images \
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
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from mapanything.utils.colmap_export import voxel_downsample_point_cloud
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.hf_utils.hf_helpers import (
    initialize_mapanything_local,
    initialize_mapanything_model,
)
from mapanything.utils.image import load_images

DEFAULT_MODEL = "pi3"
DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_MAPANYTHING_HF_MODEL = "facebook/map-anything"
DEFAULT_CONFIG_OVERRIDES = {
    "pi3": [
        "machine=aws",
        "model=pi3",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "vggt": [
        "machine=aws",
        "model=vggt",
    ],
    "da3": [
        "machine=aws",
        "model=da3",
    ],
    "mapanything": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
}
IDENTITY_MODELS = {"anycalib", "moge", "pi3", "pi3x", "vggt"}
CLASH_ENV = {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7891",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a supported benchmark model on an image folder and export the "
            "unified world-space point cloud as PLY."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["pi3", "vggt", "da3", "mapanything"],
        help="Model to run. Matches the rs_guided_dense_mv benchmark model set.",
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
        default=None,
        help=(
            "Optional local checkpoint (.pth/.pt/.safetensors). If omitted, the "
            "script uses the model's default HuggingFace weights."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mapanything_pointcloud.ply",
        help="Output PLY path, or a directory to receive mapanything_pointcloud.ply.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Hydra config path used for local-checkpoint initialization.",
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
        help="Optional model alias override for local-checkpoint initialization.",
    )
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=None,
        help="Optional Hydra override list. Defaults depend on --model.",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace model name for no-checkpoint runs. Currently used "
            "for mapanything; defaults to facebook/map-anything."
        ),
    )
    parser.add_argument(
        "--enable_clash_proxy",
        action="store_true",
        default=False,
        help=(
            "Set the same proxy env vars as 'source /etc/profile.d/clash.sh && proxy_on' "
            "before downloading HuggingFace weights."
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
        help="Use memory-efficient inference when the model exposes model.infer().",
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
        help="Apply non-ambiguous masks when the model exposes model.infer().",
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
        help="Filter depth discontinuity / normal edges when the model exposes model.infer().",
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


def resolve_config_overrides(args: argparse.Namespace):
    if args.config_overrides is not None:
        return args.config_overrides
    return list(DEFAULT_CONFIG_OVERRIDES[args.model])


def maybe_enable_clash_proxy(enable_proxy: bool):
    if not enable_proxy:
        return
    clash_path = Path("/etc/profile.d/clash.sh")
    if not clash_path.exists():
        print("Clash helper not found at /etc/profile.d/clash.sh; skipping proxy setup")
        return
    os.environ.update(CLASH_ENV)
    print("Enabled Clash proxy environment for HuggingFace downloads")


def maybe_prepare_da3_pythonpath(model_name: str):
    if model_name != "da3":
        return
    da3_src = Path("/root/autodl-tmp/Models/Depth-Anything-3/src")
    if not da3_src.exists():
        raise FileNotFoundError(
            "DA3 requires /root/autodl-tmp/Models/Depth-Anything-3/src to exist"
        )
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))
        print(f"Added DA3 dependency path: {da3_src}")


def build_local_config(args: argparse.Namespace, config_overrides) -> dict:
    local_config = {
        "path": args.config_path,
        "checkpoint_path": args.checkpoint_path,
        "config_overrides": config_overrides,
        "strict": args.strict,
    }
    if args.config_json_path is not None:
        local_config["config_json_path"] = args.config_json_path
    if args.model_str is not None:
        local_config["model_str"] = args.model_str
    return local_config


def initialize_model(args: argparse.Namespace, device: str, config_overrides):
    maybe_enable_clash_proxy(args.enable_clash_proxy)
    maybe_prepare_da3_pythonpath(args.model)

    if args.checkpoint_path:
        local_config = build_local_config(args, config_overrides)
        print(f"Initializing model from local config: {local_config}")
        model = initialize_mapanything_local(local_config, device)
        print("Successfully loaded local checkpoint")
        return model

    if args.model == "mapanything":
        hf_model_name = args.hf_model_name or DEFAULT_MAPANYTHING_HF_MODEL
        high_level_config = {
            "path": args.config_path,
            "hf_model_name": hf_model_name,
            "model_str": "mapanything",
            "config_overrides": config_overrides,
            "checkpoint_name": "model.safetensors",
            "config_name": "config.json",
        }
        print(f"Initializing model from HuggingFace defaults: {high_level_config}")
        model = initialize_mapanything_model(high_level_config, device)
        print("Successfully loaded HuggingFace weights")
        return model

    from mapanything.models import init_model_from_config

    print(f"Initializing model '{args.model}' from default wrapper weights")
    model = init_model_from_config(args.model, device=device, machine="aws").eval()
    print("Successfully loaded default wrapper weights")
    return model


def convert_views_to_identity_if_needed(views, model_name: str):
    if model_name not in IDENTITY_MODELS:
        return views

    converted_views = []
    for view in views:
        norm_type = view["data_norm_type"][0]
        if norm_type == "identity":
            converted_views.append(view)
            continue

        if norm_type not in IMAGE_NORMALIZATION_DICT:
            raise ValueError(f"Unsupported norm_type for identity conversion: {norm_type}")

        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        mean = torch.as_tensor(
            img_norm.mean,
            dtype=view["img"].dtype,
            device=view["img"].device,
        ).view(1, -1, 1, 1)
        std = torch.as_tensor(
            img_norm.std,
            dtype=view["img"].dtype,
            device=view["img"].device,
        ).view(1, -1, 1, 1)

        converted_view = dict(view)
        converted_view["img"] = (view["img"] * std + mean).clamp(0, 1)
        converted_view["data_norm_type"] = ["identity"]
        converted_views.append(converted_view)

    return converted_views


def move_views_to_device(views, device: torch.device):
    moved_views = []
    for view in views:
        moved_view = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                moved_view[key] = value.to(device)
            else:
                moved_view[key] = value
        moved_views.append(moved_view)
    return moved_views


def run_model_inference(model, views, args: argparse.Namespace):
    if hasattr(model, "infer"):
        return model.infer(
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

    model_device = next(model.parameters()).device
    return model(move_views_to_device(views, model_device))


def get_view_colors(pred, view):
    if "img_no_norm" in pred:
        image_np = pred["img_no_norm"][0].cpu().numpy()
    else:
        image_np = view["img"][0].permute(1, 2, 0).cpu().numpy()
    return np.clip(image_np * 255.0, 0, 255).astype(np.uint8)


def collect_world_space_point_cloud(
    outputs,
    views,
    apply_confidence_mask=False,
    confidence_percentile=50.0,
):
    all_points = []
    all_colors = []
    per_view_stats = []

    for view_idx, pred in enumerate(outputs):
        if "pts3d" in pred:
            pts3d_np = pred["pts3d"][0].cpu().numpy()
            export_mask = np.isfinite(pts3d_np).all(axis=-1)
            if apply_confidence_mask and "conf" in pred:
                conf_np = pred["conf"][0].cpu().numpy()
                if conf_np.ndim == 3 and conf_np.shape[-1] == 1:
                    conf_np = conf_np.squeeze(-1)
                valid_conf = conf_np[export_mask]
                if valid_conf.size > 0:
                    conf_threshold = np.percentile(valid_conf, confidence_percentile)
                    export_mask &= conf_np >= conf_threshold
        else:
            depthmap_torch = pred["depth_z"][0].squeeze(-1)
            intrinsics_torch = pred["intrinsics"][0]
            camera_pose_torch = pred["camera_poses"][0]

            pts3d_world, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )
            pts3d_np = pts3d_world.cpu().numpy()

            valid_mask_np = valid_mask.cpu().numpy()
            if "mask" in pred:
                export_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
                export_mask &= valid_mask_np
            else:
                export_mask = valid_mask_np

        colors_np = get_view_colors(pred, views[view_idx])
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


def resolve_output_path(output_path_str: str) -> Path:
    output_path = Path(output_path_str)
    if output_path.suffix.lower() == ".ply":
        return output_path
    if output_path.exists() and output_path.is_dir():
        return output_path / "mapanything_pointcloud.ply"
    if output_path.suffix == "":
        return output_path / "mapanything_pointcloud.ply"
    return output_path.with_suffix(".ply")


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_overrides = resolve_config_overrides(args)
    model = initialize_model(args, device, config_overrides)

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

    model_name = getattr(model, "name", args.model)
    views = convert_views_to_identity_if_needed(views, model_name)

    print("Running inference...")
    start_time = time()
    with torch.inference_mode():
        outputs = run_model_inference(model, views, args)
    duration = time() - start_time
    print(f"Inference finished in {duration:.3f}s")

    print("Collecting unified world-space point cloud...")
    points, colors, per_view_stats = collect_world_space_point_cloud(
        outputs,
        views,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
    )
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

    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path)
    print(f"Saved unified point cloud PLY to: {output_path}")


if __name__ == "__main__":
    main()
