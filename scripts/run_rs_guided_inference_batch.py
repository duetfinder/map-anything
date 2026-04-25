#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from benchmarking.rs_guided_dense_mv.benchmark_unified import (
    compute_aerial_scene_metrics,
    compute_joint_global_point_l1,
    compute_joint_global_pointmaps_abs_rel,
    compute_remote_height_metrics,
    compute_remote_height_metrics_affine,
    diff_metric_dict,
    model_supports_metric_outputs,
)
from benchmarking.dense_n_view.benchmark import get_all_info_for_metric_computation
from mapanything.datasets.wai.vigor_chicago import VigorChicagoWAI
from mapanything.datasets.wai.vigor_chicago_rs_aerial import VigorChicagoRSAerial
from mapanything.utils.geometry import (
    depthmap_to_world_frame,
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.hf_utils.hf_helpers import (
    initialize_mapanything_local,
    initialize_mapanything_model,
)
from mapanything.utils.image import rgb
from mapanything.utils.viz import predictions_to_glb
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

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
DEFAULT_DATA_NORM = {
    "pi3": "identity",
    "vggt": "identity",
    "da3": "dinov2",
    "mapanything": "dinov2",
}
IDENTITY_MODELS = {"anycalib", "moge", "pi3", "pi3x", "vggt"}
CLASH_ENV = {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7891",
}
IGNORE_KEYS = {
    "depthmap",
    "dataset",
    "label",
    "instance",
    "idx",
    "true_shape",
    "rng",
    "data_norm_type",
}


class ExplicitFrameVigorChicago(VigorChicagoWAI):
    def __init__(self, *args, frame_indices, **kwargs):
        self.frame_indices = [int(x) for x in frame_indices]
        super().__init__(*args, num_views=len(self.frame_indices), max_num_retries=0, **kwargs)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if len(self.frame_indices) != num_views_to_sample:
            raise ValueError(
                f"Explicit frame count ({len(self.frame_indices)}) does not match "
                f"dataset num_views ({num_views_to_sample})"
            )

        scene_name = self.scenes[sampled_idx]
        scene_root = os.path.join(self.ROOT, scene_name)
        from mapanything.utils.wai.core import load_data, load_frame

        scene_meta = load_data(os.path.join(scene_root, "scene_meta.json"), "scene_meta")
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        views = []
        for view_index in self.frame_indices:
            if view_index < 0 or view_index >= num_views_in_scene:
                raise IndexError(
                    f"Frame index {view_index} is out of range for {scene_name} "
                    f"(num_frames={num_views_in_scene})"
                )

            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,
                    camera_intrinsics=intrinsics,
                    dataset="VigorChicago",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                    frame_index=int(view_index),
                )
            )

        return views


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference for rs-guided scenes from a CSV file."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="CSV describing scenes to run.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["pi3", "vggt", "da3", "mapanything"],
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/rs_guided_vis",
    )
    parser.add_argument("--config_path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--config_json_path", type=str, default=None)
    parser.add_argument("--model_str", type=str, default=None)
    parser.add_argument("--config_overrides", nargs="*", default=None)
    parser.add_argument("--hf_model_name", type=str, default=None)
    parser.add_argument("--enable_clash_proxy", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data_norm_type", type=str, default=None)
    parser.add_argument("--aerial_root", type=str, default="/root/autodl-tmp/traindata/Crossview_wai")
    parser.add_argument(
        "--aerial_metadata_dir",
        type=str,
        default="/root/autodl-tmp/traindata/mapanything_metadata/Crossview",
    )
    parser.add_argument(
        "--remote_metadata_root",
        type=str,
        default="/root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial",
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--provider", type=str, default="Google_Satellite")
    parser.add_argument("--resolution", nargs=2, type=int, default=[518, 518])
    parser.add_argument("--principal_point_centered", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--covisibility_thres", type=float, default=0.05)
    parser.add_argument("--remote_crop_mode", type=str, default="none")
    parser.add_argument("--remote_crop_scale_min", type=float, default=0.7)
    parser.add_argument("--remote_crop_scale_max", type=float, default=1.0)
    parser.add_argument("--remote_image_resize_mode", type=str, default="nearest")
    parser.add_argument("--remote_label_resize_mode", type=str, default="nearest")
    parser.add_argument("--save_glb", action="store_true", default=False)
    return parser.parse_args()


def parse_frame_indices(value):
    if value is None:
        raise ValueError("CSV must provide frame_indices")
    text = str(value).strip()
    if not text:
        raise ValueError("frame_indices is empty")
    text = text.replace("[", "").replace("]", "")
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [p for p in parts[0].split() if p]
    return [int(part) for part in parts]


def resolve_config_overrides(args):
    if args.config_overrides is not None:
        return args.config_overrides
    return list(DEFAULT_CONFIG_OVERRIDES[args.model])


def maybe_enable_clash_proxy(enable_proxy):
    if not enable_proxy:
        return
    clash_path = Path("/etc/profile.d/clash.sh")
    if not clash_path.exists():
        print("Clash helper not found at /etc/profile.d/clash.sh; skipping proxy setup")
        return
    os.environ.update(CLASH_ENV)
    print("Enabled Clash proxy environment for HuggingFace downloads")


def maybe_prepare_da3_pythonpath(model_name):
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


def build_local_config(args, config_overrides):
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


def initialize_model(args, device, config_overrides):
    maybe_enable_clash_proxy(args.enable_clash_proxy)
    maybe_prepare_da3_pythonpath(args.model)

    if args.checkpoint_path:
        local_config = build_local_config(args, config_overrides)
        print(f"Initializing model from local config: {local_config}")
        model = initialize_mapanything_local(local_config, device)
        model.eval()
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
        model.eval()
        return model

    from mapanything.models import init_model_from_config

    print(f"Initializing model '{args.model}' from default wrapper weights")
    model = init_model_from_config(args.model, device=device, machine="aws").eval()
    return model


def normalize_cities_arg(cities):
    if cities is None:
        return None
    normalized = [str(city).strip() for city in cities if str(city).strip()]
    return normalized or None


def checkpoint_tag(args):
    if args.checkpoint_path:
        path = Path(args.checkpoint_path)
        if path.parent.name:
            return path.parent.name
        return path.stem
    return "default"


def scene_output_dir(args, scene_name):
    root = Path(args.output_root) / args.model / checkpoint_tag(args) / scene_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def flatten_dict(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flatten_dict(next_prefix, item, out)
    else:
        out[prefix] = value


def resolve_scene_name(scene_token, available_scenes):
    if scene_token in available_scenes:
        return scene_token

    matches = [scene for scene in available_scenes if scene.endswith(f"__{scene_token}")]
    if len(matches) == 1:
        return matches[0]

    matches = [scene for scene in available_scenes if scene == scene_token or scene.endswith(scene_token)]
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise KeyError(f"Could not resolve scene '{scene_token}' from dataset scenes")
    raise KeyError(f"Scene token '{scene_token}' is ambiguous: {matches}")


def find_remote_sample(dataset, scene_name):
    for idx, manifest in enumerate(dataset.base_manifests):
        if manifest["scene_name"] == scene_name:
            sample = dataset[idx]
            sample["manifest"] = manifest
            return sample
    raise KeyError(f"Remote sample for scene '{scene_name}' was not found")


def build_aerial_dataset(args, scene_name, frame_indices):
    return ExplicitFrameVigorChicago(
        ROOT=args.aerial_root,
        dataset_metadata_dir=args.aerial_metadata_dir,
        split=args.split,
        resolution=tuple(args.resolution),
        principal_point_centered=args.principal_point_centered,
        seed=args.seed,
        transform="imgnorm",
        data_norm_type=args.data_norm_type or DEFAULT_DATA_NORM[args.model],
        sample_specific_scene=True,
        specific_scene_name=scene_name,
        cities=normalize_cities_arg(args.cities),
        variable_num_views=False,
        covisibility_thres=args.covisibility_thres,
        frame_indices=frame_indices,
    )


def build_remote_dataset(args, provider_name):
    return VigorChicagoRSAerial(
        metadata_root=args.remote_metadata_root,
        split=args.split,
        providers=[provider_name],
        cities=normalize_cities_arg(args.cities),
        load_aerial_scene_meta=False,
        transform="imgnorm",
        resolution=tuple(args.resolution),
        num_augmented_crops_per_sample=1,
        crop_mode=args.remote_crop_mode,
        crop_scale_range=(args.remote_crop_scale_min, args.remote_crop_scale_max),
        image_resize_mode=args.remote_image_resize_mode,
        label_resize_mode=args.remote_label_resize_mode,
    )


def batched_scene(scene_dataset):
    loader = torch.utils.data.DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch


def prepare_batch_for_model(batch, device):
    for view in batch:
        if isinstance(view.get("idx"), (list, tuple)) and len(view["idx"]) >= 3:
            view["idx"] = view["idx"][2:]
        for name in list(view.keys()):
            if name in IGNORE_KEYS:
                continue
            if torch.is_tensor(view[name]):
                view[name] = view[name].to(device, non_blocking=True)
    return batch


def create_remote_view(remote_sample, data_norm_type, device):
    remote_image = remote_sample["remote_image"].unsqueeze(0).to(device, non_blocking=True)
    return {
        "img": remote_image,
        "data_norm_type": [data_norm_type],
    }


def tensor_image_to_uint8(view):
    image_rgb = rgb(
        view["img"][0],
        view["data_norm_type"][0],
        true_shape=tuple(int(x) for x in view["true_shape"][0].tolist()),
    )
    return np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)


def pose_from_prediction(pred):
    pose = np.eye(4, dtype=np.float32)
    rot = quaternion_to_rotation_matrix(pred["cam_quats"][0:1]).detach().cpu().numpy()[0]
    trans = pred["cam_trans"][0].detach().cpu().numpy().astype(np.float32)
    pose[:3, :3] = rot.astype(np.float32)
    pose[:3, 3] = trans
    return pose


def tensor_pose_to_numpy(pose_tensor):
    return pose_tensor[0].detach().cpu().numpy().astype(np.float32)


def build_benchmark_aligned_aerial(batch, preds):
    gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(batch, preds)

    aligned_views = []
    gt_global_colors = []
    pred_global_colors = []
    for view_idx in range(len(batch)):
        image_rgb = tensor_image_to_uint8(batch[view_idx])
        valid_mask_np = valid_masks[view_idx][0].numpy().astype(bool)
        aligned_views.append(
            {
                "gt_pose": tensor_pose_to_numpy(gt_info["poses"][view_idx]),
                "pred_pose": tensor_pose_to_numpy(pr_info["poses"][view_idx]),
                "gt_points": gt_info["pts3d"][view_idx][0].numpy().astype(np.float32),
                "pred_points": pr_info["pts3d"][view_idx][0].numpy().astype(np.float32),
                "valid_mask": valid_mask_np,
            }
        )
        gt_global_colors.append(image_rgb[valid_mask_np])
        pred_global_colors.append(image_rgb[valid_mask_np])

    gt_global_points = np.concatenate(
        [view["gt_points"][view["valid_mask"]] for view in aligned_views], axis=0
    ).astype(np.float32)
    pred_global_points = np.concatenate(
        [view["pred_points"][view["valid_mask"]] for view in aligned_views], axis=0
    ).astype(np.float32)

    return {
        "views": aligned_views,
        "global_gt_points": gt_global_points,
        "global_pred_points": pred_global_points,
        "global_gt_colors": np.concatenate(gt_global_colors, axis=0).astype(np.uint8),
        "global_pred_colors": np.concatenate(pred_global_colors, axis=0).astype(np.uint8),
    }


def build_benchmark_aligned_joint(batch, joint_preds, remote_sample):
    aerial_gt_pts_list = [view["pts3d"].detach().cpu() for view in batch]
    aerial_pr_pts_list = []
    aerial_valid_masks = []
    for view_idx, view in enumerate(batch):
        gt_pts = view["pts3d"].detach().cpu()
        pred_pts = joint_preds[view_idx]["pts3d"].detach().cpu()
        if "metric_scaling_factor" in joint_preds[view_idx]:
            pred_pts = pred_pts / joint_preds[view_idx]["metric_scaling_factor"].detach().cpu().view(-1, 1, 1, 1)
        aerial_pr_pts_list.append(pred_pts)
        aerial_valid_masks.append(
            (
                torch.isfinite(gt_pts).all(dim=-1)
                & torch.isfinite(pred_pts).all(dim=-1)
            ).detach().cpu()
        )

    gt_pts_list = [view["pts3d"].detach().cpu() for view in batch]
    pr_pts_list = [joint_preds[view_idx]["pts3d"].detach().cpu() for view_idx in range(len(batch))]
    valid_masks = [
        (
            torch.isfinite(gt_pts_list[view_idx]).all(dim=-1)
            & torch.isfinite(pr_pts_list[view_idx]).all(dim=-1)
        ).detach().cpu()
        for view_idx, view in enumerate(batch)
    ]

    gt_remote_pts = torch.from_numpy(remote_sample["remote_pointmap"]).unsqueeze(0).float()
    pr_remote_pts = joint_preds[len(batch)]["pts3d"].detach().cpu()
    remote_valid_mask = (
        torch.from_numpy(remote_sample["remote_valid_mask"])
        .unsqueeze(0)
        .bool()
    )
    remote_valid_mask = (
        remote_valid_mask
        & torch.isfinite(gt_remote_pts).all(dim=-1)
        & torch.isfinite(pr_remote_pts).all(dim=-1)
    )

    gt_pts_list.append(gt_remote_pts)
    pr_pts_list.append(pr_remote_pts)
    valid_masks.append(remote_valid_mask)

    gt_pts_norm = normalize_multiple_pointclouds(
        gt_pts_list, valid_masks=valid_masks, norm_mode="avg_dis"
    )
    pr_pts_norm = normalize_multiple_pointclouds(
        pr_pts_list, valid_masks=valid_masks, norm_mode="avg_dis"
    )

    aerial_views = []
    aerial_gt_colors = []
    aerial_pred_colors = []
    for view_idx in range(len(batch)):
        image_rgb = tensor_image_to_uint8(batch[view_idx])
        valid_mask_np = valid_masks[view_idx][0].numpy().astype(bool)
        aerial_views.append(
            {
                "gt_points": gt_pts_norm[view_idx][0].numpy().astype(np.float32),
                "pred_points": pr_pts_norm[view_idx][0].numpy().astype(np.float32),
                "valid_mask": valid_mask_np,
            }
        )
        aerial_gt_colors.append(image_rgb[valid_mask_np])
        aerial_pred_colors.append(image_rgb[valid_mask_np])

    remote_image_rgb = np.clip(
        remote_sample["remote_image"].permute(1, 2, 0).cpu().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    remote_valid_mask_np = valid_masks[-1][0].numpy().astype(bool)

    pred_camera0 = torch.eye(4, device=joint_preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0[..., :3, :3] = quaternion_to_rotation_matrix(joint_preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, 3] = joint_preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)
    gt_in_camera0 = inv(batch[0]["camera_pose"])

    gt_aerial_norm = normalize_multiple_pointclouds(
        aerial_gt_pts_list,
        valid_masks=aerial_valid_masks,
        norm_mode="avg_dis",
        ret_factor=True,
    )
    gt_norm_factor = gt_aerial_norm[-1]
    pr_aerial_norm = normalize_multiple_pointclouds(
        aerial_pr_pts_list,
        valid_masks=aerial_valid_masks,
        norm_mode="avg_dis",
        ret_factor=True,
    )
    pr_norm_factor = pr_aerial_norm[-1]

    raw_gt_remote = gt_remote_pts[0].detach().cpu().numpy().astype(np.float32)
    raw_pred_remote = pr_remote_pts[0].detach().cpu().numpy().astype(np.float32)
    remote_pose_gt = np.eye(4, dtype=np.float32)
    remote_pose_pred = np.eye(4, dtype=np.float32)

    return {
        "aerial_views": aerial_views,
        "global_gt_colors": np.concatenate(
            aerial_gt_colors + [remote_image_rgb[remote_valid_mask_np]], axis=0
        ).astype(np.uint8),
        "global_pred_colors": np.concatenate(
            aerial_pred_colors + [remote_image_rgb[remote_valid_mask_np]], axis=0
        ).astype(np.uint8),
        "remote": {
            "gt_points": gt_pts_norm[-1][0].numpy().astype(np.float32),
            "pred_points": pr_pts_norm[-1][0].numpy().astype(np.float32),
            "valid_mask": valid_masks[-1][0].numpy().astype(bool),
            "raw_gt_points": raw_gt_remote,
            "raw_pred_points": raw_pred_remote,
            "gt_pose": remote_pose_gt,
            "pred_pose": remote_pose_pred,
            "gt_in_view0_raw": geotrf(
                gt_in_camera0,
                gt_remote_pts.to(gt_in_camera0.device),
            )[0].detach().cpu().numpy().astype(np.float32),
            "pred_in_view0_raw": geotrf(
                pred_in_camera0,
                pr_remote_pts.to(pred_in_camera0.device),
            )[0].detach().cpu().numpy().astype(np.float32),
            "joint_view0_gt_points": (
                geotrf(
                    gt_in_camera0,
                    gt_remote_pts.to(gt_in_camera0.device),
                ).detach().cpu() / gt_norm_factor.detach().cpu()
            )[0].numpy().astype(np.float32),
            "joint_view0_pred_points": (
                geotrf(
                    pred_in_camera0,
                    (
                        pr_remote_pts.to(pred_in_camera0.device)
                        / joint_preds[len(batch)]["metric_scaling_factor"].detach().to(pred_in_camera0.device).view(-1, 1, 1, 1)
                    )
                    if "metric_scaling_factor" in joint_preds[len(batch)]
                    else pr_remote_pts.to(pred_in_camera0.device),
                ).detach().cpu() / pr_norm_factor.detach().cpu()
            )[0].numpy().astype(np.float32),
        },
    }


def estimate_remote_metric_scale(pred_points, valid_mask, meters_per_pixel):
    valid_mask = valid_mask.astype(bool)
    valid_points = pred_points[valid_mask]
    valid_points = valid_points[np.isfinite(valid_points).all(axis=-1)]
    if valid_points.shape[0] < 32:
        return None

    center = np.median(valid_points, axis=0)
    centered = valid_points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]

    def projected_neighbor_norm(diff, mask):
        if diff.size == 0 or not mask.any():
            return np.empty((0,), dtype=np.float32)
        diff = diff[mask]
        proj0 = diff @ basis[0]
        proj1 = diff @ basis[1]
        return np.sqrt(proj0**2 + proj1**2)

    right_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    down_valid = valid_mask[:-1, :] & valid_mask[1:, :]

    right_diff = pred_points[:, 1:, :] - pred_points[:, :-1, :]
    down_diff = pred_points[1:, :, :] - pred_points[:-1, :, :]

    right_norm = projected_neighbor_norm(right_diff, right_valid)
    down_norm = projected_neighbor_norm(down_diff, down_valid)

    samples = []
    if right_norm.size > 0:
        samples.append(right_norm)
    if down_norm.size > 0:
        samples.append(down_norm)
    if not samples:
        return None

    spacing = np.concatenate(samples, axis=0)
    spacing = spacing[np.isfinite(spacing) & (spacing > 1e-8)]
    if spacing.size == 0:
        return None

    pred_spacing = float(np.median(spacing))
    if not np.isfinite(pred_spacing) or pred_spacing <= 1e-8:
        return None

    return {
        "scale_factor": float(meters_per_pixel / pred_spacing),
        "pred_spacing_xy": pred_spacing,
        "meters_per_pixel": float(meters_per_pixel),
        "plane_basis_u": basis[0].astype(np.float32),
        "plane_basis_v": basis[1].astype(np.float32),
    }


def build_remote_metric_aligned_joint(batch, joint_preds, remote_sample):
    pred_camera0 = torch.eye(4, device=joint_preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0[..., :3, :3] = quaternion_to_rotation_matrix(joint_preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, 3] = joint_preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)
    gt_in_camera0 = inv(batch[0]["camera_pose"])

    gt_pose_quats = []
    gt_pose_trans = []
    pred_pose_quats = []
    pred_pose_trans = []
    gt_points = []
    pred_points = []
    valid_masks = []

    for view_idx in range(len(batch)):
        gt_points_curr = geotrf(gt_in_camera0, batch[view_idx]["pts3d"]).detach().cpu().numpy()[0]
        gt_points.append(gt_points_curr.astype(np.float32))
        valid_masks.append(batch[view_idx]["valid_mask"][0].detach().cpu().numpy().astype(bool))

        if view_idx == 0:
            gt_pose_quats.append(np.array([0, 0, 0, 1], dtype=np.float32))
            gt_pose_trans.append(np.array([0, 0, 0], dtype=np.float32))
        else:
            gt_quat, gt_trans = transform_pose_using_quats_and_trans_2_to_1(
                batch[0]["camera_pose_quats"],
                batch[0]["camera_pose_trans"],
                batch[view_idx]["camera_pose_quats"],
                batch[view_idx]["camera_pose_trans"],
            )
            gt_pose_quats.append(gt_quat[0].detach().cpu().numpy().astype(np.float32))
            gt_pose_trans.append(gt_trans[0].detach().cpu().numpy().astype(np.float32))

        pred_quat, pred_trans = transform_pose_using_quats_and_trans_2_to_1(
            joint_preds[0]["cam_quats"],
            joint_preds[0]["cam_trans"],
            joint_preds[view_idx]["cam_quats"],
            joint_preds[view_idx]["cam_trans"],
        )
        pred_pose_quats.append(pred_quat[0].detach().cpu().numpy().astype(np.float32))

        pred_points_curr = geotrf(pred_in_camera0, joint_preds[view_idx]["pts3d"]).detach().cpu().numpy()[0]
        if "metric_scaling_factor" in joint_preds[view_idx]:
            scale = (
                joint_preds[view_idx]["metric_scaling_factor"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            pred_points_curr = pred_points_curr / scale
            pred_trans_np = pred_trans[0].detach().cpu().numpy().astype(np.float32) / scale
        else:
            pred_trans_np = pred_trans[0].detach().cpu().numpy().astype(np.float32)
        pred_points.append(pred_points_curr.astype(np.float32))
        pred_pose_trans.append(pred_trans_np.astype(np.float32))

    pred_remote = joint_preds[len(batch)]["pts3d"].detach().cpu().numpy()[0]
    if "metric_scaling_factor" in joint_preds[len(batch)]:
        remote_raw_scale = (
            joint_preds[len(batch)]["metric_scaling_factor"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        pred_remote = pred_remote / remote_raw_scale

    pred_remote_tensor = torch.from_numpy(pred_remote).unsqueeze(0).to(pred_in_camera0.device)
    gt_remote_tensor = torch.from_numpy(remote_sample["remote_pointmap"]).unsqueeze(0).to(gt_in_camera0.device)
    pred_remote_in_view0 = geotrf(pred_in_camera0, pred_remote_tensor).detach().cpu().numpy()[0]
    gt_remote_in_view0 = geotrf(gt_in_camera0, gt_remote_tensor).detach().cpu().numpy()[0]
    remote_valid_mask = remote_sample["remote_valid_mask"].astype(bool)

    manifest = remote_sample.get("manifest", {})
    meters_per_pixel = manifest.get("meters_per_pixel")
    if meters_per_pixel is None:
        meters_per_pixel = remote_sample.get("remote_info", {}).get("meters_per_pixel")
    scale_info = (
        estimate_remote_metric_scale(pred_remote_in_view0, remote_valid_mask, float(meters_per_pixel))
        if meters_per_pixel is not None
        else None
    )

    if scale_info is None:
        scale_factor = 1.0
        scale_info = {
            "scale_factor": 1.0,
            "pred_spacing_xy": float("nan"),
            "meters_per_pixel": float("nan") if meters_per_pixel is None else float(meters_per_pixel),
        }
    else:
        scale_factor = scale_info["scale_factor"]

    aligned_views = []
    global_gt_points = []
    global_pred_points = []
    global_gt_colors = []
    global_pred_colors = []
    for view_idx in range(len(batch)):
        image_rgb = tensor_image_to_uint8(batch[view_idx])
        gt_pose = np.eye(4, dtype=np.float32)
        gt_pose[:3, :3] = quaternion_to_rotation_matrix(
            torch.from_numpy(gt_pose_quats[view_idx]).unsqueeze(0)
        ).detach().cpu().numpy()[0].astype(np.float32)
        gt_pose[:3, 3] = gt_pose_trans[view_idx]

        pred_pose = np.eye(4, dtype=np.float32)
        pred_pose[:3, :3] = quaternion_to_rotation_matrix(
            torch.from_numpy(pred_pose_quats[view_idx]).unsqueeze(0)
        ).detach().cpu().numpy()[0].astype(np.float32)
        pred_pose[:3, 3] = (pred_pose_trans[view_idx] * scale_factor).astype(np.float32)

        pred_pts_scaled = (pred_points[view_idx] * scale_factor).astype(np.float32)
        aligned_views.append(
            {
                "gt_pose": gt_pose,
                "pred_pose": pred_pose,
                "gt_points": gt_points[view_idx].astype(np.float32),
                "pred_points": pred_pts_scaled,
                "valid_mask": valid_masks[view_idx],
            }
        )
        global_gt_points.append(gt_points[view_idx][valid_masks[view_idx]])
        global_pred_points.append(pred_pts_scaled[valid_masks[view_idx]])
        global_gt_colors.append(image_rgb[valid_masks[view_idx]])
        global_pred_colors.append(image_rgb[valid_masks[view_idx]])

    remote_image_rgb = np.clip(
        remote_sample["remote_image"].permute(1, 2, 0).cpu().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    global_gt_points.append(gt_remote_in_view0[remote_valid_mask])
    global_pred_points.append((pred_remote_in_view0 * scale_factor)[remote_valid_mask])
    global_gt_colors.append(remote_image_rgb[remote_valid_mask])
    global_pred_colors.append(remote_image_rgb[remote_valid_mask])

    return {
        "scale_info": scale_info,
        "views": aligned_views,
        "remote": {
            "gt_points": gt_remote_in_view0.astype(np.float32),
            "pred_points": (pred_remote_in_view0 * scale_factor).astype(np.float32),
            "valid_mask": remote_valid_mask,
        },
        "global": {
            "gt_points": np.concatenate(global_gt_points, axis=0).astype(np.float32),
            "pred_points": np.concatenate(global_pred_points, axis=0).astype(np.float32),
            "gt_colors": np.concatenate(global_gt_colors, axis=0).astype(np.uint8),
            "pred_colors": np.concatenate(global_pred_colors, axis=0).astype(np.uint8),
        },
    }


def collect_world_space_point_cloud(outputs, views):
    all_points = []
    all_colors = []

    for view_idx, pred in enumerate(outputs):
        if "pts3d" in pred:
            pts3d_np = pred["pts3d"][0].detach().cpu().numpy()
            export_mask = np.isfinite(pts3d_np).all(axis=-1)
        else:
            depthmap_torch = pred["depth_z"][0].squeeze(-1)
            intrinsics_torch = pred["intrinsics"][0]
            camera_pose_torch = pred["camera_poses"][0]
            pts3d_world, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )
            pts3d_np = pts3d_world.cpu().numpy()
            export_mask = valid_mask.cpu().numpy()

        colors_np = tensor_image_to_uint8(views[view_idx])
        selected_points = pts3d_np[export_mask]
        selected_colors = colors_np[export_mask]
        if selected_points.shape[0] == 0:
            continue
        all_points.append(selected_points)
        all_colors.append(selected_colors)

    if not all_points:
        raise RuntimeError("No valid points remained after masking; cannot export point cloud.")

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)


def collect_world_space_gt_point_cloud(views, remote_sample=None):
    all_points = []
    all_colors = []

    for view in views:
        pts3d_np = view["pts3d"][0].detach().cpu().numpy()
        valid_mask = view["valid_mask"][0].detach().cpu().numpy().astype(bool)
        colors_np = tensor_image_to_uint8(view)
        selected_points = pts3d_np[valid_mask]
        selected_colors = colors_np[valid_mask]
        if selected_points.shape[0] == 0:
            continue
        all_points.append(selected_points)
        all_colors.append(selected_colors)

    if remote_sample is not None:
        remote_points = np.asarray(remote_sample["remote_pointmap"], dtype=np.float32)
        remote_mask = np.asarray(remote_sample["remote_valid_mask"], dtype=bool)
        remote_colors = np.clip(
            remote_sample["remote_image"].permute(1, 2, 0).cpu().numpy() * 255.0,
            0,
            255,
        ).astype(np.uint8)
        selected_points = remote_points[remote_mask]
        selected_colors = remote_colors[remote_mask]
        if selected_points.shape[0] > 0:
            all_points.append(selected_points)
            all_colors.append(selected_colors)

    if not all_points:
        raise RuntimeError("No valid GT points remained after masking; cannot export point cloud.")

    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)


def save_pointcloud_ply(path, points, colors):
    trimesh.PointCloud(vertices=points, colors=colors).export(path)


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def scene_bundle_from_outputs(scene_name, frame_indices, provider_name, batch, remote_sample, aerial_preds, rs_preds, joint_preds):
    gt_points, gt_colors = collect_world_space_gt_point_cloud(batch, remote_sample=remote_sample)
    aerial_points, aerial_colors = collect_world_space_point_cloud(aerial_preds, batch)
    joint_points, joint_colors = collect_world_space_point_cloud(joint_preds[: len(batch)], batch)
    aerial_aligned = build_benchmark_aligned_aerial(batch, aerial_preds)
    joint_aligned = build_benchmark_aligned_aerial(batch, joint_preds[: len(batch)])
    joint_global_aligned = build_benchmark_aligned_joint(batch, joint_preds, remote_sample)
    joint_remote_metric = build_remote_metric_aligned_joint(batch, joint_preds, remote_sample)

    views_payload = []
    for view_idx, view in enumerate(batch):
        gt_points = view["pts3d"][0].detach().cpu().numpy()
        gt_valid = view["valid_mask"][0].detach().cpu().numpy().astype(bool)
        aerial_pred_points = aerial_preds[view_idx]["pts3d"][0].detach().cpu().numpy()
        joint_pred_points = joint_preds[view_idx]["pts3d"][0].detach().cpu().numpy()
        views_payload.append(
            {
                "frame_index": int(frame_indices[view_idx]),
                "instance": str(view["instance"][0]),
                "image_rgb": tensor_image_to_uint8(view),
                "camera_intrinsics": view["camera_intrinsics"][0].detach().cpu().numpy().astype(np.float32),
                "gt_pose": view["camera_pose"][0].detach().cpu().numpy().astype(np.float32),
                "aerial_pred_pose": pose_from_prediction(aerial_preds[view_idx]),
                "joint_pred_pose": pose_from_prediction(joint_preds[view_idx]),
                "gt_points": gt_points.astype(np.float32),
                "gt_valid_mask": gt_valid,
                "aerial_pred_points": aerial_pred_points.astype(np.float32),
                "joint_pred_points": joint_pred_points.astype(np.float32),
                "benchmark_aligned": {
                    "aerial_only": {
                        "gt_pose": aerial_aligned["views"][view_idx]["gt_pose"],
                        "pred_pose": aerial_aligned["views"][view_idx]["pred_pose"],
                        "gt_points": aerial_aligned["views"][view_idx]["gt_points"],
                        "pred_points": aerial_aligned["views"][view_idx]["pred_points"],
                        "valid_mask": aerial_aligned["views"][view_idx]["valid_mask"],
                    },
                    "joint": {
                        "gt_pose": joint_aligned["views"][view_idx]["gt_pose"],
                        "pred_pose": joint_aligned["views"][view_idx]["pred_pose"],
                        "gt_points": joint_aligned["views"][view_idx]["gt_points"],
                        "pred_points": joint_aligned["views"][view_idx]["pred_points"],
                        "valid_mask": joint_aligned["views"][view_idx]["valid_mask"],
                    },
                    "joint_global": {
                        "gt_points": joint_global_aligned["aerial_views"][view_idx]["gt_points"],
                        "pred_points": joint_global_aligned["aerial_views"][view_idx]["pred_points"],
                        "valid_mask": joint_global_aligned["aerial_views"][view_idx]["valid_mask"],
                    },
                    "remote_metric_joint": {
                        "gt_pose": joint_remote_metric["views"][view_idx]["gt_pose"],
                        "pred_pose": joint_remote_metric["views"][view_idx]["pred_pose"],
                        "gt_points": joint_remote_metric["views"][view_idx]["gt_points"],
                        "pred_points": joint_remote_metric["views"][view_idx]["pred_points"],
                        "valid_mask": joint_remote_metric["views"][view_idx]["valid_mask"],
                    },
                },
            }
        )

    remote_image = np.clip(
        remote_sample["remote_image"].permute(1, 2, 0).cpu().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)

    bundle = {
        "metadata": {
            "scene_name": scene_name,
            "provider": provider_name,
            "frame_indices": [int(x) for x in frame_indices],
        },
        "views": views_payload,
        "remote": {
            "image_rgb": remote_image,
            "gt_points": remote_sample["remote_pointmap"].astype(np.float32),
            "gt_valid_mask": remote_sample["remote_valid_mask"].astype(bool),
            "rs_only_pred_points": rs_preds[0]["pts3d"][0].detach().cpu().numpy().astype(np.float32),
            "joint_pred_points": joint_preds[len(batch)]["pts3d"][0].detach().cpu().numpy().astype(np.float32),
            "remote_provider": remote_sample["remote_provider"],
            "remote_projection_type": remote_sample["remote_projection_type"],
            "remote_crop_box_xyxy": remote_sample["remote_crop_box_xyxy"].astype(np.int32),
            "benchmark_aligned": {
                "joint": {
                    "gt_points": joint_global_aligned["remote"]["joint_view0_gt_points"],
                    "pred_points": joint_global_aligned["remote"]["joint_view0_pred_points"],
                    "valid_mask": joint_global_aligned["remote"]["valid_mask"],
                },
                "joint_global": joint_global_aligned["remote"],
            },
            "remote_metric_aligned": joint_remote_metric["remote"],
            "meters_per_pixel": (
                remote_sample.get("manifest", {}).get("meters_per_pixel")
                or remote_sample.get("remote_info", {}).get("meters_per_pixel")
            ),
        },
        "global_pointclouds": {
            "gt": {
                "points": gt_points.astype(np.float32),
                "colors": gt_colors.astype(np.uint8),
            },
            "aerial_only": {
                "points": aerial_points.astype(np.float32),
                "colors": aerial_colors.astype(np.uint8),
            },
            "joint": {
                "points": joint_points.astype(np.float32),
                "colors": joint_colors.astype(np.uint8),
            },
            "benchmark_aligned": {
                "aerial_only": {
                    "gt_points": aerial_aligned["global_gt_points"],
                    "pred_points": aerial_aligned["global_pred_points"],
                    "gt_colors": aerial_aligned["global_gt_colors"],
                    "pred_colors": aerial_aligned["global_pred_colors"],
                },
                "joint": {
                    "gt_points": joint_aligned["global_gt_points"],
                    "pred_points": joint_aligned["global_pred_points"],
                    "gt_colors": joint_aligned["global_gt_colors"],
                    "pred_colors": joint_aligned["global_pred_colors"],
                },
                "remote_metric_joint": {
                    "gt_points": joint_remote_metric["global"]["gt_points"],
                    "pred_points": joint_remote_metric["global"]["pred_points"],
                    "gt_colors": joint_remote_metric["global"]["gt_colors"],
                    "pred_colors": joint_remote_metric["global"]["pred_colors"],
                    "scale_info": joint_remote_metric["scale_info"],
                },
            },
        },
    }
    return bundle


def run_scene(model, device, args, row, available_scenes):
    scene_token = row.get("scene_name") or row.get("location")
    if not scene_token:
        raise ValueError("CSV row must contain scene_name or location")
    scene_name = resolve_scene_name(scene_token, available_scenes)
    frame_indices = parse_frame_indices(row.get("frame_indices"))
    provider_name = row.get("provider") or args.provider

    aerial_dataset = build_aerial_dataset(args, scene_name, frame_indices)
    batch = batched_scene(aerial_dataset)
    remote_dataset = build_remote_dataset(args, provider_name)
    remote_sample = find_remote_sample(remote_dataset, scene_name)

    batch = prepare_batch_for_model(batch, device)
    remote_view = create_remote_view(remote_sample, args.data_norm_type or DEFAULT_DATA_NORM[args.model], device)

    start = time()
    with torch.inference_mode():
        aerial_preds = model(batch)
        rs_preds = model([remote_view])
        joint_preds = model(batch + [remote_view])
    inference_seconds = time() - start

    aerial_metrics = compute_aerial_scene_metrics(batch, aerial_preds)[scene_name]
    joint_aerial_metrics = compute_aerial_scene_metrics(batch, joint_preds[: len(batch)])[scene_name]

    rs_supports_metric_outputs = model_supports_metric_outputs(rs_preds)
    joint_supports_metric_outputs = model_supports_metric_outputs(joint_preds)

    rs_pts = rs_preds[0]["pts3d"][0].detach().cpu().numpy()
    rs_metrics = compute_remote_height_metrics_affine(
        remote_sample["remote_height_map"],
        rs_pts,
        remote_sample["remote_valid_mask"].astype(bool),
    )
    if rs_supports_metric_outputs:
        rs_metrics.update(
            compute_remote_height_metrics(
                remote_sample["remote_height_map"],
                rs_pts,
                remote_sample["remote_valid_mask"].astype(bool),
            )
        )

    joint_rs_pts = joint_preds[len(batch)]["pts3d"][0].detach().cpu().numpy()
    joint_rs_metrics = compute_remote_height_metrics_affine(
        remote_sample["remote_height_map"],
        joint_rs_pts,
        remote_sample["remote_valid_mask"].astype(bool),
    )
    if joint_supports_metric_outputs:
        joint_rs_metrics.update(
            compute_remote_height_metrics(
                remote_sample["remote_height_map"],
                joint_rs_pts,
                remote_sample["remote_valid_mask"].astype(bool),
            )
        )

    joint_metrics = {
        **joint_aerial_metrics,
        **joint_rs_metrics,
        "joint_global_point_l1": (
            compute_joint_global_point_l1(batch=batch, joint_preds=joint_preds, remote_sample=remote_sample)
            if joint_supports_metric_outputs
            else float("nan")
        ),
        "joint_global_pointmaps_abs_rel": compute_joint_global_pointmaps_abs_rel(
            batch=batch,
            joint_preds=joint_preds,
            remote_sample=remote_sample,
        ),
    }

    result = {
        "scene_name": scene_name,
        "provider": provider_name,
        "frame_indices": frame_indices,
        "inference_seconds": float(inference_seconds),
        "aerial_only": aerial_metrics,
        "rs_only": rs_metrics,
        "joint": joint_metrics,
        "improvement": {
            "aerial_vs_aerial_only": diff_metric_dict(joint_aerial_metrics, aerial_metrics),
            "rs_vs_rs_only": diff_metric_dict(joint_rs_metrics, rs_metrics),
        },
    }

    bundle = scene_bundle_from_outputs(
        scene_name=scene_name,
        frame_indices=frame_indices,
        provider_name=provider_name,
        batch=batch,
        remote_sample=remote_sample,
        aerial_preds=aerial_preds,
        rs_preds=rs_preds,
        joint_preds=joint_preds,
    )
    bundle["metrics"] = {
        "aerial_only": result["aerial_only"],
        "rs_only": result["rs_only"],
        "joint": result["joint"],
        "improvement": result["improvement"],
        "debug": {
            "remote_metric_scale": (
                bundle["global_pointclouds"]
                .get("benchmark_aligned", {})
                .get("remote_metric_joint", {})
                .get("scale_info", {})
            ),
        },
        "metadata": {
            "scene_name": scene_name,
            "provider": provider_name,
            "frame_indices": frame_indices,
            "inference_seconds": float(inference_seconds),
        },
    }
    return result, bundle


def write_scene_outputs(args, result, bundle):
    out_dir = scene_output_dir(args, result["scene_name"])
    bundle_path = out_dir / "scene_bundle.pt"
    metrics_path = out_dir / "metrics.json"
    aerial_ply_path = out_dir / "aerial_only_global.ply"
    joint_ply_path = out_dir / "joint_global.ply"

    torch.save(bundle, bundle_path)
    metrics_path.write_text(json.dumps(make_json_safe(bundle["metrics"]), indent=4))
    save_pointcloud_ply(
        aerial_ply_path,
        bundle["global_pointclouds"]["aerial_only"]["points"],
        bundle["global_pointclouds"]["aerial_only"]["colors"],
    )
    save_pointcloud_ply(
        joint_ply_path,
        bundle["global_pointclouds"]["joint"]["points"],
        bundle["global_pointclouds"]["joint"]["colors"],
    )

    if args.save_glb:
        glb_predictions = {
            "world_points": np.stack([view["joint_pred_points"] for view in bundle["views"]], axis=0),
            "images": np.stack([view["image_rgb"] for view in bundle["views"]], axis=0).astype(np.float32) / 255.0,
            "final_masks": np.stack(
                [np.isfinite(view["joint_pred_points"]).all(axis=-1) for view in bundle["views"]],
                axis=0,
            ),
        }
        predictions_to_glb(glb_predictions, as_mesh=False).export(out_dir / "joint_points.glb")

    return {
        "scene_name": result["scene_name"],
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "aerial_ply_path": str(aerial_ply_path),
        "joint_ply_path": str(joint_ply_path),
    }


def write_batch_summary(args, rows):
    root = Path(args.output_root) / args.model / checkpoint_tag(args)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "batch_results.json"
    csv_path = root / "batch_results.csv"
    json_path.write_text(json.dumps(make_json_safe(rows), indent=4))

    flat_rows = []
    for row in rows:
        flat = {}
        flatten_dict("", row, flat)
        flat_rows.append(flat)

    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_overrides = resolve_config_overrides(args)
    model = initialize_model(args, device, config_overrides)

    available_scene_dataset = VigorChicagoWAI(
        ROOT=args.aerial_root,
        dataset_metadata_dir=args.aerial_metadata_dir,
        split=args.split,
        resolution=tuple(args.resolution),
        principal_point_centered=args.principal_point_centered,
        seed=args.seed,
        transform="imgnorm",
        data_norm_type=args.data_norm_type or DEFAULT_DATA_NORM[args.model],
        cities=normalize_cities_arg(args.cities),
        variable_num_views=False,
        num_views=2,
        covisibility_thres=args.covisibility_thres,
        max_num_retries=0,
    )
    available_scenes = list(available_scene_dataset.scenes)

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    if not csv_rows:
        raise ValueError(f"No rows found in CSV: {args.csv_path}")

    batch_results = []
    for row_idx, row in enumerate(csv_rows):
        print(f"[{row_idx + 1}/{len(csv_rows)}] Running row: {row}")
        result, bundle = run_scene(model, device, args, row, available_scenes)
        artifact_paths = write_scene_outputs(args, result, bundle)
        result["artifacts"] = artifact_paths
        batch_results.append(result)

    write_batch_summary(args, batch_results)


if __name__ == "__main__":
    main()
