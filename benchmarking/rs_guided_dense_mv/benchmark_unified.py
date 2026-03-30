# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Unified RS-Aerial benchmark.

Current executable scope:
- Aerial-only metrics on paired scenes
- RS-only height metrics on paired scenes
- Joint aerial+RS forward inference on paired scenes
- joint_global_point_l1
"""

import ast
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf

from benchmarking.dense_n_view.benchmark import (
    build_dataset,
    get_all_info_for_metric_computation,
)
from mapanything.models import init_model
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
)
from mapanything.utils.metrics import (
    calculate_auc_np,
    evaluate_ate,
    l2_distance_of_unit_ray_directions_to_angular_error,
    m_rel_ae,
    se3_to_relative_pose_error,
)
from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


def resolve_resolution(resolution_cfg):
    if isinstance(resolution_cfg, str):
        parsed = ast.literal_eval(resolution_cfg)
    else:
        parsed = resolution_cfg
    return list(parsed)


def build_data_loaders(args):
    aerial_loader = build_dataset(
        args.dataset.test_dataset_aerial, args.batch_size, args.dataset.num_workers
    )
    remote_loader = build_dataset(
        args.dataset.test_dataset_remote, args.batch_size, args.dataset.num_workers
    )
    return aerial_loader, remote_loader


def point_l1_metric(gt_pts, pr_pts, valid_mask):
    if not valid_mask.any():
        return float("nan")
    diff = np.abs(pr_pts - gt_pts).sum(axis=-1)
    return float(np.mean(diff[valid_mask]))


def model_supports_metric_outputs(preds):
    return len(preds) > 0 and "metric_scaling_factor" in preds[0]


def get_metric_space_pointmaps(batch, preds):
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]

    in_camera0 = inv(batch[0]["camera_pose"])
    pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0 = pred_camera0.repeat(batch_size, 1, 1)
    pred_camera0[..., :3, :3] = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    gt_pts_metric = []
    pr_pts_metric = []
    for i in range(n_views):
        gt_pts_metric.append(geotrf(in_camera0, batch[i]["pts3d"]).cpu())
        pr_pts_metric.append(geotrf(pred_in_camera0, preds[i]["pts3d"]).detach().cpu())
    return gt_pts_metric, pr_pts_metric


def compute_aerial_scene_metrics(batch, preds):
    n_views = len(batch)
    gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(batch, preds)
    supports_metric_outputs = model_supports_metric_outputs(preds)
    if supports_metric_outputs:
        gt_pts_metric, pr_pts_metric = get_metric_space_pointmaps(batch, preds)
    else:
        gt_pts_metric, pr_pts_metric = None, None

    batch_metrics = {}
    batch_size = batch[0]["img"].shape[0]
    for batch_idx in range(batch_size):
        scene = batch[0]["label"][batch_idx]

        pointmaps_abs_rel_across_views = []
        z_depth_abs_rel_across_views = []
        ray_dirs_err_deg_across_views = []
        metric_point_l1_across_views = []
        gt_poses_curr_set = []
        pr_poses_curr_set = []

        for view_idx in range(n_views):
            valid_mask_curr_view = valid_masks[view_idx][batch_idx].numpy().astype(bool)

            pointmaps_abs_rel_curr_view = m_rel_ae(
                gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
                pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
                mask=valid_mask_curr_view,
            )
            z_depth_abs_rel_curr_view = m_rel_ae(
                gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
                pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
                mask=valid_mask_curr_view,
            )
            if supports_metric_outputs:
                metric_point_l1_curr_view = point_l1_metric(
                    gt_pts_metric[view_idx][batch_idx].numpy(),
                    pr_pts_metric[view_idx][batch_idx].numpy(),
                    valid_mask_curr_view,
                )
            else:
                metric_point_l1_curr_view = float("nan")

            pointmaps_abs_rel_across_views.append(pointmaps_abs_rel_curr_view)
            z_depth_abs_rel_across_views.append(z_depth_abs_rel_curr_view)
            metric_point_l1_across_views.append(metric_point_l1_curr_view)

            ray_dirs_l2 = torch.norm(
                gt_info["ray_directions"][view_idx][batch_idx]
                - pr_info["ray_directions"][view_idx][batch_idx],
                dim=-1,
            )
            ray_dirs_err_deg = l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2)
            ray_dirs_err_deg_across_views.append(torch.mean(ray_dirs_err_deg).cpu().numpy())

            gt_poses_curr_set.append(gt_info["poses"][view_idx][batch_idx])
            pr_poses_curr_set.append(pr_info["poses"][view_idx][batch_idx])

        pose_ate_curr_set = float(
            evaluate_ate(gt_traj=gt_poses_curr_set, est_traj=pr_poses_curr_set).item()
        )
        gt_poses_curr_set = torch.stack(gt_poses_curr_set)
        pr_poses_curr_set = torch.stack(pr_poses_curr_set)
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
            pred_se3=pr_poses_curr_set,
            gt_se3=gt_poses_curr_set,
            num_frames=pr_poses_curr_set.shape[0],
        )
        pose_auc_5_curr_set, _ = calculate_auc_np(
            rel_rangle_deg.cpu().numpy(),
            rel_tangle_deg.cpu().numpy(),
            max_threshold=5,
        )

        metric_scale_abs_rel = float("nan")
        if (
            supports_metric_outputs
            and gt_info["metric_scale"] is not None
            and pr_info["metric_scale"] is not None
        ):
            gt_metric_scale_curr_set = float(np.asarray(gt_info["metric_scale"][batch_idx].numpy()).reshape(-1)[0])
            pr_metric_scale_curr_set = float(np.asarray(pr_info["metric_scale"][batch_idx].numpy()).reshape(-1)[0])
            metric_scale_abs_rel = float(
                np.abs(pr_metric_scale_curr_set - gt_metric_scale_curr_set)
                / gt_metric_scale_curr_set
            )

        batch_metrics[scene] = {
            "pointmaps_abs_rel": float(np.mean(pointmaps_abs_rel_across_views)),
            "z_depth_abs_rel": float(np.mean(z_depth_abs_rel_across_views)),
            "pose_ate_rmse": pose_ate_curr_set,
            "pose_auc_5": float(pose_auc_5_curr_set * 100.0),
            "ray_dirs_err_deg": float(np.mean(ray_dirs_err_deg_across_views)),
            "metric_scale_abs_rel": metric_scale_abs_rel,
            "metric_point_l1": float(np.mean(metric_point_l1_across_views)),
        }

    return batch_metrics


def compute_remote_height_metrics(gt_height, pred_pts, valid_mask):
    pred_height = pred_pts[..., 2]
    overlap = valid_mask & np.isfinite(gt_height) & np.isfinite(pred_height)
    if not overlap.any():
        return {
            "rs_height_mae": float("nan"),
            "rs_height_rmse": float("nan"),
            "rs_z_offset": float("nan"),
        }

    z_offset = float(np.mean(gt_height[overlap] - pred_height[overlap]))
    pred_height_aligned = pred_height + z_offset
    height_err = pred_height_aligned[overlap] - gt_height[overlap]
    return {
        "rs_height_mae": float(np.mean(np.abs(height_err))),
        "rs_height_rmse": float(np.sqrt(np.mean(np.square(height_err)))),
        "rs_z_offset": z_offset,
    }




def compute_remote_height_metrics_affine(gt_height, pred_pts, valid_mask):
    pred_height = pred_pts[..., 2]
    overlap = valid_mask & np.isfinite(gt_height) & np.isfinite(pred_height)
    if not overlap.any():
        return {
            "rs_height_mae_affine": float("nan"),
            "rs_height_rmse_affine": float("nan"),
            "rs_z_scale_affine": float("nan"),
            "rs_z_offset_affine": float("nan"),
        }

    pred_vec = pred_height[overlap].reshape(-1)
    gt_vec = gt_height[overlap].reshape(-1)
    design = np.stack([pred_vec, np.ones_like(pred_vec)], axis=1)
    scale, offset = np.linalg.lstsq(design, gt_vec, rcond=None)[0]
    pred_height_aligned = pred_height * scale + offset
    height_err = pred_height_aligned[overlap] - gt_height[overlap]
    return {
        "rs_height_mae_affine": float(np.mean(np.abs(height_err))),
        "rs_height_rmse_affine": float(np.sqrt(np.mean(np.square(height_err)))),
        "rs_z_scale_affine": float(scale),
        "rs_z_offset_affine": float(offset),
    }


def compute_joint_global_point_l1(batch, joint_preds, remote_sample):
    total_error = 0.0
    total_count = 0

    for view_idx, view in enumerate(batch):
        gt_pts = view["pts3d"][0].detach().cpu().numpy()
        pr_pts = joint_preds[view_idx]["pts3d"][0].detach().cpu().numpy()
        valid_mask = np.isfinite(gt_pts).all(axis=-1) & np.isfinite(pr_pts).all(axis=-1)
        if valid_mask.any():
            diff = np.abs(pr_pts - gt_pts).sum(axis=-1)
            total_error += float(diff[valid_mask].sum())
            total_count += int(valid_mask.sum())

    gt_remote_pts = remote_sample["remote_pointmap"]
    pr_remote_pts = joint_preds[len(batch)]["pts3d"][0].detach().cpu().numpy()
    remote_valid_mask = remote_sample["remote_valid_mask"].astype(bool)
    valid_mask = (
        remote_valid_mask
        & np.isfinite(gt_remote_pts).all(axis=-1)
        & np.isfinite(pr_remote_pts).all(axis=-1)
    )
    if valid_mask.any():
        diff = np.abs(pr_remote_pts - gt_remote_pts).sum(axis=-1)
        total_error += float(diff[valid_mask].sum())
        total_count += int(valid_mask.sum())

    if total_count == 0:
        return float("nan")
    return float(total_error / total_count)


def compute_joint_global_pointmaps_abs_rel(batch, joint_preds, remote_sample):
    gt_pts_list = [view["pts3d"].detach().cpu() for view in batch]
    pr_pts_list = [joint_preds[view_idx]["pts3d"].detach().cpu() for view_idx in range(len(batch))]
    valid_masks = [
        (
            torch.isfinite(view["pts3d"]).all(dim=-1)
            & torch.isfinite(joint_preds[view_idx]["pts3d"]).all(dim=-1)
        ).detach().cpu()
        for view_idx, view in enumerate(batch)
    ]

    gt_remote_pts = torch.from_numpy(remote_sample["remote_pointmap"]).unsqueeze(0).float()
    pr_remote_pts = joint_preds[len(batch)]["pts3d"].detach().cpu()
    remote_valid_mask = torch.from_numpy(remote_sample["remote_valid_mask"]).unsqueeze(0).bool()
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

    total_error = 0.0
    total_count = 0
    for gt_pts, pr_pts, valid_mask in zip(gt_pts_norm, pr_pts_norm, valid_masks):
        gt_np = gt_pts[0].numpy()
        pr_np = pr_pts[0].numpy()
        valid_np = valid_mask[0].numpy().astype(bool)
        gt_norm = np.linalg.norm(gt_np, axis=-1)
        combined_mask = valid_np & (gt_norm > 0)
        if not combined_mask.any():
            continue
        rel_ae = np.linalg.norm(pr_np - gt_np, axis=-1) / np.clip(gt_norm, 1e-8, None)
        total_error += float(rel_ae[combined_mask].sum())
        total_count += int(combined_mask.sum())

    if total_count == 0:
        return float("nan")
    return float(total_error / total_count)


def select_items(data, indices):
    if torch.is_tensor(data):
        return data[indices]
    if isinstance(data, list):
        return [data[i] for i in indices]
    if isinstance(data, tuple):
        return tuple(data[i] for i in indices)
    return data


def select_batch_indices(batch, indices):
    selected_batch = []
    for view in batch:
        selected_view = {}
        for key, value in view.items():
            selected_view[key] = select_items(value, indices)
        selected_batch.append(selected_view)
    return selected_batch


def select_prediction_sample(preds, sample_idx):
    selected_preds = []
    for pred in preds:
        selected_pred = {}
        for key, value in pred.items():
            if torch.is_tensor(value):
                selected_pred[key] = value[sample_idx : sample_idx + 1]
            else:
                selected_pred[key] = value
        selected_preds.append(selected_pred)
    return selected_preds


def aggregate_scene_metrics(per_scene_results):
    if not per_scene_results:
        return {}
    metric_names = sorted({k for v in per_scene_results.values() for k in v.keys()})
    aggregated = {}
    for metric_name in metric_names:
        values = [scene_metrics[metric_name] for scene_metrics in per_scene_results.values()]
        finite_values = [v for v in values if np.isfinite(v)]
        aggregated[metric_name] = float(np.mean(finite_values)) if finite_values else float("nan")
    return aggregated


def diff_metric_dict(new_metrics, baseline_metrics):
    diff = {}
    for key, value in new_metrics.items():
        base = baseline_metrics.get(key)
        if base is None or not np.isfinite(value) or not np.isfinite(base):
            diff[key] = float("nan")
        else:
            diff[key] = float(value - base)
    return diff


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = not args.disable_cudnn_benchmark

    if args.amp:
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    aerial_loader, remote_loader = build_data_loaders(args)

    remote_samples_by_scene = {
        remote_loader.dataset[idx]["scene_name"]: remote_loader.dataset[idx]
        for idx in range(len(remote_loader.dataset))
    }

    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)

    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(args.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    aerial_per_scene = {}
    rs_per_scene = {}
    joint_per_scene = {}
    improvement_aerial = {}
    improvement_rs = {}

    for batch in aerial_loader:
        scene_names = list(batch[0]["label"])
        valid_indices = [
            sample_idx
            for sample_idx, scene_name in enumerate(scene_names)
            if scene_name in remote_samples_by_scene
        ]
        if not valid_indices:
            continue
        if len(valid_indices) != len(scene_names):
            batch = select_batch_indices(batch, valid_indices)
            scene_names = [scene_names[i] for i in valid_indices]

        remote_samples = [remote_samples_by_scene[scene_name] for scene_name in scene_names]

        for view in batch:
            view["idx"] = view["idx"][2:]

        ignore_keys = {
            "depthmap",
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
            "data_norm_type",
        }
        for view in batch:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        remote_image = torch.stack(
            [remote_sample["remote_image"] for remote_sample in remote_samples], dim=0
        ).to(device, non_blocking=True)
        remote_view = {
            "img": remote_image,
            "data_norm_type": [args.model.data_norm_type] * len(remote_samples),
        }

        with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
            aerial_preds = model(batch)
            rs_preds = model([remote_view])
            joint_preds = model(batch + [remote_view])

        aerial_metrics_by_scene = compute_aerial_scene_metrics(batch, aerial_preds)
        joint_aerial_metrics_by_scene = compute_aerial_scene_metrics(
            batch, joint_preds[: len(batch)]
        )
        rs_supports_metric_outputs = model_supports_metric_outputs(rs_preds)
        joint_supports_metric_outputs = model_supports_metric_outputs(joint_preds)

        for sample_idx, scene in enumerate(scene_names):
            remote_sample = remote_samples[sample_idx]
            aerial_metrics = aerial_metrics_by_scene[scene]
            aerial_per_scene[scene] = aerial_metrics

            gt_height = remote_sample["remote_height_map"]
            valid_mask = remote_sample["remote_valid_mask"].astype(bool)

            rs_pts = rs_preds[0]["pts3d"][sample_idx].detach().cpu().numpy()
            rs_metrics = compute_remote_height_metrics_affine(
                gt_height,
                rs_pts,
                valid_mask,
            )
            if rs_supports_metric_outputs:
                rs_metrics.update(
                    compute_remote_height_metrics(
                        gt_height,
                        rs_pts,
                        valid_mask,
                    )
                )
            rs_per_scene[scene] = rs_metrics

            joint_aerial_metrics = joint_aerial_metrics_by_scene[scene]
            joint_rs_pts = joint_preds[len(batch)]["pts3d"][sample_idx].detach().cpu().numpy()
            joint_rs_metrics = compute_remote_height_metrics_affine(
                gt_height,
                joint_rs_pts,
                valid_mask,
            )
            if joint_supports_metric_outputs:
                joint_rs_metrics.update(
                    compute_remote_height_metrics(
                        gt_height,
                        joint_rs_pts,
                        valid_mask,
                    )
                )

            single_batch = select_batch_indices(batch, [sample_idx])
            single_joint_preds = select_prediction_sample(joint_preds, sample_idx)
            joint_per_scene[scene] = {
                **joint_aerial_metrics,
                **joint_rs_metrics,
                "joint_global_point_l1": (
                    compute_joint_global_point_l1(
                        batch=single_batch,
                        joint_preds=single_joint_preds,
                        remote_sample=remote_sample,
                    )
                    if joint_supports_metric_outputs
                    else float("nan")
                ),
                "joint_global_pointmaps_abs_rel": compute_joint_global_pointmaps_abs_rel(
                    batch=single_batch,
                    joint_preds=single_joint_preds,
                    remote_sample=remote_sample,
                ),
            }
            improvement_aerial[scene] = diff_metric_dict(joint_aerial_metrics, aerial_metrics)
            improvement_rs[scene] = diff_metric_dict(joint_rs_metrics, rs_metrics)

    paired_scenes = sorted(set(aerial_per_scene.keys()) & set(rs_per_scene.keys()) & set(joint_per_scene.keys()))

    per_scene_results = {}
    for scene in paired_scenes:
        per_scene_results[scene] = {
            "aerial_only": aerial_per_scene[scene],
            "rs_only": rs_per_scene[scene],
            "joint": joint_per_scene[scene],
            "improvement": {
                "aerial_vs_aerial_only": improvement_aerial[scene],
                "rs_vs_rs_only": improvement_rs[scene],
            },
        }

    result = {
        "metadata": {
            "benchmark_name": "RS-Aerial Reconstruction Benchmark",
            "paired_scene_count": len(paired_scenes),
            "aerial_scene_count": len(aerial_per_scene),
            "rs_scene_count": len(rs_per_scene),
            "resolution": resolve_resolution(args.dataset.resolution_val),
            "joint_forward_implemented": True,
            "joint_metrics_implemented": True,
            "joint_metric_names": [
                "joint_global_point_l1",
                "joint_global_pointmaps_abs_rel",
            ],
        },
        "aerial_only": {
            "per_scene": {scene: aerial_per_scene[scene] for scene in paired_scenes},
            "average": aggregate_scene_metrics(
                {scene: aerial_per_scene[scene] for scene in paired_scenes}
            ),
        },
        "rs_only": {
            "per_scene": {scene: rs_per_scene[scene] for scene in paired_scenes},
            "average": aggregate_scene_metrics(
                {scene: rs_per_scene[scene] for scene in paired_scenes}
            ),
        },
        "joint": {
            "per_scene": {scene: joint_per_scene[scene] for scene in paired_scenes},
            "average": aggregate_scene_metrics(
                {scene: joint_per_scene[scene] for scene in paired_scenes}
            ),
        },
        "improvement": {
            "aerial_vs_aerial_only": {
                "per_scene": {scene: improvement_aerial[scene] for scene in paired_scenes},
                "average": aggregate_scene_metrics(
                    {scene: improvement_aerial[scene] for scene in paired_scenes}
                ),
            },
            "rs_vs_rs_only": {
                "per_scene": {scene: improvement_rs[scene] for scene in paired_scenes},
                "average": aggregate_scene_metrics(
                    {scene: improvement_rs[scene] for scene in paired_scenes}
                ),
            },
        },
        "per_scene_results": per_scene_results,
    }

    with open(os.path.join(args.output_dir, "rs_aerial_benchmark_results.json"), "w") as f:
        json.dump(result, f, indent=4)

    with open(os.path.join(args.output_dir, "rs_aerial_per_scene_results.json"), "w") as f:
        json.dump(per_scene_results, f, indent=4)

    print("Aerial-only average results:")
    for metric_name, metric_value in result["aerial_only"]["average"].items():
        print(f"{metric_name}: {metric_value}")
    print("RS-only average results:")
    for metric_name, metric_value in result["rs_only"]["average"].items():
        print(f"{metric_name}: {metric_value}")
    print("Joint average results:")
    for metric_name, metric_value in result["joint"]["average"].items():
        print(f"{metric_name}: {metric_value}")
    print("Improvement over aerial-only:")
    for metric_name, metric_value in result["improvement"]["aerial_vs_aerial_only"]["average"].items():
        print(f"{metric_name}: {metric_value}")
    print("Improvement over rs-only:")
    for metric_name, metric_value in result["improvement"]["rs_vs_rs_only"]["average"].items():
        print(f"{metric_name}: {metric_value}")
    print("Benchmark metadata:")
    print(json.dumps(result["metadata"], indent=4))


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rs_aerial_benchmark",
)
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()
