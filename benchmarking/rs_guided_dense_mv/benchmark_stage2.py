# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Stage-2 RS-Aerial benchmark skeleton.

Current scope:
- evaluate aerial multi-view metrics on scenes that also have remote supervision
- evaluate remote-view geometry metrics on the same paired scenes
- merge both result groups into a single per-scene/per-dataset output

Current cross-view metric:
- crossview_pointmap_gap_abs: absolute gap between aerial and remote pointmap errors on the
  same scene. This is the first frame-invariant cross-view metric that is valid before a shared
  aerial/remote global frame is introduced into model inference.

Not implemented yet:
- shared-frame geometric consistency metrics between aerial and remote predictions
"""

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
from benchmarking.rs_guided_dense_mv.benchmark_stage1 import normalized_remote_metrics
from mapanything.models import init_model
from mapanything.utils.metrics import (
    calculate_auc_np,
    evaluate_ate,
    l2_distance_of_unit_ray_directions_to_angular_error,
    m_rel_ae,
    se3_to_relative_pose_error,
)
from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


def build_data_loaders(args):
    aerial_loader = build_dataset(
        args.dataset.test_dataset_aerial, args.batch_size, args.dataset.num_workers
    )
    remote_loader = build_dataset(
        args.dataset.test_dataset_remote, args.batch_size, args.dataset.num_workers
    )
    return aerial_loader, remote_loader


def compute_aerial_scene_metrics(batch, preds):
    n_views = len(batch)
    gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(batch, preds)

    batch_metrics = {}
    batch_size = batch[0]["img"].shape[0]
    for batch_idx in range(batch_size):
        scene = batch[0]["label"][batch_idx]

        pointmaps_abs_rel_across_views = []
        z_depth_abs_rel_across_views = []
        ray_dirs_err_deg_across_views = []
        gt_poses_curr_set = []
        pr_poses_curr_set = []

        for view_idx in range(n_views):
            valid_mask_curr_view = valid_masks[view_idx][batch_idx].numpy()

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

            pointmaps_abs_rel_across_views.append(pointmaps_abs_rel_curr_view)
            z_depth_abs_rel_across_views.append(z_depth_abs_rel_curr_view)

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

        batch_metrics[scene] = {
            "pointmaps_abs_rel": float(np.mean(pointmaps_abs_rel_across_views)),
            "z_depth_abs_rel": float(np.mean(z_depth_abs_rel_across_views)),
            "pose_ate_rmse": pose_ate_curr_set,
            "pose_auc_5": float(pose_auc_5_curr_set * 100.0),
            "ray_dirs_err_deg": float(np.mean(ray_dirs_err_deg_across_views)),
        }

    return batch_metrics


def aggregate_scene_metrics(per_scene_results):
    metric_names = []
    for scene_metrics in per_scene_results.values():
        metric_names.extend(scene_metrics.keys())
    metric_names = sorted(set(metric_names))

    aggregated = {}
    for metric_name in metric_names:
        values = []
        for scene_metrics in per_scene_results.values():
            metric_value = scene_metrics.get(metric_name)
            if metric_value is None:
                continue
            values.append(metric_value)
        aggregated[metric_name] = float(np.mean(values)) if values else float("nan")
    return aggregated


def add_crossview_metrics(per_scene_results):
    for scene_metrics in per_scene_results.values():
        if "pointmaps_abs_rel" in scene_metrics and "rs_pointmap_abs_rel" in scene_metrics:
            scene_metrics["crossview_pointmap_gap_abs"] = float(
                abs(scene_metrics["pointmaps_abs_rel"] - scene_metrics["rs_pointmap_abs_rel"])
            )


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
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    aerial_loader, remote_loader = build_data_loaders(args)

    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)

    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(args.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    per_scene_results = {}

    for batch in aerial_loader:
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
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
            preds = model(batch)

        aerial_scene_metrics = compute_aerial_scene_metrics(batch, preds)
        for scene, scene_metrics in aerial_scene_metrics.items():
            per_scene_results.setdefault(scene, {}).update(scene_metrics)

    for batch in remote_loader:
        scene = batch["scene_name"][0]
        remote_image = batch["remote_image"].to(device, non_blocking=True)
        view = {
            "img": remote_image,
            "data_norm_type": [args.model.data_norm_type],
        }

        with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
            preds = model([view])

        pred_pts = preds[0]["pts3d"][0].detach().cpu().numpy()
        gt_pts = batch["remote_pointmap"][0].numpy()
        valid_mask = batch["remote_valid_mask"][0].numpy().astype(bool)
        remote_metrics = normalized_remote_metrics(
            gt_pts=gt_pts,
            pred_pts=pred_pts,
            valid_mask=valid_mask,
        )
        per_scene_results.setdefault(scene, {}).update(remote_metrics)

    common_scene_results = {
        scene_name: scene_metrics
        for scene_name, scene_metrics in per_scene_results.items()
        if all(
            key in scene_metrics
            for key in [
                "pointmaps_abs_rel",
                "z_depth_abs_rel",
                "pose_ate_rmse",
                "pose_auc_5",
                "ray_dirs_err_deg",
                "rs_pointmap_abs_rel",
                "rs_height_mae",
                "rs_height_rmse",
            ]
        )
    }
    add_crossview_metrics(common_scene_results)

    with open(os.path.join(args.output_dir, "VigorChicagoRSAerialJoint_per_scene_results.json"), "w") as f:
        json.dump(common_scene_results, f, indent=4)

    avg_results = aggregate_scene_metrics(common_scene_results)
    with open(
        os.path.join(args.output_dir, "VigorChicagoRSAerialJoint_avg_across_all_scenes.json"),
        "w",
    ) as f:
        json.dump(avg_results, f, indent=4)

    metadata = {
        "benchmark_stage": "stage2_skeleton",
        "crossview_metrics_computed": True,
        "crossview_metric_names": ["crossview_pointmap_gap_abs"],
        "crossview_metric_definition": (
            "Absolute gap between aerial pointmaps_abs_rel and remote rs_pointmap_abs_rel on the "
            "same scene. Lower is better."
        ),
        "common_scene_count": len(common_scene_results),
        "aerial_scene_count": len(aerial_loader.dataset.scenes),
        "remote_scene_count": len(remote_loader.dataset),
    }
    with open(os.path.join(args.output_dir, "benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Average results across all paired scenes:")
    for metric_name, metric_value in avg_results.items():
        print(f"{metric_name}: {metric_value}")
    print("Benchmark metadata:")
    print(json.dumps(metadata, indent=4))


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rs_aerial_stage2_benchmark",
)
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()
