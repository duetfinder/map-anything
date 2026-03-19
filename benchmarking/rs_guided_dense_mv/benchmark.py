# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Minimal benchmark for dense multi-view reconstruction with remote-sensing top-view reference.
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


@torch.no_grad()
def benchmark(args):
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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

    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        dataset.split("(")[0]: build_dataset(
            dataset, args.batch_size, args.dataset.num_workers
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
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

    per_dataset_results = {}

    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)

        per_scene_results = {}
        for dataset_scene in data_loader.dataset.dataset.scenes:
            per_scene_results[dataset_scene] = {
                "pointmaps_abs_rel": [],
                "z_depth_abs_rel": [],
                "pose_ate_rmse": [],
                "pose_auc_5": [],
                "ray_dirs_err_deg": [],
            }

        for batch in data_loader:
            n_views = len(batch)
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

            gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(batch, preds)

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
                    ray_dirs_err_deg = l2_distance_of_unit_ray_directions_to_angular_error(
                        ray_dirs_l2
                    )
                    ray_dirs_err_deg_across_views.append(
                        torch.mean(ray_dirs_err_deg).cpu().numpy()
                    )

                    gt_poses_curr_set.append(gt_info["poses"][view_idx][batch_idx])
                    pr_poses_curr_set.append(pr_info["poses"][view_idx][batch_idx])

                pointmaps_abs_rel_curr_set = float(np.mean(pointmaps_abs_rel_across_views))
                z_depth_abs_rel_curr_set = float(np.mean(z_depth_abs_rel_across_views))
                ray_dirs_err_deg_curr_set = float(np.mean(ray_dirs_err_deg_across_views))

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
                pose_auc_5_curr_set = float(pose_auc_5_curr_set * 100.0)

                per_scene_results[scene]["pointmaps_abs_rel"].append(pointmaps_abs_rel_curr_set)
                per_scene_results[scene]["z_depth_abs_rel"].append(z_depth_abs_rel_curr_set)
                per_scene_results[scene]["ray_dirs_err_deg"].append(ray_dirs_err_deg_curr_set)
                per_scene_results[scene]["pose_ate_rmse"].append(pose_ate_curr_set)
                per_scene_results[scene]["pose_auc_5"].append(pose_auc_5_curr_set)

        with open(
            os.path.join(args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"),
            "w",
        ) as f:
            json.dump(per_scene_results, f, indent=4)

        across_dataset_results = {}
        for scene, scene_metrics in per_scene_results.items():
            for metric_name, metric_values in scene_metrics.items():
                if metric_name not in across_dataset_results:
                    across_dataset_results[metric_name] = []
                across_dataset_results[metric_name].extend(metric_values)

        for metric_name, metric_values in across_dataset_results.items():
            if metric_values:
                across_dataset_results[metric_name] = float(np.mean(metric_values))
            else:
                across_dataset_results[metric_name] = float("nan")

        with open(
            os.path.join(
                args.output_dir, f"{benchmark_dataset_name}_avg_across_all_scenes.json"
            ),
            "w",
        ) as f:
            json.dump(across_dataset_results, f, indent=4)

        print("Average results across all scenes for dataset: ", benchmark_dataset_name)
        for metric_name, metric_value in across_dataset_results.items():
            print(f"{metric_name}: {metric_value}")

        per_dataset_results[benchmark_dataset_name] = across_dataset_results

    average_results = {}
    metric_names = per_dataset_results[next(iter(per_dataset_results))].keys()
    for metric_name in metric_names:
        values = [
            per_dataset_results[dataset_name][metric_name]
            for dataset_name in per_dataset_results
            if np.isfinite(per_dataset_results[dataset_name][metric_name])
        ]
        average_results[metric_name] = float(np.mean(values)) if values else float("nan")
    per_dataset_results["Average"] = average_results

    print("Benchmarking Done! ...")
    print("Average results across all datasets:")
    for metric_name, metric_value in average_results.items():
        print(f"{metric_name}: {metric_value}")

    with open(os.path.join(args.output_dir, "per_dataset_results.json"), "w") as f:
        json.dump(per_dataset_results, f, indent=4)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rs_guided_dense_mv_benchmark",
)
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()
