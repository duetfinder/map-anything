# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Stage-1 RS-Aerial benchmark: evaluate remote-sensing image geometry only on scenes that already
have remote image + per-pixel pointmap labels.
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

from mapanything.datasets import get_test_data_loader
from mapanything.models import init_model
from mapanything.utils.geometry import normalize_multiple_pointclouds
from mapanything.utils.metrics import m_rel_ae
from mapanything.utils.misc import StreamToLogger

log = logging.getLogger(__name__)


def build_dataset(dataset, batch_size, num_workers):
    print("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )
    print("Dataset length: ", len(loader))
    return loader


def normalized_remote_metrics(gt_pts, pred_pts, valid_mask):
    valid_mask = valid_mask & np.isfinite(gt_pts).all(axis=-1) & np.isfinite(pred_pts).all(axis=-1)

    gt_pts_t = torch.from_numpy(gt_pts).unsqueeze(0)
    pred_pts_t = torch.from_numpy(pred_pts).unsqueeze(0)
    valid_mask_t = torch.from_numpy(valid_mask).unsqueeze(0)

    pred_norm_out = normalize_multiple_pointclouds(
        [pred_pts_t], [valid_mask_t], norm_mode="avg_dis", ret_factor=True
    )
    gt_norm_out = normalize_multiple_pointclouds(
        [gt_pts_t], [valid_mask_t], norm_mode="avg_dis", ret_factor=True
    )
    pred_pts_norm = pred_norm_out[0][0].numpy()
    gt_pts_norm = gt_norm_out[0][0].numpy()

    if not valid_mask.any():
        return {
            "rs_pointmap_abs_rel": float("nan"),
            "rs_height_mae": float("nan"),
            "rs_height_rmse": float("nan"),
        }

    rs_pointmap_abs_rel = m_rel_ae(gt=gt_pts_norm, pred=pred_pts_norm, mask=valid_mask)

    pred_z = pred_pts_norm[..., 2]
    gt_z = gt_pts_norm[..., 2]
    overlap = valid_mask & np.isfinite(pred_z) & np.isfinite(gt_z)
    if overlap.any():
        z_err = pred_z[overlap] - gt_z[overlap]
        rs_height_mae = float(np.mean(np.abs(z_err)))
        rs_height_rmse = float(np.sqrt(np.mean(np.square(z_err))))
    else:
        rs_height_mae = float("nan")
        rs_height_rmse = float("nan")

    return {
        "rs_pointmap_abs_rel": float(rs_pointmap_abs_rel),
        "rs_height_mae": rs_height_mae,
        "rs_height_rmse": rs_height_rmse,
    }


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
        dataset.split("(")[0]: build_dataset(dataset, args.batch_size, args.dataset.num_workers)
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

        per_scene_results = {}
        for batch in data_loader:
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

            metrics = normalized_remote_metrics(
                gt_pts=gt_pts,
                pred_pts=pred_pts,
                valid_mask=valid_mask,
            )
            per_scene_results[scene] = metrics

        with open(
            os.path.join(args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"),
            "w",
        ) as f:
            json.dump(per_scene_results, f, indent=4)

        across_dataset_results = {}
        metric_names = next(iter(per_scene_results.values())).keys()
        for metric_name in metric_names:
            values = [scene_metrics[metric_name] for scene_metrics in per_scene_results.values()]
            across_dataset_results[metric_name] = float(np.mean(values))

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
    metric_names = next(iter(per_dataset_results.values())).keys()
    for metric_name in metric_names:
        values = [dataset_metrics[metric_name] for dataset_metrics in per_dataset_results.values()]
        average_results[metric_name] = float(np.mean(values))
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
    config_name="rs_aerial_stage1_benchmark",
)
def execute_benchmarking(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()
