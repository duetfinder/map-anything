# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Unified RS-Aerial benchmark.

Current executable scope:
- Aerial-only metrics on paired scenes
- RS-only height and pointmap metrics on paired scenes
- Joint aerial+RS forward inference on paired scenes
- joint_global_pointmaps_abs_rel
"""

import ast
import json
import logging
import os
import re
import sys
import warnings
from functools import lru_cache
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
from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
)
from mapanything.utils.hf_utils.hf_helpers import (
    initialize_mapanything_local,
    initialize_mapanything_model,
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

DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_MAPANYTHING_HF_MODEL = "facebook/map-anything"
SUPPORTED_MODEL_INPUTS = {
    "pi3",
    "pi3_modality_embedding",
    "pi3_modality_embedding_remote_head",
    "vggt",
    "vggt_omega",
    "da3",
    "mapanything",
    "mapanything_rs_joint",
}
DEFAULT_CONFIG_OVERRIDES = {
    "pi3": [
        "machine=aws",
        "model=pi3",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "pi3_modality_embedding": [
        "machine=aws",
        "model=pi3_modality_embedding",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "pi3_modality_embedding_remote_head": [
        "machine=aws",
        "model=pi3_modality_embedding_remote_head",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "vggt": [
        "machine=aws",
        "model=vggt",
    ],
    "vggt_omega": [
        "machine=aws",
        "model=vggt_omega",
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
    "mapanything_rs_joint": [
        "machine=aws",
        "model=mapanything_rs_joint",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
}
CLASH_ENV = {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7891",
}


def cfg_get(args, key, default=None):
    if args is None:
        return default
    if isinstance(args, DictConfig):
        return args.get(key, default)
    return getattr(args, key, default)


def resolve_requested_model_name(args):
    requested_model = cfg_get(args, "model_input")
    if requested_model:
        return str(requested_model)
    model_cfg = cfg_get(args, "model")
    if model_cfg is not None:
        model_config = cfg_get(model_cfg, "model_config")
        if model_config is not None:
            model_name = cfg_get(model_config, "name")
            if model_name:
                return str(model_name)
        model_str = cfg_get(model_cfg, "model_str")
        if model_str:
            return str(model_str)
    raise ValueError("Could not resolve model name from config")


def is_raw_vggt_checkpoint(args, model_name):
    checkpoint_path = cfg_get(args, "checkpoint_path")
    if not checkpoint_path:
        return False
    checkpoint_path = Path(str(checkpoint_path))
    checkpoint_str = str(checkpoint_path)
    if model_name == "vggt":
        return checkpoint_path.name == "model.pt" and "checkpoints/vggt" in checkpoint_str
    if model_name == "vggt_omega":
        return (
            checkpoint_path.name in {"vggt_omega_1b_512.pt", "model.pt"}
            and "checkpoints/vggt_omega" in checkpoint_str
        )
    return False


@lru_cache(maxsize=32)
def checkpoint_has_key_prefix(checkpoint_path, *prefixes):
    if not checkpoint_path:
        return False
    checkpoint_path = Path(str(checkpoint_path))
    if not checkpoint_path.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("Failed to inspect checkpoint %s: %s", checkpoint_path, exc)
        return False
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_prefix = any(
        any(str(key).startswith(prefix) for prefix in prefixes)
        for key in state_dict.keys()
    )
    del ckpt, state_dict
    return has_prefix


def resolve_vggt_output_heads(args, model_name):
    if model_name != "vggt":
        return None, None

    ordinary_head = cfg_get(args, "vggt_ordinary_output_head")
    remote_head = cfg_get(args, "vggt_remote_output_head")
    export_mode = cfg_get(args, "vggt_export_mode")

    if export_mode == "mixed":
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "point"
    elif export_mode == "depth_all":
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "depth"
    elif export_mode == "point_all":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "point"
    elif export_mode == "ordinary_point_remote_depth":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "depth"

    return ordinary_head, remote_head


def is_vggt_p7_p5b_projection_aux_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return False
    checkpoint_path_lower = str(checkpoint_path).lower()
    return (
        "p7_vggt_p5b_shared_norm_projection_aux" in checkpoint_path_lower
        or "p7_chicago_newyork_full_p5b_joint" in checkpoint_path_lower
        or "p7_allcities_p5b_joint" in checkpoint_path_lower
        or "p7_allcities_p5b_parallel_token_aux" in checkpoint_path_lower
        or "p7_p5b_parallel_token_aux" in checkpoint_path_lower
        or "p7_proj_moge_aux" in checkpoint_path_lower
        or "p7_proj_moge_denseheight" in checkpoint_path_lower
        or "p7_proj_moge_pmheight" in checkpoint_path_lower
        or "p7_proj_moge_robustpm" in checkpoint_path_lower
        or "p7_proj_denseh" in checkpoint_path_lower
        or "p7_proj_headonly" in checkpoint_path_lower
        or "p7_proj_robust" in checkpoint_path_lower
        or "p7_proj_tokenres" in checkpoint_path_lower
        or "p7_proj_views" in checkpoint_path_lower
        or "overlappm" in checkpoint_path_lower
        or checkpoint_has_key_prefix(
            checkpoint_path,
            "remote_projection_aux_token_",
            "remote_projection_aux_image_stem.",
            "remote_projection_aux_head.",
        )
    )


def resolve_vggt_projection_aux_source(checkpoint_path):
    checkpoint_path_lower = str(checkpoint_path or "").lower()
    if (
        "parallel_token_aux" in checkpoint_path_lower
        or "parallel_tokens_aux" in checkpoint_path_lower
        or "private_tokens" in checkpoint_path_lower
        or "p7_proj_moge_pmheight" in checkpoint_path_lower
        or "p7_proj_moge_robustpm" in checkpoint_path_lower
        or "p7_proj_denseh" in checkpoint_path_lower
        or "p7_proj_headonly" in checkpoint_path_lower
        or "p7_proj_robust" in checkpoint_path_lower
        or "p7_proj_tokenres" in checkpoint_path_lower
        or "p7_proj_views" in checkpoint_path_lower
        or "overlappm" in checkpoint_path_lower
        or checkpoint_has_key_prefix(
            checkpoint_path,
            "remote_projection_aux_token_",
        )
    ):
        return "tokens"
    return "pointmap"


def is_vggt_p7_token_residual_checkpoint(checkpoint_path):
    checkpoint_path_lower = str(checkpoint_path or "").lower()
    return (
        "p7_proj_tokenres" in checkpoint_path_lower
        or "tokenres" in checkpoint_path_lower
        or "token_residual" in checkpoint_path_lower
        or checkpoint_has_key_prefix(checkpoint_path, "remote_projection_aux_token_residual")
    )


def is_vggt_pre_aggregator_view_type_bias_checkpoint(checkpoint_path):
    return checkpoint_has_key_prefix(checkpoint_path, "pre_aggregator_view_type_embedding.")


def is_vggt_remote_to_aerial_gated_residual_checkpoint(checkpoint_path):
    return checkpoint_has_key_prefix(
        checkpoint_path, "remote_to_aerial_gate", "remote_to_aerial_residual."
    )


def resolve_vggt_late_fusion_type(checkpoint_path):
    checkpoint_path_lower = str(checkpoint_path or "").lower()
    if checkpoint_has_key_prefix(checkpoint_path, "remote_to_aerial_late_film."):
        return "film"
    if checkpoint_has_key_prefix(
        checkpoint_path,
        "remote_to_aerial_late_cross_attention.",
        "remote_to_aerial_late_query_norm.",
        "remote_to_aerial_late_key_value_norm.",
    ):
        return "cross_attention"
    if "crossattn" in checkpoint_path_lower or "cross_attention" in checkpoint_path_lower:
        return "cross_attention"
    if "film" in checkpoint_path_lower:
        return "film"
    return "none"


def is_vggt_split_remote_aggregator_checkpoint(checkpoint_path):
    return resolve_vggt_late_fusion_type(checkpoint_path) != "none" or checkpoint_has_key_prefix(
        checkpoint_path, "remote_aggregator."
    )


def is_vggt_remote_private_aggregator_checkpoint(checkpoint_path):
    return checkpoint_has_key_prefix(checkpoint_path, "remote_aggregator.")


def resolve_config_overrides(args, model_name):
    config_overrides = cfg_get(args, "config_overrides")
    if config_overrides is not None:
        overrides = list(config_overrides)
    else:
        if model_name not in DEFAULT_CONFIG_OVERRIDES:
            raise ValueError(
                f"Unsupported model input '{model_name}'. Supported: {sorted(SUPPORTED_MODEL_INPUTS)}"
            )
        overrides = list(DEFAULT_CONFIG_OVERRIDES[model_name])

    checkpoint_path = cfg_get(args, "checkpoint_path")
    if is_raw_vggt_checkpoint(args, model_name):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=true",
                f"model.model_config.custom_ckpt_path={checkpoint_path}",
            ]
        )

    if cfg_get(args, "vggt_joint_remote_export", False):
        if model_name != "vggt":
            raise ValueError("vggt_joint_remote_export is only supported with vggt")
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=false",
                "model.model_config.use_point_head_for_remote=true",
            ]
        )

    ordinary_head, remote_head = resolve_vggt_output_heads(args, model_name)
    if ordinary_head is not None:
        overrides.append(f"model.model_config.ordinary_output_head={ordinary_head}")
    if remote_head is not None:
        overrides.append(f"model.model_config.remote_output_head={remote_head}")
    if model_name == "vggt" and cfg_get(args, "vggt_use_remote_private_point_head", False):
        overrides.extend(
            [
                "model.model_config.use_remote_private_point_head=true",
                "model.model_config.output_point_head_for_consistency=true",
            ]
        )

    if model_name == "vggt" and checkpoint_path and is_vggt_remote_private_aggregator_checkpoint(checkpoint_path):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=false",
                "model.model_config.use_point_head_for_remote=true",
                "model.model_config.ordinary_output_head=depth",
                "model.model_config.remote_output_head=point",
                "model.model_config.use_remote_private_point_head=true",
                "model.model_config.output_point_head_for_consistency=true",
                "model.model_config.use_split_remote_aggregator=true",
                "model.model_config.use_remote_private_aggregator=true",
                "model.model_config.remote_to_aerial_late_fusion_type=none",
                "model.model_config.protect_ordinary_heads_from_remote=true",
            ]
        )

    if model_name == "vggt" and checkpoint_path:
        if is_vggt_p7_p5b_projection_aux_checkpoint(checkpoint_path):
            use_pre_aggregator_bias = is_vggt_pre_aggregator_view_type_bias_checkpoint(
                checkpoint_path
            )
            use_remote_gated_residual = is_vggt_remote_to_aerial_gated_residual_checkpoint(
                checkpoint_path
            )
            use_split_remote_aggregator = is_vggt_split_remote_aggregator_checkpoint(
                checkpoint_path
            )
            late_fusion_type = resolve_vggt_late_fusion_type(checkpoint_path)
            overrides.extend(
                [
                    "model.model_config.load_pretrained_weights=false",
                    "model.model_config.load_custom_ckpt=false",
                    "model.model_config.use_point_head_for_remote=true",
                    "model.model_config.ordinary_output_head=depth",
                    "model.model_config.remote_output_head=point",
                    "model.model_config.use_remote_private_point_head=true",
                    "model.model_config.output_point_head_for_consistency=true",
                    "model.model_config.use_view_type_bias=false",
                    f"model.model_config.use_pre_aggregator_view_type_bias={str(use_pre_aggregator_bias).lower()}",
                    f"model.model_config.use_remote_to_aerial_gated_residual={str(use_remote_gated_residual).lower()}",
                    "model.model_config.remote_to_aerial_residual_hidden_scale=0.25",
                    "model.model_config.remote_to_aerial_gate_init=0.0",
                    f"model.model_config.use_split_remote_aggregator={str(use_split_remote_aggregator).lower()}",
                    f"model.model_config.remote_to_aerial_late_fusion_type={late_fusion_type}",
                    "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                    "model.model_config.remote_to_aerial_late_fusion_gate_init=0.0",
                    "model.model_config.remote_to_aerial_cross_attention_heads=8",
                    "model.model_config.remote_to_aerial_max_remote_tokens=256",
                    "model.model_config.protect_ordinary_heads_from_remote=false",
                    "model.model_config.use_remote_projection_aux_head=true",
                    "model.model_config.remote_projection_aux_hidden_dim=96",
                    f"model.model_config.remote_projection_aux_source={resolve_vggt_projection_aux_source(checkpoint_path)}",
                    "model.model_config.remote_projection_aux_detach_pointmap=false",
                    "model.model_config.remote_projection_aux_use_rgb=true",
                    "model.model_config.remote_projection_aux_use_coord=true",
                    "model.model_config.remote_projection_aux_image_stem_dim=32",
                    "model.model_config.remote_projection_aux_positive_slope=true",
                    "model.model_config.remote_projection_aux_slope_init=0.1",
                    "model.model_config.remote_projection_aux_num_blocks=6",
                ]
            )
            if is_vggt_p7_token_residual_checkpoint(checkpoint_path):
                overrides.extend(
                    [
                        "model.model_config.use_remote_projection_aux_token_residual=true",
                        "model.model_config.remote_projection_aux_token_residual_hidden_scale=0.25",
                        "model.model_config.remote_projection_aux_token_residual_gate_init=0.01",
                    ]
                )

    if (
        model_name == "pi3_modality_embedding_remote_head"
        and checkpoint_path
        and "p7_pi3_remote_head_projection_aux" in str(checkpoint_path).lower()
    ):
        checkpoint_path_lower = str(checkpoint_path).lower()
        rel_match = re.search(r"relscale([0-9]+(?:p[0-9]+)?)", checkpoint_path_lower)
        offset_match = re.search(r"offsetscale([0-9]+(?:p[0-9]+)?)", checkpoint_path_lower)
        rel_scale = float(rel_match.group(1).replace("p", ".")) if rel_match else 1.0
        offset_scale = float(offset_match.group(1).replace("p", ".")) if offset_match else 1.0
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.use_remote_projection_aux_head=true",
                "model.model_config.remote_projection_aux_hidden_dim=96",
                "model.model_config.remote_projection_aux_use_rgb=true",
                "model.model_config.remote_projection_aux_use_coord=true",
                "model.model_config.remote_projection_aux_image_stem_dim=32",
                "model.model_config.remote_projection_aux_positive_slope=true",
                "model.model_config.remote_projection_aux_slope_init=0.1",
                "model.model_config.remote_projection_aux_num_blocks=6",
                f"model.model_config.remote_projection_aux_rel_height_output_scale={rel_scale}",
                f"model.model_config.remote_projection_aux_offset_output_scale={offset_scale}",
            ]
        )

    return overrides


def resolve_effective_model_name(args, requested_model_name):
    checkpoint_path = cfg_get(args, "checkpoint_path")
    if requested_model_name != "pi3" or not checkpoint_path:
        return requested_model_name

    checkpoint_path_lower = str(checkpoint_path).lower()
    if (
        "pi3_modality_embedding_remote_head" in checkpoint_path_lower
        or "p7_pi3_remote_head_projection_aux" in checkpoint_path_lower
    ):
        print(
            "Auto-detected Pi3 variant from checkpoint path: "
            "pi3_modality_embedding_remote_head"
        )
        return "pi3_modality_embedding_remote_head"
    if (
        "pi3_modality_embedding" in checkpoint_path_lower
        or "p3_pi3_freeze_shared" in checkpoint_path_lower
    ):
        print(
            "Auto-detected Pi3 variant from checkpoint path: "
            "pi3_modality_embedding"
        )
        return "pi3_modality_embedding"
    return requested_model_name


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


def build_local_config(
    args,
    config_overrides,
    requested_model_name,
    effective_model_name,
):
    checkpoint_path = cfg_get(args, "checkpoint_path")
    legacy_pretrained = cfg_get(cfg_get(args, "model"), "pretrained")
    if checkpoint_path is None:
        checkpoint_path = legacy_pretrained
    local_config = {
        "path": cfg_get(args, "config_path", DEFAULT_CONFIG_PATH),
        "checkpoint_path": checkpoint_path,
        "config_overrides": config_overrides,
        "strict": bool(cfg_get(args, "strict", False)),
        "model_str": cfg_get(args, "model_str") or effective_model_name,
    }

    # Legacy benchmark scripts may pass model.model_config.* Hydra overrides
    # together with model.pretrained. Preserve that resolved config when the
    # requested and effective model are the same. New checkpoint_path-based
    # calls use the export-style override flags instead.
    if (
        legacy_pretrained
        and checkpoint_path == legacy_pretrained
        and requested_model_name == effective_model_name
        and cfg_get(args, "config_overrides") is None
    ):
        local_config["model_config"] = cfg_get(cfg_get(args, "model"), "model_config")

    config_json_path = cfg_get(args, "config_json_path")
    if config_json_path is not None:
        local_config["config_json_path"] = config_json_path
    return local_config


def initialize_benchmark_model(args, device):
    requested_model_name = resolve_requested_model_name(args)
    effective_model_name = resolve_effective_model_name(args, requested_model_name)
    config_overrides = resolve_config_overrides(args, effective_model_name)
    maybe_enable_clash_proxy(cfg_get(args, "enable_clash_proxy", False))
    maybe_prepare_da3_pythonpath(effective_model_name)

    checkpoint_path = cfg_get(args, "checkpoint_path")
    legacy_pretrained = cfg_get(cfg_get(args, "model"), "pretrained")
    if checkpoint_path or legacy_pretrained:
        if checkpoint_path is None:
            print(
                "Using legacy model.pretrained checkpoint input. Prefer "
                "checkpoint_path=... for parity with scripts/export_pointcloud_ply.py."
            )
        if is_raw_vggt_checkpoint(args, effective_model_name):
            print(
                f"Detected raw {effective_model_name} checkpoint; loading it through "
                "model.model_config.custom_ckpt_path."
            )
        local_config = build_local_config(
            args,
            config_overrides,
            requested_model_name,
            effective_model_name,
        )
        print(f"Initializing model from local config: {local_config}")
        model = initialize_mapanything_local(local_config, device)
        return model.eval()

    if effective_model_name == "mapanything":
        hf_model_name = cfg_get(args, "hf_model_name") or DEFAULT_MAPANYTHING_HF_MODEL
        high_level_config = {
            "path": cfg_get(args, "config_path", DEFAULT_CONFIG_PATH),
            "hf_model_name": hf_model_name,
            "model_str": "mapanything",
            "config_overrides": config_overrides,
            "checkpoint_name": "model.safetensors",
            "config_name": "config.json",
        }
        print(f"Initializing model from HuggingFace defaults: {high_level_config}")
        model = initialize_mapanything_model(high_level_config, device)
        return model.eval()

    model_cfg = cfg_get(args, "model")
    current_model_name = cfg_get(cfg_get(model_cfg, "model_config"), "name")
    if requested_model_name == effective_model_name == current_model_name:
        from mapanything.models import init_model

        print(f"Initializing model '{effective_model_name}' from Hydra benchmark config")
        model = init_model(
            cfg_get(model_cfg, "model_str"),
            cfg_get(model_cfg, "model_config"),
            torch_hub_force_reload=False,
        )
        model.to(device)
        return model.eval()

    from mapanything.models import init_model_from_config

    print(f"Initializing model '{effective_model_name}' from default wrapper weights")
    model = init_model_from_config(
        effective_model_name, device=device, machine="aws"
    )
    return model.eval()


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


def compute_remote_pointmap_metrics(gt_pts, pred_pts, valid_mask):
    overlap = (
        valid_mask
        & np.isfinite(gt_pts).all(axis=-1)
        & np.isfinite(pred_pts).all(axis=-1)
    )
    if not overlap.any():
        return {
            "rs_point_l1": float("nan"),
            "rs_point_l1_centered": float("nan"),
            "rs_point_l1_scale_aligned": float("nan"),
            "rs_point_scale_aligned_scale": float("nan"),
            "rs_point_abs_rel": float("nan"),
            "rs_point_abs_rel_centered": float("nan"),
            "rs_point_abs_rel_scale_aligned": float("nan"),
        }

    gt_vec = gt_pts[overlap]
    pred_vec = pred_pts[overlap]
    point_err = np.linalg.norm(pred_vec - gt_vec, axis=-1)

    gt_center = gt_vec.mean(axis=0, keepdims=True)
    pred_center = pred_vec.mean(axis=0, keepdims=True)
    centered_err = np.linalg.norm(
        (pred_vec - pred_center) - (gt_vec - gt_center),
        axis=-1,
    )

    pred_centered = pred_vec - pred_center
    gt_centered = gt_vec - gt_center
    denom = float(np.sum(pred_centered * pred_centered))
    if denom > 1e-12:
        scale = max(float(np.sum(pred_centered * gt_centered) / denom), 1e-8)
        pred_scale_aligned = pred_centered * scale
        scale_aligned_err = np.linalg.norm(pred_scale_aligned - gt_centered, axis=-1)
        scale_aligned_l1 = float(np.mean(scale_aligned_err))
    else:
        scale = float("nan")
        scale_aligned_l1 = float("nan")

    gt_norm = np.linalg.norm(gt_vec, axis=-1)
    abs_rel = point_err / np.clip(gt_norm, 1e-8, None)
    gt_centered_norm = np.linalg.norm(gt_centered, axis=-1)
    centered_abs_rel = centered_err / np.clip(gt_centered_norm, 1e-8, None)
    if denom > 1e-12:
        scale_aligned_abs_rel = scale_aligned_err / np.clip(gt_centered_norm, 1e-8, None)
        scale_aligned_abs_rel = float(np.mean(scale_aligned_abs_rel))
    else:
        scale_aligned_abs_rel = float("nan")

    return {
        "rs_point_l1": float(np.mean(point_err)),
        "rs_point_l1_centered": float(np.mean(centered_err)),
        "rs_point_l1_scale_aligned": scale_aligned_l1,
        "rs_point_scale_aligned_scale": scale,
        "rs_point_abs_rel": float(np.mean(abs_rel)),
        "rs_point_abs_rel_centered": float(np.mean(centered_abs_rel)),
        "rs_point_abs_rel_scale_aligned": scale_aligned_abs_rel,
    }


def apply_aux_point_residual_to_remote_pts(
    pred,
    sample_idx,
    norm_mode="avg_dis",
    apply_z=False,
):
    """Apply P7 aux normalized xy residual to one remote pointmap prediction."""
    if "remote_projection_offset_xy_pred" not in pred:
        return pred["pts3d"][sample_idx]

    pts = pred["pts3d"][sample_idx : sample_idx + 1]
    offset_xy = pred["remote_projection_offset_xy_pred"][sample_idx : sample_idx + 1]
    valid_mask = torch.isfinite(pts).all(dim=-1)
    pts_norm, norm_factor = normalize_multiple_pointclouds(
        [pts],
        valid_masks=[valid_mask],
        norm_mode=norm_mode,
        ret_factor=True,
    )
    corrected_norm = pts_norm.clone()
    corrected_norm[..., :2] = corrected_norm[..., :2] + offset_xy.to(corrected_norm)
    if apply_z and "remote_projection_rel_height_pred" in pred:
        z_residual = pred["remote_projection_rel_height_pred"][
            sample_idx : sample_idx + 1
        ].to(corrected_norm)
        corrected_norm[..., 2] = corrected_norm[..., 2] + z_residual
    corrected = corrected_norm * norm_factor
    if not apply_z:
        corrected[..., 2] = pts[..., 2]
    return corrected[0]


def get_joint_remote_metric_space_pointmaps(batch, joint_preds, remote_sample):
    pred_camera0 = torch.eye(4, device=joint_preds[0]["cam_quats"].device).unsqueeze(0)
    pred_camera0[..., :3, :3] = quaternion_to_rotation_matrix(joint_preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, 3] = joint_preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)
    gt_in_camera0 = inv(batch[0]["camera_pose"])

    gt_pts_list = []
    pr_pts_list = []
    valid_masks = []

    for view_idx, view in enumerate(batch):
        gt_pts = geotrf(gt_in_camera0, view["pts3d"]).detach().cpu()
        pr_pts = geotrf(pred_in_camera0, joint_preds[view_idx]["pts3d"]).detach().cpu()
        if "metric_scaling_factor" in joint_preds[view_idx]:
            pr_pts = pr_pts / joint_preds[view_idx]["metric_scaling_factor"].detach().cpu().view(-1, 1, 1, 1)

        gt_mask = view["valid_mask"].detach().cpu().bool()
        valid_mask = (
            gt_mask
            & torch.isfinite(gt_pts).all(dim=-1)
            & torch.isfinite(pr_pts).all(dim=-1)
        )
        gt_pts_list.append(gt_pts)
        pr_pts_list.append(pr_pts)
        valid_masks.append(valid_mask)

    gt_remote_pts = torch.from_numpy(remote_sample["remote_pointmap"]).unsqueeze(0).float()
    pr_remote_pts = joint_preds[len(batch)]["pts3d"].detach().cpu()
    if "metric_scaling_factor" in joint_preds[len(batch)]:
        pr_remote_pts = (
            pr_remote_pts
            / joint_preds[len(batch)]["metric_scaling_factor"].detach().cpu().view(-1, 1, 1, 1)
        )
    gt_remote_pts = geotrf(gt_in_camera0.detach().cpu(), gt_remote_pts)
    pr_remote_pts = geotrf(pred_in_camera0.detach().cpu(), pr_remote_pts)
    remote_valid_mask = torch.from_numpy(remote_sample["remote_valid_mask"]).unsqueeze(0).bool()
    remote_valid_mask = (
        remote_valid_mask
        & torch.isfinite(gt_remote_pts).all(dim=-1)
        & torch.isfinite(pr_remote_pts).all(dim=-1)
    )

    gt_pts_list.append(gt_remote_pts)
    pr_pts_list.append(pr_remote_pts)
    valid_masks.append(remote_valid_mask)
    return gt_pts_list, pr_pts_list, valid_masks


def compute_joint_global_pointmaps_abs_rel(batch, joint_preds, remote_sample):
    gt_pts_list, pr_pts_list, valid_masks = get_joint_remote_metric_space_pointmaps(
        batch=batch,
        joint_preds=joint_preds,
        remote_sample=remote_sample,
    )

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


def select_items(data, indices, batch_size):
    if torch.is_tensor(data):
        if data.ndim > 0 and data.shape[0] == batch_size:
            return data[indices]
        return data
    if isinstance(data, list):
        if len(data) == batch_size:
            return [data[i] for i in indices]
        return data
    if isinstance(data, tuple):
        if len(data) == batch_size:
            return tuple(data[i] for i in indices)
        return data
    return data


def select_batch_indices(batch, indices):
    selected_batch = []
    for view in batch:
        batch_size = None
        for value in view.values():
            if torch.is_tensor(value) and value.ndim > 0:
                batch_size = value.shape[0]
                break
        selected_view = {}
        for key, value in view.items():
            selected_view[key] = select_items(value, indices, batch_size)
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


def resolve_remote_control_modes(args):
    modes = cfg_get(args, "remote_control_modes", ["same"])
    if modes is None:
        return []
    if isinstance(modes, str):
        modes = modes.strip()
        if not modes or modes.lower() == "none":
            return []
        if modes.startswith("[") and modes.endswith("]"):
            inner = modes[1:-1].strip()
            modes = [] if not inner else [mode.strip().strip("'\"") for mode in inner.split(",")]
        else:
            modes = [mode.strip() for mode in modes.split(",")]

    valid_modes = {"same", "blank", "shuffled"}
    resolved_modes = []
    for mode in modes:
        mode = str(mode).strip()
        if not mode:
            continue
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported remote control mode '{mode}'. Supported: {sorted(valid_modes)}"
            )
        if mode not in resolved_modes:
            resolved_modes.append(mode)
    return resolved_modes


def make_remote_view(remote_image, args, batch_size):
    return {
        "img": remote_image,
        "data_norm_type": [args.model.data_norm_type] * batch_size,
        "instance": "remote",
    }


def build_blank_remote_image(remote_image, blank_value):
    return torch.full_like(remote_image, fill_value=float(blank_value))


def choose_shuffled_remote_sample(scene_name, remote_scene_names, remote_samples_by_scene):
    if len(remote_scene_names) <= 1:
        return scene_name, remote_samples_by_scene[scene_name]

    scene_idx = remote_scene_names.index(scene_name) if scene_name in remote_scene_names else -1
    for offset in range(1, len(remote_scene_names) + 1):
        candidate_scene = remote_scene_names[(scene_idx + offset) % len(remote_scene_names)]
        if candidate_scene != scene_name:
            return candidate_scene, remote_samples_by_scene[candidate_scene]
    return scene_name, remote_samples_by_scene[scene_name]


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
    remote_scene_names = sorted(remote_samples_by_scene.keys())
    remote_control_modes = resolve_remote_control_modes(args)
    blank_remote_value = float(cfg_get(args, "blank_remote_value", 0.5))

    model = initialize_benchmark_model(args, device)
    model.to(device)

    aerial_per_scene = {}
    rs_per_scene = {}
    joint_per_scene = {}
    improvement_aerial = {}
    improvement_rs = {}
    remote_control_per_scene = {mode: {} for mode in remote_control_modes}
    remote_control_improvement_aerial = {mode: {} for mode in remote_control_modes}
    remote_control_sources = {}

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
        remote_view = make_remote_view(remote_image, args, len(remote_samples))

        control_remote_views = {"same": remote_view}
        control_remote_source_scenes = {
            scene_name: {"same": scene_name} for scene_name in scene_names
        }
        if "blank" in remote_control_modes:
            blank_remote_image = build_blank_remote_image(remote_image, blank_remote_value)
            control_remote_views["blank"] = make_remote_view(
                blank_remote_image, args, len(remote_samples)
            )
            for scene_name in scene_names:
                control_remote_source_scenes[scene_name]["blank"] = "blank"
        if "shuffled" in remote_control_modes:
            shuffled_samples = []
            for scene_name in scene_names:
                shuffled_scene, shuffled_sample = choose_shuffled_remote_sample(
                    scene_name, remote_scene_names, remote_samples_by_scene
                )
                shuffled_samples.append(shuffled_sample)
                control_remote_source_scenes[scene_name]["shuffled"] = shuffled_scene
            shuffled_remote_image = torch.stack(
                [sample["remote_image"] for sample in shuffled_samples], dim=0
            ).to(device, non_blocking=True)
            control_remote_views["shuffled"] = make_remote_view(
                shuffled_remote_image, args, len(shuffled_samples)
            )

        with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
            aerial_preds = model(batch)
            rs_preds = model([remote_view])
            joint_preds = model(batch + [remote_view])
            control_joint_preds = {"same": joint_preds}
            for control_mode in remote_control_modes:
                if control_mode == "same":
                    continue
                control_joint_preds[control_mode] = model(
                    batch + [control_remote_views[control_mode]]
                )

        aerial_metrics_by_scene = compute_aerial_scene_metrics(batch, aerial_preds)
        joint_aerial_metrics_by_scene = compute_aerial_scene_metrics(
            batch, joint_preds[: len(batch)]
        )
        control_aerial_metrics_by_scene = {"same": joint_aerial_metrics_by_scene}
        for control_mode, control_preds in control_joint_preds.items():
            if control_mode == "same":
                continue
            control_aerial_metrics_by_scene[control_mode] = compute_aerial_scene_metrics(
                batch, control_preds[: len(batch)]
            )
        rs_supports_metric_outputs = model_supports_metric_outputs(rs_preds)
        joint_supports_metric_outputs = model_supports_metric_outputs(joint_preds)
        use_aux_point_residual_metric = bool(
            cfg_get(args, "remote_point_metric_use_aux_point_residual", False)
        )
        aux_point_residual_norm_mode = str(
            cfg_get(args, "remote_point_metric_aux_point_residual_norm_mode", "avg_dis")
        )
        aux_point_residual_apply_z = bool(
            cfg_get(args, "remote_point_metric_aux_point_residual_apply_z", False)
        )

        for sample_idx, scene in enumerate(scene_names):
            remote_sample = remote_samples[sample_idx]
            aerial_metrics = aerial_metrics_by_scene[scene]
            aerial_per_scene[scene] = aerial_metrics

            gt_height = remote_sample["remote_height_map"]
            gt_pointmap = remote_sample["remote_pointmap"]
            valid_mask = remote_sample["remote_valid_mask"].astype(bool)

            if use_aux_point_residual_metric:
                rs_pts_tensor = apply_aux_point_residual_to_remote_pts(
                    rs_preds[0],
                    sample_idx,
                    norm_mode=aux_point_residual_norm_mode,
                    apply_z=aux_point_residual_apply_z,
                )
            else:
                rs_pts_tensor = rs_preds[0]["pts3d"][sample_idx]
            rs_pts = rs_pts_tensor.detach().cpu().numpy()
            rs_metrics = compute_remote_height_metrics_affine(
                gt_height,
                rs_pts,
                valid_mask,
            )
            rs_metrics.update(
                compute_remote_pointmap_metrics(
                    gt_pointmap,
                    rs_pts,
                    valid_mask,
                )
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
            if use_aux_point_residual_metric:
                joint_rs_pts_tensor = apply_aux_point_residual_to_remote_pts(
                    joint_preds[len(batch)],
                    sample_idx,
                    norm_mode=aux_point_residual_norm_mode,
                    apply_z=aux_point_residual_apply_z,
                )
            else:
                joint_rs_pts_tensor = joint_preds[len(batch)]["pts3d"][sample_idx]
            joint_rs_pts = joint_rs_pts_tensor.detach().cpu().numpy()
            joint_rs_metrics = compute_remote_height_metrics_affine(
                gt_height,
                joint_rs_pts,
                valid_mask,
            )
            joint_rs_metrics.update(
                compute_remote_pointmap_metrics(
                    gt_pointmap,
                    joint_rs_pts,
                    valid_mask,
                )
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
                "joint_global_pointmaps_abs_rel": compute_joint_global_pointmaps_abs_rel(
                    batch=single_batch,
                    joint_preds=single_joint_preds,
                    remote_sample=remote_sample,
                ),
            }
            improvement_aerial[scene] = diff_metric_dict(joint_aerial_metrics, aerial_metrics)
            improvement_rs[scene] = diff_metric_dict(joint_rs_metrics, rs_metrics)
            remote_control_sources[scene] = control_remote_source_scenes[scene]
            for control_mode in remote_control_modes:
                control_aerial_metrics = control_aerial_metrics_by_scene[control_mode][scene]
                remote_control_per_scene[control_mode][scene] = control_aerial_metrics
                remote_control_improvement_aerial[control_mode][scene] = diff_metric_dict(
                    control_aerial_metrics, aerial_metrics
                )

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
            "remote_controls": {
                mode: {
                    "remote_source_scene": remote_control_sources.get(scene, {}).get(mode),
                    "joint_aerial": remote_control_per_scene[mode][scene],
                    "aerial_vs_aerial_only": remote_control_improvement_aerial[mode][scene],
                }
                for mode in remote_control_modes
                if scene in remote_control_per_scene[mode]
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
                "joint_global_pointmaps_abs_rel",
            ],
            "remote_control_modes": remote_control_modes,
            "blank_remote_value": blank_remote_value,
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
        "remote_controls": {
            "joint_aerial": {
                mode: {
                    "per_scene": {
                        scene: remote_control_per_scene[mode][scene]
                        for scene in paired_scenes
                        if scene in remote_control_per_scene[mode]
                    },
                    "average": aggregate_scene_metrics(
                        {
                            scene: remote_control_per_scene[mode][scene]
                            for scene in paired_scenes
                            if scene in remote_control_per_scene[mode]
                        }
                    ),
                }
                for mode in remote_control_modes
            },
            "aerial_vs_aerial_only": {
                mode: {
                    "per_scene": {
                        scene: remote_control_improvement_aerial[mode][scene]
                        for scene in paired_scenes
                        if scene in remote_control_improvement_aerial[mode]
                    },
                    "average": aggregate_scene_metrics(
                        {
                            scene: remote_control_improvement_aerial[mode][scene]
                            for scene in paired_scenes
                            if scene in remote_control_improvement_aerial[mode]
                        }
                    ),
                }
                for mode in remote_control_modes
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
    if remote_control_modes:
        print("Remote-control joint aerial averages:")
        for control_mode, control_result in result["remote_controls"]["joint_aerial"].items():
            print(f"[{control_mode}]")
            for metric_name, metric_value in control_result["average"].items():
                print(f"{metric_name}: {metric_value}")
        print("Remote-control improvement over aerial-only:")
        for control_mode, control_result in result["remote_controls"]["aerial_vs_aerial_only"].items():
            print(f"[{control_mode}]")
            for metric_name, metric_value in control_result["average"].items():
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
