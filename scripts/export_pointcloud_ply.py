#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Export a unified world-space point cloud from an image folder.

Supported benchmark models from bash_scripts/benchmark/rs_guided_dense_mv:
- pi3
- pi3_modality_embedding
- pi3_modality_embedding_remote_head
- vggt
- vggt_omega
- da3
- mapanything
- mapanything_rs_joint

Include:

pi3：
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/pi3 \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_base/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_base \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_modality_embedding/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_modality_embedding \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_freeze_shared/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_freeze_shared \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding_remote_head \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_modality_embedding_remote_head/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_modality_embedding_remote_head \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_zero_covis/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_zero_covis \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_low_covis/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/p3_pi3_low_covis \

mapanything:
python scripts/export_pointcloud_ply.py \
    --model mapanything \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/mapanything

mapanything_rs_joint:
# P4 MapAnything RS-joint checkpoints are supported for export. Use
# --model mapanything_rs_joint and pass the trained checkpoint. If the input
# folder contains a satellite / map image, mark it with --remote_view_names
# or --remote_view_indices so that it is routed through the remote direct
# pointmap head. Unmarked views use the ordinary MapAnything aerial branch.
# Filename metadata used for --remote_view_names is stripped before calling
# MapAnything.infer(), whose input validator only accepts model-facing keys.
python scripts/export_pointcloud_ply.py \
    --model mapanything_rs_joint \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/mapanything_p4_rs_joint \
    --remote_view_names zimage.png

# Baseline comparison with the original MapAnything checkpoint.
python scripts/export_pointcloud_ply.py \
    --model mapanything \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/mapanything_base


vggt:
# Original VGGT, matching the wrapper path used by the benchmark. This detects
# /outputs/checkpoints/vggt/model.pt as a raw VGGT state_dict and loads it via
# VGGTWrapper.model.load_state_dict(...), not as a MapAnything training ckpt.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/vggt/model.pt \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt \
&& \
# p5b default mixed export: ordinary views use camera+depth, remote uses point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5b_mixed \
    --vggt_joint_remote_export \
    --vggt_export_mode mixed \
    --remote_view_names image.png \
&& \
# p5b diagnostic: force every view through camera+depth.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5b_depth_all \
    --vggt_joint_remote_export \
    --vggt_export_mode depth_all \
    --remote_view_names image.png \
&& \
# p5b diagnostic: force every view through point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5b_point_all \
    --vggt_joint_remote_export \
    --vggt_export_mode point_all \
    --remote_view_names image.png \
&& \
# p5c
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5c_mixed \
    --vggt_joint_remote_export \
    --vggt_export_mode mixed \
    --config_overrides machine=aws model=vggt model.model_config.use_view_type_bias=true \
    --remote_view_names image.png \
&& \
# p5d remote-private point head + consistency checkpoint.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5d_vggt_remote_point_head_consistency/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5d_mixed \
    --vggt_joint_remote_export \
    --vggt_use_remote_private_point_head \
    --vggt_export_mode mixed \
    --remote_view_names image.png \
&& \
# p5e default mixed export: ordinary views use camera+depth, remote uses point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5e_mixed \
    --vggt_joint_remote_export \
    --vggt_ordinary_output_head depth \
    --vggt_remote_output_head point \
    --vggt_use_remote_private_point_head \
    --remote_view_names image.png

# p5f-lite: early view-type embedding + remote-to-aerial gated residual.
# If --export_remote_control_modes is set, one PLY is written per mode with
# suffixes such as *_same.ply and *_blank.ply. shuffled also requires
# --shuffled_remote_image_path.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5f_vggt_lite_early_bias_gated_residual/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p5f_lite_mixed \
    --vggt_p5f_lite_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p6a conditional remote adapter: official raw VGGT base + split late cross-attn.
# P6A uses remote as a conditioning input. By default this exports ordinary-view
# points only, because protected split heads predict remote points in a separate
# split frame. Add --include_remote_points only for debugging that remote branch.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6a_vggt_raw_base_conditional_remote_adapter/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p6a_raw_ordinary \
    --vggt_p6a_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p6b joint remote alignment: ordinary views use camera+depth, remote views use
# the trained remote point path. Private-head and shared-head checkpoints are
# auto-detected from the checkpoint path; use --vggt_p6b_export for clarity.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p6b_private_mixed \
    --vggt_p6b_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 projection-aux: split late fusion plus remote projection auxiliary heads.
# Default export writes ordinary-view reconstruction under remote conditioning;
# add --include_remote_points only to inspect the remote branch itself.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_projection_aux_split_late_fusion/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p7_projection_aux_ordinary \
    --vggt_p7_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 remote-head projection-aux: p5d-style separate remote point head plus
# projection auxiliary multitask learning, without split/late fusion.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p7_remote_head_projection_aux_mixed \
    --vggt_p7_remote_head_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

vggt_omega:
# Fine-tuned VGGT-Omega Crossview checkpoint. VGGT-Omega uses patch_size=16, so
# use resolution_set=512 or another 16-aligned fixed_size.
python scripts/export_pointcloud_ply.py \
    --model vggt_omega \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_omega_finetuned \
    --resolution_set 512 \
    --remote_view_names image.png \
&& \
# Raw released VGGT-Omega checkpoint before Crossview fine-tuning.
python scripts/export_pointcloud_ply.py \
    --model vggt_omega \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --image_folder /root/autodl-tmp/test/scence/125 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_omega_raw \
    --resolution_set 512
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
from mapanything.utils.image import heif_support_enabled, load_images

DEFAULT_MODEL = "pi3"
DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_MAPANYTHING_HF_MODEL = "facebook/map-anything"
SUPPORTED_MODELS = [
    "pi3",
    "pi3_modality_embedding",
    "pi3_modality_embedding_remote_head",
    "vggt",
    "vggt_omega",
    "da3",
    "mapanything",
    "mapanything_rs_joint",
]
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
IDENTITY_MODELS = {
    "anycalib",
    "moge",
    "pi3",
    "pi3_modality_embedding",
    "pi3_modality_embedding_remote_head",
    "pi3x",
    "vggt",
    "vggt_omega",
}
CLASH_ENV = {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7891",
}
REMOTE_INSTANCE_VALUE = "remote"


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
        choices=SUPPORTED_MODELS,
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
        "--vggt_joint_remote_export",
        action="store_true",
        default=False,
        help=(
            "Special handling for VGGT p5b/p5c RS-joint checkpoints: disable "
            "wrapper-side pretrained/custom init, enable remote point-head routing, "
            "and enable remote-view tagging for mixed export."
        ),
    )
    parser.add_argument(
        "--vggt_export_mode",
        type=str,
        default=None,
        choices=["mixed", "depth_all", "point_all", "ordinary_point_remote_depth"],
        help=(
            "Convenience VGGT output-head mode. mixed=ordinary depth and remote "
            "point; depth_all=all camera+depth; point_all=all point_head; "
            "ordinary_point_remote_depth swaps the mixed assignment."
        ),
    )
    parser.add_argument(
        "--vggt_ordinary_output_head",
        type=str,
        default=None,
        choices=["depth", "point"],
        help="Explicit output head for non-remote VGGT views.",
    )
    parser.add_argument(
        "--vggt_remote_output_head",
        type=str,
        default=None,
        choices=["auto", "depth", "point"],
        help="Explicit output head for remote VGGT views.",
    )
    parser.add_argument(
        "--vggt_use_remote_private_point_head",
        action="store_true",
        default=False,
        help=(
            "Enable VGGT remote private point_head when exporting p5d checkpoints."
        ),
    )
    parser.add_argument(
        "--vggt_p5f_lite_export",
        action="store_true",
        default=False,
        help=(
            "Enable the p5f-lite VGGT export preset: mixed ordinary/remote heads, "
            "remote private point head, pre-aggregator view-type embedding, and "
            "remote-to-aerial gated residual. This is also auto-enabled when "
            "checkpoint_path contains p5f_vggt_lite."
        ),
    )
    parser.add_argument(
        "--vggt_p6a_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P6A VGGT export preset: mixed ordinary/remote heads, "
            "remote private point head, split remote aggregator, late remote-to-aerial "
            "cross-attention, and protected ordinary heads. This is also auto-enabled "
            "when checkpoint_path contains p6a_vggt."
        ),
    )
    parser.add_argument(
        "--vggt_p6b_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P6B VGGT export preset: mixed ordinary/remote heads and "
            "P6B-specific remote point routing. Private-head, shared-head, and "
            "viewtype variants are auto-detected from checkpoint_path when possible. "
            "This is also auto-enabled when checkpoint_path contains p6b_vggt."
        ),
    )
    parser.add_argument(
        "--vggt_p7_projection_aux_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 VGGT projection-aux export preset: mixed ordinary/remote "
            "heads, split remote aggregator, late remote-to-aerial fusion, protected "
            "ordinary heads, and remote projection auxiliary heads. This is also "
            "auto-enabled when checkpoint_path contains p7_vggt_projection_aux or "
            "p7_projection_aux."
        ),
    )
    parser.add_argument(
        "--vggt_p7_remote_head_projection_aux_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 remote-head projection-aux export preset: mixed "
            "ordinary/remote heads, remote private point head, and remote projection "
            "auxiliary heads, without split aggregator or late fusion. This is also "
            "auto-enabled when checkpoint_path contains "
            "p7_vggt_remote_head_projection_aux or p7_remote_head_projection_aux."
        ),
    )
    parser.add_argument(
        "--vggt_projection_aux_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for P7 remote projection auxiliary heads.",
    )
    parser.add_argument(
        "--vggt_projection_aux_detach_pointmap",
        action="store_true",
        default=False,
        help="Detach the pointmap input before the P7 projection auxiliary heads.",
    )
    parser.add_argument(
        "--vggt_projection_aux_use_rgb",
        action="store_true",
        default=False,
        help="Condition the P7 projection auxiliary pixel head on remote RGB plus pointmap.",
    )
    parser.add_argument(
        "--vggt_projection_aux_use_coord",
        action="store_true",
        default=False,
        help="Condition the P7 projection auxiliary pixel head on normalized image coordinates.",
    )
    parser.add_argument(
        "--vggt_projection_aux_positive_slope",
        action="store_true",
        default=False,
        help="Constrain the P7 projection auxiliary global slope prediction to be positive.",
    )
    parser.add_argument(
        "--vggt_projection_aux_slope_init",
        type=float,
        default=0.1,
        help="Initial positive global slope for the P7 projection auxiliary head.",
    )
    parser.add_argument(
        "--vggt_projection_aux_num_blocks",
        type=int,
        default=0,
        help="Number of residual conv blocks in the P7 projection auxiliary pixel head.",
    )
    parser.add_argument(
        "--include_remote_points",
        action="store_true",
        default=False,
        help=(
            "Also write marked remote-view points to the PLY. For P6A/P7 split-protected "
            "exports, remote points are skipped by default because they are predicted "
            "in a separate split frame and are not directly aligned with ordinary points."
        ),
    )
    parser.add_argument(
        "--vggt_late_fusion_type",
        type=str,
        default="cross_attention",
        choices=["none", "film", "cross_attention"],
        help="Late remote-to-aerial fusion type used by P5h/P6A/P7 VGGT exports.",
    )
    parser.add_argument(
        "--vggt_late_gate_init",
        type=float,
        default=1e-3,
        help="Late fusion gate init used to build P6A export wrapper before loading checkpoint.",
    )
    parser.add_argument(
        "--vggt_max_remote_tokens",
        type=int,
        default=256,
        help="Maximum remote tokens for VGGT late cross-attention exports.",
    )
    parser.add_argument(
        "--vggt_cross_attention_heads",
        type=int,
        default=8,
        help="Number of heads for VGGT late remote-to-aerial cross-attention exports.",
    )
    parser.add_argument(
        "--force_remote_instance",
        action="store_true",
        default=False,
        help=(
            "Force every loaded view to use instance='remote'. Useful when exporting "
            "from RS-joint checkpoints that route remote views through a dedicated "
            "point head."
        ),
    )
    parser.add_argument(
        "--remote_view_indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "0-based indices of input images that should be treated as remote views "
            "for RS-joint export. Unspecified views remain ordinary views."
        ),
    )
    parser.add_argument(
        "--remote_view_names",
        nargs="*",
        default=None,
        help=(
            "Basenames of input images that should be treated as remote views for "
            "RS-joint export."
        ),
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
    parser.add_argument(
        "--export_remote_control_modes",
        nargs="*",
        choices=["same", "blank", "shuffled"],
        default=None,
        help=(
            "Optional remote-control visualization modes. same uses the marked "
            "remote image, blank replaces marked remote views with a constant image, "
            "and shuffled replaces them with --shuffled_remote_image_path. When set, "
            "one PLY is exported per mode."
        ),
    )
    parser.add_argument(
        "--blank_remote_value",
        type=float,
        default=0.5,
        help="Pixel value used for blank remote-control exports after identity conversion.",
    )
    parser.add_argument(
        "--shuffled_remote_image_path",
        type=str,
        default=None,
        help="Image path used to replace marked remote views in shuffled control exports.",
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


def is_raw_vggt_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    return checkpoint_path.name == "model.pt" and "checkpoints/vggt" in str(
        checkpoint_path
    )


def is_raw_vggt_omega_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt_omega" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    return checkpoint_path.name in {"vggt_omega_1b_512.pt", "model.pt"} and "checkpoints/vggt_omega" in str(
        checkpoint_path
    )


def is_p5f_lite_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p5f_vggt_lite" in checkpoint_path


def is_p6a_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p6a_vggt" in checkpoint_path


def is_p6b_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p6b_vggt" in checkpoint_path


def is_p7_split_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p7_vggt_projection_aux" in checkpoint_path or "p7_projection_aux" in checkpoint_path


def is_p7_remote_head_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return (
        "p7_vggt_remote_head_projection_aux" in checkpoint_path
        or "p7_remote_head_projection_aux" in checkpoint_path
    )


def is_p7_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    return is_p7_split_projection_aux_checkpoint(args) or is_p7_remote_head_projection_aux_checkpoint(args)


def is_p6b_shared_head_checkpoint(args: argparse.Namespace) -> bool:
    if not is_p6b_checkpoint(args):
        return False
    return "shared_head" in str(args.checkpoint_path).lower()


def is_p6b_viewtype_checkpoint(args: argparse.Namespace) -> bool:
    if not is_p6b_checkpoint(args):
        return False
    return "viewtype" in str(args.checkpoint_path).lower()


def use_p5f_lite_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p5f_lite_export or is_p5f_lite_checkpoint(args)
    )


def use_p6a_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p6a_export or is_p6a_checkpoint(args)
    )


def use_p6b_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p6b_export or is_p6b_checkpoint(args)
    )


def use_p7_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p7_projection_aux_export or is_p7_split_projection_aux_checkpoint(args)
    )


def use_p7_remote_head_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p7_remote_head_projection_aux_export
        or is_p7_remote_head_projection_aux_checkpoint(args)
    )


def use_any_p7_projection_aux_export(args: argparse.Namespace) -> bool:
    return use_p7_projection_aux_export(args) or use_p7_remote_head_projection_aux_export(args)


def resolve_vggt_late_fusion_type(args: argparse.Namespace) -> str:
    checkpoint_path = str(args.checkpoint_path or "").lower()
    if use_p7_projection_aux_export(args):
        if "no_fusion" in checkpoint_path:
            return "none"
        if "film" in checkpoint_path:
            return "film"
        if "crossattn" in checkpoint_path or "cross_attention" in checkpoint_path:
            return "cross_attention"
    return args.vggt_late_fusion_type


def use_vggt_remote_private_point_head(args: argparse.Namespace) -> bool:
    if args.model != "vggt":
        return False
    if args.vggt_use_remote_private_point_head:
        return True
    if use_p5f_lite_export(args) or use_p6a_export(args) or use_any_p7_projection_aux_export(args):
        return True
    if use_p6b_export(args):
        return not is_p6b_shared_head_checkpoint(args)
    return False


def resolve_vggt_output_heads(args: argparse.Namespace):
    if args.model != "vggt":
        return None, None

    ordinary_head = args.vggt_ordinary_output_head
    remote_head = args.vggt_remote_output_head

    if args.vggt_export_mode == "mixed" or use_p5f_lite_export(args) or use_p6a_export(args) or use_p6b_export(args) or use_any_p7_projection_aux_export(args):
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "point"
    elif args.vggt_export_mode == "depth_all":
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "depth"
    elif args.vggt_export_mode == "point_all":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "point"
    elif args.vggt_export_mode == "ordinary_point_remote_depth":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "depth"

    return ordinary_head, remote_head


def resolve_config_overrides(args: argparse.Namespace):
    if args.config_overrides is not None:
        overrides = list(args.config_overrides)
    else:
        overrides = list(DEFAULT_CONFIG_OVERRIDES[args.model])

    if is_raw_vggt_checkpoint(args):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=true",
                f"model.model_config.custom_ckpt_path={args.checkpoint_path}",
            ]
        )

    if is_raw_vggt_omega_checkpoint(args):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=true",
                f"model.model_config.custom_ckpt_path={args.checkpoint_path}",
            ]
        )

    use_vggt_joint_remote_export = (
        args.vggt_joint_remote_export or use_p5f_lite_export(args) or use_p6a_export(args) or use_p6b_export(args) or use_any_p7_projection_aux_export(args)
    )
    if use_vggt_joint_remote_export:
        if args.model != "vggt":
            raise ValueError("VGGT joint remote export presets are only supported with --model vggt")
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=false",
                "model.model_config.use_point_head_for_remote=true",
            ]
        )

    if use_p5f_lite_export(args):
        overrides.extend(
            [
                "model.model_config.use_pre_aggregator_view_type_bias=true",
                "model.model_config.use_remote_to_aerial_gated_residual=true",
                "model.model_config.remote_to_aerial_residual_hidden_scale=0.25",
                "model.model_config.remote_to_aerial_gate_init=0.0",
            ]
        )

    if use_p6a_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=true",
                "model.model_config.use_split_remote_aggregator=true",
                f"model.model_config.remote_to_aerial_late_fusion_type={args.vggt_late_fusion_type}",
                "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                f"model.model_config.remote_to_aerial_late_fusion_gate_init={args.vggt_late_gate_init}",
                f"model.model_config.remote_to_aerial_cross_attention_heads={args.vggt_cross_attention_heads}",
                f"model.model_config.remote_to_aerial_max_remote_tokens={args.vggt_max_remote_tokens}",
                "model.model_config.protect_ordinary_heads_from_remote=true",
            ]
        )

    if use_p7_projection_aux_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=true",
                "model.model_config.use_split_remote_aggregator=true",
                f"model.model_config.remote_to_aerial_late_fusion_type={resolve_vggt_late_fusion_type(args)}",
                "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                f"model.model_config.remote_to_aerial_late_fusion_gate_init={args.vggt_late_gate_init}",
                f"model.model_config.remote_to_aerial_cross_attention_heads={args.vggt_cross_attention_heads}",
                f"model.model_config.remote_to_aerial_max_remote_tokens={args.vggt_max_remote_tokens}",
                "model.model_config.protect_ordinary_heads_from_remote=true",
                "model.model_config.use_remote_projection_aux_head=true",
                f"model.model_config.remote_projection_aux_hidden_dim={args.vggt_projection_aux_hidden_dim}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                f"model.model_config.remote_projection_aux_use_rgb={str(args.vggt_projection_aux_use_rgb).lower()}",
                f"model.model_config.remote_projection_aux_use_coord={str(args.vggt_projection_aux_use_coord).lower()}",
                f"model.model_config.remote_projection_aux_positive_slope={str(args.vggt_projection_aux_positive_slope).lower()}",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                f"model.model_config.remote_projection_aux_num_blocks={args.vggt_projection_aux_num_blocks}",
            ]
        )

    if use_p7_remote_head_projection_aux_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=false",
                "model.model_config.use_pre_aggregator_view_type_bias=false",
                "model.model_config.use_remote_to_aerial_gated_residual=false",
                "model.model_config.use_split_remote_aggregator=false",
                "model.model_config.output_point_head_for_consistency=false",
                "model.model_config.use_remote_projection_aux_head=true",
                f"model.model_config.remote_projection_aux_hidden_dim={args.vggt_projection_aux_hidden_dim}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                f"model.model_config.remote_projection_aux_use_rgb={str(args.vggt_projection_aux_use_rgb).lower()}",
                f"model.model_config.remote_projection_aux_use_coord={str(args.vggt_projection_aux_use_coord).lower()}",
                f"model.model_config.remote_projection_aux_positive_slope={str(args.vggt_projection_aux_positive_slope).lower()}",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                f"model.model_config.remote_projection_aux_num_blocks={args.vggt_projection_aux_num_blocks}",
            ]
        )

    if use_p6b_export(args) and is_p6b_viewtype_checkpoint(args):
        overrides.append("model.model_config.use_view_type_bias=true")

    ordinary_head, remote_head = resolve_vggt_output_heads(args)
    if ordinary_head is not None:
        overrides.append(f"model.model_config.ordinary_output_head={ordinary_head}")
    if remote_head is not None:
        overrides.append(f"model.model_config.remote_output_head={remote_head}")
    if use_vggt_remote_private_point_head(args):
        overrides.append("model.model_config.use_remote_private_point_head=true")
        if not use_p7_remote_head_projection_aux_export(args):
            overrides.append("model.model_config.output_point_head_for_consistency=true")

    return overrides


def resolve_effective_model_name(args: argparse.Namespace) -> str:
    if args.model != "pi3" or not args.checkpoint_path:
        return args.model

    checkpoint_path_lower = str(args.checkpoint_path).lower()
    if "pi3_modality_embedding_remote_head" in checkpoint_path_lower:
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
    return args.model


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


def build_local_config(
    args: argparse.Namespace,
    config_overrides,
    effective_model_name: str,
) -> dict:
    local_config = {
        "path": args.config_path,
        "checkpoint_path": args.checkpoint_path,
        "config_overrides": config_overrides,
        "strict": args.strict,
        "model_str": args.model_str or effective_model_name,
    }
    if args.config_json_path is not None:
        local_config["config_json_path"] = args.config_json_path
    return local_config


def initialize_model(
    args: argparse.Namespace,
    device: str,
    config_overrides,
    effective_model_name: str,
):
    maybe_enable_clash_proxy(args.enable_clash_proxy)
    maybe_prepare_da3_pythonpath(effective_model_name)

    if args.checkpoint_path:
        if is_raw_vggt_checkpoint(args):
            print(
                "Detected raw VGGT checkpoint; loading it through "
                "model.model_config.custom_ckpt_path before the compatibility "
                "local-checkpoint load."
            )
        local_config = build_local_config(args, config_overrides, effective_model_name)
        print(f"Initializing model from local config: {local_config}")
        model = initialize_mapanything_local(local_config, device)
        print("Successfully loaded local checkpoint")
        return model

    if effective_model_name == "mapanything":
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

    print(
        f"Initializing model '{effective_model_name}' from default wrapper weights"
    )
    model = init_model_from_config(
        effective_model_name, device=device, machine="aws"
    ).eval()
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


def list_loaded_image_names(image_folder: str, stride: int):
    supported_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_extensions += [".heic", ".heif"]
    supported_extensions = tuple(supported_extensions)

    names = []
    for idx, name in enumerate(sorted(os.listdir(image_folder))):
        if idx % stride != 0:
            continue
        if not name.lower().endswith(supported_extensions):
            continue
        names.append(name)
    return names


def annotate_view_source_names(views, image_folder: str, stride: int):
    source_names = list_loaded_image_names(image_folder, stride)
    if len(source_names) != len(views):
        print(
            "Warning: loaded image-name count does not match view count; "
            "remote name selection may be unreliable."
        )
    for idx, view in enumerate(views):
        if idx < len(source_names):
            view["source_name"] = source_names[idx]
    return views


def maybe_assign_remote_instances(views, args: argparse.Namespace):
    use_joint_remote_logic = (
        args.force_remote_instance
        or args.vggt_joint_remote_export
        or args.vggt_p5f_lite_export
        or args.vggt_p6a_export
        or args.vggt_p6b_export
        or args.vggt_p7_projection_aux_export
        or args.vggt_p7_remote_head_projection_aux_export
        or is_p5f_lite_checkpoint(args)
        or is_p6a_checkpoint(args)
        or is_p6b_checkpoint(args)
        or is_p7_projection_aux_checkpoint(args)
        or args.model == "mapanything_rs_joint"
        or bool(args.remote_view_indices)
        or bool(args.remote_view_names)
    )
    if not use_joint_remote_logic:
        return views

    remote_indices = set(args.remote_view_indices or [])
    remote_names = {name for name in (args.remote_view_names or [])}

    if args.force_remote_instance or (
        (args.vggt_joint_remote_export or use_p5f_lite_export(args) or use_p6a_export(args) or use_p6b_export(args) or use_any_p7_projection_aux_export(args))
        and not remote_indices
        and not remote_names
    ):
        remote_indices = set(range(len(views)))

    forced_views = []
    remote_assignments = []
    for idx, view in enumerate(views):
        forced_view = dict(view)
        source_name = forced_view.get("source_name")
        is_remote = idx in remote_indices or (
            source_name is not None and source_name in remote_names
        )
        if is_remote:
            forced_view["instance"] = [REMOTE_INSTANCE_VALUE]
            remote_assignments.append((idx, source_name or f"view_{idx}"))
        forced_views.append(forced_view)

    if remote_assignments:
        print("Assigned remote views:")
        for idx, source_name in remote_assignments:
            print(f"  - idx={idx} name={source_name}")
    else:
        print("No views were marked as remote; export will use ordinary view logic.")

    return forced_views


def is_remote_view(view) -> bool:
    instance = view.get("instance")
    if isinstance(instance, (list, tuple)) and len(instance) > 0:
        instance = instance[0]
    return instance == REMOTE_INSTANCE_VALUE


def get_remote_view_indices(views):
    return [idx for idx, view in enumerate(views) if is_remote_view(view)]


def get_output_head_name(pred):
    return pred.get("vggt_output_head", pred.get("vggt_omega_output_head", "default"))


def should_skip_remote_points(args: argparse.Namespace) -> bool:
    return (use_p6a_export(args) or use_p7_projection_aux_export(args)) and not args.include_remote_points


def copy_views(views):
    return [dict(view) for view in views]


def make_blank_remote_control_views(views, remote_indices, blank_value: float):
    control_views = copy_views(views)
    for idx in remote_indices:
        control_views[idx] = dict(control_views[idx])
        control_views[idx]["img"] = torch.full_like(
            control_views[idx]["img"], fill_value=float(blank_value)
        )
        control_views[idx]["source_name"] = f"blank::{control_views[idx].get('source_name', idx)}"
    return control_views


def load_shuffled_remote_view_like(args, remote_view, model_name: str):
    if args.shuffled_remote_image_path is None:
        raise ValueError(
            "--shuffled_remote_image_path is required when exporting shuffled remote controls"
        )
    if not Path(args.shuffled_remote_image_path).exists():
        raise FileNotFoundError(
            f"shuffled remote image not found: {args.shuffled_remote_image_path}"
        )

    _, _, height, width = remote_view["img"].shape
    loaded = load_images(
        [args.shuffled_remote_image_path],
        resize_mode="fixed_size",
        size=(width, height),
        resolution_set=args.resolution_set,
    )
    loaded = convert_views_to_identity_if_needed(loaded, model_name)
    replacement = loaded[0]
    replacement["source_name"] = Path(args.shuffled_remote_image_path).name
    return replacement


def make_shuffled_remote_control_views(views, remote_indices, args, model_name: str):
    control_views = copy_views(views)
    for idx in remote_indices:
        replacement = load_shuffled_remote_view_like(args, control_views[idx], model_name)
        control_view = dict(control_views[idx])
        control_view["img"] = replacement["img"]
        control_view["true_shape"] = replacement.get("true_shape", control_view.get("true_shape"))
        control_view["source_name"] = f"shuffled::{replacement.get('source_name', idx)}"
        control_views[idx] = control_view
    return control_views


def build_remote_control_view_variants(views, args: argparse.Namespace, model_name: str):
    modes = args.export_remote_control_modes
    if not modes:
        return [(None, views)]

    remote_indices = get_remote_view_indices(views)
    if not remote_indices:
        raise ValueError(
            "--export_remote_control_modes requires at least one marked remote view. "
            "Use --remote_view_names, --remote_view_indices, or --force_remote_instance."
        )

    variants = []
    for mode in modes:
        if mode == "same":
            variants.append((mode, views))
        elif mode == "blank":
            variants.append(
                (mode, make_blank_remote_control_views(views, remote_indices, args.blank_remote_value))
            )
        elif mode == "shuffled":
            variants.append(
                (mode, make_shuffled_remote_control_views(views, remote_indices, args, model_name))
            )
        else:
            raise ValueError(f"Unsupported remote control mode: {mode}")
    return variants


def strip_export_only_view_keys(views):
    stripped_views = []
    for view in views:
        stripped_view = dict(view)
        stripped_view.pop("source_name", None)
        stripped_views.append(stripped_view)
    return stripped_views


def run_model_inference(model, views, args: argparse.Namespace):
    views_for_model = strip_export_only_view_keys(views)
    if hasattr(model, "infer"):
        return model.infer(
            views_for_model,
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
    return model(move_views_to_device(views_for_model, model_device))


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
    skip_remote_points=False,
):
    all_points = []
    all_colors = []
    per_view_stats = []

    for view_idx, pred in enumerate(outputs):
        if skip_remote_points and view_idx < len(views) and is_remote_view(views[view_idx]):
            per_view_stats.append(
                {
                    "view_idx": view_idx,
                    "points": 0,
                    "head": get_output_head_name(pred),
                    "skipped": "remote_split_frame",
                }
            )
            continue

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
                "head": get_output_head_name(pred),
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


def resolve_variant_output_path(output_path_str: str, variant_name: str | None) -> Path:
    output_path = resolve_output_path(output_path_str)
    if variant_name is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_{variant_name}{output_path.suffix}")


def export_point_cloud_for_views(model, views, args: argparse.Namespace, output_path: Path, label: str | None):
    if label:
        print(f"Running inference for remote-control mode: {label}")
    else:
        print("Running inference...")
    start_time = time()
    with torch.inference_mode():
        outputs = run_model_inference(model, views, args)
    duration = time() - start_time
    print(f"Inference finished in {duration:.3f}s")

    skip_remote_points = should_skip_remote_points(args)
    if skip_remote_points:
        print(
            "P6A/P7 split/protected export: skipping remote-view points by default; "
            "remote views still condition ordinary-view predictions. Use "
            "--include_remote_points only for branch debugging."
        )
    print("Collecting unified world-space point cloud...")
    points, colors, per_view_stats = collect_world_space_point_cloud(
        outputs,
        views,
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        skip_remote_points=skip_remote_points,
    )
    for stat in per_view_stats:
        skipped = stat.get("skipped")
        if skipped:
            print(
                f"View {stat['view_idx']}: skipped ({skipped}, head={stat['head']})"
            )
        else:
            print(
                f"View {stat['view_idx']}: kept {stat['points']} points "
                f"(head={stat['head']})"
            )
    print(f"Total points before downsampling: {points.shape[0]}")

    if args.voxel_downsample:
        points, colors = voxel_downsample_point_cloud(
            points,
            colors,
            voxel_fraction=args.voxel_fraction,
            voxel_size=args.voxel_size,
        )
        print(f"Total points after downsampling: {points.shape[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path)
    print(f"Saved unified point cloud PLY to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.model == "vggt_omega" and args.resize_mode == "fixed_mapping" and args.resolution_set != 512:
        print(
            "VGGT-Omega uses patch_size=16; overriding fixed_mapping "
            f"resolution_set {args.resolution_set} -> 512."
        )
        args.resolution_set = 512
    effective_model_name = resolve_effective_model_name(args)
    if effective_model_name == "mapanything_rs_joint" and not args.checkpoint_path:
        raise ValueError(
            "--model mapanything_rs_joint requires --checkpoint_path because there "
            "is no default HuggingFace checkpoint for this local RS-joint variant."
        )
    if effective_model_name != args.model:
        print(
            f"Resolved export model '{args.model}' -> '{effective_model_name}' "
            "based on the checkpoint path"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    original_model_name = args.model
    args.model = effective_model_name
    config_overrides = resolve_config_overrides(args)
    model = initialize_model(args, device, config_overrides, effective_model_name)
    args.model = original_model_name

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
    views = annotate_view_source_names(views, args.image_folder, args.stride)

    model_name = getattr(model, "name", effective_model_name)
    views = convert_views_to_identity_if_needed(views, model_name)
    views = maybe_assign_remote_instances(views, args)

    variants = build_remote_control_view_variants(views, args, model_name)
    for variant_name, variant_views in variants:
        output_path = resolve_variant_output_path(args.output_path, variant_name)
        export_point_cloud_for_views(
            model=model,
            views=variant_views,
            args=args,
            output_path=output_path,
            label=variant_name,
        )


if __name__ == "__main__":
    main()
