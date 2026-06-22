#!/usr/bin/env python3
"""Run Crossview RS guided benchmarks for selected Crossview checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path(
    "/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview"
)
DEFAULT_OUT_ROOT = Path(
    "/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/"
    "rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8"
)
RUNNER = "bash_scripts/benchmark/rs_guided_dense_mv/run_crossview_finetuned_unified.sh"
RESULT_JSON = "rs_aerial_benchmark_results.json"


RES_518 = [
    "dataset.resolution_train=${dataset.resolution_options.518_1_00_ar}",
    "dataset.resolution_val=${dataset.resolution_options.518_1_00_ar}",
]
RES_512 = [
    "dataset.resolution_train=${dataset.resolution_options.512_1_00_ar}",
    "dataset.resolution_val=${dataset.resolution_options.512_1_00_ar}",
]


@dataclass(frozen=True)
class Job:
    label: str
    model_name: str
    ckpt: str
    args: list[str] = field(default_factory=list)

    @property
    def uses_checkpoint(self) -> bool:
        return self.ckpt not in {"", "none", "__none__"}

    @property
    def ckpt_path(self) -> Path:
        return TRAIN_ROOT / self.ckpt

    @property
    def ckpt_env(self) -> str:
        return str(self.ckpt_path) if self.uses_checkpoint else "none"


def label_from_checkpoint_path(checkpoint_path: Path) -> str:
    rel = checkpoint_path.relative_to(TRAIN_ROOT)
    stem = checkpoint_path.stem.replace("checkpoint-", "")
    return "_".join(rel.parts[:-1] + (stem,)).replace("-", "_")


def model_name_from_checkpoint_path(checkpoint_path: Path) -> str:
    rel = checkpoint_path.relative_to(TRAIN_ROOT)
    family = rel.parts[0]
    lower = str(rel).lower()
    if family == "pi3":
        if "remote_head" in lower or "projection_aux" in lower:
            return "pi3_modality_embedding_remote_head"
        if "modality_embedding" in lower or "freeze_shared" in lower:
            return "pi3_modality_embedding"
        return "pi3"
    if family == "vggt_omega":
        return "vggt_omega"
    return family


def args_from_checkpoint_path(checkpoint_path: Path) -> list[str]:
    model_name = model_name_from_checkpoint_path(checkpoint_path)
    if model_name == "vggt_omega":
        return RES_512
    return RES_518


def discover_checkpoint_jobs(
    registered_jobs: list[Job],
    exclude_patterns: list[str],
) -> list[Job]:
    registered_paths = {job.ckpt_path.resolve() for job in registered_jobs}
    labels = {job.label for job in registered_jobs}
    discovered = []
    for checkpoint_path in sorted(TRAIN_ROOT.glob("**/checkpoint-*.pth")):
        checkpoint_path = checkpoint_path.resolve()
        lower = str(checkpoint_path).lower()
        if checkpoint_path in registered_paths:
            continue
        if any(pattern and pattern in lower for pattern in exclude_patterns):
            continue
        rel = checkpoint_path.relative_to(TRAIN_ROOT)
        label = label_from_checkpoint_path(checkpoint_path)
        if label in labels:
            suffix = str(len(labels))
            label = f"{label}_{suffix}"
        labels.add(label)
        discovered.append(
            Job(
                label=label,
                model_name=model_name_from_checkpoint_path(checkpoint_path),
                ckpt=str(rel),
                args=args_from_checkpoint_path(checkpoint_path),
            )
        )
    return discovered


def cfg(model: str, *overrides: str, omega: bool = False) -> list[str]:
    return [
        f'config_overrides=["machine=aws","model={model}"]',
        *overrides,
        *(RES_512 if omega else RES_518),
    ]


P5B = cfg("vggt")
P5C = cfg("vggt", "++model.model_config.use_view_type_bias=true")
P5D = cfg(
    "vggt",
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
)
P5E = cfg(
    "vggt",
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    "++model.model_config.use_point_head_for_remote=true",
    "++model.model_config.use_view_type_bias=true",
    "++model.model_config.ordinary_output_head=depth",
    "++model.model_config.remote_output_head=point",
    "++model.model_config.use_remote_private_point_head=true",
    "++model.model_config.output_point_head_for_consistency=true",
)
P5F = cfg(
    "vggt",
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    "++model.model_config.use_pre_aggregator_view_type_bias=true",
    "++model.model_config.use_remote_to_aerial_gated_residual=true",
    "++model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
    "++model.model_config.remote_to_aerial_late_fusion_gate_init=0.0",
)


def p5g(fusion: str, protected: bool) -> list[str]:
    return cfg(
        "vggt",
        "vggt_use_remote_private_point_head=true",
        "vggt_joint_remote_export=true",
        "vggt_export_mode=mixed",
        "++model.model_config.use_split_remote_aggregator=true",
        f"++model.model_config.remote_to_aerial_late_fusion_type={fusion}",
        "++model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
        "++model.model_config.remote_to_aerial_late_fusion_gate_init=0.0",
        "++model.model_config.remote_to_aerial_cross_attention_heads=8",
        "++model.model_config.remote_to_aerial_max_remote_tokens=256",
        f"++model.model_config.protect_ordinary_from_remote={str(protected).lower()}",
    )


def p5h(fusion: str) -> list[str]:
    return cfg(
        "vggt",
        "vggt_use_remote_private_point_head=true",
        "vggt_joint_remote_export=true",
        "vggt_export_mode=mixed",
        "++model.model_config.use_view_type_bias=true",
        "++model.model_config.use_split_remote_aggregator=true",
        f"++model.model_config.remote_to_aerial_late_fusion_type={fusion}",
        "++model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
        "++model.model_config.remote_to_aerial_late_fusion_gate_init=0.0",
        "++model.model_config.remote_to_aerial_cross_attention_heads=8",
        "++model.model_config.remote_to_aerial_max_remote_tokens=256",
        "++model.model_config.protect_ordinary_heads_from_remote=true",
    )


P6A = cfg(
    "vggt",
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    "++model.model_config.use_view_type_bias=true",
    "++model.model_config.use_split_remote_aggregator=true",
    "++model.model_config.remote_to_aerial_late_fusion_type=cross_attention",
    "++model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
    "++model.model_config.remote_to_aerial_late_fusion_gate_init=1e-3",
    "++model.model_config.remote_to_aerial_cross_attention_heads=8",
    "++model.model_config.remote_to_aerial_max_remote_tokens=256",
    "++model.model_config.protect_ordinary_heads_from_remote=true",
)
P6B = cfg(
    "vggt",
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    "++model.model_config.use_split_remote_aggregator=false",
    "++model.model_config.protect_ordinary_heads_from_remote=false",
    "++model.model_config.use_view_type_bias=false",
    "++model.model_config.use_pre_aggregator_view_type_bias=false",
    "++model.model_config.use_remote_to_aerial_gated_residual=false",
    "++model.model_config.remote_to_aerial_late_fusion_type=none",
)
P7 = [
    *P6B,
    "++model.model_config.use_remote_projection_aux_head=true",
    "++model.model_config.remote_projection_aux_hidden_dim=64",
]
P7_P5B_SHARED_NORM_PROJECTION_AUX = [
    (
        'config_overrides=["machine=aws","model=vggt",'
        '"model.model_config.use_split_remote_aggregator=false",'
        '"model.model_config.protect_ordinary_heads_from_remote=false",'
        '"model.model_config.use_view_type_bias=false",'
        '"model.model_config.use_pre_aggregator_view_type_bias=false",'
        '"model.model_config.use_remote_to_aerial_gated_residual=false",'
        '"model.model_config.remote_to_aerial_late_fusion_type=none",'
        '"model.model_config.use_remote_projection_aux_head=true",'
        '"model.model_config.remote_projection_aux_hidden_dim=96",'
        '"model.model_config.remote_projection_aux_detach_pointmap=false",'
        '"model.model_config.remote_projection_aux_use_rgb=true",'
        '"model.model_config.remote_projection_aux_use_coord=true",'
        '"model.model_config.remote_projection_aux_image_stem_dim=32",'
        '"model.model_config.remote_projection_aux_positive_slope=true",'
        '"model.model_config.remote_projection_aux_slope_init=0.1",'
        '"model.model_config.remote_projection_aux_num_blocks=6"]'
    ),
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    *RES_518,
]
P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX = [
    (
        'config_overrides=["machine=aws","model=vggt",'
        '"model.model_config.use_split_remote_aggregator=false",'
        '"model.model_config.protect_ordinary_heads_from_remote=false",'
        '"model.model_config.use_view_type_bias=false",'
        '"model.model_config.use_pre_aggregator_view_type_bias=false",'
        '"model.model_config.use_remote_to_aerial_gated_residual=false",'
        '"model.model_config.remote_to_aerial_late_fusion_type=none",'
        '"model.model_config.use_remote_projection_aux_head=true",'
        '"model.model_config.remote_projection_aux_hidden_dim=96",'
        '"model.model_config.remote_projection_aux_source=tokens",'
        '"model.model_config.remote_projection_aux_detach_pointmap=false",'
        '"model.model_config.remote_projection_aux_use_rgb=true",'
        '"model.model_config.remote_projection_aux_use_coord=true",'
        '"model.model_config.remote_projection_aux_image_stem_dim=32",'
        '"model.model_config.remote_projection_aux_positive_slope=true",'
        '"model.model_config.remote_projection_aux_slope_init=0.1",'
        '"model.model_config.remote_projection_aux_num_blocks=6"]'
    ),
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    *RES_518,
]
P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX = [
    (
        'config_overrides=["machine=aws","model=vggt",'
        '"model.model_config.use_split_remote_aggregator=false",'
        '"model.model_config.protect_ordinary_heads_from_remote=false",'
        '"model.model_config.use_view_type_bias=false",'
        '"model.model_config.use_pre_aggregator_view_type_bias=false",'
        '"model.model_config.use_remote_to_aerial_gated_residual=false",'
        '"model.model_config.remote_to_aerial_late_fusion_type=none",'
        '"model.model_config.use_remote_private_point_head=true",'
        '"model.model_config.output_point_head_for_consistency=true",'
        '"model.model_config.use_remote_projection_aux_head=true",'
        '"model.model_config.remote_projection_aux_hidden_dim=96",'
        '"model.model_config.remote_projection_aux_source=tokens",'
        '"model.model_config.remote_projection_aux_detach_pointmap=false",'
        '"model.model_config.remote_projection_aux_use_rgb=true",'
        '"model.model_config.remote_projection_aux_use_coord=true",'
        '"model.model_config.remote_projection_aux_image_stem_dim=32",'
        '"model.model_config.remote_projection_aux_positive_slope=true",'
        '"model.model_config.remote_projection_aux_slope_init=0.1",'
        '"model.model_config.remote_projection_aux_num_blocks=6"]'
    ),
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    *RES_518,
]
P7_P5E_PRIVATE_VIEWTYPE_PROJECTION_AUX = [
    (
        'config_overrides=["machine=aws","model=vggt",'
        '"model.model_config.use_point_head_for_remote=true",'
        '"model.model_config.use_view_type_bias=true",'
        '"model.model_config.use_pre_aggregator_view_type_bias=false",'
        '"model.model_config.use_remote_to_aerial_gated_residual=false",'
        '"model.model_config.use_split_remote_aggregator=false",'
        '"model.model_config.protect_ordinary_heads_from_remote=false",'
        '"model.model_config.remote_to_aerial_late_fusion_type=none",'
        '"model.model_config.ordinary_output_head=depth",'
        '"model.model_config.remote_output_head=point",'
        '"model.model_config.use_remote_private_point_head=true",'
        '"model.model_config.output_point_head_for_consistency=false",'
        '"model.model_config.use_remote_projection_aux_head=true",'
        '"model.model_config.remote_projection_aux_hidden_dim=96",'
        '"model.model_config.remote_projection_aux_detach_pointmap=false",'
        '"model.model_config.remote_projection_aux_use_rgb=true",'
        '"model.model_config.remote_projection_aux_use_coord=true",'
        '"model.model_config.remote_projection_aux_image_stem_dim=32",'
        '"model.model_config.remote_projection_aux_positive_slope=true",'
        '"model.model_config.remote_projection_aux_slope_init=0.1",'
        '"model.model_config.remote_projection_aux_num_blocks=6"]'
    ),
    "vggt_use_remote_private_point_head=true",
    "vggt_joint_remote_export=true",
    "vggt_export_mode=mixed",
    *RES_518,
]


JOBS: list[Job] = [
    Job("pi3_raw_pretrained_image_input", "pi3", "none", RES_518),
    Job("vggt_raw_pretrained_image_input", "vggt", "none", P5B),
    Job("pi3_p3_base", "pi3", "pi3/p3_pi3_base/checkpoint-best.pth", RES_518),
    Job(
        "pi3_p3_freeze_shared",
        "pi3_modality_embedding",
        "pi3/p3_pi3_freeze_shared/checkpoint-best.pth",
        RES_518,
    ),
    Job(
        "pi3_p3_modality_embedding",
        "pi3_modality_embedding",
        "pi3/p3_pi3_modality_embedding/checkpoint-best.pth",
        RES_518,
    ),
    Job(
        "pi3_p3_modality_embedding_remote_head",
        "pi3_modality_embedding_remote_head",
        "pi3/p3_pi3_modality_embedding_remote_head/checkpoint-best.pth",
        RES_518,
    ),
    Job("pi3_p3_zero_covis", "pi3", "pi3/p3_pi3_zero_covis/checkpoint-best.pth", RES_518),
    Job("vggt_p5b_shared_norm", "vggt", "vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth", P5B),
    Job("vggt_p5b_shared_norm_2", "vggt", "vggt/p5b_vggt_joint_shared_all_shared_norm_2/checkpoint-best.pth", P5B),
    Job("vggt_p5c_viewtype", "vggt", "vggt/p5c_vggt_joint_shared_all_viewtype/checkpoint-best.pth", P5C),
    Job("vggt_p5d_remote_point_head_consistency", "vggt", "vggt/p5d_vggt_remote_point_head_consistency/checkpoint-best.pth", P5D),
    Job("vggt_p5e_remote_head_attention_viewtype", "vggt", "vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth", P5E),
    Job("vggt_p5f_lite_early_bias_gated_residual", "vggt", "vggt/p5f_vggt_lite_early_bias_gated_residual/checkpoint-final.pth", P5F),
    Job("vggt_p5g_crossattn_split_remote", "vggt", "vggt/p5g_vggt_crossattn_split_remote/checkpoint-best.pth", p5g("cross_attention", False)),
    Job("vggt_p5g_film_split_remote", "vggt", "vggt/p5g_vggt_film_split_remote/checkpoint-best.pth", p5g("film", False)),
    Job("vggt_p5g_no_fusion_fixedfreeze_protected", "vggt", "vggt/p5g_vggt_no_fusion_fixedfreeze_protected/checkpoint-best.pth", p5g("none", True)),
    Job("vggt_p5g_no_fusion_split_remote", "vggt", "vggt/p5g_vggt_no_fusion_split_remote/checkpoint-best.pth", p5g("none", False)),
    Job("vggt_p5h_crossattn_protected", "vggt", "vggt/p5h_vggt_p5e_base_crossattn_protected/checkpoint-best.pth", p5h("cross_attention")),
    Job("vggt_p5h_film_protected", "vggt", "vggt/p5h_vggt_p5e_base_film_protected/checkpoint-best.pth", p5h("film")),
    Job("vggt_p5h_film_unfreeze_viewtype_protected", "vggt", "vggt/p5h_vggt_p5e_base_film_unfreeze_viewtype_protected/checkpoint-best.pth", p5h("film")),
    Job("vggt_p6a_raw_base_conditional_remote_adapter", "vggt", "vggt/p6a_vggt_raw_base_conditional_remote_adapter/checkpoint-best.pth", P6A),
    Job("vggt_p6b_private_head_1", "vggt", "vggt/p6b_vggt_joint_remote_alignment_private_head_1/checkpoint-best.pth", P6B),
    Job("vggt_p6b_private_head_2", "vggt", "vggt/p6b_vggt_joint_remote_alignment_private_head_2/checkpoint-best.pth", P6B),
    Job("vggt_p6b_private_head_w03_bs5_static_remoteonly", "vggt", "vggt/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly/checkpoint-best.pth", P6B),
    Job("vggt_p7_remote_head_projection_aux_trunk", "vggt", "vggt/p7_vggt_remote_head_projection_aux_trunk/checkpoint-best.pth", P7),
    Job("vggt_p7_p5b_shared_norm_projection_aux_full_2city", "vggt", "vggt/p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu/checkpoint-final.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_best", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun/checkpoint-best.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_final", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun/checkpoint-final.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_h0005_final", "vggt", "vggt/p7_allcities_p5b_joint_pm4_h0005_aux_h075_lowover15_warm_p5bfinal_e6_b8_2gpu/checkpoint-final.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_to4v_e4_b8_2gpu/checkpoint-final.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_nocrop_warmbest_e8_b8_2gpu/checkpoint-best.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu/checkpoint-best.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final", "vggt", "vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu/checkpoint-final.pth", P7_P5B_SHARED_NORM_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_preservep5b_h035_best", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_preservep5b_h035_e6_b9_4gpu/checkpoint-best.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_preservep5b_h035_final", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_preservep5b_h035_e6_b9_4gpu/checkpoint-final.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_recoverp5b_h035_best", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_recoverp5b_h035_warmpreserve_e8_b9_4gpu/checkpoint-best.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_recoverp5b_h035_final", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_recoverp5b_h035_warmpreserve_e8_b9_4gpu/checkpoint-final.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_p5b_anchor_h035_best", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_p5b_anchor_h035_e4_b10_4gpu/checkpoint-best.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_parallel_token_aux_p5b_anchor_h035_final", "vggt", "vggt/p7_allcities_p5b_parallel_token_aux_p5b_anchor_h035_e4_b10_4gpu/checkpoint-final.pth", P7_P5B_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_remote_parallel_token_aux_recoverp5b_h035_best", "vggt", "vggt/p7_allcities_p5b_private_remote_parallel_token_aux_recoverp5b_h035_e6_b9_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_remote_parallel_token_aux_recoverp5b_h035_final", "vggt", "vggt/p7_allcities_p5b_private_remote_parallel_token_aux_recoverp5b_h035_e6_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_best", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e6_b9_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_final", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e6_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_best", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_b9_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_final", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_best", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_e4_b9_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_final", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_e4_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_zheight001_final", "vggt", "vggt/p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_zheight001_e3_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_oldp7_trunk_p5b_remote_head_final", "vggt", "vggt/p7_diagnostic_oldp7_trunk_p5b_remote_head/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_best", "vggt", "vggt/p7_allcities_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_e3_b9_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_final", "vggt", "vggt/p7_allcities_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_e3_b9_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_best", "vggt", "vggt/p7_allcities_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b16_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_final", "vggt", "vggt/p7_allcities_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b16_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best", "vggt", "vggt/p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b32_4gpu/checkpoint-best.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5b_warm_privatehead_freeze_remotehead_aux_h035_height001_trunklr5e8_final", "vggt", "vggt/p7_diagnostic_p5b_warm_privatehead_freeze_remotehead_aux_h035_height001_trunklr5e8_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5b_warm_privatehead_frozen_trunk_remotehead_auxonly_h035_height001_final", "vggt", "vggt/p7_diagnostic_p5b_warm_privatehead_frozen_trunk_remotehead_auxonly_h035_height001_e4_b32_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_oldp7_frozen_trunk_train_remotehead_aux_h035_height001_final", "vggt", "vggt/p7_diagnostic_oldp7_frozen_trunk_train_remotehead_aux_h035_height001_e4_b32_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_final", "vggt", "vggt/p7_oldp7_train_remotehead_nonreentrant_aggtail2lr1e7_raw001_gradz005_paramanchor500k_lowlr3e6_h003_e2_b24_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_e4_final", "vggt", "vggt/p7_oldp7_train_remotehead_nonreentrant_aggtail2lr1e7_raw001_gradz005_paramanchor500k_lowlr3e6_h003_e4_b24_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_heads_aux_h035_height001_final", "vggt", "vggt/p7_diagnostic_p5bhead_frozen_trunk_remotehead_heads_aux_h035_height001_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherz5_trunklr2e7_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherz5_trunklr2e7_e3_b8_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherxyz5_trunklr2e7_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherxyz5_trunklr2e7_e3_b8_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final", "vggt", "vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_e3_b10_4gpu/checkpoint-final.pth", P7_P5B_PRIVATE_PARALLEL_TOKEN_PROJECTION_AUX),
    Job("vggt_p7_p5e_private_viewtype_projection_aux_allcities_best", "vggt", "vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static/checkpoint-best.pth", P7_P5E_PRIVATE_VIEWTYPE_PROJECTION_AUX),
    Job("vggt_p7_p5e_private_viewtype_projection_aux_allcities_final", "vggt", "vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static/checkpoint-final.pth", P7_P5E_PRIVATE_VIEWTYPE_PROJECTION_AUX),
    Job("vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_best", "vggt", "vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-best.pth", P7_P5E_PRIVATE_VIEWTYPE_PROJECTION_AUX),
    Job("vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final", "vggt", "vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-final.pth", P7_P5E_PRIVATE_VIEWTYPE_PROJECTION_AUX),
    Job("vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final", "vggt", "vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank005_e8_b8_2gpu/checkpoint-final.pth", p5h("film")),
    Job("vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final", "vggt", "vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank05_gate005_e6_b8_2gpu/checkpoint-final.pth", p5h("film")),
    Job("vggt_p7_p5h_film_diffblank_rank02_gate005_allcities_final", "vggt", "vggt/p7_allcities_p5h_film_diffblank_rank02_gate005_e4_b8_2gpu/checkpoint-final.pth", p5h("film")),
    Job("vggt_omega_p1_joint_depth_512", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
    Job("vggt_omega_p1_joint_depth_512_1gpu_2v", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
    Job("vggt_omega_p1_joint_depth_512_all", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
]


def run_job(
    job: Job,
    out_root: Path,
    cuda_device: str,
    force: bool,
    num_views: int,
    remote_control_modes: str,
    batch_size: int,
    scene_list_path: Path | None,
    max_scenes: int | None,
) -> dict[str, object]:
    output_dir = out_root / f"v{num_views}" / job.label
    result_path = output_dir / RESULT_JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and not force:
        return {
            "label": job.label,
            "status": "skipped_existing",
            "output_dir": str(output_dir),
            "result_json": str(result_path),
            "model_name": job.model_name,
            "ckpt": job.ckpt_env,
            "num_views": num_views,
            "batch_size": batch_size,
            "scene_list_path": str(scene_list_path) if scene_list_path else None,
            "max_scenes": max_scenes,
        }

    env = os.environ.copy()
    env.update(
        {
            "NUM_VIEWS": str(num_views),
            "BATCH_SIZE": str(batch_size),
            "REMOTE_OVERFIT_NUM_SETS": str(max_scenes) if max_scenes else "null",
            "REMOTE_CONTROL_MODES": remote_control_modes,
            "CUDA_DEVICE": cuda_device,
            "MODEL_NAME": job.model_name,
            "CKPT_PATH": job.ckpt_env,
            "OUTPUT_DIR": str(output_dir),
        }
    )
    if scene_list_path:
        env["SCENE_LIST_PATH"] = str(scene_list_path)
    cmd = ["bash", RUNNER, *job.args]
    started = time.time()
    log_path = output_dir / "codex_run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(json.dumps({k: env[k] for k in sorted(env) if k in {
            "NUM_VIEWS", "BATCH_SIZE", "REMOTE_OVERFIT_NUM_SETS",
            "REMOTE_CONTROL_MODES", "CUDA_DEVICE", "MODEL_NAME",
            "CKPT_PATH", "OUTPUT_DIR", "SCENE_LIST_PATH",
        }}, indent=2) + "\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "label": job.label,
        "status": "ok" if proc.returncode == 0 and result_path.exists() else "failed",
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "output_dir": str(output_dir),
        "result_json": str(result_path) if result_path.exists() else None,
        "log": str(log_path),
        "model_name": job.model_name,
        "ckpt": job.ckpt_env,
        "num_views": num_views,
        "batch_size": batch_size,
        "scene_list_path": str(scene_list_path) if scene_list_path else None,
        "max_scenes": max_scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated devices for parallel execution, e.g. 0,1. Overrides --cuda-device.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument(
        "--scene-list-path",
        type=Path,
        default=None,
        help="Optional .npy scene list. Use this for fixed-scene NUM_VIEWS sweeps.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional remote overfit/truncation count. Leave unset with fixed scene lists.",
    )
    parser.add_argument("--remote-control-modes", default="none")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--discover-checkpoints", action="store_true")
    parser.add_argument(
        "--exclude-patterns",
        default="debug,smoke,overfit,probe,p7,p8",
        help="Comma-separated lowercase substrings excluded by --discover-checkpoints.",
    )
    parser.add_argument(
        "--exclude-label-patterns",
        default="p7,p8",
        help="Comma-separated lowercase substrings excluded from registered/discovered job labels and checkpoint paths.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional job labels to run/check instead of the full JOBS list.",
    )
    args = parser.parse_args()

    selected_jobs = JOBS
    if args.discover_checkpoints:
        exclude_patterns = [
            item.strip().lower()
            for item in args.exclude_patterns.split(",")
            if item.strip()
        ]
        discovered_jobs = discover_checkpoint_jobs(JOBS, exclude_patterns)
        selected_jobs = [*selected_jobs, *discovered_jobs]
        print(f"Discovered {len(discovered_jobs)} additional checkpoint jobs.")
    exclude_label_patterns = [
        item.strip().lower()
        for item in args.exclude_label_patterns.split(",")
        if item.strip()
    ]
    if exclude_label_patterns:
        before = len(selected_jobs)
        selected_jobs = [
            job
            for job in selected_jobs
            if not any(
                pattern in job.label.lower() or pattern in job.ckpt.lower()
                for pattern in exclude_label_patterns
            )
        ]
        print(f"Excluded {before - len(selected_jobs)} jobs by label/path patterns: {exclude_label_patterns}")
    if args.only:
        requested = set(args.only)
        selected_jobs = [job for job in selected_jobs if job.label in requested]
        missing_labels = requested - {job.label for job in selected_jobs}
        if missing_labels:
            print("Unknown job labels:", file=sys.stderr)
            for label in sorted(missing_labels):
                print(label, file=sys.stderr)
            return 2

    missing = []
    for job in selected_jobs:
        result_paths = [
            args.out_root / f"v{num_views}" / job.label / RESULT_JSON
            for num_views in args.num_views
        ]
        if all(path.exists() for path in result_paths) and not args.force:
            continue
        if job.uses_checkpoint and not job.ckpt_path.exists():
            missing.append((job.label, str(job.ckpt_path)))
    if missing:
        print("Missing checkpoints:", file=sys.stderr)
        for label, path in missing:
            print(f"{label}: {path}", file=sys.stderr)
        if args.skip_missing:
            missing_labels = {label for label, _ in missing}
            selected_jobs = [job for job in selected_jobs if job.label not in missing_labels]
        else:
            return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.scene_list_path and not args.scene_list_path.exists():
        print(f"Scene list does not exist: {args.scene_list_path}", file=sys.stderr)
        return 2
    summary_path = args.out_root / "run_summary.jsonl"
    cuda_devices = (
        [dev.strip() for dev in args.cuda_devices.split(",") if dev.strip()]
        if args.cuda_devices
        else [str(args.cuda_device)]
    )
    workers = args.workers or len(cuda_devices)
    tasks = [
        (num_views, job)
        for num_views in args.num_views
        for job in selected_jobs
    ]
    print(
        f"Running {len(tasks)} tasks ({len(selected_jobs)} jobs x {len(args.num_views)} num_views). "
        f"Output: {args.out_root}. Devices: {','.join(cuda_devices)}. "
        f"Workers: {workers}. Batch size: {args.batch_size}. "
        f"Scene list: {args.scene_list_path or 'default'}"
    )
    with summary_path.open("a", encoding="utf-8") as summary:
        def submit_task(task_idx: int, num_views: int, job: Job) -> dict[str, object]:
            cuda_device = cuda_devices[(task_idx - 1) % len(cuda_devices)]
            print(
                f"[{task_idx}/{len(tasks)}] views={num_views} {job.label} on cuda:{cuda_device}",
                flush=True,
            )
            return run_job(
                job,
                args.out_root,
                cuda_device,
                args.force,
                num_views,
                args.remote_control_modes,
                args.batch_size,
                args.scene_list_path,
                args.max_scenes,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(submit_task, idx, num_views, job): (idx, num_views, job)
                for idx, (num_views, job) in enumerate(tasks, 1)
            }
            for future in as_completed(future_to_task):
                _, num_views, job = future_to_task[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "label": job.label,
                        "num_views": num_views,
                        "status": "failed_exception",
                        "error": str(exc),
                    }
                summary.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary.flush()
                print(
                    f"  -> views={record.get('num_views')} {record['label']} "
                    f"{record['status']} ({record.get('seconds', 0)}s)",
                    flush=True,
                )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
