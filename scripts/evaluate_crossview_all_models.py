#!/usr/bin/env python3
"""Run the New York RS guided mini benchmark for all Crossview checkpoints."""

from __future__ import annotations

import argparse
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
    "rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls"
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
    def ckpt_path(self) -> Path:
        return TRAIN_ROOT / self.ckpt


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


JOBS: list[Job] = [
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
    Job("vggt_omega_p1_joint_depth_512", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
    Job("vggt_omega_p1_joint_depth_512_1gpu_2v", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
    Job("vggt_omega_p1_joint_depth_512_all", "vggt_omega", "vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth", cfg("vggt_omega", omega=True)),
]


def run_job(job: Job, out_root: Path, cuda_device: str, force: bool) -> dict[str, object]:
    output_dir = out_root / job.label
    result_path = output_dir / RESULT_JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and not force:
        return {
            "label": job.label,
            "status": "skipped_existing",
            "output_dir": str(output_dir),
            "result_json": str(result_path),
        }

    env = os.environ.copy()
    env.update(
        {
            "NUM_VIEWS": "4",
            "BATCH_SIZE": "1",
            "REMOTE_OVERFIT_NUM_SETS": "10",
            "REMOTE_CONTROL_MODES": "[same,blank,shuffled]",
            "CUDA_DEVICE": cuda_device,
            "MODEL_NAME": job.model_name,
            "CKPT_PATH": str(job.ckpt_path),
            "OUTPUT_DIR": str(output_dir),
        }
    )
    cmd = ["bash", RUNNER, *job.args]
    started = time.time()
    log_path = output_dir / "codex_run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(json.dumps({k: env[k] for k in sorted(env) if k in {
            "NUM_VIEWS", "BATCH_SIZE", "REMOTE_OVERFIT_NUM_SETS",
            "REMOTE_CONTROL_MODES", "CUDA_DEVICE", "MODEL_NAME",
            "CKPT_PATH", "OUTPUT_DIR",
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
        "ckpt": str(job.ckpt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    missing = [str(job.ckpt_path) for job in JOBS if not job.ckpt_path.exists()]
    if missing:
        print("Missing checkpoints:", file=sys.stderr)
        for path in missing:
            print(path, file=sys.stderr)
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "run_summary.jsonl"
    print(f"Running {len(JOBS)} jobs. Output: {args.out_root}")
    with summary_path.open("a", encoding="utf-8") as summary:
        for idx, job in enumerate(JOBS, 1):
            print(f"[{idx}/{len(JOBS)}] {job.label}", flush=True)
            record = run_job(job, args.out_root, args.cuda_device, args.force)
            summary.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary.flush()
            print(f"  -> {record['status']} ({record.get('seconds', 0)}s)", flush=True)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
