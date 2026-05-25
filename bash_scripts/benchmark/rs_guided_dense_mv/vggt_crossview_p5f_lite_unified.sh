#!/bin/bash

set -euo pipefail

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5f_vggt_lite_early_bias_gated_residual/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5f_lite_unified'}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR REMOTE_CONTROL_MODES

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_use_remote_private_point_head=true
    vggt_export_mode=mixed
    'config_overrides=["machine=aws","model=vggt","model.model_config.use_pre_aggregator_view_type_bias=true","model.model_config.use_remote_to_aerial_gated_residual=true","model.model_config.remote_to_aerial_residual_hidden_scale=0.25","model.model_config.remote_to_aerial_gate_init=0.0"]'
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
