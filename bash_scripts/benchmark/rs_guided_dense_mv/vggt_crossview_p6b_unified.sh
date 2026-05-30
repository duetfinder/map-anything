#!/bin/bash

set -euo pipefail

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6b_vggt_joint_remote_alignment_private_head_w03}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/${EXPERIMENT_NAME}/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/'${EXPERIMENT_NAME}'_unified'}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-true}
USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS:-false}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR REMOTE_CONTROL_MODES

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_use_remote_private_point_head=${USE_REMOTE_PRIVATE_POINT_HEAD}
    vggt_export_mode=mixed
    'config_overrides=["machine=aws","model=vggt","model.model_config.use_split_remote_aggregator=false","model.model_config.protect_ordinary_heads_from_remote=false","model.model_config.use_remote_to_aerial_gated_residual=false","model.model_config.remote_to_aerial_late_fusion_type=none","model.model_config.use_pre_aggregator_view_type_bias=false","model.model_config.use_view_type_bias='${USE_VIEW_TYPE_BIAS}'"]'
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
