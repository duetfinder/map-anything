#!/bin/bash

set -euo pipefail

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_official_raw_conditional_remote_adapter}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/${EXPERIMENT_NAME}/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/'${EXPERIMENT_NAME}'_mini_controls'}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
MAX_REMOTE_TOKENS=${MAX_REMOTE_TOKENS:-256}
CROSS_ATTENTION_HEADS=${CROSS_ATTENTION_HEADS:-8}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR REMOTE_CONTROL_MODES

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_use_remote_private_point_head=true
    vggt_export_mode=mixed
    'config_overrides=["machine=aws","model=vggt","model.model_config.use_view_type_bias=true","model.model_config.use_split_remote_aggregator=true","model.model_config.remote_to_aerial_late_fusion_type='"${FUSION_TYPE}"'","model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25","model.model_config.remote_to_aerial_late_fusion_gate_init=1e-3","model.model_config.remote_to_aerial_cross_attention_heads='"${CROSS_ATTENTION_HEADS}"'","model.model_config.remote_to_aerial_max_remote_tokens='"${MAX_REMOTE_TOKENS}"'","model.model_config.protect_ordinary_heads_from_remote='"${PROTECT_ORDINARY_HEADS}"'"]'
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
