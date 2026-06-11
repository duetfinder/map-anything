#!/bin/bash

set -euo pipefail

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
FUSION_TYPE=${FUSION_TYPE:-film}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p7h_newyork_2v_e6}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/${EXPERIMENT_NAME}/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/'${EXPERIMENT_NAME}'_mini_controls'}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
MAX_REMOTE_TOKENS=${MAX_REMOTE_TOKENS:-256}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
CROSS_ATTENTION_HEADS=${CROSS_ATTENTION_HEADS:-8}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR REMOTE_CONTROL_MODES

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_use_remote_private_point_head=true
    vggt_export_mode=mixed
    'config_overrides=["machine=aws","model=vggt","model.model_config.use_split_remote_aggregator=true","model.model_config.remote_to_aerial_late_fusion_type='"${FUSION_TYPE}"'","model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25","model.model_config.remote_to_aerial_late_fusion_gate_init=0.0","model.model_config.remote_to_aerial_cross_attention_heads='"${CROSS_ATTENTION_HEADS}"'","model.model_config.remote_to_aerial_max_remote_tokens='"${MAX_REMOTE_TOKENS}"'","model.model_config.protect_ordinary_heads_from_remote='"${PROTECT_ORDINARY_HEADS}"'","model.model_config.ordinary_output_head=depth","model.model_config.remote_output_head=point","model.model_config.use_remote_private_point_head=true","model.model_config.output_point_head_for_consistency=true","model.model_config.use_remote_projection_aux_head=true","model.model_config.remote_projection_aux_hidden_dim=96","model.model_config.remote_projection_aux_detach_pointmap=false","model.model_config.remote_projection_aux_use_rgb=true","model.model_config.remote_projection_aux_use_coord=true","model.model_config.remote_projection_aux_image_stem_dim=32","model.model_config.remote_projection_aux_positive_slope=true","model.model_config.remote_projection_aux_slope_init=0.1","model.model_config.remote_projection_aux_num_blocks=6"]'
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
