#!/bin/bash

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5d_vggt_remote_point_head_consistency/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5d_unified'}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_use_remote_private_point_head=true
    vggt_export_mode=mixed
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
