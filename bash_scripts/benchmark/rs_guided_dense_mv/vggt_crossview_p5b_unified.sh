#!/bin/bash

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_loss_only/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5b_unified'}
export MODEL_NAME NUM_VIEWS CKPT_PATH OUTPUT_DIR

MODEL_EXTRA_ARGS=(
    vggt_joint_remote_export=true
    vggt_export_mode=mixed
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}" "$@"
