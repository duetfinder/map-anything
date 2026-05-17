#!/bin/bash

MODEL_NAME=vggt
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5c_unified'}

MODEL_EXTRA_ARGS=(
    model.model_config.use_point_head_for_remote=true
    model.model_config.use_view_type_bias=true
)

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh" "${MODEL_EXTRA_ARGS[@]}"
