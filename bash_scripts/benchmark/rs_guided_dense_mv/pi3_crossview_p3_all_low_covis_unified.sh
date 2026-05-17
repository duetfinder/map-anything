#!/bin/bash

MODEL_NAME=pi3
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/vigor_chicago/p3_pi3_joint_input_500_2gpu_all_low_covis/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/pi3_crossview_p3_all_low_covis_unified'}

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh"
