#!/bin/bash

MODEL_NAME=pi3
NUM_VIEWS=${NUM_VIEWS:-2}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/vigor_chicago/baselines/p1_pi3_baseline_500_pretrained_2gpu/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/pi3_crossview_p1_unified'}

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh"
