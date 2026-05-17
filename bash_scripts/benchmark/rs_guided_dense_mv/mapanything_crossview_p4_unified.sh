#!/bin/bash

MODEL_NAME=mapanything_rs_joint
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/mapanything_crossview_p4_unified'}

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh"
