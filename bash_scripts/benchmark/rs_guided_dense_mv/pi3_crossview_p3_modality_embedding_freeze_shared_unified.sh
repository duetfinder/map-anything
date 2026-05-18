#!/bin/bash

MODEL_NAME=pi3_modality_embedding
NUM_VIEWS=${NUM_VIEWS:-4}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_freeze_shared/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/pi3_crossview_p3_modality_embedding_freeze_shared_unified'}

bash "$(dirname "$0")/run_crossview_finetuned_unified.sh"
