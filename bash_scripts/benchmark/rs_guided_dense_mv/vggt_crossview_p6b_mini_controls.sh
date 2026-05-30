#!/bin/bash

set -euo pipefail

NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
REMOTE_OVERFIT_NUM_SETS=${REMOTE_OVERFIT_NUM_SETS:-10}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6b_vggt_joint_remote_alignment_private_head_w03}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/'${EXPERIMENT_NAME}'_mini_controls'}
export NUM_VIEWS BATCH_SIZE REMOTE_OVERFIT_NUM_SETS REMOTE_CONTROL_MODES EXPERIMENT_NAME OUTPUT_DIR

bash "$(dirname "$0")/vggt_crossview_p6b_unified.sh" "$@"
