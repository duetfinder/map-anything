#!/bin/bash

set -euo pipefail

NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
REMOTE_OVERFIT_NUM_SETS=${REMOTE_OVERFIT_NUM_SETS:-5}
REMOTE_CONTROL_MODES=${REMOTE_CONTROL_MODES:-[same,blank,shuffled]}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5f_lite_mini_controls'}
export NUM_VIEWS BATCH_SIZE REMOTE_OVERFIT_NUM_SETS REMOTE_CONTROL_MODES OUTPUT_DIR

bash "$(dirname "$0")/vggt_crossview_p5f_lite_unified.sh" "$@"
