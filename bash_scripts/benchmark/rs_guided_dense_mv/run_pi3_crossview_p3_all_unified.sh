#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CITY=${CITY:-newyork}
export REMOTE_PROVIDER=${REMOTE_PROVIDER:-Google_Satellite}
export COVISIBILITY_THRES=${COVISIBILITY_THRES:-0.0}
export VIEW_SAMPLING_MODE=${VIEW_SAMPLING_MODE:-connected} # low_covis
export CUDA_DEVICE=${CUDA_DEVICE:-0}

EXPERIMENTS=(
    "pi3_crossview_p3_all_unified.sh pi3_crossview_p3_base"
    "pi3_crossview_p3_modality_embedding_unified.sh pi3_crossview_p3_modality_embedding"
    "pi3_crossview_p3_modality_embedding_freeze_shared_unified.sh pi3_crossview_p3_modality_embedding_freeze_shared"
    "pi3_crossview_p3_modality_embedding_remote_head_unified.sh pi3_crossview_p3_modality_embedding_remote_head"
    "pi3_crossview_p3_all_zero_covis_unified.sh pi3_crossview_p3_zero_covis"
)

# Format: "batch_size num_views". Matches pi3_unified_sweep.sh unless overridden
# by editing this file for a narrower run.
RUN_CONFIGS=(
    "40 2"
    "40 4"
    "20 8"
    "10 16"
    "8 24"
    "8 32"
    "2 40"
)

for item in "${EXPERIMENTS[@]}"; do
    read -r experiment output_prefix <<< "${item}"
    for cfg in "${RUN_CONFIGS[@]}"; do
        read -r batch_size num_views <<< "${cfg}"
        export BATCH_SIZE="${batch_size}"
        export NUM_VIEWS="${num_views}"
        export OUTPUT_DIR='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/'"${CITY}"'/'"${output_prefix}"'_'"${num_views}"'v'
        echo "Running ${experiment}: city=${CITY} provider=${REMOTE_PROVIDER} covis=${COVISIBILITY_THRES} sampling=${VIEW_SAMPLING_MODE} batch_size=${BATCH_SIZE} num_views=${NUM_VIEWS}"
        bash "${SCRIPT_DIR}/${experiment}"
    done
done
