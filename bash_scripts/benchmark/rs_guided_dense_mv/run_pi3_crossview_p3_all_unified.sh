#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
    pi3_crossview_p3_all_unified.sh
    pi3_crossview_p3_modality_embedding_unified.sh
    pi3_crossview_p3_modality_embedding_freeze_shared_unified.sh
    pi3_crossview_p3_modality_embedding_remote_head_unified.sh
    pi3_crossview_p3_all_low_covis_unified.sh
    pi3_crossview_p3_all_zero_covis_unified.sh
)

for experiment in "${EXPERIMENTS[@]}"; do
    echo "Running ${experiment}"
    bash "${SCRIPT_DIR}/${experiment}"
done
