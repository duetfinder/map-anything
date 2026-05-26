#!/bin/bash
set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5g_vggt_crossattn_split_remote}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/p5g_vggt_split_late_fusion.sh" "$@"
