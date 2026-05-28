#!/bin/bash
set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_crossattn_protected}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/p5h_vggt_p5e_base_split_late_fusion.sh" "$@"
