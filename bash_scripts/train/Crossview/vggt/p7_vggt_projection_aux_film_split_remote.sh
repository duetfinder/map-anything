#!/bin/bash
set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-film}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p7_vggt_projection_aux_film_split_remote}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/p7_vggt_projection_aux_split_late_fusion.sh" "$@"
