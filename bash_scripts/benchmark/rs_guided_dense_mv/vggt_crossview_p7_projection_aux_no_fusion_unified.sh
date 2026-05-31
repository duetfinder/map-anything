#!/bin/bash

set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-none}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p7_vggt_projection_aux_no_fusion_split_remote}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/vggt_crossview_p7_projection_aux_unified.sh" "$@"
