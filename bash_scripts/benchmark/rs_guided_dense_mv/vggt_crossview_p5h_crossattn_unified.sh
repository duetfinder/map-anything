#!/bin/bash

set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_crossattn_protected}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/vggt_crossview_p5h_unified.sh" "$@"
