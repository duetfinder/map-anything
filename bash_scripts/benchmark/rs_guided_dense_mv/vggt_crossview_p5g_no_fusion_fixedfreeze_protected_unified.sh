#!/bin/bash

set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-none}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5g_vggt_no_fusion_fixedfreeze_protected}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
export FUSION_TYPE EXPERIMENT_NAME PROTECT_ORDINARY_HEADS

bash "$(dirname "$0")/vggt_crossview_p5g_unified.sh" "$@"
