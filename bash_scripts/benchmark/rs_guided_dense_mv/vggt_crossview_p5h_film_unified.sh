#!/bin/bash

set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-film}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_film_protected}
export FUSION_TYPE EXPERIMENT_NAME

bash "$(dirname "$0")/vggt_crossview_p5h_unified.sh" "$@"
