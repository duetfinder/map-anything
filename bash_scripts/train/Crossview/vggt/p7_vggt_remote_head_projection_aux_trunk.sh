#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p7_remote_head_projection_aux_trunk}
export OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk'}
exec bash "${SCRIPT_DIR}/p7_vggt_remote_head_projection_aux.sh" "$@"
