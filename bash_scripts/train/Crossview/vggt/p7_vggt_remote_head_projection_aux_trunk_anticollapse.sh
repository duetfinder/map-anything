#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p7_remote_head_projection_aux_trunk}
export LOSS_CONFIG=${LOSS_CONFIG:-vggt_loss_rs_joint_p7_remote_head_projection_aux_anticollapse}
export OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk_anticollapse'}
export LAMBDA_PROJ_REL_HEIGHT=${LAMBDA_PROJ_REL_HEIGHT:-0.5}
export LAMBDA_PROJ_OFFSET=${LAMBDA_PROJ_OFFSET:-2.0}
export LAMBDA_PROJ_GLOBAL_DIR=${LAMBDA_PROJ_GLOBAL_DIR:-0.1}
export LAMBDA_PROJ_GLOBAL_SLOPE=${LAMBDA_PROJ_GLOBAL_SLOPE:-0.5}
export LAMBDA_PROJ_CONSISTENCY=${LAMBDA_PROJ_CONSISTENCY:-0.0}
export PROJ_OFFSET_USE_TILT_MASK=${PROJ_OFFSET_USE_TILT_MASK:-true}
export PROJ_CONSISTENCY_USE_TILT_MASK=${PROJ_CONSISTENCY_USE_TILT_MASK:-true}
export PROJ_OFFSET_MIN_MAGNITUDE=${PROJ_OFFSET_MIN_MAGNITUDE:-0.5}
export PROJ_CONSISTENCY_MIN_MAGNITUDE=${PROJ_CONSISTENCY_MIN_MAGNITUDE:-0.5}
exec bash "${SCRIPT_DIR}/p7_vggt_remote_head_projection_aux.sh" "$@"
