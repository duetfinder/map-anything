#!/bin/bash
set -euo pipefail
export TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p6b_joint_remote_alignment_shared_head}
export P6B_VARIANT=${P6B_VARIANT:-shared_head}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6b_vggt_joint_remote_alignment_shared_head_w03}
export USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-false}
export USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS:-false}
export POINT_HEAD_LR=${POINT_HEAD_LR:-5e-06}
export REMOTE_POINT_HEAD_LR=${REMOTE_POINT_HEAD_LR:-0}
bash "$(dirname "$0")/p6b_vggt_joint_remote_alignment.sh" "$@"
