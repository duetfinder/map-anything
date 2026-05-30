#!/bin/bash
set -euo pipefail
export TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p6b_joint_remote_alignment_private_head_viewtype}
export P6B_VARIANT=${P6B_VARIANT:-private_head_viewtype}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6b_vggt_joint_remote_alignment_private_head_viewtype_w03}
export USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-true}
export USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS:-true}
export POINT_HEAD_LR=${POINT_HEAD_LR:-0}
export REMOTE_POINT_HEAD_LR=${REMOTE_POINT_HEAD_LR:-2e-05}
export VIEW_TYPE_LR=${VIEW_TYPE_LR:-5e-05}
export DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH:-true}
export DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-false}
bash "$(dirname "$0")/p6b_vggt_joint_remote_alignment.sh" "$@"
