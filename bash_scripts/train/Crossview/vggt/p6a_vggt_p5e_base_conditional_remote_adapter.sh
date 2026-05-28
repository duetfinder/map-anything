#!/bin/bash
set -euo pipefail

# Historical control: P6A on top of the p5e remote-head/view-type checkpoint.
# Use this only as a comparison against the raw-base P6A main experiment.
BASE_CKPT=${BASE_CKPT:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-false}
CUSTOM_CKPT_PATH=${CUSTOM_CKPT_PATH:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_p5e_base_conditional_remote_adapter}

export BASE_CKPT LOAD_PRETRAINED_WEIGHTS LOAD_CUSTOM_CKPT CUSTOM_CKPT_PATH EXPERIMENT_NAME

bash "$(dirname "$0")/p6a_vggt_conditional_remote_adapter.sh" "$@"
