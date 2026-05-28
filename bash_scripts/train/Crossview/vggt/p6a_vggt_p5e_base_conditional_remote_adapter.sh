#!/bin/bash
set -euo pipefail

# Historical control: P6A on top of the p5e remote-head/view-type checkpoint.
# Use this only as a comparison against the raw-base P6A main experiment.
BASE_CKPT=${BASE_CKPT:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_p5e_base_conditional_remote_adapter}

export BASE_CKPT EXPERIMENT_NAME

bash "$(dirname "$0")/p6a_vggt_conditional_remote_adapter.sh" "$@"
