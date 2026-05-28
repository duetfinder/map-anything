#!/bin/bash

set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-none}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_no_fusion_protected}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
export FUSION_TYPE EXPERIMENT_NAME CKPT_PATH PROTECT_ORDINARY_HEADS

bash "$(dirname "$0")/vggt_crossview_p5h_unified.sh" "$@"
