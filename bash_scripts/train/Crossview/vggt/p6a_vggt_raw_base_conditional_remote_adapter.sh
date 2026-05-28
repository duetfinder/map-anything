#!/bin/bash
set -euo pipefail

# Explicit raw/map-anything benchmark base entrypoint for P6A.
BASE_CKPT=${BASE_CKPT:-null}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-true}
CUSTOM_CKPT_PATH=${CUSTOM_CKPT_PATH:-/root/autodl-tmp/outputs/checkpoints/vggt/model.pt}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_raw_base_conditional_remote_adapter}

export BASE_CKPT LOAD_PRETRAINED_WEIGHTS LOAD_CUSTOM_CKPT CUSTOM_CKPT_PATH EXPERIMENT_NAME

bash "$(dirname "$0")/p6a_vggt_conditional_remote_adapter.sh" "$@"
