#!/bin/bash
set -euo pipefail

# Explicit raw/map-anything benchmark base entrypoint for P6A.
BASE_CKPT=${BASE_CKPT:-/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_raw_base_conditional_remote_adapter}

export BASE_CKPT EXPERIMENT_NAME

bash "$(dirname "$0")/p6a_vggt_conditional_remote_adapter.sh" "$@"
