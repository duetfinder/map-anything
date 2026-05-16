#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec bash "${REPO_ROOT}/bash_scripts/train/Crossview/pi3/p2a_pi3_rs_only_loss_ablation_2gpu.sh" "$@"
