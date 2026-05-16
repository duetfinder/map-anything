#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec bash "${REPO_ROOT}/bash_scripts/train/Crossview/pi3/p3_pi3_joint_input_500_2gpu_all_zero_covis.sh" "$@"
