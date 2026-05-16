#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec bash "${REPO_ROOT}/bash_scripts/train/Crossview/pi3/p0_pi3_data_sanity_500_pretrained_2gpu.sh" "$@"
