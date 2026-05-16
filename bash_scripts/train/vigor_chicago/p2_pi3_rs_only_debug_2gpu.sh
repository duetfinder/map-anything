#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec bash "${REPO_ROOT}/bash_scripts/train/Crossview/pi3/p2_pi3_rs_only_debug_2gpu.sh" "$@"
