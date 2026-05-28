#!/bin/bash
set -euo pipefail

# Diagnostic ablation only: weak ranking should not be treated as the main P6
# training objective. It tests whether explicit scene-specific pressure helps.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_conditional_remote_adapter_weak_ranking}
RANKING_WEIGHT=${RANKING_WEIGHT:-0.05}
RANKING_MARGIN=${RANKING_MARGIN:-0.005}
PRESERVE_WEIGHT=${PRESERVE_WEIGHT:-0.05}

export EXPERIMENT_NAME RANKING_WEIGHT RANKING_MARGIN PRESERVE_WEIGHT

bash "$(dirname "$0")/p6a_vggt_conditional_remote_adapter.sh" "$@"
