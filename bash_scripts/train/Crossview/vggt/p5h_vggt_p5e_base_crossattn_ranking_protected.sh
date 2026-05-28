#!/bin/bash
set -euo pipefail

FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_crossattn_ranking_protected}
RANKING_LOSS_WEIGHT=${RANKING_LOSS_WEIGHT:-0.2}
RANKING_MARGIN=${RANKING_MARGIN:-0.01}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-20}
export FUSION_TYPE EXPERIMENT_NAME BATCH_SIZE EPOCHS

BASE_NUM_GPUS=${NUM_GPUS:-5}

bash "$(dirname "$0")/p5h_vggt_p5e_base_split_late_fusion.sh" "${BASE_NUM_GPUS}" "$@" \
  train_params.remote_control_ranking_loss_weight=${RANKING_LOSS_WEIGHT} \
  train_params.remote_control_ranking_margin=${RANKING_MARGIN} \
  train_params.remote_control_ranking_modes='[blank,shuffled]' \
  train_params.remote_control_blank_value=0.5
