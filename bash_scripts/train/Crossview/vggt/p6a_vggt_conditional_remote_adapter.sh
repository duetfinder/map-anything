#!/bin/bash
set -euo pipefail

# P6A: protected conditional remote adapter. No ranking by default;
# same/blank/shuffled controls should be run as evaluation.
FUSION_TYPE=${FUSION_TYPE:-cross_attention}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6a_vggt_raw_base_conditional_remote_adapter}
LATE_GATE_INIT=${LATE_GATE_INIT:-1e-3}
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-0.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
PRESERVE_WEIGHT=${PRESERVE_WEIGHT:-0.05}
GATE_L1_WEIGHT=${GATE_L1_WEIGHT:-1e-03}
WEIGHTED_DELTA_L2_WEIGHT=${WEIGHTED_DELTA_L2_WEIGHT:-1e-04}
RANKING_WEIGHT=${RANKING_WEIGHT:-0.0}
RANKING_MARGIN=${RANKING_MARGIN:-0.0}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p6_conditional_remote_adapter}
BASE_CKPT=${BASE_CKPT:-null}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-true}
CUSTOM_CKPT_PATH=${CUSTOM_CKPT_PATH:-/root/autodl-tmp/outputs/checkpoints/vggt/model.pt}

export FUSION_TYPE EXPERIMENT_NAME LATE_GATE_INIT LAMBDA_REMOTE_PM LAMBDA_REMOTE_H PROTECT_ORDINARY_HEADS BASE_CKPT LOAD_PRETRAINED_WEIGHTS LOAD_CUSTOM_CKPT CUSTOM_CKPT_PATH

BASE_NUM_GPUS=${NUM_GPUS:-${1:-4}}

bash "$(dirname "$0")/p5h_vggt_p5e_base_split_late_fusion.sh" "${BASE_NUM_GPUS}" \
    train_params=${TRAIN_PARAMS} \
    train_params.remote_blank_preserve_loss_weight=${PRESERVE_WEIGHT} \
    train_params.remote_late_gate_l1_weight=${GATE_L1_WEIGHT} \
    train_params.remote_late_weighted_delta_l2_weight=${WEIGHTED_DELTA_L2_WEIGHT} \
    train_params.remote_control_ranking_loss_weight=${RANKING_WEIGHT} \
    train_params.remote_control_ranking_margin=${RANKING_MARGIN} \
    "${@:2}"
