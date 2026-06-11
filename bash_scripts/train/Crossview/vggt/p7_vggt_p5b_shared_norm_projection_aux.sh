#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-4}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29516}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-0}
KEEP_FREQ=${KEEP_FREQ:-0}
PRINT_FREQ=${PRINT_FREQ:-20}
RS_PROVIDER=${RS_PROVIDER:-Google_Satellite,Bing_Satellite}
REMOTE_PROVIDER_SAMPLING_MODE=${REMOTE_PROVIDER_SAMPLING_MODE:-random}
REMOTE_TRAIN_CROP_MODE=${REMOTE_TRAIN_CROP_MODE:-random_scale_offset}
REMOTE_VAL_CROP_MODE=${REMOTE_VAL_CROP_MODE:-random_scale_offset}
REMOTE_TEST_CROP_MODE=${REMOTE_TEST_CROP_MODE:-none}
REMOTE_CROP_SCALE_MIN=${REMOTE_CROP_SCALE_MIN:-0.6}
REMOTE_CROP_SCALE_MAX=${REMOTE_CROP_SCALE_MAX:-1.0}
REMOTE_IMAGE_RESIZE_MODE=${REMOTE_IMAGE_RESIZE_MODE:-nearest}
REMOTE_LABEL_RESIZE_MODE=${REMOTE_LABEL_RESIZE_MODE:-nearest}
REMOTE_NUM_VIEWS=${REMOTE_NUM_VIEWS:-1}
TRAIN_CITIES=${TRAIN_CITIES:-[chicago,newyork]} # Chicago and New York have the best coverage of both aerial and street-level imagery, so we train on both by default.
VAL_CITIES=${VAL_CITIES:-[chicago,newyork]}
TEST_CITIES=${TEST_CITIES:-[chicago,newyork]}

# P5B shared-normalization reconstruction losses.
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-4.0}
LAMBDA_REMOTE_RAW_PM=${LAMBDA_REMOTE_RAW_PM:-0.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
REMOTE_POINTMAP_TOP_N_PERCENT=${REMOTE_POINTMAP_TOP_N_PERCENT:-0.0}
LAMBDA_REMOTE_PM_GRAD=${LAMBDA_REMOTE_PM_GRAD:-0.0}
REMOTE_PM_GRAD_SCALES=${REMOTE_PM_GRAD_SCALES:-4}
REMOTE_PM_GRAD_CHANNELS=${REMOTE_PM_GRAD_CHANNELS:-z}
LAMBDA_REMOTE_MOGE_GRAD=${LAMBDA_REMOTE_MOGE_GRAD:-0.0}
LAMBDA_REMOTE_MOGE_EDGE=${LAMBDA_REMOTE_MOGE_EDGE:-0.0}
LAMBDA_REMOTE_PM_MOGE_HEIGHT=${LAMBDA_REMOTE_PM_MOGE_HEIGHT:-0.0}
REMOTE_PM_MOGE_HEIGHT_PRIOR_MIN_WEIGHT=${REMOTE_PM_MOGE_HEIGHT_PRIOR_MIN_WEIGHT:-0.02}
REMOTE_PM_MOGE_HEIGHT_GROUND_QUANTILE=${REMOTE_PM_MOGE_HEIGHT_GROUND_QUANTILE:-0.2}
LAMBDA_REMOTE_OVERLAP_PM=${LAMBDA_REMOTE_OVERLAP_PM:-0.0}
REMOTE_OVERLAP_DEPTH_TOL=${REMOTE_OVERLAP_DEPTH_TOL:-3.0}
REMOTE_OVERLAP_REL_DEPTH_TOL=${REMOTE_OVERLAP_REL_DEPTH_TOL:-0.05}
REMOTE_OVERLAP_MIN_PIXELS=${REMOTE_OVERLAP_MIN_PIXELS:-64}
REMOTE_OVERLAP_MIN_DEPTH=${REMOTE_OVERLAP_MIN_DEPTH:-0.001}
REMOTE_MOGE_PRIOR_MIN_WEIGHT=${REMOTE_MOGE_PRIOR_MIN_WEIGHT:-0.02}
REMOTE_MOGE_EDGE_TEMPERATURE=${REMOTE_MOGE_EDGE_TEMPERATURE:-10.0}
REMOTE_MOGE_EDGE_THRESHOLD=${REMOTE_MOGE_EDGE_THRESHOLD:-0.5}
LAMBDA_REMOTE_Z_DIST=${LAMBDA_REMOTE_Z_DIST:-0.0}
LAMBDA_REMOTE_HIGH_Z=${LAMBDA_REMOTE_HIGH_Z:-0.0}
REMOTE_HIGH_Z_QUANTILE=${REMOTE_HIGH_Z_QUANTILE:-0.8}
REMOTE_HIGH_Z_MIN_PIXELS=${REMOTE_HIGH_Z_MIN_PIXELS:-16}
SCALE_REMOTE_BY_NUM_VIEWS=${SCALE_REMOTE_BY_NUM_VIEWS:-true}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-false}
REMOTE_COMPARE_GT_IN_VIEW0_ONLY=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}
REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE:-aerial_avg_dis}

# P7 projection auxiliary path. Defaults come from the successful
# newyork top128 capacity_grad aux-only diagnosis, adjusted for main joint training.
LAMBDA_PROJ_REL_HEIGHT=${LAMBDA_PROJ_REL_HEIGHT:-0.5}
LAMBDA_PROJ_OFFSET=${LAMBDA_PROJ_OFFSET:-1.5}
LAMBDA_PROJ_OFFSET_MAG=${LAMBDA_PROJ_OFFSET_MAG:-0.0}
LAMBDA_PROJ_OFFSET_DIR=${LAMBDA_PROJ_OFFSET_DIR:-0.0}
LAMBDA_PROJ_GLOBAL_DIR=${LAMBDA_PROJ_GLOBAL_DIR:-0.0}
LAMBDA_PROJ_GLOBAL_SLOPE=${LAMBDA_PROJ_GLOBAL_SLOPE:-0.1}
LAMBDA_PROJ_GLOBAL_VECTOR=${LAMBDA_PROJ_GLOBAL_VECTOR:-0.0}
LAMBDA_PROJ_CONSISTENCY=${LAMBDA_PROJ_CONSISTENCY:-0.0}
PROJ_GLOBAL_TARGET_FROM_POINTMAP=${PROJ_GLOBAL_TARGET_FROM_POINTMAP:-false}
PROJ_GLOBAL_TARGET_MIN_REL_HEIGHT=${PROJ_GLOBAL_TARGET_MIN_REL_HEIGHT:-0.0}
PROJ_GLOBAL_TARGET_MIN_PIXELS=${PROJ_GLOBAL_TARGET_MIN_PIXELS:-64}
LAMBDA_PROJ_MOGE_GRAD=${LAMBDA_PROJ_MOGE_GRAD:-0.0}
LAMBDA_PROJ_MOGE_EDGE=${LAMBDA_PROJ_MOGE_EDGE:-0.0}
LAMBDA_PROJ_MOGE_HEIGHT=${LAMBDA_PROJ_MOGE_HEIGHT:-0.0}
PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT:-0.02}
PROJ_MOGE_HEIGHT_GROUND_QUANTILE=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE:-0.2}
PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK=${PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK:-true}
PROJ_MOGE_PRIOR_MIN_WEIGHT=${PROJ_MOGE_PRIOR_MIN_WEIGHT:-0.02}
PROJ_MOGE_EDGE_TEMPERATURE=${PROJ_MOGE_EDGE_TEMPERATURE:-10.0}
PROJ_MOGE_EDGE_THRESHOLD=${PROJ_MOGE_EDGE_THRESHOLD:-0.5}
PROJ_GLOBAL_DIR_FROM_OFFSET=${PROJ_GLOBAL_DIR_FROM_OFFSET:-true}
PROJ_REL_HEIGHT_SCALE=${PROJ_REL_HEIGHT_SCALE:-1.0}
PROJ_REL_HEIGHT_SCALE_MODE=${PROJ_REL_HEIGHT_SCALE_MODE:-gt_pointmap_norm}
PROJ_REL_HEIGHT_SCALE_QUANTILE=${PROJ_REL_HEIGHT_SCALE_QUANTILE:-0.9}
PROJ_REL_HEIGHT_CLIP=${PROJ_REL_HEIGHT_CLIP:-0.0}
PROJ_REL_HEIGHT_MIN=${PROJ_REL_HEIGHT_MIN:-0.0}
PROJ_REL_HEIGHT_USE_TILT_MASK=${PROJ_REL_HEIGHT_USE_TILT_MASK:-false}
PROJ_REL_HEIGHT_TARGET_WEIGHT=${PROJ_REL_HEIGHT_TARGET_WEIGHT:-0.0}
PROJ_REL_HEIGHT_TARGET_WEIGHT_GAMMA=${PROJ_REL_HEIGHT_TARGET_WEIGHT_GAMMA:-1.0}
PROJ_REL_HEIGHT_AFFINE_WEIGHT=${PROJ_REL_HEIGHT_AFFINE_WEIGHT:-0.25}
PROJ_REL_HEIGHT_AFFINE_DETACH_FIT=${PROJ_REL_HEIGHT_AFFINE_DETACH_FIT:-true}
PROJ_REL_HEIGHT_AFFINE_MIN_PIXELS=${PROJ_REL_HEIGHT_AFFINE_MIN_PIXELS:-16}
PROJ_REL_HEIGHT_BALANCED_WEIGHT=${PROJ_REL_HEIGHT_BALANCED_WEIGHT:-0.8}
PROJ_REL_HEIGHT_BALANCED_QUANTILES=${PROJ_REL_HEIGHT_BALANCED_QUANTILES:-[0.5,0.8]}
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=${PROJ_REL_HEIGHT_CONTRAST_WEIGHT:-0.6}
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=${PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT:-1.0}
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=${PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT:-0.5}
PROJ_REL_HEIGHT_LOW_OVERPRED_START_EPOCH=${PROJ_REL_HEIGHT_LOW_OVERPRED_START_EPOCH:-2.0}
PROJ_REL_HEIGHT_LOW_OVERPRED_RAMP_EPOCHS=${PROJ_REL_HEIGHT_LOW_OVERPRED_RAMP_EPOCHS:-2.0}
PROJ_DENSE_REL_HEIGHT_WEIGHT=${PROJ_DENSE_REL_HEIGHT_WEIGHT:-0.0}
PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK=${PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK:-true}
PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT=${PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT:-0.25}
PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE=${PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE:-0.5}
PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE=${PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE:-0.0}
PROJ_DENSE_GLOBAL_OFFSET_WEIGHT=${PROJ_DENSE_GLOBAL_OFFSET_WEIGHT:-0.0}
PROJ_DENSE_GLOBAL_OFFSET_LOW_WEIGHT=${PROJ_DENSE_GLOBAL_OFFSET_LOW_WEIGHT:-0.5}
PROJ_DENSE_GLOBAL_OFFSET_LOW_QUANTILE=${PROJ_DENSE_GLOBAL_OFFSET_LOW_QUANTILE:-0.5}
PROJ_OFFSET_SCALE=${PROJ_OFFSET_SCALE:-32.0}
PROJ_OFFSET_TARGET_WEIGHT=${PROJ_OFFSET_TARGET_WEIGHT:-0.0}
PROJ_OFFSET_TARGET_WEIGHT_GAMMA=${PROJ_OFFSET_TARGET_WEIGHT_GAMMA:-1.0}
PROJ_OFFSET_BALANCED_WEIGHT=${PROJ_OFFSET_BALANCED_WEIGHT:-1.0}
PROJ_OFFSET_BALANCED_QUANTILES=${PROJ_OFFSET_BALANCED_QUANTILES:-[0.5,0.8]}
PROJ_OFFSET_CONTRAST_WEIGHT=${PROJ_OFFSET_CONTRAST_WEIGHT:-0.6}
PROJ_OFFSET_BUCKET_MEAN_WEIGHT=${PROJ_OFFSET_BUCKET_MEAN_WEIGHT:-1.5}
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=${PROJ_OFFSET_LOW_OVERPRED_WEIGHT:-1.0}
PROJ_OFFSET_LOW_OVERPRED_START_EPOCH=${PROJ_OFFSET_LOW_OVERPRED_START_EPOCH:-2.0}
PROJ_OFFSET_LOW_OVERPRED_RAMP_EPOCHS=${PROJ_OFFSET_LOW_OVERPRED_RAMP_EPOCHS:-2.0}
PROJ_CONSISTENCY_USE_LOSS_SPACE=${PROJ_CONSISTENCY_USE_LOSS_SPACE:-false}
PROJ_OFFSET_USE_TILT_MASK=${PROJ_OFFSET_USE_TILT_MASK:-false}
PROJ_CONSISTENCY_USE_TILT_MASK=${PROJ_CONSISTENCY_USE_TILT_MASK:-false}
PROJ_OFFSET_MIN_MAGNITUDE=${PROJ_OFFSET_MIN_MAGNITUDE:-0.0}
PROJ_CONSISTENCY_MIN_MAGNITUDE=${PROJ_CONSISTENCY_MIN_MAGNITUDE:-0.0}
PROJ_RECON_OFFSET_TO_GT=${PROJ_RECON_OFFSET_TO_GT:-0.0}
PROJ_RECON_GLOBAL_TO_GT=${PROJ_RECON_GLOBAL_TO_GT:-0.0}
PROJ_RECON_OFFSET_TO_POINT_DETACH=${PROJ_RECON_OFFSET_TO_POINT_DETACH:-0.0}
PROJ_RECON_GLOBAL_TO_POINT_DETACH=${PROJ_RECON_GLOBAL_TO_POINT_DETACH:-0.0}
PROJ_RECON_TO_GT_USE_POINTMAP_NORM=${PROJ_RECON_TO_GT_USE_POINTMAP_NORM:-false}
PROJ_RECON_TO_GT_HIGH_Z_QUANTILE=${PROJ_RECON_TO_GT_HIGH_Z_QUANTILE:-0.0}
PROJ_RECON_TO_GT_HIGH_Z_MIN_PIXELS=${PROJ_RECON_TO_GT_HIGH_Z_MIN_PIXELS:-16}
PROJ_GRID_GLOBAL_TO_GT=${PROJ_GRID_GLOBAL_TO_GT:-0.0}
PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE:-0.0}
PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_MIN_PIXELS=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_MIN_PIXELS:-16}

REMOTE_PROJECTION_AUX_HIDDEN_DIM=${REMOTE_PROJECTION_AUX_HIDDEN_DIM:-96}
REMOTE_PROJECTION_AUX_SOURCE=${REMOTE_PROJECTION_AUX_SOURCE:-pointmap}
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP:-false}
REMOTE_PROJECTION_AUX_USE_RGB=${REMOTE_PROJECTION_AUX_USE_RGB:-true}
REMOTE_PROJECTION_AUX_USE_COORD=${REMOTE_PROJECTION_AUX_USE_COORD:-true}
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM:-32}
REMOTE_PROJECTION_AUX_POSITIVE_SLOPE=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE:-true}
REMOTE_PROJECTION_AUX_SLOPE_INIT=${REMOTE_PROJECTION_AUX_SLOPE_INIT:-0.1}
REMOTE_PROJECTION_AUX_NUM_BLOCKS=${REMOTE_PROJECTION_AUX_NUM_BLOCKS:-6}
USE_REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL=${USE_REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL:-false}
REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_HIDDEN_SCALE=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_HIDDEN_SCALE:-0.25}
REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_GATE_INIT=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_GATE_INIT:-0.0}
USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-false}
USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS:-false}
USE_PRE_AGGREGATOR_VIEW_TYPE_BIAS=${USE_PRE_AGGREGATOR_VIEW_TYPE_BIAS:-false}
USE_REMOTE_TO_AERIAL_GATED_RESIDUAL=${USE_REMOTE_TO_AERIAL_GATED_RESIDUAL:-false}
REMOTE_TO_AERIAL_RESIDUAL_HIDDEN_SCALE=${REMOTE_TO_AERIAL_RESIDUAL_HIDDEN_SCALE:-0.25}
REMOTE_TO_AERIAL_GATE_INIT=${REMOTE_TO_AERIAL_GATE_INIT:-0.0}
USE_SPLIT_REMOTE_AGGREGATOR=${USE_SPLIT_REMOTE_AGGREGATOR:-false}
REMOTE_TO_AERIAL_LATE_FUSION_TYPE=${REMOTE_TO_AERIAL_LATE_FUSION_TYPE:-none}
REMOTE_TO_AERIAL_LATE_FUSION_HIDDEN_SCALE=${REMOTE_TO_AERIAL_LATE_FUSION_HIDDEN_SCALE:-0.25}
REMOTE_TO_AERIAL_LATE_FUSION_GATE_INIT=${REMOTE_TO_AERIAL_LATE_FUSION_GATE_INIT:-0.0}
REMOTE_TO_AERIAL_CROSS_ATTENTION_HEADS=${REMOTE_TO_AERIAL_CROSS_ATTENTION_HEADS:-8}
REMOTE_TO_AERIAL_MAX_REMOTE_TOKENS=${REMOTE_TO_AERIAL_MAX_REMOTE_TOKENS:-256}
PROTECT_ORDINARY_HEADS_FROM_REMOTE=${PROTECT_ORDINARY_HEADS_FROM_REMOTE:-false}
REMOTE_OUTPUT_HEAD=${REMOTE_OUTPUT_HEAD:-auto}

PRETRAINED_CKPT=${PRETRAINED_CKPT:-/root/autodl-tmp/outputs/checkpoints/vggt/model.pt}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-auto}
RESUME=${RESUME:-false}
WARMSTART_CKPT=${WARMSTART_CKPT:-null}
TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p7_p5b_shared_norm_projection_aux}
LOSS_CONFIG=${LOSS_CONFIG:-vggt_loss_rs_joint_p7_remote_head_projection_aux}

OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux'}

if [ -n "${PRETRAINED_CKPT}" ] && [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "PRETRAINED_CKPT does not exist: ${PRETRAINED_CKPT}" >&2
    exit 1
fi

if [ "${BATCH_SIZE}" -lt "${NUM_VIEWS}" ]; then
    echo "BATCH_SIZE (${BATCH_SIZE}) < NUM_VIEWS (${NUM_VIEWS}); overriding BATCH_SIZE to ${NUM_VIEWS} so validation batch size stays valid." >&2
    BATCH_SIZE=${NUM_VIEWS}
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export HF_HOME=${HF_HOME:-/root/autodl-tmp/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_CLI_ARGS=("${@:2}")
if [ "${LOAD_CUSTOM_CKPT}" = "auto" ]; then
    if [ -n "${PRETRAINED_CKPT}" ]; then
        LOAD_CUSTOM_CKPT=true
    else
        LOAD_CUSTOM_CKPT=false
    fi
fi

if [ "${LOAD_CUSTOM_CKPT}" = "true" ]; then
    LOAD_PRETRAINED_WEIGHTS=false
    echo "LOAD_CUSTOM_CKPT=true"
    echo "Using local VGGT checkpoint: ${PRETRAINED_CKPT}"
elif [ "${LOAD_PRETRAINED_WEIGHTS}" = "true" ]; then
    echo "LOAD_CUSTOM_CKPT=false"
    echo "LOAD_PRETRAINED_WEIGHTS=true"
    echo "Using Hugging Face VGGT initialization path."
else
    echo "LOAD_CUSTOM_CKPT=false"
    echo "LOAD_PRETRAINED_WEIGHTS=false"
    echo "Starting from random initialization."
fi

echo "P7 projection_aux fused into P5B shared-norm joint training"
echo "TRAIN_CITIES=${TRAIN_CITIES} VAL_CITIES=${VAL_CITIES} TEST_CITIES=${TEST_CITIES}"
echo "REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE} LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM} LAMBDA_REMOTE_RAW_PM=${LAMBDA_REMOTE_RAW_PM}"
echo "REMOTE_PM_GRAD weight=${LAMBDA_REMOTE_PM_GRAD} scales=${REMOTE_PM_GRAD_SCALES} channels=${REMOTE_PM_GRAD_CHANNELS}"
echo "REMOTE_MOGE prior grad=${LAMBDA_REMOTE_MOGE_GRAD} edge=${LAMBDA_REMOTE_MOGE_EDGE} pm_height=${LAMBDA_REMOTE_PM_MOGE_HEIGHT} min_weight=${REMOTE_MOGE_PRIOR_MIN_WEIGHT} pm_height_min_weight=${REMOTE_PM_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} pm_height_ground_q=${REMOTE_PM_MOGE_HEIGHT_GROUND_QUANTILE} edge_temp=${REMOTE_MOGE_EDGE_TEMPERATURE} edge_threshold=${REMOTE_MOGE_EDGE_THRESHOLD}"
echo "REMOTE_OVERLAP_PM weight=${LAMBDA_REMOTE_OVERLAP_PM} depth_tol=${REMOTE_OVERLAP_DEPTH_TOL} rel_depth_tol=${REMOTE_OVERLAP_REL_DEPTH_TOL} min_pixels=${REMOTE_OVERLAP_MIN_PIXELS}"
echo "REMOTE_Z_DIST weight=${LAMBDA_REMOTE_Z_DIST}"
echo "REMOTE_HIGH_Z weight=${LAMBDA_REMOTE_HIGH_Z} quantile=${REMOTE_HIGH_Z_QUANTILE} min_pixels=${REMOTE_HIGH_Z_MIN_PIXELS}"
echo "AUX source=${REMOTE_PROJECTION_AUX_SOURCE} hidden=${REMOTE_PROJECTION_AUX_HIDDEN_DIM} image_stem=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM} blocks=${REMOTE_PROJECTION_AUX_NUM_BLOCKS} detach=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP}"
echo "AUX token residual enabled=${USE_REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL} hidden_scale=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_HIDDEN_SCALE} gate_init=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_GATE_INIT}"
echo "PROJ rel_height=${LAMBDA_PROJ_REL_HEIGHT} offset=${LAMBDA_PROJ_OFFSET} slope=${LAMBDA_PROJ_GLOBAL_SLOPE} global_vector=${LAMBDA_PROJ_GLOBAL_VECTOR} consistency=${LAMBDA_PROJ_CONSISTENCY}"
echo "PROJ_GLOBAL_TARGET_FROM_POINTMAP=${PROJ_GLOBAL_TARGET_FROM_POINTMAP} min_rel_height=${PROJ_GLOBAL_TARGET_MIN_REL_HEIGHT} min_pixels=${PROJ_GLOBAL_TARGET_MIN_PIXELS}"
echo "PROJ dense_rel_height=${PROJ_DENSE_REL_HEIGHT_WEIGHT} exclude_hard=${PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK} low_weight=${PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT} low_q=${PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE} min_abs_q=${PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE}"
echo "PROJ dense_global_offset=${PROJ_DENSE_GLOBAL_OFFSET_WEIGHT} low_weight=${PROJ_DENSE_GLOBAL_OFFSET_LOW_WEIGHT} low_q=${PROJ_DENSE_GLOBAL_OFFSET_LOW_QUANTILE}"
echo "PROJ_MOGE prior grad=${LAMBDA_PROJ_MOGE_GRAD} edge=${LAMBDA_PROJ_MOGE_EDGE} height=${LAMBDA_PROJ_MOGE_HEIGHT} min_weight=${PROJ_MOGE_PRIOR_MIN_WEIGHT} height_min_weight=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} height_ground_q=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE} edge_temp=${PROJ_MOGE_EDGE_TEMPERATURE} edge_threshold=${PROJ_MOGE_EDGE_THRESHOLD}"
echo "PROJ_REL_HEIGHT_SCALE_MODE=${PROJ_REL_HEIGHT_SCALE_MODE} PROJ_REL_HEIGHT_SCALE_QUANTILE=${PROJ_REL_HEIGHT_SCALE_QUANTILE} PROJ_OFFSET_SCALE=${PROJ_OFFSET_SCALE}"
echo "PROJ_RECON offset_gt=${PROJ_RECON_OFFSET_TO_GT} global_gt=${PROJ_RECON_GLOBAL_TO_GT} offset_point_detach=${PROJ_RECON_OFFSET_TO_POINT_DETACH} global_point_detach=${PROJ_RECON_GLOBAL_TO_POINT_DETACH}"
echo "PROJ_RECON_TO_GT_USE_POINTMAP_NORM=${PROJ_RECON_TO_GT_USE_POINTMAP_NORM}"
echo "PROJ_RECON_TO_GT_HIGH_Z quantile=${PROJ_RECON_TO_GT_HIGH_Z_QUANTILE} min_pixels=${PROJ_RECON_TO_GT_HIGH_Z_MIN_PIXELS}"
echo "PROJ_GRID_GLOBAL_TO_GT weight=${PROJ_GRID_GLOBAL_TO_GT} high_z_q=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE} min_pixels=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_MIN_PIXELS}"
echo "USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD}"
echo "USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS} USE_PRE_AGGREGATOR_VIEW_TYPE_BIAS=${USE_PRE_AGGREGATOR_VIEW_TYPE_BIAS} REMOTE_OUTPUT_HEAD=${REMOTE_OUTPUT_HEAD}"
echo "REMOTE_TO_AERIAL gated=${USE_REMOTE_TO_AERIAL_GATED_RESIDUAL} hidden_scale=${REMOTE_TO_AERIAL_RESIDUAL_HIDDEN_SCALE} gate_init=${REMOTE_TO_AERIAL_GATE_INIT} split=${USE_SPLIT_REMOTE_AGGREGATOR}"
echo "REMOTE_TO_AERIAL late_fusion=${REMOTE_TO_AERIAL_LATE_FUSION_TYPE} late_hidden_scale=${REMOTE_TO_AERIAL_LATE_FUSION_HIDDEN_SCALE} late_gate_init=${REMOTE_TO_AERIAL_LATE_FUSION_GATE_INIT} heads=${REMOTE_TO_AERIAL_CROSS_ATTENTION_HEADS} max_remote_tokens=${REMOTE_TO_AERIAL_MAX_REMOTE_TOKENS} protect_ordinary=${PROTECT_ORDINARY_HEADS_FROM_REMOTE}"
echo "TRAIN_PARAMS=${TRAIN_PARAMS} LOSS_CONFIG=${LOSS_CONFIG} WARMSTART_CKPT=${WARMSTART_CKPT}"

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_rs_joint_518 \
    dataset.num_workers=${NUM_WORKERS} \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.train.cities=${TRAIN_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.val.cities=${VAL_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.test.cities=${TEST_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_providers=[${RS_PROVIDER}] \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_providers=[${RS_PROVIDER}] \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_providers=[${RS_PROVIDER}] \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_crop_mode=${REMOTE_TRAIN_CROP_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_crop_mode=${REMOTE_VAL_CROP_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_crop_mode=${REMOTE_TEST_CROP_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_crop_scale_range=[${REMOTE_CROP_SCALE_MIN},${REMOTE_CROP_SCALE_MAX}] \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_crop_scale_range=[${REMOTE_CROP_SCALE_MIN},${REMOTE_CROP_SCALE_MAX}] \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_crop_scale_range=[${REMOTE_CROP_SCALE_MIN},${REMOTE_CROP_SCALE_MAX}] \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_image_resize_mode=${REMOTE_IMAGE_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_image_resize_mode=${REMOTE_IMAGE_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_image_resize_mode=${REMOTE_IMAGE_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_label_resize_mode=${REMOTE_LABEL_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_label_resize_mode=${REMOTE_LABEL_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_label_resize_mode=${REMOTE_LABEL_RESIZE_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_num_views=${REMOTE_NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_num_views=${REMOTE_NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_num_views=${REMOTE_NUM_VIEWS} \
    loss=${LOSS_CONFIG} \
    loss.remote_pointmap_loss_weight=${LAMBDA_REMOTE_PM} \
    loss.remote_raw_pointmap_loss_weight=${LAMBDA_REMOTE_RAW_PM} \
    loss.remote_height_loss_weight=${LAMBDA_REMOTE_H} \
    loss.remote_pointmap_top_n_percent=${REMOTE_POINTMAP_TOP_N_PERCENT} \
    loss.remote_pointmap_gradient_loss_weight=${LAMBDA_REMOTE_PM_GRAD} \
    loss.remote_pointmap_gradient_scales=${REMOTE_PM_GRAD_SCALES} \
    loss.remote_pointmap_gradient_channels=${REMOTE_PM_GRAD_CHANNELS} \
    loss.remote_moge_gradient_loss_weight=${LAMBDA_REMOTE_MOGE_GRAD} \
    loss.remote_moge_edge_loss_weight=${LAMBDA_REMOTE_MOGE_EDGE} \
    loss.remote_pointmap_moge_height_loss_weight=${LAMBDA_REMOTE_PM_MOGE_HEIGHT} \
    loss.remote_pointmap_moge_height_prior_min_weight=${REMOTE_PM_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} \
    loss.remote_pointmap_moge_height_ground_quantile=${REMOTE_PM_MOGE_HEIGHT_GROUND_QUANTILE} \
    loss.remote_overlap_pointmap_loss_weight=${LAMBDA_REMOTE_OVERLAP_PM} \
    loss.remote_overlap_depth_tolerance=${REMOTE_OVERLAP_DEPTH_TOL} \
    loss.remote_overlap_relative_depth_tolerance=${REMOTE_OVERLAP_REL_DEPTH_TOL} \
    loss.remote_overlap_min_pixels=${REMOTE_OVERLAP_MIN_PIXELS} \
    loss.remote_overlap_min_depth=${REMOTE_OVERLAP_MIN_DEPTH} \
    loss.remote_moge_prior_min_weight=${REMOTE_MOGE_PRIOR_MIN_WEIGHT} \
    loss.remote_moge_edge_temperature=${REMOTE_MOGE_EDGE_TEMPERATURE} \
    loss.remote_moge_edge_threshold=${REMOTE_MOGE_EDGE_THRESHOLD} \
    loss.remote_pointmap_z_distribution_loss_weight=${LAMBDA_REMOTE_Z_DIST} \
    loss.remote_pointmap_high_z_loss_weight=${LAMBDA_REMOTE_HIGH_Z} \
    loss.remote_pointmap_high_z_quantile=${REMOTE_HIGH_Z_QUANTILE} \
    loss.remote_pointmap_high_z_min_pixels=${REMOTE_HIGH_Z_MIN_PIXELS} \
    loss.remote_pointmap_norm_mode=${REMOTE_POINTMAP_NORM_MODE} \
    loss.scale_remote_loss_by_num_aerial_views=${SCALE_REMOTE_BY_NUM_VIEWS} \
    loss.remote_compare_in_view0_frame=${REMOTE_COMPARE_IN_VIEW0} \
    loss.remote_compare_gt_in_view0_frame_only=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY} \
    loss.remote_detach_pose_for_view0_align=${REMOTE_DETACH_POSE_ALIGN} \
    loss.remote_projection_rel_height_loss_weight=${LAMBDA_PROJ_REL_HEIGHT} \
    loss.remote_projection_offset_loss_weight=${LAMBDA_PROJ_OFFSET} \
    loss.remote_projection_offset_mag_loss_weight=${LAMBDA_PROJ_OFFSET_MAG} \
    loss.remote_projection_offset_dir_loss_weight=${LAMBDA_PROJ_OFFSET_DIR} \
    loss.remote_projection_global_dir_loss_weight=${LAMBDA_PROJ_GLOBAL_DIR} \
    loss.remote_projection_global_slope_loss_weight=${LAMBDA_PROJ_GLOBAL_SLOPE} \
    loss.remote_projection_global_vector_loss_weight=${LAMBDA_PROJ_GLOBAL_VECTOR} \
    loss.remote_projection_consistency_loss_weight=${LAMBDA_PROJ_CONSISTENCY} \
    loss.remote_projection_global_target_from_pointmap=${PROJ_GLOBAL_TARGET_FROM_POINTMAP} \
    loss.remote_projection_global_target_min_rel_height=${PROJ_GLOBAL_TARGET_MIN_REL_HEIGHT} \
    loss.remote_projection_global_target_min_pixels=${PROJ_GLOBAL_TARGET_MIN_PIXELS} \
    loss.remote_projection_moge_gradient_loss_weight=${LAMBDA_PROJ_MOGE_GRAD} \
    loss.remote_projection_moge_edge_loss_weight=${LAMBDA_PROJ_MOGE_EDGE} \
    loss.remote_projection_moge_height_loss_weight=${LAMBDA_PROJ_MOGE_HEIGHT} \
    loss.remote_projection_moge_height_prior_min_weight=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} \
    loss.remote_projection_moge_height_ground_quantile=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE} \
    loss.remote_projection_moge_height_exclude_hard_mask=${PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK} \
    loss.remote_projection_moge_prior_min_weight=${PROJ_MOGE_PRIOR_MIN_WEIGHT} \
    loss.remote_projection_moge_edge_temperature=${PROJ_MOGE_EDGE_TEMPERATURE} \
    loss.remote_projection_moge_edge_threshold=${PROJ_MOGE_EDGE_THRESHOLD} \
    loss.remote_projection_global_dir_from_offset=${PROJ_GLOBAL_DIR_FROM_OFFSET} \
    loss.remote_projection_rel_height_scale=${PROJ_REL_HEIGHT_SCALE} \
    loss.remote_projection_rel_height_scale_mode=${PROJ_REL_HEIGHT_SCALE_MODE} \
    loss.remote_projection_rel_height_scale_quantile=${PROJ_REL_HEIGHT_SCALE_QUANTILE} \
    loss.remote_projection_rel_height_clip=${PROJ_REL_HEIGHT_CLIP} \
    loss.remote_projection_rel_height_min=${PROJ_REL_HEIGHT_MIN} \
    loss.remote_projection_rel_height_use_tilt_mask=${PROJ_REL_HEIGHT_USE_TILT_MASK} \
    loss.remote_projection_rel_height_target_weight=${PROJ_REL_HEIGHT_TARGET_WEIGHT} \
    loss.remote_projection_rel_height_target_weight_gamma=${PROJ_REL_HEIGHT_TARGET_WEIGHT_GAMMA} \
    loss.remote_projection_rel_height_affine_loss_weight=${PROJ_REL_HEIGHT_AFFINE_WEIGHT} \
    loss.remote_projection_rel_height_affine_detach_fit=${PROJ_REL_HEIGHT_AFFINE_DETACH_FIT} \
    loss.remote_projection_rel_height_affine_min_pixels=${PROJ_REL_HEIGHT_AFFINE_MIN_PIXELS} \
    loss.remote_projection_rel_height_balanced_loss_weight=${PROJ_REL_HEIGHT_BALANCED_WEIGHT} \
    loss.remote_projection_rel_height_balanced_quantiles=${PROJ_REL_HEIGHT_BALANCED_QUANTILES} \
    loss.remote_projection_rel_height_contrast_loss_weight=${PROJ_REL_HEIGHT_CONTRAST_WEIGHT} \
    loss.remote_projection_rel_height_bucket_mean_loss_weight=${PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT} \
    loss.remote_projection_rel_height_low_overpred_loss_weight=${PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT} \
    loss.remote_projection_rel_height_low_overpred_start_epoch=${PROJ_REL_HEIGHT_LOW_OVERPRED_START_EPOCH} \
    loss.remote_projection_rel_height_low_overpred_ramp_epochs=${PROJ_REL_HEIGHT_LOW_OVERPRED_RAMP_EPOCHS} \
    loss.remote_projection_dense_rel_height_loss_weight=${PROJ_DENSE_REL_HEIGHT_WEIGHT} \
    loss.remote_projection_dense_rel_height_exclude_hard_mask=${PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK} \
    loss.remote_projection_dense_rel_height_low_weight=${PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT} \
    loss.remote_projection_dense_rel_height_low_quantile=${PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE} \
    loss.remote_projection_dense_rel_height_min_abs_quantile=${PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE} \
    loss.remote_projection_dense_global_offset_loss_weight=${PROJ_DENSE_GLOBAL_OFFSET_WEIGHT} \
    loss.remote_projection_dense_global_offset_low_weight=${PROJ_DENSE_GLOBAL_OFFSET_LOW_WEIGHT} \
    loss.remote_projection_dense_global_offset_low_quantile=${PROJ_DENSE_GLOBAL_OFFSET_LOW_QUANTILE} \
    loss.remote_projection_offset_scale=${PROJ_OFFSET_SCALE} \
    loss.remote_projection_offset_target_weight=${PROJ_OFFSET_TARGET_WEIGHT} \
    loss.remote_projection_offset_target_weight_gamma=${PROJ_OFFSET_TARGET_WEIGHT_GAMMA} \
    loss.remote_projection_offset_balanced_loss_weight=${PROJ_OFFSET_BALANCED_WEIGHT} \
    loss.remote_projection_offset_balanced_quantiles=${PROJ_OFFSET_BALANCED_QUANTILES} \
    loss.remote_projection_offset_contrast_loss_weight=${PROJ_OFFSET_CONTRAST_WEIGHT} \
    loss.remote_projection_offset_bucket_mean_loss_weight=${PROJ_OFFSET_BUCKET_MEAN_WEIGHT} \
    loss.remote_projection_offset_low_overpred_loss_weight=${PROJ_OFFSET_LOW_OVERPRED_WEIGHT} \
    loss.remote_projection_offset_low_overpred_start_epoch=${PROJ_OFFSET_LOW_OVERPRED_START_EPOCH} \
    loss.remote_projection_offset_low_overpred_ramp_epochs=${PROJ_OFFSET_LOW_OVERPRED_RAMP_EPOCHS} \
    loss.remote_projection_consistency_use_loss_space=${PROJ_CONSISTENCY_USE_LOSS_SPACE} \
    loss.remote_projection_offset_use_tilt_mask=${PROJ_OFFSET_USE_TILT_MASK} \
    loss.remote_projection_consistency_use_tilt_mask=${PROJ_CONSISTENCY_USE_TILT_MASK} \
    loss.remote_projection_offset_min_magnitude=${PROJ_OFFSET_MIN_MAGNITUDE} \
    loss.remote_projection_consistency_min_magnitude=${PROJ_CONSISTENCY_MIN_MAGNITUDE} \
    loss.remote_projection_reconstruct_offset_to_gt_loss_weight=${PROJ_RECON_OFFSET_TO_GT} \
    loss.remote_projection_reconstruct_global_to_gt_loss_weight=${PROJ_RECON_GLOBAL_TO_GT} \
    loss.remote_projection_reconstruct_offset_to_point_detach_loss_weight=${PROJ_RECON_OFFSET_TO_POINT_DETACH} \
    loss.remote_projection_reconstruct_global_to_point_detach_loss_weight=${PROJ_RECON_GLOBAL_TO_POINT_DETACH} \
    loss.remote_projection_reconstruct_to_gt_use_pointmap_norm=${PROJ_RECON_TO_GT_USE_POINTMAP_NORM} \
    loss.remote_projection_reconstruct_to_gt_high_z_quantile=${PROJ_RECON_TO_GT_HIGH_Z_QUANTILE} \
    loss.remote_projection_reconstruct_to_gt_high_z_min_pixels=${PROJ_RECON_TO_GT_HIGH_Z_MIN_PIXELS} \
    loss.remote_projection_grid_global_to_gt_loss_weight=${PROJ_GRID_GLOBAL_TO_GT} \
    loss.remote_projection_grid_global_to_gt_high_z_quantile=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE} \
    loss.remote_projection_grid_global_to_gt_high_z_min_pixels=${PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_MIN_PIXELS} \
    model=vggt \
    model.model_config.load_pretrained_weights=${LOAD_PRETRAINED_WEIGHTS} \
    model.model_config.load_custom_ckpt=${LOAD_CUSTOM_CKPT} \
    model.model_config.custom_ckpt_path=${PRETRAINED_CKPT} \
    model.model_config.use_point_head_for_remote=true \
    model.model_config.use_view_type_bias=${USE_VIEW_TYPE_BIAS} \
    model.model_config.use_pre_aggregator_view_type_bias=${USE_PRE_AGGREGATOR_VIEW_TYPE_BIAS} \
    model.model_config.use_remote_to_aerial_gated_residual=${USE_REMOTE_TO_AERIAL_GATED_RESIDUAL} \
    model.model_config.remote_to_aerial_residual_hidden_scale=${REMOTE_TO_AERIAL_RESIDUAL_HIDDEN_SCALE} \
    model.model_config.remote_to_aerial_gate_init=${REMOTE_TO_AERIAL_GATE_INIT} \
    model.model_config.use_split_remote_aggregator=${USE_SPLIT_REMOTE_AGGREGATOR} \
    model.model_config.remote_to_aerial_late_fusion_type=${REMOTE_TO_AERIAL_LATE_FUSION_TYPE} \
    model.model_config.remote_to_aerial_late_fusion_hidden_scale=${REMOTE_TO_AERIAL_LATE_FUSION_HIDDEN_SCALE} \
    model.model_config.remote_to_aerial_late_fusion_gate_init=${REMOTE_TO_AERIAL_LATE_FUSION_GATE_INIT} \
    model.model_config.remote_to_aerial_cross_attention_heads=${REMOTE_TO_AERIAL_CROSS_ATTENTION_HEADS} \
    model.model_config.remote_to_aerial_max_remote_tokens=${REMOTE_TO_AERIAL_MAX_REMOTE_TOKENS} \
    model.model_config.protect_ordinary_heads_from_remote=${PROTECT_ORDINARY_HEADS_FROM_REMOTE} \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=${REMOTE_OUTPUT_HEAD} \
    model.model_config.use_remote_private_point_head=${USE_REMOTE_PRIVATE_POINT_HEAD} \
    model.model_config.output_point_head_for_consistency=false \
    model.model_config.use_remote_projection_aux_head=true \
    model.model_config.remote_projection_aux_hidden_dim=${REMOTE_PROJECTION_AUX_HIDDEN_DIM} \
    model.model_config.remote_projection_aux_source=${REMOTE_PROJECTION_AUX_SOURCE} \
    model.model_config.remote_projection_aux_detach_pointmap=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP} \
    model.model_config.remote_projection_aux_use_rgb=${REMOTE_PROJECTION_AUX_USE_RGB} \
    model.model_config.remote_projection_aux_use_coord=${REMOTE_PROJECTION_AUX_USE_COORD} \
    model.model_config.remote_projection_aux_image_stem_dim=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM} \
    model.model_config.remote_projection_aux_positive_slope=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE} \
    model.model_config.remote_projection_aux_slope_init=${REMOTE_PROJECTION_AUX_SLOPE_INIT} \
    model.model_config.remote_projection_aux_num_blocks=${REMOTE_PROJECTION_AUX_NUM_BLOCKS} \
    model.model_config.use_remote_projection_aux_token_residual=${USE_REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL} \
    model.model_config.remote_projection_aux_token_residual_hidden_scale=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_HIDDEN_SCALE} \
    model.model_config.remote_projection_aux_token_residual_gate_init=${REMOTE_PROJECTION_AUX_TOKEN_RESIDUAL_GATE_INIT} \
    train_params=${TRAIN_PARAMS} \
    train_params.epochs=${EPOCHS} \
    train_params.warmup_epochs=${WARMUP_EPOCHS} \
    train_params.eval_freq=${EVAL_FREQ} \
    train_params.save_freq=${SAVE_FREQ} \
    train_params.keep_freq=${KEEP_FREQ} \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=${PRINT_FREQ} \
    train_params.resume=${RESUME} \
    train_params.warmstart_ckpt=${WARMSTART_CKPT} \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]}"
