#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-2}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
MASTER_PORT=${MASTER_PORT:-29516}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
KEEP_FREQ=${KEEP_FREQ:-10}
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
TRAIN_CITIES=${TRAIN_CITIES:-[chicago]}
VAL_CITIES=${VAL_CITIES:-[chicago]}
TEST_CITIES=${TEST_CITIES:-[chicago]}

# P5B shared-normalization reconstruction losses.
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-4.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
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
LAMBDA_PROJ_CONSISTENCY=${LAMBDA_PROJ_CONSISTENCY:-0.0}
PROJ_GLOBAL_DIR_FROM_OFFSET=${PROJ_GLOBAL_DIR_FROM_OFFSET:-true}
PROJ_REL_HEIGHT_SCALE=${PROJ_REL_HEIGHT_SCALE:-1.0}
PROJ_REL_HEIGHT_SCALE_MODE=${PROJ_REL_HEIGHT_SCALE_MODE:-gt_pointmap_norm}
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

REMOTE_PROJECTION_AUX_HIDDEN_DIM=${REMOTE_PROJECTION_AUX_HIDDEN_DIM:-96}
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP:-false}
REMOTE_PROJECTION_AUX_USE_RGB=${REMOTE_PROJECTION_AUX_USE_RGB:-true}
REMOTE_PROJECTION_AUX_USE_COORD=${REMOTE_PROJECTION_AUX_USE_COORD:-true}
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM:-32}
REMOTE_PROJECTION_AUX_POSITIVE_SLOPE=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE:-true}
REMOTE_PROJECTION_AUX_SLOPE_INIT=${REMOTE_PROJECTION_AUX_SLOPE_INIT:-0.1}
REMOTE_PROJECTION_AUX_NUM_BLOCKS=${REMOTE_PROJECTION_AUX_NUM_BLOCKS:-6}
USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-false}

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
echo "REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE} LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM}"
echo "AUX hidden=${REMOTE_PROJECTION_AUX_HIDDEN_DIM} image_stem=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM} blocks=${REMOTE_PROJECTION_AUX_NUM_BLOCKS} detach=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP}"
echo "PROJ rel_height=${LAMBDA_PROJ_REL_HEIGHT} offset=${LAMBDA_PROJ_OFFSET} slope=${LAMBDA_PROJ_GLOBAL_SLOPE} consistency=${LAMBDA_PROJ_CONSISTENCY}"
echo "PROJ_REL_HEIGHT_SCALE_MODE=${PROJ_REL_HEIGHT_SCALE_MODE} PROJ_OFFSET_SCALE=${PROJ_OFFSET_SCALE}"
echo "USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD}"
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
    loss.remote_height_loss_weight=${LAMBDA_REMOTE_H} \
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
    loss.remote_projection_consistency_loss_weight=${LAMBDA_PROJ_CONSISTENCY} \
    loss.remote_projection_global_dir_from_offset=${PROJ_GLOBAL_DIR_FROM_OFFSET} \
    loss.remote_projection_rel_height_scale=${PROJ_REL_HEIGHT_SCALE} \
    loss.remote_projection_rel_height_scale_mode=${PROJ_REL_HEIGHT_SCALE_MODE} \
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
    model=vggt \
    model.model_config.load_pretrained_weights=${LOAD_PRETRAINED_WEIGHTS} \
    model.model_config.load_custom_ckpt=${LOAD_CUSTOM_CKPT} \
    model.model_config.custom_ckpt_path=${PRETRAINED_CKPT} \
    model.model_config.use_point_head_for_remote=true \
    model.model_config.use_view_type_bias=false \
    model.model_config.use_pre_aggregator_view_type_bias=false \
    model.model_config.use_remote_to_aerial_gated_residual=false \
    model.model_config.use_split_remote_aggregator=false \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=auto \
    model.model_config.use_remote_private_point_head=${USE_REMOTE_PRIVATE_POINT_HEAD} \
    model.model_config.output_point_head_for_consistency=false \
    model.model_config.use_remote_projection_aux_head=true \
    model.model_config.remote_projection_aux_hidden_dim=${REMOTE_PROJECTION_AUX_HIDDEN_DIM} \
    model.model_config.remote_projection_aux_detach_pointmap=${REMOTE_PROJECTION_AUX_DETACH_POINTMAP} \
    model.model_config.remote_projection_aux_use_rgb=${REMOTE_PROJECTION_AUX_USE_RGB} \
    model.model_config.remote_projection_aux_use_coord=${REMOTE_PROJECTION_AUX_USE_COORD} \
    model.model_config.remote_projection_aux_image_stem_dim=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM} \
    model.model_config.remote_projection_aux_positive_slope=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE} \
    model.model_config.remote_projection_aux_slope_init=${REMOTE_PROJECTION_AUX_SLOPE_INIT} \
    model.model_config.remote_projection_aux_num_blocks=${REMOTE_PROJECTION_AUX_NUM_BLOCKS} \
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
