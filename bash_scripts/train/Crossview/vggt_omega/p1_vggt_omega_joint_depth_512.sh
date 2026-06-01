#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-4}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
KEEP_FREQ=${KEEP_FREQ:-10}
PRINT_FREQ=${PRINT_FREQ:-20}
LOSS_CONFIG=${LOSS_CONFIG:-vggt_loss_rs_joint}
LR=${LR:-1e-05}
MIN_LR=${MIN_LR:-1e-07}
SCHEDULE_TYPE=${SCHEDULE_TYPE:-linear_warmup_half_cycle_cosine_decay}
TRAIN_CITIES=${TRAIN_CITIES:-chicago}
VAL_CITIES=${VAL_CITIES:-${TRAIN_CITIES}}
TEST_CITIES=${TEST_CITIES:-${VAL_CITIES}}
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
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-4.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
SCALE_REMOTE_BY_NUM_VIEWS=${SCALE_REMOTE_BY_NUM_VIEWS:-true}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-false}
REMOTE_COMPARE_GT_IN_VIEW0_ONLY=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-auto}
RESUME=${RESUME:-false}

OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512'}

if [ "${LOAD_CUSTOM_CKPT}" != "false" ] && [ -n "${PRETRAINED_CKPT}" ] && [ ! -f "${PRETRAINED_CKPT}" ]; then
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
    echo "LOAD_CUSTOM_CKPT=true"
    echo "Using local VGGT-Omega checkpoint: ${PRETRAINED_CKPT}"
else
    echo "LOAD_CUSTOM_CKPT=false"
    echo "Starting VGGT-Omega from random initialization."
fi

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_rs_joint_512 \
    dataset.num_workers=${NUM_WORKERS} \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.train.cities=[${TRAIN_CITIES}] \
    dataset.vigor_chicago_joint_rs_aerial.val.cities=[${VAL_CITIES}] \
    dataset.vigor_chicago_joint_rs_aerial.test.cities=[${TEST_CITIES}] \
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
    loss.scale_remote_loss_by_num_aerial_views=${SCALE_REMOTE_BY_NUM_VIEWS} \
    loss.remote_compare_in_view0_frame=${REMOTE_COMPARE_IN_VIEW0} \
    loss.remote_compare_gt_in_view0_frame_only=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY} \
    loss.remote_detach_pose_for_view0_align=${REMOTE_DETACH_POSE_ALIGN} \
    model=vggt_omega \
    model.model_config.load_custom_ckpt=${LOAD_CUSTOM_CKPT} \
    model.model_config.custom_ckpt_path=${PRETRAINED_CKPT} \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=depth \
    train_params=vggt_omega_finetune \
    train_params.epochs=${EPOCHS} \
    train_params.lr=${LR} \
    train_params.min_lr=${MIN_LR} \
    train_params.warmup_epochs=${WARMUP_EPOCHS} \
    train_params.schedule_type=${SCHEDULE_TYPE} \
    train_params.eval_freq=${EVAL_FREQ} \
    train_params.save_freq=${SAVE_FREQ} \
    train_params.keep_freq=${KEEP_FREQ} \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=${PRINT_FREQ} \
    train_params.resume=${RESUME} \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]}"
