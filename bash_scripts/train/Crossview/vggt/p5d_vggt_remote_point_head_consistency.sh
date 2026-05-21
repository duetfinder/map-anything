#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-2}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
KEEP_FREQ=${KEEP_FREQ:-5}
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
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-4.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
LAMBDA_BRANCH_CONSISTENCY=${LAMBDA_BRANCH_CONSISTENCY:-0.2}
BRANCH_CONSISTENCY_NORM_MODE=${BRANCH_CONSISTENCY_NORM_MODE:-null}
BRANCH_CONSISTENCY_DETACH_DEPTH=${BRANCH_CONSISTENCY_DETACH_DEPTH:-true}
REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE:-aerial_avg_dis}
SCALE_REMOTE_BY_NUM_VIEWS=${SCALE_REMOTE_BY_NUM_VIEWS:-true}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-false}
REMOTE_COMPARE_GT_IN_VIEW0_ONLY=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-/root/autodl-tmp/outputs/checkpoints/vggt/model.pt}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-auto}
RESUME=${RESUME:-false}

OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt/p5d_vggt_remote_point_head_consistency'}

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

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_rs_joint_518 \
    dataset.num_workers=${NUM_WORKERS} \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.train.cities=[chicago] \
    dataset.vigor_chicago_joint_rs_aerial.val.cities=[chicago] \
    dataset.vigor_chicago_joint_rs_aerial.test.cities=[chicago] \
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
    loss=vggt_loss_rs_joint_p5d \
    loss.remote_pointmap_loss_weight=${LAMBDA_REMOTE_PM} \
    loss.remote_height_loss_weight=${LAMBDA_REMOTE_H} \
    loss.remote_pointmap_norm_mode=${REMOTE_POINTMAP_NORM_MODE} \
    loss.scale_remote_loss_by_num_aerial_views=${SCALE_REMOTE_BY_NUM_VIEWS} \
    loss.remote_compare_in_view0_frame=${REMOTE_COMPARE_IN_VIEW0} \
    loss.remote_compare_gt_in_view0_frame_only=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY} \
    loss.remote_detach_pose_for_view0_align=${REMOTE_DETACH_POSE_ALIGN} \
    loss.branch_consistency_loss_weight=${LAMBDA_BRANCH_CONSISTENCY} \
    loss.branch_consistency_norm_mode=${BRANCH_CONSISTENCY_NORM_MODE} \
    loss.branch_consistency_detach_depth_target=${BRANCH_CONSISTENCY_DETACH_DEPTH} \
    model=vggt \
    model.model_config.load_pretrained_weights=${LOAD_PRETRAINED_WEIGHTS} \
    model.model_config.load_custom_ckpt=${LOAD_CUSTOM_CKPT} \
    model.model_config.custom_ckpt_path=${PRETRAINED_CKPT} \
    model.model_config.use_point_head_for_remote=true \
    model.model_config.use_view_type_bias=false \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=point \
    model.model_config.use_remote_private_point_head=true \
    model.model_config.output_point_head_for_consistency=true \
    train_params=vggt_finetune \
    train_params.epochs=${EPOCHS} \
    train_params.warmup_epochs=${WARMUP_EPOCHS} \
    train_params.eval_freq=${EVAL_FREQ} \
    train_params.save_freq=${SAVE_FREQ} \
    train_params.keep_freq=${KEEP_FREQ} \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=${PRINT_FREQ} \
    train_params.resume=${RESUME} \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]}"
