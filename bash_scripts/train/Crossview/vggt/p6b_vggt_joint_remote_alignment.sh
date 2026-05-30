#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-2}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
NUM_WORKERS=${NUM_WORKERS:-6}
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-0}
KEEP_FREQ=${KEEP_FREQ:-0}
PRINT_FREQ=${PRINT_FREQ:-20}
DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH:-false}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-true}
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

# P6B target: remote view point output is supervised in ordinary view0 frame.
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-0.3}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
LAMBDA_BRANCH_CONSISTENCY=${LAMBDA_BRANCH_CONSISTENCY:-0.0}
BRANCH_CONSISTENCY_NORM_MODE=${BRANCH_CONSISTENCY_NORM_MODE:-null}
BRANCH_CONSISTENCY_DETACH_DEPTH=${BRANCH_CONSISTENCY_DETACH_DEPTH:-true}
REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE:-aerial_avg_dis}
SCALE_REMOTE_BY_NUM_VIEWS=${SCALE_REMOTE_BY_NUM_VIEWS:-true}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-false}
REMOTE_COMPARE_GT_IN_VIEW0_ONLY=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}

P6B_VARIANT=${P6B_VARIANT:-private_head}
USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD:-true}
USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS:-false}
POINT_HEAD_LR=${POINT_HEAD_LR:-0}
REMOTE_POINT_HEAD_LR=${REMOTE_POINT_HEAD_LR:-2e-05}
VIEW_TYPE_LR=${VIEW_TYPE_LR:-0}
TRAIN_PARAMS=${TRAIN_PARAMS:-vggt_p6b_joint_remote_alignment_private_head}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p6b_vggt_joint_remote_alignment_private_head_w03}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/vggt/'${EXPERIMENT_NAME}}

PRETRAINED_CKPT=${PRETRAINED_CKPT:-/root/autodl-tmp/outputs/checkpoints/vggt/model.pt}
LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
LOAD_CUSTOM_CKPT=${LOAD_CUSTOM_CKPT:-auto}
RESUME=${RESUME:-false}

if [ -n "${PRETRAINED_CKPT}" ] && [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "PRETRAINED_CKPT does not exist: ${PRETRAINED_CKPT}" >&2
    exit 1
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
else
    echo "LOAD_CUSTOM_CKPT=false"
    echo "LOAD_PRETRAINED_WEIGHTS=false"
fi

echo "P6B_VARIANT=${P6B_VARIANT}"
echo "NUM_GPUS=${NUM_GPUS} CUDA_DEVICES=${CUDA_DEVICES} BATCH_SIZE=${BATCH_SIZE} NUM_VIEWS=${NUM_VIEWS}"
echo "LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM} REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE}"
echo "USE_REMOTE_PRIVATE_POINT_HEAD=${USE_REMOTE_PRIVATE_POINT_HEAD} USE_VIEW_TYPE_BIAS=${USE_VIEW_TYPE_BIAS}"
echo "TRAIN_PARAMS=${TRAIN_PARAMS}"
echo "POINT_HEAD_LR=${POINT_HEAD_LR} REMOTE_POINT_HEAD_LR=${REMOTE_POINT_HEAD_LR} VIEW_TYPE_LR=${VIEW_TYPE_LR}"
echo "DDP_STATIC_GRAPH=${DDP_STATIC_GRAPH} DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"

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
    model.pretrained=null \
    model.model_config.load_pretrained_weights=${LOAD_PRETRAINED_WEIGHTS} \
    model.model_config.load_custom_ckpt=${LOAD_CUSTOM_CKPT} \
    model.model_config.custom_ckpt_path=${PRETRAINED_CKPT} \
    model.model_config.use_point_head_for_remote=true \
    model.model_config.use_split_remote_aggregator=false \
    model.model_config.protect_ordinary_heads_from_remote=false \
    model.model_config.use_view_type_bias=${USE_VIEW_TYPE_BIAS} \
    model.model_config.use_pre_aggregator_view_type_bias=false \
    model.model_config.use_remote_to_aerial_gated_residual=false \
    model.model_config.remote_to_aerial_late_fusion_type=none \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=point \
    model.model_config.use_remote_private_point_head=${USE_REMOTE_PRIVATE_POINT_HEAD} \
    model.model_config.output_point_head_for_consistency=false \
    train_params=${TRAIN_PARAMS} \
    train_params.epochs=${EPOCHS} \
    train_params.warmup_epochs=${WARMUP_EPOCHS} \
    train_params.eval_freq=${EVAL_FREQ} \
    train_params.save_freq=${SAVE_FREQ} \
    train_params.keep_freq=${KEEP_FREQ} \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=${PRINT_FREQ} \
    train_params.ddp_static_graph=${DDP_STATIC_GRAPH} \
    train_params.ddp_find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS} \
    train_params.resume=${RESUME} \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]}"
