#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-4}}
MASTER_PORT=${MASTER_PORT:-29500}
CUDA_DEVICES=${CUDA_DEVICES:-1,2,3,4}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-40}
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
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-0.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
LAMBDA_BRANCH_CONSISTENCY=${LAMBDA_BRANCH_CONSISTENCY:-0.0}
BRANCH_CONSISTENCY_NORM_MODE=${BRANCH_CONSISTENCY_NORM_MODE:-null}
BRANCH_CONSISTENCY_DETACH_DEPTH=${BRANCH_CONSISTENCY_DETACH_DEPTH:-true}
REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE:-aerial_avg_dis}
SCALE_REMOTE_BY_NUM_VIEWS=${SCALE_REMOTE_BY_NUM_VIEWS:-true}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-false}
REMOTE_COMPARE_GT_IN_VIEW0_ONLY=${REMOTE_COMPARE_GT_IN_VIEW0_ONLY:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}
BASE_CKPT=${BASE_CKPT:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth}
RESUME=${RESUME:-false}
FUSION_TYPE=${FUSION_TYPE:-cross_attention}
LATE_GATE_INIT=${LATE_GATE_INIT:-0.0}
LATE_HIDDEN_SCALE=${LATE_HIDDEN_SCALE:-0.25}
CROSS_ATTENTION_HEADS=${CROSS_ATTENTION_HEADS:-8}
MAX_REMOTE_TOKENS=${MAX_REMOTE_TOKENS:-256}
PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS:-true}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-p5h_vggt_p5e_base_${FUSION_TYPE}_protected}
OUTPUT_DIR=${OUTPUT_DIR:-"\${root_experiments_dir}/mapanything/training/Crossview/vggt/${EXPERIMENT_NAME}"}

if [ -n "${BASE_CKPT}" ] && [ ! -f "${BASE_CKPT}" ]; then
    echo "BASE_CKPT does not exist: ${BASE_CKPT}" >&2
    exit 1
fi

if [ "${BATCH_SIZE}" -lt "${NUM_VIEWS}" ]; then
    echo "BATCH_SIZE (${BATCH_SIZE}) < NUM_VIEWS (${NUM_VIEWS}); overriding BATCH_SIZE to ${NUM_VIEWS}." >&2
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

echo "P5h p5e-base split remote late fusion"
echo "BASE_CKPT=${BASE_CKPT}"
echo "FUSION_TYPE=${FUSION_TYPE}"
echo "PROTECT_ORDINARY_HEADS=${PROTECT_ORDINARY_HEADS}"
echo "REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE}"
echo "REMOTE_NUM_VIEWS=${REMOTE_NUM_VIEWS}"
echo "MAX_REMOTE_TOKENS=${MAX_REMOTE_TOKENS}"

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}" \
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
    model.pretrained=${BASE_CKPT} \
    model.model_config.load_pretrained_weights=false \
    model.model_config.load_custom_ckpt=false \
    model.model_config.use_point_head_for_remote=true \
    model.model_config.use_view_type_bias=true \
    model.model_config.use_pre_aggregator_view_type_bias=false \
    model.model_config.use_remote_to_aerial_gated_residual=false \
    model.model_config.use_split_remote_aggregator=true \
    model.model_config.remote_to_aerial_late_fusion_type=${FUSION_TYPE} \
    model.model_config.remote_to_aerial_late_fusion_hidden_scale=${LATE_HIDDEN_SCALE} \
    model.model_config.remote_to_aerial_late_fusion_gate_init=${LATE_GATE_INIT} \
    model.model_config.remote_to_aerial_cross_attention_heads=${CROSS_ATTENTION_HEADS} \
    model.model_config.remote_to_aerial_max_remote_tokens=${MAX_REMOTE_TOKENS} \
    model.model_config.protect_ordinary_heads_from_remote=${PROTECT_ORDINARY_HEADS} \
    model.model_config.ordinary_output_head=depth \
    model.model_config.remote_output_head=point \
    model.model_config.use_remote_private_point_head=true \
    model.model_config.output_point_head_for_consistency=true \
    train_params=vggt_p5h_frozen_late_fusion \
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
