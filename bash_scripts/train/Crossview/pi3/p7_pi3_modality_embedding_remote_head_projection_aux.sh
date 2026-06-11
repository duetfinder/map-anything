#!/bin/bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-${1:-4}}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29524}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-30}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
EVAL_FREQ=${EVAL_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-0}
KEEP_FREQ=${KEEP_FREQ:-0}
PRINT_FREQ=${PRINT_FREQ:-20}
TRAIN_LR=${TRAIN_LR:-1e-5}
TRAIN_MIN_LR=${TRAIN_MIN_LR:-1e-7}
RS_PROVIDER=${RS_PROVIDER:-Google_Satellite,Bing_Satellite}
REMOTE_PROVIDER_SAMPLING_MODE=${REMOTE_PROVIDER_SAMPLING_MODE:-random}
REMOTE_TRAIN_CROP_MODE=${REMOTE_TRAIN_CROP_MODE:-random_scale_offset}
REMOTE_VAL_CROP_MODE=${REMOTE_VAL_CROP_MODE:-random_scale_offset}
REMOTE_TEST_CROP_MODE=${REMOTE_TEST_CROP_MODE:-none}
REMOTE_CROP_SCALE_MIN=${REMOTE_CROP_SCALE_MIN:-0.6}
REMOTE_CROP_SCALE_MAX=${REMOTE_CROP_SCALE_MAX:-1.0}
REMOTE_IMAGE_RESIZE_MODE=${REMOTE_IMAGE_RESIZE_MODE:-nearest}
REMOTE_LABEL_RESIZE_MODE=${REMOTE_LABEL_RESIZE_MODE:-nearest}
TRAIN_CITIES=${TRAIN_CITIES:-[chicago,newyork,sanfrancisco,seattle]}
VAL_CITIES=${VAL_CITIES:-[newyork]}
TEST_CITIES=${TEST_CITIES:-[newyork]}
TRAIN_SCENE_LIST_PATH=${TRAIN_SCENE_LIST_PATH:-null}
TRAIN_OVERFIT_NUM_SETS=${TRAIN_OVERFIT_NUM_SETS:-80}
VAL_OVERFIT_NUM_SETS=${VAL_OVERFIT_NUM_SETS:-16}
TEST_OVERFIT_NUM_SETS=${TEST_OVERFIT_NUM_SETS:-16}

LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-6.0}
LAMBDA_REMOTE_RAW_PM=${LAMBDA_REMOTE_RAW_PM:-0.0}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.0}
REMOTE_COMPARE_IN_VIEW0=${REMOTE_COMPARE_IN_VIEW0:-true}
REMOTE_DETACH_POSE_ALIGN=${REMOTE_DETACH_POSE_ALIGN:-false}
REMOTE_POINTMAP_NORM_MODE=${REMOTE_POINTMAP_NORM_MODE:-avg_dis}

LAMBDA_PROJ_REL_HEIGHT=${LAMBDA_PROJ_REL_HEIGHT:-0.35}
LAMBDA_PROJ_OFFSET=${LAMBDA_PROJ_OFFSET:-0.75}
LAMBDA_PROJ_GLOBAL_DIR=${LAMBDA_PROJ_GLOBAL_DIR:-0.0}
LAMBDA_PROJ_GLOBAL_SLOPE=${LAMBDA_PROJ_GLOBAL_SLOPE:-0.05}
LAMBDA_PROJ_CONSISTENCY=${LAMBDA_PROJ_CONSISTENCY:-0.0}
PROJ_REL_HEIGHT_SCALE=${PROJ_REL_HEIGHT_SCALE:-1.0}
PROJ_REL_HEIGHT_SCALE_MODE=${PROJ_REL_HEIGHT_SCALE_MODE:-fixed}
PROJ_REL_HEIGHT_SCALE_QUANTILE=${PROJ_REL_HEIGHT_SCALE_QUANTILE:-0.9}
PROJ_REL_HEIGHT_CLIP=${PROJ_REL_HEIGHT_CLIP:-0.0}
PROJ_REL_HEIGHT_AFFINE_WEIGHT=${PROJ_REL_HEIGHT_AFFINE_WEIGHT:-0.0}
PROJ_REL_HEIGHT_BALANCED_WEIGHT=${PROJ_REL_HEIGHT_BALANCED_WEIGHT:-0.0}
PROJ_REL_HEIGHT_BALANCED_QUANTILES=${PROJ_REL_HEIGHT_BALANCED_QUANTILES:-[0.5,0.8]}
PROJ_OFFSET_SCALE=${PROJ_OFFSET_SCALE:-1.0}
LAMBDA_PROJ_MOGE_GRAD=${LAMBDA_PROJ_MOGE_GRAD:-0.02}
LAMBDA_PROJ_MOGE_EDGE=${LAMBDA_PROJ_MOGE_EDGE:-0.005}
LAMBDA_PROJ_MOGE_HEIGHT=${LAMBDA_PROJ_MOGE_HEIGHT:-0.0}
PROJ_MOGE_PRIOR_MIN_WEIGHT=${PROJ_MOGE_PRIOR_MIN_WEIGHT:-0.03}
PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT:-0.03}
PROJ_MOGE_HEIGHT_GROUND_QUANTILE=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE:-0.2}
PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK=${PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK:-true}
PROJ_MOGE_EDGE_TEMPERATURE=${PROJ_MOGE_EDGE_TEMPERATURE:-10.0}
PROJ_MOGE_EDGE_THRESHOLD=${PROJ_MOGE_EDGE_THRESHOLD:-0.5}

REMOTE_PROJECTION_AUX_HIDDEN_DIM=${REMOTE_PROJECTION_AUX_HIDDEN_DIM:-96}
REMOTE_PROJECTION_AUX_USE_RGB=${REMOTE_PROJECTION_AUX_USE_RGB:-true}
REMOTE_PROJECTION_AUX_USE_COORD=${REMOTE_PROJECTION_AUX_USE_COORD:-true}
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM:-32}
REMOTE_PROJECTION_AUX_POSITIVE_SLOPE=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE:-true}
REMOTE_PROJECTION_AUX_SLOPE_INIT=${REMOTE_PROJECTION_AUX_SLOPE_INIT:-0.1}
REMOTE_PROJECTION_AUX_NUM_BLOCKS=${REMOTE_PROJECTION_AUX_NUM_BLOCKS:-6}
REMOTE_PROJECTION_AUX_REL_HEIGHT_OUTPUT_SCALE=${REMOTE_PROJECTION_AUX_REL_HEIGHT_OUTPUT_SCALE:-1.0}
REMOTE_PROJECTION_AUX_OFFSET_OUTPUT_SCALE=${REMOTE_PROJECTION_AUX_OFFSET_OUTPUT_SCALE:-1.0}

LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS:-false}
WARMSTART_CKPT=${WARMSTART_CKPT:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_modality_embedding_remote_head/checkpoint-final.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/Crossview/pi3/p7_pi3_remote_head_projection_aux'}

if [ "${WARMSTART_CKPT}" != "null" ] && [ ! -f "${WARMSTART_CKPT}" ]; then
    echo "WARMSTART_CKPT does not exist: ${WARMSTART_CKPT}" >&2
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

echo "P7 PI3 modality-embedding remote-head projection-aux training"
echo "TRAIN_CITIES=${TRAIN_CITIES} VAL_CITIES=${VAL_CITIES} TRAIN_SCENE_LIST_PATH=${TRAIN_SCENE_LIST_PATH}"
echo "REMOTE_PM=${LAMBDA_REMOTE_PM} REMOTE_RAW_PM=${LAMBDA_REMOTE_RAW_PM} REMOTE_H=${LAMBDA_REMOTE_H}"
echo "PROJ rel_height=${LAMBDA_PROJ_REL_HEIGHT} offset=${LAMBDA_PROJ_OFFSET} slope=${LAMBDA_PROJ_GLOBAL_SLOPE}"
echo "PROJ_REL_HEIGHT scale_mode=${PROJ_REL_HEIGHT_SCALE_MODE} scale=${PROJ_REL_HEIGHT_SCALE} quantile=${PROJ_REL_HEIGHT_SCALE_QUANTILE} clip=${PROJ_REL_HEIGHT_CLIP}"
echo "PROJ_REL_HEIGHT balanced_weight=${PROJ_REL_HEIGHT_BALANCED_WEIGHT} balanced_quantiles=${PROJ_REL_HEIGHT_BALANCED_QUANTILES}"
echo "PROJ_MOGE grad=${LAMBDA_PROJ_MOGE_GRAD} edge=${LAMBDA_PROJ_MOGE_EDGE} height=${LAMBDA_PROJ_MOGE_HEIGHT} min_weight=${PROJ_MOGE_PRIOR_MIN_WEIGHT} height_min_weight=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} height_ground_q=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE}"
echo "PROJ_AUX output_scale rel_height=${REMOTE_PROJECTION_AUX_REL_HEIGHT_OUTPUT_SCALE} offset=${REMOTE_PROJECTION_AUX_OFFSET_OUTPUT_SCALE}"
echo "LOAD_PRETRAINED_WEIGHTS=${LOAD_PRETRAINED_WEIGHTS} WARMSTART_CKPT=${WARMSTART_CKPT}"
echo "TRAIN_LR=${TRAIN_LR} TRAIN_MIN_LR=${TRAIN_MIN_LR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_rs_joint_518 \
    dataset.num_workers=${NUM_WORKERS} \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.train.cities=${TRAIN_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.val.cities=${VAL_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.test.cities=${TEST_CITIES} \
    dataset.vigor_chicago_joint_rs_aerial.train.scene_list_path=${TRAIN_SCENE_LIST_PATH} \
    dataset.vigor_chicago_joint_rs_aerial.train.overfit_num_sets=${TRAIN_OVERFIT_NUM_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.val.overfit_num_sets=${VAL_OVERFIT_NUM_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.test.overfit_num_sets=${TEST_OVERFIT_NUM_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_provider_sampling_mode=${REMOTE_PROVIDER_SAMPLING_MODE} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_providers=[${RS_PROVIDER}] \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_providers=[${RS_PROVIDER}] \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_providers=[${RS_PROVIDER}] \
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
    loss=pi3_loss_rs_joint_projection_aux \
    loss.remote_pointmap_loss_weight=${LAMBDA_REMOTE_PM} \
    loss.remote_raw_pointmap_loss_weight=${LAMBDA_REMOTE_RAW_PM} \
    loss.remote_height_loss_weight=${LAMBDA_REMOTE_H} \
    loss.remote_compare_in_view0_frame=${REMOTE_COMPARE_IN_VIEW0} \
    loss.remote_detach_pose_for_view0_align=${REMOTE_DETACH_POSE_ALIGN} \
    loss.remote_pointmap_norm_mode=${REMOTE_POINTMAP_NORM_MODE} \
    loss.remote_projection_rel_height_loss_weight=${LAMBDA_PROJ_REL_HEIGHT} \
    loss.remote_projection_offset_loss_weight=${LAMBDA_PROJ_OFFSET} \
    loss.remote_projection_global_dir_loss_weight=${LAMBDA_PROJ_GLOBAL_DIR} \
    loss.remote_projection_global_slope_loss_weight=${LAMBDA_PROJ_GLOBAL_SLOPE} \
    loss.remote_projection_consistency_loss_weight=${LAMBDA_PROJ_CONSISTENCY} \
    loss.remote_projection_rel_height_scale=${PROJ_REL_HEIGHT_SCALE} \
    loss.remote_projection_rel_height_scale_mode=${PROJ_REL_HEIGHT_SCALE_MODE} \
    loss.remote_projection_rel_height_scale_quantile=${PROJ_REL_HEIGHT_SCALE_QUANTILE} \
    loss.remote_projection_rel_height_clip=${PROJ_REL_HEIGHT_CLIP} \
    loss.remote_projection_rel_height_affine_loss_weight=${PROJ_REL_HEIGHT_AFFINE_WEIGHT} \
    loss.remote_projection_rel_height_balanced_loss_weight=${PROJ_REL_HEIGHT_BALANCED_WEIGHT} \
    loss.remote_projection_rel_height_balanced_quantiles=${PROJ_REL_HEIGHT_BALANCED_QUANTILES} \
    loss.remote_projection_offset_scale=${PROJ_OFFSET_SCALE} \
    loss.remote_projection_moge_gradient_loss_weight=${LAMBDA_PROJ_MOGE_GRAD} \
    loss.remote_projection_moge_edge_loss_weight=${LAMBDA_PROJ_MOGE_EDGE} \
    loss.remote_projection_moge_height_loss_weight=${LAMBDA_PROJ_MOGE_HEIGHT} \
    loss.remote_projection_moge_height_prior_min_weight=${PROJ_MOGE_HEIGHT_PRIOR_MIN_WEIGHT} \
    loss.remote_projection_moge_height_ground_quantile=${PROJ_MOGE_HEIGHT_GROUND_QUANTILE} \
    loss.remote_projection_moge_height_exclude_hard_mask=${PROJ_MOGE_HEIGHT_EXCLUDE_HARD_MASK} \
    loss.remote_projection_moge_prior_min_weight=${PROJ_MOGE_PRIOR_MIN_WEIGHT} \
    loss.remote_projection_moge_edge_temperature=${PROJ_MOGE_EDGE_TEMPERATURE} \
    loss.remote_projection_moge_edge_threshold=${PROJ_MOGE_EDGE_THRESHOLD} \
    model=pi3_modality_embedding_remote_head \
    model.model_config.load_pretrained_weights=${LOAD_PRETRAINED_WEIGHTS} \
    model.model_config.use_remote_projection_aux_head=true \
    model.model_config.remote_projection_aux_hidden_dim=${REMOTE_PROJECTION_AUX_HIDDEN_DIM} \
    model.model_config.remote_projection_aux_use_rgb=${REMOTE_PROJECTION_AUX_USE_RGB} \
    model.model_config.remote_projection_aux_use_coord=${REMOTE_PROJECTION_AUX_USE_COORD} \
    model.model_config.remote_projection_aux_image_stem_dim=${REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM} \
    model.model_config.remote_projection_aux_positive_slope=${REMOTE_PROJECTION_AUX_POSITIVE_SLOPE} \
    model.model_config.remote_projection_aux_slope_init=${REMOTE_PROJECTION_AUX_SLOPE_INIT} \
    model.model_config.remote_projection_aux_num_blocks=${REMOTE_PROJECTION_AUX_NUM_BLOCKS} \
    model.model_config.remote_projection_aux_rel_height_output_scale=${REMOTE_PROJECTION_AUX_REL_HEIGHT_OUTPUT_SCALE} \
    model.model_config.remote_projection_aux_offset_output_scale=${REMOTE_PROJECTION_AUX_OFFSET_OUTPUT_SCALE} \
    train_params=pi3_finetune \
    train_params.epochs=${EPOCHS} \
    train_params.warmup_epochs=${WARMUP_EPOCHS} \
    train_params.eval_freq=${EVAL_FREQ} \
    train_params.save_freq=${SAVE_FREQ} \
    train_params.keep_freq=${KEEP_FREQ} \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=${PRINT_FREQ} \
    train_params.lr=${TRAIN_LR} \
    train_params.min_lr=${TRAIN_MIN_LR} \
    train_params.resume=false \
    train_params.warmstart_ckpt=${WARMSTART_CKPT} \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_CLI_ARGS[@]}"
