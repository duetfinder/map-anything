#!/bin/bash

NUM_GPUS=${1:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
TRAIN_SETS=${TRAIN_SETS:-160}
VAL_SETS=${VAL_SETS:-8}
TEST_SETS=${TEST_SETS:-8}
LOSS_CONFIG=${LOSS_CONFIG:-pi3_rs_only_pointmap_height_exclude_top5_loss}
RS_PROVIDER=${RS_PROVIDER:-all}
RS_VAL_PROVIDER=${RS_VAL_PROVIDER:-${RS_PROVIDER}}
RS_TEST_PROVIDER=${RS_TEST_PROVIDER:-${RS_PROVIDER}}
RS_TRAIN_CROP_MODE=${RS_TRAIN_CROP_MODE:-random_scale_offset}
RS_VAL_CROP_MODE=${RS_VAL_CROP_MODE:-random_scale_offset}
RS_TEST_CROP_MODE=${RS_TEST_CROP_MODE:-none}
RS_TRAIN_NUM_AUG_CROPS=${RS_TRAIN_NUM_AUG_CROPS:-1}
RS_VAL_NUM_AUG_CROPS=${RS_VAL_NUM_AUG_CROPS:-1}
RS_TEST_NUM_AUG_CROPS=${RS_TEST_NUM_AUG_CROPS:-1}
RS_CROP_SCALE_MIN=${RS_CROP_SCALE_MIN:-0.6}
RS_CROP_SCALE_MAX=${RS_CROP_SCALE_MAX:-1.0}
RS_IMAGE_RESIZE_MODE=${RS_IMAGE_RESIZE_MODE:-nearest}
RS_LABEL_RESIZE_MODE=${RS_LABEL_RESIZE_MODE:-nearest}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/vigor_chicago/p2a/'${LOSS_CONFIG}'/L1'}

MIN_REQUIRED_TRAIN_SETS=$((BATCH_SIZE * NUM_GPUS))
if [ "${TRAIN_SETS}" -lt "${MIN_REQUIRED_TRAIN_SETS}" ]; then
    echo "TRAIN_SETS (${TRAIN_SETS}) must be >= BATCH_SIZE * NUM_GPUS (${BATCH_SIZE} * ${NUM_GPUS} = ${MIN_REQUIRED_TRAIN_SETS}) for the distributed dynamic sampler." >&2
    exit 1
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node "${NUM_GPUS}"     scripts/train.py     machine=autodl_vigor     dataset=vigor_chicago_rs_518     loss=${LOSS_CONFIG}     model=pi3     model.model_config.load_pretrained_weights=true     dataset.vigor_chicago_rs.train.overfit_num_sets=${TRAIN_SETS}     dataset.vigor_chicago_rs.val.overfit_num_sets=${VAL_SETS}     dataset.vigor_chicago_rs.test.overfit_num_sets=${TEST_SETS}     dataset.vigor_chicago_rs.train.providers=[${RS_PROVIDER}]     dataset.vigor_chicago_rs.val.providers=[${RS_VAL_PROVIDER}]     dataset.vigor_chicago_rs.test.providers=[${RS_TEST_PROVIDER}]     dataset.vigor_chicago_rs.train.crop_mode=${RS_TRAIN_CROP_MODE}     dataset.vigor_chicago_rs.val.crop_mode=${RS_VAL_CROP_MODE}     dataset.vigor_chicago_rs.test.crop_mode=${RS_TEST_CROP_MODE}     dataset.vigor_chicago_rs.train.num_augmented_crops_per_sample=${RS_TRAIN_NUM_AUG_CROPS}     dataset.vigor_chicago_rs.val.num_augmented_crops_per_sample=${RS_VAL_NUM_AUG_CROPS}     dataset.vigor_chicago_rs.test.num_augmented_crops_per_sample=${RS_TEST_NUM_AUG_CROPS}     dataset.vigor_chicago_rs.train.crop_scale_range=[${RS_CROP_SCALE_MIN},${RS_CROP_SCALE_MAX}]     dataset.vigor_chicago_rs.val.crop_scale_range=[${RS_CROP_SCALE_MIN},${RS_CROP_SCALE_MAX}]     dataset.vigor_chicago_rs.test.crop_scale_range=[${RS_CROP_SCALE_MIN},${RS_CROP_SCALE_MAX}]     dataset.vigor_chicago_rs.train.image_resize_mode=${RS_IMAGE_RESIZE_MODE}     dataset.vigor_chicago_rs.val.image_resize_mode=${RS_IMAGE_RESIZE_MODE}     dataset.vigor_chicago_rs.test.image_resize_mode=${RS_IMAGE_RESIZE_MODE}     dataset.vigor_chicago_rs.train.label_resize_mode=${RS_LABEL_RESIZE_MODE}     dataset.vigor_chicago_rs.val.label_resize_mode=${RS_LABEL_RESIZE_MODE}     dataset.vigor_chicago_rs.test.label_resize_mode=${RS_LABEL_RESIZE_MODE}     train_params=pi3_finetune     train_params.epochs=50     train_params.warmup_epochs=0     train_params.eval_freq=5     train_params.save_freq=10     train_params.keep_freq=10     train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE}     train_params.print_freq=1     train_params.resume=false     hydra.run.dir="${OUTPUT_DIR}"
