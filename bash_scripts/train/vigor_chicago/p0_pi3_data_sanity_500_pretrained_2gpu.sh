#!/bin/bash

NUM_GPUS=${1:-2}
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
TRAIN_SETS=${TRAIN_SETS:-2}
VAL_SETS=${VAL_SETS:-1}
TEST_SETS=${TEST_SETS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/vigor_chicago/debug/p0_pi3_data_sanity_500_pretrained_2gpu'}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_500_518 \
    dataset.num_workers=0 \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_wai.train.overfit_num_sets=${TRAIN_SETS} \
    dataset.vigor_chicago_wai.val.overfit_num_sets=${VAL_SETS} \
    dataset.vigor_chicago_wai.test.overfit_num_sets=${TEST_SETS} \
    dataset.vigor_chicago_wai.train.transform=imgnorm \
    loss=pi3_loss \
    model=pi3 \
    model.model_config.load_pretrained_weights=true \
    train_params=pi3_finetune \
    train_params.epochs=1 \
    train_params.warmup_epochs=0 \
    train_params.eval_freq=0 \
    train_params.save_freq=0 \
    train_params.keep_freq=0 \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=1 \
    train_params.resume=false \
    hydra.run.dir="${OUTPUT_DIR}"
