#!/bin/bash

NUM_GPUS=${1:-1}

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_50_518 \
    loss=pi3_loss \
    model=pi3 \
    train_params=pi3_finetune \
    train_params.epochs=10 \
    train_params.warmup_epochs=1 \
    train_params.keep_freq=5 \
    train_params.eval_freq=1 \
    train_params.max_num_of_imgs_per_gpu=8 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/vigor_chicago_50/pi3_finetune'
