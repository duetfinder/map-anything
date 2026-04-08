#!/bin/bash

set -euo pipefail

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Enable the local proxy before the first pretrained-weight download.
source /etc/profile.d/clash.sh
proxy_on

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_50_518 \
    dataset.num_workers=0 \
    dataset.num_views=2 \
    dataset.vigor_chicago_wai.train.overfit_num_sets=2 \
    dataset.vigor_chicago_wai.val.overfit_num_sets=1 \
    dataset.vigor_chicago_wai.test.overfit_num_sets=1 \
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
    train_params.max_num_of_imgs_per_gpu=2 \
    train_params.print_freq=1 \
    train_params.resume=false \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/vigor_chicago_50/pi3_smoke_pretrained_1gpu'
