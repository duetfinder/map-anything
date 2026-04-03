#!/bin/bash

NUM_GPUS=${1:-2}
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
TRAIN_SETS=${TRAIN_SETS:-16}
VAL_SETS=${VAL_SETS:-8}
TEST_SETS=${TEST_SETS:-8}
RS_PROVIDER=${RS_PROVIDER:-Google_Satellite}
LAMBDA_REMOTE_PM=${LAMBDA_REMOTE_PM:-0.1}
LAMBDA_REMOTE_H=${LAMBDA_REMOTE_H:-0.01}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/training/vigor_chicago/p3_joint_input_debug'}

MIN_REQUIRED_TRAIN_SETS=$((BATCH_SIZE * NUM_GPUS))
if [ "${TRAIN_SETS}" -lt "${MIN_REQUIRED_TRAIN_SETS}" ]; then
    echo "TRAIN_SETS (${TRAIN_SETS}) must be >= BATCH_SIZE * NUM_GPUS (${BATCH_SIZE} * ${NUM_GPUS} = ${MIN_REQUIRED_TRAIN_SETS}) for the distributed dynamic sampler." >&2
    exit 1
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node "${NUM_GPUS}" \
    scripts/train.py \
    machine=autodl_vigor \
    dataset=vigor_chicago_rs_joint_518 \
    dataset.num_workers=0 \
    dataset.num_views=${NUM_VIEWS} \
    dataset.vigor_chicago_joint_rs_aerial.train.overfit_num_sets=${TRAIN_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.val.overfit_num_sets=${VAL_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.test.overfit_num_sets=${TEST_SETS} \
    dataset.vigor_chicago_joint_rs_aerial.train.remote_provider=${RS_PROVIDER} \
    dataset.vigor_chicago_joint_rs_aerial.val.remote_provider=${RS_PROVIDER} \
    dataset.vigor_chicago_joint_rs_aerial.test.remote_provider=${RS_PROVIDER} \
    loss=pi3_loss_rs_joint \
    loss.remote_pointmap_loss_weight=${LAMBDA_REMOTE_PM} \
    loss.remote_height_loss_weight=${LAMBDA_REMOTE_H} \
    model=pi3 \
    model.model_config.load_pretrained_weights=true \
    train_params=pi3_finetune \
    train_params.epochs=1 \
    train_params.warmup_epochs=0 \
    train_params.eval_freq=1 \
    train_params.save_freq=1 \
    train_params.keep_freq=1 \
    train_params.max_num_of_imgs_per_gpu=${BATCH_SIZE} \
    train_params.print_freq=1 \
    train_params.resume=false \
    hydra.run.dir="${OUTPUT_DIR}"
