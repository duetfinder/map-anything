#!/bin/bash

export HYDRA_FULL_ERROR=1
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
CKPT_PATH=${CKPT_PATH:-/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/vigor_chicago/p2a/pi3_rs_only_pointmap_height_exclude_top5_loss/L3/checkpoint-best.pth}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/pi3_chicago500_finetuned_p2aL3_unified'}

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

PYTHONPATH=. python3 \
    benchmarking/rs_guided_dense_mv/benchmark_unified.py \
    machine=autodl_vigor \
    dataset=benchmark_vigor_chicago_rs_aerial \
    dataset.num_views=$NUM_VIEWS \
    dataset.num_workers=0 \
    batch_size=$BATCH_SIZE \
    model=pi3 \
    model.pretrained="$CKPT_PATH" \
    hydra.run.dir="$OUTPUT_DIR"
