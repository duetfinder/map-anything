#!/bin/bash

export HYDRA_FULL_ERROR=1
NUM_VIEWS=${NUM_VIEWS:-32}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/mapanything_unified'}

if [ -z "$MAPANYTHING_CKPT" ]; then
    echo "Please set MAPANYTHING_CKPT to a MapAnything checkpoint path."
    echo "Example: export MAPANYTHING_CKPT=/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth"
    exit 1
fi

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python3 \
    benchmarking/rs_guided_dense_mv/benchmark_unified.py \
    machine=autodl_vigor \
    dataset=benchmark_vigor_chicago_rs_aerial \
    dataset.num_views=$NUM_VIEWS \
    dataset.num_workers=0 \
    batch_size=$BATCH_SIZE \
    model=mapanything \
    model/task=images_only \
    model.encoder.uses_torch_hub=false \
    model.pretrained="$MAPANYTHING_CKPT" \
    hydra.run.dir="$OUTPUT_DIR" \
    dataset.vigor_chicago_wai.val.cities=[newyork] \
    dataset.vigor_chicago_rs_aerial_benchmark.remote.cities=[newyork]
