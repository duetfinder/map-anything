#!/bin/bash

export HYDRA_FULL_ERROR=1
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/da3_unified'}

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
    model=da3 \
    hydra.run.dir="$OUTPUT_DIR"
