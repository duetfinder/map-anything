#!/bin/bash

export HYDRA_FULL_ERROR=1

python3 \
    benchmarking/rs_guided_dense_mv/benchmark.py \
    machine=autodl_vigor \
    dataset=benchmark_vigor_chicago_rs_guided_518 \
    dataset.num_workers=4 \
    dataset.num_views=4 \
    batch_size=2 \
    model=vggt \
    hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/vggt'
