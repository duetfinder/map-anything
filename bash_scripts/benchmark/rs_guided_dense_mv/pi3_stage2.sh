#!/bin/bash

export HYDRA_FULL_ERROR=1

PYTHONPATH=. python3 \
    benchmarking/rs_guided_dense_mv/benchmark_stage2.py \
    machine=autodl_vigor \
    dataset=benchmark_vigor_chicago_rs_aerial_stage2 \
    dataset.num_workers=0 \
    batch_size=1 \
    model=pi3 \
    hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/pi3_stage2'
