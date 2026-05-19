#!/bin/bash

set -euo pipefail

export HYDRA_FULL_ERROR=1
NUM_VIEWS=${NUM_VIEWS:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
REMOTE_PROVIDER=${REMOTE_PROVIDER:-Google_Satellite}
CITY=${CITY:-newyork}
COVISIBILITY_THRES=${COVISIBILITY_THRES:-0.0}
VIEW_SAMPLING_MODE=${VIEW_SAMPLING_MODE:-connected}
CUDA_DEVICE=${CUDA_DEVICE:-0}

if [ -z "${MODEL_NAME:-}" ]; then
    echo "MODEL_NAME must be set." >&2
    exit 1
fi

if [ -z "${CKPT_PATH:-}" ]; then
    echo "CKPT_PATH must be set." >&2
    exit 1
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "OUTPUT_DIR must be set." >&2
    exit 1
fi

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 \
    benchmarking/rs_guided_dense_mv/benchmark_unified.py \
    machine=autodl_vigor \
    dataset=benchmark_vigor_chicago_rs_aerial \
    dataset.num_views=${NUM_VIEWS} \
    dataset.num_workers=0 \
    dataset.vigor_chicago_rs_aerial_benchmark.remote.providers=[${REMOTE_PROVIDER}] \
    dataset.vigor_chicago_wai.val.covisibility_thres=${COVISIBILITY_THRES} \
    dataset.vigor_chicago_wai.val.view_sampling_mode=${VIEW_SAMPLING_MODE} \
    batch_size=${BATCH_SIZE} \
    model=${MODEL_NAME} \
    model.pretrained="${CKPT_PATH}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    dataset.vigor_chicago_wai.val.cities=[${CITY}] \
    dataset.vigor_chicago_rs_aerial_benchmark.remote.cities=[${CITY}] \
    "$@"
