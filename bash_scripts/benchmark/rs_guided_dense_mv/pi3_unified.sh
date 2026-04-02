#!/bin/bash

export HYDRA_FULL_ERROR=1
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
REMOTE_PROVIDER=${REMOTE_PROVIDER:-Google_Satellite}
REMOTE_CROP_MODE=${REMOTE_CROP_MODE:-none}
REMOTE_NUM_AUG_CROPS=${REMOTE_NUM_AUG_CROPS:-1}
REMOTE_CROP_SCALE_MIN=${REMOTE_CROP_SCALE_MIN:-0.7}
REMOTE_CROP_SCALE_MAX=${REMOTE_CROP_SCALE_MAX:-1.0}
REMOTE_IMAGE_RESIZE_MODE=${REMOTE_IMAGE_RESIZE_MODE:-nearest}
REMOTE_LABEL_RESIZE_MODE=${REMOTE_LABEL_RESIZE_MODE:-nearest}
OUTPUT_DIR=${OUTPUT_DIR:-'${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/pi3_unified'}

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

PYTHONPATH=. python3     benchmarking/rs_guided_dense_mv/benchmark_unified.py     machine=autodl_vigor     dataset=benchmark_vigor_chicago_rs_aerial     dataset.num_views=$NUM_VIEWS     dataset.num_workers=0     dataset.vigor_chicago_rs_aerial_benchmark.remote.provider=${REMOTE_PROVIDER}     dataset.vigor_chicago_rs_aerial_benchmark.remote.crop_mode=${REMOTE_CROP_MODE}     dataset.vigor_chicago_rs_aerial_benchmark.remote.num_augmented_crops_per_sample=${REMOTE_NUM_AUG_CROPS}     dataset.vigor_chicago_rs_aerial_benchmark.remote.crop_scale_range=[${REMOTE_CROP_SCALE_MIN},${REMOTE_CROP_SCALE_MAX}]     dataset.vigor_chicago_rs_aerial_benchmark.remote.image_resize_mode=${REMOTE_IMAGE_RESIZE_MODE}     dataset.vigor_chicago_rs_aerial_benchmark.remote.label_resize_mode=${REMOTE_LABEL_RESIZE_MODE}     batch_size=$BATCH_SIZE     model=pi3     hydra.run.dir="$OUTPUT_DIR"
