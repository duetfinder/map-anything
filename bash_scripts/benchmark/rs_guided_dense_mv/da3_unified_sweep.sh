#!/bin/bash

export HYDRA_FULL_ERROR=1

run_configs=(
    "1 2"
    "1 4"
    "1 8"
    "1 16"
    "1 24"
    "1 32"
    "1 40"
)

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

for cfg in "${run_configs[@]}"; do
    read -r batch_size num_views <<< "$cfg"
    echo "Running DA3 unified benchmark with batch_size=$batch_size num_views=$num_views"
    NUM_VIEWS=$num_views \
    BATCH_SIZE=$batch_size \
    OUTPUT_DIR='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/da3_unified_'"${num_views}"'v' \
    bash /root/autodl-tmp/Models/map-anything/bash_scripts/benchmark/rs_guided_dense_mv/da3_unified.sh
done
