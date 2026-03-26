#!/bin/bash

export HYDRA_FULL_ERROR=1

if [ -z "$MAPANYTHING_CKPT" ]; then
    echo "Please set MAPANYTHING_CKPT to a MapAnything checkpoint path."
    echo "Example: export MAPANYTHING_CKPT=/path/to/checkpoint-last.pth"
    exit 1
fi

# Format: "batch_size num_views"
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
    echo "Running MapAnything unified benchmark with batch_size=$batch_size num_views=$num_views"
    NUM_VIEWS=$num_views \
    BATCH_SIZE=$batch_size \
    OUTPUT_DIR='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/mapanything_unified_'"${num_views}"'v' \
    bash /root/autodl-tmp/Models/map-anything/bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified.sh
done
