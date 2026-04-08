#!/bin/bash

export HYDRA_FULL_ERROR=1

# Format: "batch_size num_views"
run_configs=(
    "10 2"
    "10 4"
    "10 8"
    "5 16"
    "4 24"
    "2 32"
    "1 40"
)

if [ -f /etc/profile.d/clash.sh ]; then
    source /etc/profile.d/clash.sh
    proxy_on >/dev/null 2>&1 || true
fi

for cfg in "${run_configs[@]}"; do
    read -r batch_size num_views <<< "$cfg"
    echo "Running Pi3 unified benchmark with batch_size=$batch_size num_views=$num_views"
    NUM_VIEWS=$num_views \
    BATCH_SIZE=$batch_size \
    OUTPUT_DIR='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/pi3_chicago500_finetuned_p3_unified_'"${num_views}"'v' \
    bash /root/autodl-tmp/Models/map-anything/bash_scripts/benchmark/rs_guided_dense_mv/pi3_chicago500_finetuned_p3_unified.sh
done


