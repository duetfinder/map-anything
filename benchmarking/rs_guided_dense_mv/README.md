# RS-Aerial Reconstruction Benchmark

当前 benchmark 采用分阶段实现，组织形式尽量保持与 `dense_n_view` 一致。

## 当前结构

- `benchmark.py`
  - Stage-0 baseline
  - 仅评测空中透视多视图重建
- `benchmark_stage1.py`
  - Stage-1 remote-view geometry benchmark
  - 仅评测遥感视图几何重建

## 当前指标

### Stage-0

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`

### Stage-1

- `rs_pointmap_abs_rel`
- `rs_height_mae`
- `rs_height_rmse`

## 重要说明

- Stage-1 当前不是 joint benchmark
- 因此 Stage-1 的结果文件只包含遥感视图几何指标
- 如果需要在同一个 benchmark 中同时输出空中视图与遥感视图指标，需要后续实现 Stage-2

## 配置与脚本

### Stage-0

- 配置：`configs/rs_guided_dense_mv_benchmark.yaml`
- 数据配置：`configs/dataset/benchmark_vigor_chicago_rs_guided_518.yaml`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3.sh`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/vggt.sh`

### Stage-1

- 配置：`configs/rs_aerial_stage1_benchmark.yaml`
- 数据配置：`configs/dataset/benchmark_vigor_chicago_rs_aerial_stage1.yaml`
- metadata 准备：`scripts/prepare_rs_aerial_benchmark_metadata.py`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage1.sh`

更完整的中文设计说明见：

- `RS_GUIDED_DENSE_MV_BENCHMARK_DESIGN_CN.md`
