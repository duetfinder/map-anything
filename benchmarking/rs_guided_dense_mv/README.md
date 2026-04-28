# RS-Aerial Reconstruction Benchmark

当前正式入口已经收敛为统一 benchmark，组织形式尽量保持与 `dense_n_view` 一致。

## 正式入口

- `benchmark_unified.py`
  - 正式 benchmark 入口
  - 在 paired scenes 上统一输出 `aerial_only`、`rs_only` 和 `joint` 结果结构
  - 当前已经实现：
    - `aerial_only`：空中多视图重建指标
    - `rs_only`：遥感高度指标
    - `joint`：空中视图 + 遥感视图联合前向后的两类任务指标
    - `improvement`：`joint` 相对 `aerial_only` / `rs_only` 的提升量
  - 当前 joint 全局几何指标：
    - `joint_global_pointmaps_abs_rel`

## 当前正式指标

### Aerial-only

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`
- `metric_scale_abs_rel`
- `metric_point_l1`

### RS-only

- `rs_height_mae`
- `rs_height_rmse`

### Joint

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`
- `metric_scale_abs_rel`
- `metric_point_l1`
- `rs_height_mae`
- `rs_height_rmse`
- `joint_global_pointmaps_abs_rel`

其中：

- `joint_global_pointmaps_abs_rel` 现在表示“`joint` 的 `pointmaps_abs_rel` 扩展版”
- 做法是把 aerial 多视图和 remote 点图都先分别切到各自的 `view0` 参考系
- 再对 GT / Pred 各自做联合 `avg_dis` normalization
- 最后在这个统一相对几何空间里统计所有 aerial + remote 点的相对误差

## 已修正的历史错误

旧版 unified benchmark 里：

- `joint_global_point_l1`
- `joint_global_pointmaps_abs_rel`

都没有沿用 `pointmaps_abs_rel` 的 `view0` 相对对齐逻辑，而是直接在全局系或直接 joint normalization 后比较。这和 aerial benchmark 的指标定义不一致，也会让可视化结果和指标语义脱节。

当前修正后：

- 删除 `joint_global_point_l1`
- 保留并重定义 `joint_global_pointmaps_abs_rel`
- 将其作为 `pointmaps_abs_rel + remote` 的唯一正式 joint 全局几何指标

## 配置与脚本

- 配置：`configs/rs_aerial_benchmark.yaml`
- 数据配置：`configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml`
- metadata 准备：`scripts/prepare_rs_aerial_benchmark_metadata.py`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified.sh`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/vggt_unified.sh`
- 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified.sh`
- 批量脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified_sweep.sh`
- 批量脚本：`bash_scripts/benchmark/rs_guided_dense_mv/vggt_unified_sweep.sh`
- 批量脚本：`bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified_sweep.sh`

## 历史实现

下面几个文件仍保留，作为统一 benchmark 的实现参考或迁移来源，不再作为正式对外入口：

- `benchmark.py`
- `benchmark_stage1.py`
- `benchmark_stage2.py`

更完整的中文设计说明见：

- `RS_GUIDED_DENSE_MV_BENCHMARK_DESIGN_CN.md`

## 结果可视化

- 脚本：`scripts/visualize_rs_aerial_benchmark.py`
- 输入：`rs_aerial_benchmark_results.json`
- 输出：
  - `visualized/summary.md`
  - `visualized/per_scene_compact.csv`
  - `visualized/scene_ranking.csv`

## 运行说明

- 三个 unified 脚本都会在运行前尝试执行：`source /etc/profile.d/clash.sh && proxy_on`，用于加速首次权重下载。
- 帧数由 `dataset.num_views` 控制；当前脚本通过环境变量 `NUM_VIEWS` 覆盖，因此不需要改 YAML。
- 单次运行示例：`NUM_VIEWS=4 BATCH_SIZE=1 bash bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified.sh`
- 批量运行示例：`bash bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified_sweep.sh`
- `VGGT` 当前框架内的预训练入口固定为 `facebook/VGGT-1B`。如果要切换到别的 VGGT 规模，需要修改 `mapanything/models/external/vggt/__init__.py` 里硬编码的 `from_pretrained(...)` repo id，现有配置文件本身还没有暴露这个开关。
- `Pi3` 当前框架内的预训练入口固定为 `yyfz233/Pi3`。`decoder_size` 只在 `load_pretrained_weights=false` 时用于从零初始化结构，不能直接切换到另一套预训练尺度。
- `MapAnything` 的结构规模由 `configs/model/mapanything*.yaml` 的默认组合决定，例如 `model=mapanything` 使用 giant encoder，`model=mapanything_v1` 使用 large encoder；但 benchmark 运行仍需要对应结构的本地 checkpoint：`export MAPANYTHING_CKPT=/path/to/checkpoint-last.pth`。
- 如果某个模型不支持某个指标，结果文件里该指标保留为空值或 `NaN`，不再单独拆表。
