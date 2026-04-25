# RS Guided 可视化说明

本文档说明当前 `rs_guided_dense_mv` 可视化/导出流程，以及 viewer 中使用的几种对齐方式。

## 整体结构

当前相关脚本有 3 个：

- `scripts/run_rs_guided_inference_batch.py`
  - 读取 CSV
  - 一次初始化模型
  - 逐场景运行 `aerial_only`、`rs_only`、`joint`
  - 直接复用 benchmark 里的指标计算函数
  - 为每个场景导出 `scene_bundle.pt`、`metrics.json`、`*.ply`
  - 为整批任务导出汇总 `batch_results.json/csv`
- `scripts/visualize_rs_guided_scene.py`
  - 读取 `scene_bundle.pt`
  - 把几何、位姿、指标写入 Rerun recording
  - 按不同对齐模式组织显示树
- `scripts/serve_rs_guided_scene.sh`
  - 先把 `scene_bundle.pt` 转成 `.rrd`
  - 再启动浏览器版 Rerun viewer

## 输入 CSV

最小 CSV 格式：

```csv
scene_name,frame_indices
newyork__location_471,"0,2,4,6"
```

当前支持字段：

- `scene_name` 或 `location`
- `frame_indices`
- `provider`

推荐始终写完整 `scene_name`，例如：

- `newyork__location_471`

不要只写：

- `471`
- `location_471`

因为不同城市下可能会重名。

## 数据 split 范围

当前数据集 scene 编号范围如下：

- `train`: `location_1` 到 `location_400`
- `val`: `location_401` 到 `location_450`
- `test`: `location_451` 到 `location_500`

例如：

- `newyork__location_434` 属于 `val`
- `newyork__location_471` 属于 `test`

## 输出目录

单个场景会输出：

- `scene_bundle.pt`
- `metrics.json`
- `aerial_only_global.ply`
- `joint_global.ply`

整批任务会输出：

- `batch_results.json`
- `batch_results.csv`

## scene_bundle.pt 包含什么

`scene_bundle.pt` 目前包含这些主字段：

- `metadata`
  - scene 名称、provider、frame indices
- `metrics`
  - `aerial_only`
  - `rs_only`
  - `joint`
  - `improvement`
- `views`
  - 每帧 RGB 图像
  - raw GT / prediction 点图和位姿
  - benchmark 对齐后的几何
  - remote metric 对齐后的几何
- `remote`
  - 遥感 RGB 图像
  - raw GT / prediction 点图
  - 对齐后的 remote 几何
- `global_pointclouds`
  - raw fused 点云
  - benchmark 对齐后的全局点云
  - remote metric 对齐后的全局点云

## Viewer 顶层分组

网页里主要分成三组：

- `modes/remote_metric`
- `modes/benchmark`
- `debug/raw`

含义分别如下。

### `modes/benchmark`

这是和 benchmark 定义保持一致的可视化模式。

特点：

- 坐标系统一切到 `view0` 参考系
- GT 和预测分别转到各自的 `view0` 参考系
- 点云使用 benchmark 相同的 `avg_dis` joint normalization
- pose translation 使用同一个归一化因子缩放
- `joint_global` 使用和 `joint_global_pointmaps_abs_rel` 相同的 joint normalization 思路

这个模式的用途是：

- 看几何关系是否和 benchmark 评价一致
- 对照 `pointmaps_abs_rel`、`pose_ate_rmse` 等指标理解结果

这个模式不是米制，不代表真实绝对尺度。

实现来源：

- `benchmarking/dense_n_view/benchmark.py:get_all_info_for_metric_computation`
- `benchmarking/rs_guided_dense_mv/benchmark_unified.py:compute_joint_global_pointmaps_abs_rel`

### `modes/remote_metric`

这是基于遥感先验尺度恢复的可视化模式。

特点：

- 坐标系仍然切到 `view0`
- 不改 rotation，只恢复 translation / pointmap 的尺度
- 尺度信息来自遥感图的 `meters_per_pixel`
- 使用预测 remote pointmap 在水平面 `xy` 上的相邻像素距离估计一个全局 `scale_factor`

当前做法：

1. 先把 `joint` 预测结果转到预测的 `view0` 坐标系
2. 对 remote prediction：
   - 取有效像素
   - 计算横向/纵向相邻像素的 `xy` 距离
   - 取中位数作为预测像素间距
3. 用
   - `scale_factor = meters_per_pixel / predicted_spacing_xy`
4. 把这个 `scale_factor` 统一乘到：
   - `joint` aerial pointmaps
   - `joint` remote pointmap
   - `joint` pose translations

这个模式的用途是：

- 看更接近真实世界米制尺度的结果
- 直观判断全局点云大小是否合理

这个模式不用于 benchmark 打分，它是 visualization mode。

### `debug/raw`

这是原始调试模式，保留未对齐的输出，主要用于排错。

包含：

- raw aerial-only fused 点云
- raw joint fused 点云
- raw 每帧 GT / prediction 点图
- raw remote GT / `rs_only` / `joint` 点图

这个分组主要用于确认：

- 模型原始输出坐标系
- exporter 是否正确
- 对齐前后的差别

## 两种对齐方式的区别

### 方式 1：benchmark 对齐

目标：

- 和 benchmark 的指标定义完全一致

特点：

- 在相对几何上可比
- 不恢复真实尺度
- 更适合解释 benchmark 分数

### 方式 2：remote metric 对齐

目标：

- 借助遥感图的已知像素物理分辨率恢复近似米制尺度

特点：

- 坐标系仍然切到 `view0`
- 尺度由 remote `meters_per_pixel` 恢复
- 更适合直观观察全局几何大小

## 遥感元数据来源

当前原始数据里有两个相关文件：

- scene 级：
  - `Crossview_rs/<scene>/map_metadata.json`
- provider 级：
  - `Crossview_rs/<scene>/<provider>/info.json`

当前实现优先使用 manifest 里已有的：

- `meters_per_pixel`

如果 manifest 没有，再回退到 remote `info.json`。

对于当前的 metric-scale 恢复，`meters_per_pixel` 已经足够。

## 浏览器部署

推荐直接使用：

```bash
bash scripts/serve_rs_guided_scene.sh /path/to/scene_bundle.pt
```

它会做两件事：

1. 生成 `/path/to/scene_bundle.rrd`
2. 启动：

```bash
rerun /path/to/scene_bundle.rrd --web-viewer
```

然后在浏览器中打开对应地址。

## 当前建议的查看顺序

看结果时建议按这个顺序：

1. 先看 `modes/remote_metric`
   - 用来判断全局点云尺度和整体结构
2. 再看 `modes/benchmark`
   - 用来和 benchmark 指标对应
3. 最后才看 `debug/raw`
   - 只用于排查模型输出或 exporter 问题

## 注意事项

- 旧版 `scene_bundle.pt` 不包含新的 aligned 字段
- 每次对齐逻辑更新后，需要重新跑 batch exporter
- 当前 Rerun viewer 主要是调试型 viewer，不是最终产品级 dashboard
