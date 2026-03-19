# RS-Aerial Reconstruction Benchmark 设计文档

本文档定义新的 benchmark：`RS-Aerial Reconstruction Benchmark`。

该任务用于评估：

- 空中透视图像
- 遥感图像

在同一场景中的联合三维重建能力。

## 1. 任务定义

这个任务不是普通的多视角重建，也不是简单的“顶视图一致性检查”。

更准确地说，它要评测的是：

- 模型能否联合处理不同投影几何类型的图像
- 并将它们统一到同一个全局三维重建中

其中：

- 空中视图：标准透视投影
- 遥感视图：通过 `exp_005` 构建出的特殊投影几何视图

遥感图像虽然不是普通 pinhole camera 图像，但它在 benchmark 中依然应被视为一种“正式输入视图”，因为它具备：

- 图像输入
- 每像素对应的几何标签

因此它的 benchmark 身份应该是：

- 特殊视图
- 而不是后处理参考图

## 2. 数据来源

### 2.1 空中视图

来自：

- `outputs/experiments/exp_001_reconstrc/vigor_chicago_processed/location_x`

当前已经有：

- RGB 图像
- 深度
- 内参
- 外参

并已接入 WAI 风格的 `MapAnything` 数据流。

### 2.2 遥感视图

来自：

- `outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/`

这部分数据的本质不是普通 top-view 图，而是：

- 遥感图像
- 遥感图像逐像素对应的全局三维几何

这意味着它应该被建模成：

- 一种新的 view type

而不是简单的 scene-level top-view 参考。

当前 `exp_005` 已经输出了可直接作为 benchmark 标签的逐像素几何：

- `pixel_to_point_map.npz`
  - 键名：`xyz`
  - 形状：`(H, W, 3)`
  - 含义：遥感图像逐像素对应的全局三维点
  - 无效像素为 `NaN`
- `valid_mask.npy`
  - 形状：`(H, W)`
  - 含义：遥感视图的有效像素掩码
- `height_map.npy`
  - 形状：`(H, W)`
  - 含义：逐像素高度监督
- `info.json`
  - 包含 `meters_per_pixel`、`coverage_meters`、`ground_z_approx`、`angle` 等投影元信息

因此，遥感视图在 Stage-1 中已经具备“图像 + 逐像素点图 + 掩码 + 高度”的监督形式。

## 3. benchmark 的正确目标

每个 `location_x` 的 benchmark 目标应包括三部分：

### 3.1 空中视图重建指标

用于评估普通多视角重建质量：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`

### 3.2 遥感视图几何指标

用于评估模型对遥感视图的重建能力：

- `rs_pointmap_abs_rel`
- `rs_height_mae`
- `rs_height_rmse`

其中：

- `rs_pointmap_abs_rel` 的形式和 `pointmaps_abs_rel` 相同
- 区别只在于作用对象不同：
  - `pointmaps_abs_rel` 面向空中透视视图
  - `rs_pointmap_abs_rel` 面向遥感视图

### 3.3 跨视图全局一致性指标

这是该任务最关键、但也最复杂的一部分。

第一版建议目标是：

- 评测空中视图预测出来的全局点
- 与遥感视图预测出来的全局点
- 是否位于同一个一致的全局几何中

可选候选定义包括：

1. 直接比较空中与遥感预测点的重叠区域距离误差
2. 比较二者预测出的相对位姿关系
3. 把二者联合成一个整体点云，与整体 GT 一起比较

其中第 3 种最符合最终目标，但实现最复杂。

## 4. 为什么删掉之前的 top-view 指标

之前临时实现过一版 top-view 指标：

- `topview_height_mae`
- `topview_height_rmse`
- `topview_occupancy_iou`
- `topview_coverage`

这些指标现在被删除，原因是它们依赖了错误假设：

- 假设空中视图预测点云可以直接投影到遥感高度图平面后做逐像素比较

这个假设不成立，因为：

- 遥感图像的几何不是简单的正交顶视图
- 其中包含建筑偏移等特殊投影关系
- `exp_005` 已经说明了这部分映射过程本身就是特殊构建的

因此：

- 这些 top-view 指标不应该作为主 benchmark 指标
- 最多只能作为临时可视化检查工具

## 5. 现阶段 benchmark 的实现策略

为了保证工程推进稳定，benchmark 分阶段实现。

### Stage-0：可运行 baseline

当前只实现：

- 空中视图标准重建指标

也就是：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`

这一阶段的目标只是：

- 建立 benchmark 入口
- 对齐现有 `MapAnything` benchmark 风格
- 提供后续扩展基础

### Stage-1：遥感视图正式接入

这一阶段新增：

- 遥感视图作为 dataset 中的正式 view
- `rs_pointmap_abs_rel`
- `rs_height_mae / rmse`

这一阶段不再依赖后处理投影比较，而是直接使用：

- 遥感原图
- `pixel_to_point_map.npz['xyz']`
- `valid_mask.npy`
- `height_map.npy`

### Stage-2：联合跨视图一致性评测

这一阶段新增：

- 空中视图与遥感视图的全局重建一致性指标

## 6. 关于 metric 与归一化

`MapAnything` 现有 `dense_n_view` benchmark 中，大多数主指标不是直接在原始米制坐标下计算的，而是在联合归一化后计算。

核心函数是：

- `normalize_multiple_pointclouds(..., norm_mode="avg_dis")`

其含义是：

- 对一个 multi-view set 内的所有点云联合归一化
- 归一化因子是所有有效点到原点距离的平均值

因此：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`

本质上都不是直接米制误差，而是归一化几何误差。

真正更接近 metric scale 的，是：

- `metric_scale_abs_rel`

它评估的是：

- 模型输出的 metric scale 因子是否准确

而不是点云本身的直接米制误差。

## 7. 关于 pointmaps_abs_rel 的对齐方式

在 `dense_n_view` benchmark 中：

1. 所有视图先被转换到 `view0` 参考系
2. 所有视图的点云一起参与联合归一化
3. 再逐视图计算误差并求平均

所以它不是：

- 每张图单独缩放、单独对齐后再比较

而是：

- 整个 multi-view set 一起做联合归一化

这也意味着：

- 如果某一张图的位姿预测很差
- 它会同时影响该图自身的 pointmap 误差
- 也会破坏整个多视图集合的全局一致性

因此 `pointmaps_abs_rel` 和位姿误差是耦合的。

## 8. WAI 数据格式约束

WAI 本质上规定的是：

- scene 目录结构
- `scene_meta.json`
- frame-level modalities
- scene-level modalities

它并不强制所有视图都必须是普通 RGB + depth 的 pinhole camera。

因此，只要我们在 `scene_meta` 中清楚描述：

- 图像路径
- 对应几何标签
- modality 路径
- 额外的 projection metadata

那么把数据整理成 paired scene 仍然可以兼容 WAI 的扩展思路。

换句话说：

- paired scene 格式不违反 WAI
- 只要它仍然遵守 scene-level / frame-level modality 的组织方式

## 9. Stage-1 数据 manifest

为减少对现有 aerial WAI 数据的侵入，Stage-1 先不直接重写整个 WAI scene，而是增加一层 manifest，把：

- aerial WAI scene
- remote image
- remote pointmap labels

绑定起来。

每个 scene 的 manifest 建议至少包含：

- `scene_name`
- `aerial_scene_dir`
- `aerial_scene_meta`
- `remote_provider`
- `remote_scene_dir`
- `remote_image_path`
- `remote_pointmap_path`
- `remote_valid_mask_path`
- `remote_height_map_path`
- `remote_info_path`
- `remote_projection_type`

当前已经开始采用这种 manifest 组织形式。

## 10. 当前代码实现的收缩

由于之前的 top-view 指标定义不正确，当前 benchmark 执行实现已经收缩为：

- 仅保留空中视图标准指标

这样可以保证：

- 当前输出是合理的
- benchmark 主流程仍可运行
- 不会继续产生误导性 top-view 指标

## 11. 结论

当前 benchmark 的正式定义应固定为：

- 名称：`RS-Aerial Reconstruction Benchmark`
- 目标：评测空中视图与遥感视图的联合三维重建能力
- 当前可运行实现：Stage-0，仅评测空中视图标准指标
- 后续扩展方向：
  1. 把遥感图像作为正式 view 接入
  2. 增加遥感视图几何指标
  3. 增加跨视图全局一致性指标

这是目前最稳妥、也最符合任务本质的路线。


## 12. 当前代码结构与运行方式

当前 benchmark 在 `Models/map-anything` 中分为两层可执行实现。

### 12.1 Stage-0：空中视图 baseline benchmark

作用：

- 复用现有 `VigorChicagoWAI`
- 仅评测空中透视多视图重建
- 作为后续 RS-Aerial benchmark 的 baseline

对应文件：

- 入口脚本：`benchmarking/rs_guided_dense_mv/benchmark.py`
- 顶层配置：`configs/rs_guided_dense_mv_benchmark.yaml`
- 数据集配置：`configs/dataset/benchmark_vigor_chicago_rs_guided_518.yaml`
- 运行脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3.sh`
- 运行脚本：`bash_scripts/benchmark/rs_guided_dense_mv/vggt.sh`

当前输出指标：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`

### 12.2 Stage-1：遥感视图几何 benchmark

作用：

- 将遥感图像作为正式输入视图送入模型
- 使用 `exp_005` 输出的逐像素点图作为监督
- 单独评测模型对遥感视图的几何重建能力

对应文件：

- metadata 准备脚本：`scripts/prepare_rs_aerial_benchmark_metadata.py`
- dataset loader：`mapanything/datasets/wai/vigor_chicago_rs_aerial.py`
- 入口脚本：`benchmarking/rs_guided_dense_mv/benchmark_stage1.py`
- 顶层配置：`configs/rs_aerial_stage1_benchmark.yaml`
- 数据集配置：`configs/dataset/benchmark_vigor_chicago_rs_aerial_stage1.yaml`
- 运行脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage1.sh`

当前输出指标：

- `rs_pointmap_abs_rel`
- `rs_height_mae`
- `rs_height_rmse`

重要说明：

- Stage-1 当前是“遥感视图单独几何评测”，不是联合 benchmark
- 因此结果文件里只会有遥感视图几何指标，不会同时输出空中视图指标
- 如果要同时输出空中与遥感指标，需要进入后续 Stage-2 的 joint benchmark 实现

## 13. 当前数据处理链路

### 13.1 空中视图数据

空中视图来自：

- `outputs/experiments/exp_001_reconstrc/vigor_chicago_processed`

已经转换为：

- `outputs/dataset/vigor_chicago_wai`

并生成 split / metadata：

- `outputs/dataset/mapanything_metadata/vigor_chicago`

### 13.2 遥感视图数据

遥感标签来自：

- `outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/`

当前正式使用的监督文件：

- `pixel_to_point_map.npz['xyz']`：逐像素全局三维点
- `valid_mask.npy`：有效像素掩码
- `height_map.npy`：逐像素高度
- `info.json`：投影元信息

遥感原始图像当前从：

- `dataset/Vigor/map/chicago_subset_2000/location_x/<provider>.png`

读取。

### 13.3 Stage-1 manifest

为避免重写整个 WAI scene，当前 Stage-1 先使用 manifest 组织 paired data。

manifest 根目录：

- `outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial`

当前已生成内容包括：

- `train/vigor_chicago_rs_aerial_train.json`
- `train/vigor_chicago_rs_aerial_scene_list_train.npy`
- 每个 scene 的单独 manifest
- 缺失 scene 列表
- `summary.json`

当前可用覆盖：

- train 请求 40 个 scene，可用 7 个
- val 请求 5 个 scene，可用 0 个
- test 请求 5 个 scene，可用 0 个

这意味着当前 Stage-1 只能在这 7 个 scene 上运行。

## 14. 当前已完成的运行验证

### 14.1 Stage-0 最小测试

已验证：

- `pi3` 在 Stage-0 benchmark 上可以正常运行
- 输出目录：`outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/pi3_smoke_stage0`

输出指标：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`

### 14.2 Stage-1 最小测试

已验证：

- `pi3` 在 7 个可用遥感 scene 上可以正常运行 Stage-1 benchmark
- 输出目录：`outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/pi3_stage1_7scenes_v2`

该目录只包含遥感视图几何指标，这是当前设计预期，不是遗漏。

当前平均结果：

- `rs_pointmap_abs_rel: 1.4527127572468348`
- `rs_height_mae: 1.033700440611158`
- `rs_height_rmse: 1.1238195385251726`

重要说明：

- 这些结果当前是在归一化后的点图上计算，不是严格米制误差
- 其作用是先验证 benchmark 主体、数据接口、模型前向和标签比较链路已经打通

## 15. 下一步

后续建议按下面顺序推进：

1. 扩充 `exp_005` 覆盖范围，使 val / test split 也具备遥感标签
2. 为 `vggt` 增加 Stage-1 运行脚本并形成横向对比
3. 设计并实现 Stage-2 joint benchmark，使同一 scene 中的 aerial 指标与 remote 指标可以同时报告
4. 在 Stage-2 中再定义跨视图全局一致性指标
