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

### Stage-2：联合 benchmark 骨架

这一阶段的最小目标不是立刻定义复杂的跨视图一致性公式，而是先把：

- 同一批 paired scenes 的空中视图指标
- 同一批 paired scenes 的遥感视图指标

统一写入同一个结果文件。

因此当前 Stage-2 的最小定义是：

- 输入：paired scenes 上的 aerial multi-view + remote image
- 输出：同一 scene 下同时报告 aerial metrics 与 remote metrics
- 暂不输出真正的 cross-view consistency 指标

这一阶段新增：

- Stage-2 joint benchmark 入口
- Stage-2 joint dataset 过滤逻辑
- 单个结果文件中的 paired-scene 指标汇总

当前暂不实现：

- 跨视图重叠区域距离误差
- 联合点云对齐后的整体 pointmap 误差
- 遥感视图位姿相关指标

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

1. 所有视图先被转换到 $view0$ 参考系
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
- 当前可运行实现：
  1. Stage-0：空中视图标准指标
  2. Stage-1：遥感视图几何指标
  3. Stage-2：paired-scene joint benchmark + `crossview_pointmap_gap_abs`
- 后续扩展方向：
  1. 引入共享全局坐标系下的 aerial / remote 联合推理
  2. 增加真正的点云重叠区域一致性指标
  3. 增加联合点云整体误差

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


### 12.3 Stage-2：联合 benchmark 骨架

作用：

- 在同一批 paired scenes 上同时计算 aerial metrics 和 remote metrics
- 保持输出形式与现有 benchmark 一致，按 scene / dataset 汇总
- 为后续真正的 cross-view consistency 指标预留位置

对应文件：

- 入口脚本：`benchmarking/rs_guided_dense_mv/benchmark_stage2.py`
- 顶层配置：`configs/rs_aerial_stage2_benchmark.yaml`
- 数据集配置：`configs/dataset/benchmark_vigor_chicago_rs_aerial_stage2.yaml`
- 运行脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage2.sh`

当前输出指标：

- 空中视图：`pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_ate_rmse`、`pose_auc_5`、`ray_dirs_err_deg`
- 遥感视图：`rs_pointmap_abs_rel`、`rs_height_mae`、`rs_height_rmse`

当前边界：

- 只对同时具备 aerial / remote 数据的 paired scenes 统计结果
- 当前已实现第一个真正的 cross-view 指标：`crossview_pointmap_gap_abs`
- 该指标定义为同一 scene 上 `pointmaps_abs_rel` 与 `rs_pointmap_abs_rel` 的绝对差，越小越好
- 之所以先采用该定义，是因为它对当前 aerial / remote 分别推理、输出坐标系不共享的现状保持不变性
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


## 16. 当前确认的评测协议（2026-03-22）

本节用于覆盖前面偏探索性的部分，作为当前版本的正式 benchmark 思路。

### 16.1 Aerial-only 指标

Aerial-only 的主指标保留为：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`
- `pose_auc_5`
- `ray_dirs_err_deg`
- `metric_scale_abs_rel`

此外，Aerial-only 还应增加一项 metric-space 点云误差，当前建议使用：

- `metric_point_l1`

选择 `metric_point_l1` 而不是“不归一化的 `pointmaps_abs_rel`”，原因是：

- `pointmaps_abs_rel` 的定义是相对误差，本质仍然会被每个点的 GT 距离归一化
- 即使不做 multi-view 的联合归一化，它也不是直接的米制误差
- 对真实尺度数据来说，`metric_point_l1` 的单位直接是米，解释最直接
- `metric_point_l1` 能和 `metric_scale_abs_rel` 形成互补：
  - `metric_scale_abs_rel` 评估整体尺度因子是否准确
  - `metric_point_l1` 评估真实尺度下三维点到底偏了多少米

因此当前建议是：

- 保留归一化几何指标用于和现有 MapAnything benchmark 对齐
- 额外增加 `metric_point_l1` 作为真实尺度主指标之一

### 16.2 `metric_scale_abs_rel` 与 `metric_point_l1` 的区别

两者不等价，分别回答不同问题：

- `metric_scale_abs_rel`
  - 关注：模型恢复出来的“整体尺度因子”准不准
  - 不直接反映局部几何形状误差
- `metric_point_l1`
  - 关注：真实米制空间中，点云的绝对位置误差有多大
  - 同时受尺度、位姿、局部几何影响

因此 benchmark 中两者都应保留，而不是互相替代。

### 16.3 `is_metric_scale=True`

当前 VIGOR Chicago aerial dataset 中已经显式设置：

- `Models/map-anything/mapanything/datasets/wai/vigor_chicago.py`

其中：

- `self.is_metric_scale = True`

这与当前数据实际情况一致，因为：

- 深度和点云都是真实尺度
- 相机位姿也在真实尺度下定义

### 16.4 RS-only 指标

RS-only 不再把 `rs_pointmap_abs_rel` 作为主指标。

原因是：

- 遥感视图不是标准 pinhole 投影
- `pixel_to_point_map.npz['xyz']` 的几何标签中，真正有判别意义的是高度结构
- 对当前任务而言，RS 预测结果在 z 轴上允许存在一个整体偏移自由度
- 因此直接比较完整 3D pointmap 不够合理

当前建议 RS-only 的正式指标为：

- `rs_height_mae`
- `rs_height_rmse`

并在计算前做每个 scene 的 z-offset 对齐：

- 仅在有效像素上估计一个全局偏移 $b$
- 用 $z^{pr}_{\mathrm{aligned}} = z^{pr} + b$ 后再计算高度误差

这组指标更符合“卫星图像高度恢复能力”的任务定义。

### 16.5 Joint 输入下的评测组织

最终 benchmark 应分三组实验：

- `Aerial-only`
- `RS-only`
- `Joint aerial + RS`

其中 `Joint aerial + RS` 下需要比较两类结果：

1. 联合输入后，Aerial 自身任务是否提升
   - 相对 `Aerial-only` 对比 Aerial 指标
2. 联合输入后，RS 自身任务是否提升
   - 相对 `RS-only` 对比 RS 高度指标

也就是说，joint benchmark 的第一目的不是只报告一个融合分数，而是回答：

- 联合重建是否让 aerial 重建变好
- 联合重建是否让 RS 高度恢复变好

### 16.6 Joint 空间一致性评测

在 `Joint aerial + RS` 下，还需要单独评测：

- 空中视图和卫星视图的预测结果是否进入了同一个全局坐标空间

这里必须注意：

- 现有 dense benchmark 里的 aerial 点云通常是在 $view0$ 参考系下比较
- 但卫星标签 `pixel_to_point_map.npz['xyz']` 本身就在全局坐标系下

因此在 joint 空间一致性评测中：

- Aerial 预测点云不能停留在 $view0$ 坐标系
- 必须再转换到全局坐标系下
- 再与 RS 点云以及整体 GT 共同比较

这部分后续应新增真正的 joint metric，例如：

- `joint_global_point_l1`
- 或其他全局坐标系下的联合点云误差

当前不再把 `crossview_pointmap_gap_abs` 作为正式目标指标。

### 16.7 分辨率协议

benchmark 必须固定分辨率协议，否则结果不可比。

当前第一版建议统一使用：

- aerial 输入：`518 x 518`
- RS 输入：`518 x 518`

这样做的原因是：

- 先与现有 MapAnything / Pi3 / VGGT wrapper 兼容
- 先保证 benchmark 稳定、可复现

但需要明确说明，MapAnything 在 resize / crop 时并不是“对所有标签做连续插值”。

现有实现中：

- 图像使用高质量插值缩放（缩小时 Lanczos，放大时 bicubic）
- 深度图使用最近邻缩放
- 其他几何量若作为 `additional_quantities` 传入，也使用最近邻缩放
- 相机内参会同步更新

对应实现见：

- `Models/map-anything/mapanything/utils/cropping.py`
- `Models/map-anything/mapanything/datasets/base/base_dataset.py`

这套设计的目的就是避免你担心的问题：

- 不对深度或离散几何标签做双线性插值
- 尽量避免生成“悬空点”或无物理意义的中间值

但风险并没有完全消失，尤其对 RS 的逐像素点图更敏感。因此：

- aerial 数据可以继续沿用 MapAnything 现有的 resize/crop 协议
- RS 数据建议在 benchmark 中显式固定：
  - 图像：双线性
  - `valid_mask`：最近邻
  - `height_map`：最近邻
  - `pixel_to_point_map`：最近邻

后续如果发现 `518 x 518` 对 RS 高度标签损伤过大，再单独增加高分辨率协议版本，但不应在同一个 benchmark 版本中混用分辨率。


## 17. Benchmark 指标表（当前建议版）

本节把当前已经讨论达成一致的指标整理成正式表格，作为后续实现和实验报告的直接依据。

说明：

- `Aerial-only`：只输入空中透视图像
- `RS-only`：只输入卫星影像
- `Joint aerial + RS`：同时输入空中透视图像和卫星影像
- “是否 metric”表示该指标是否直接工作在真实尺度语义下，而不是仅在归一化几何空间下比较
- “主指标/辅助指标”表示当前 benchmark 设计中的优先级，不代表辅助指标不重要，只表示第一版报告中不一定必须作为主表呈现

### 17.1 Aerial-only 指标表

| 指标名 | 任务阶段 | 定义对象 | 计算空间 / 对齐方式 | 是否 metric | 指标含义 | 当前建议角色 |
|---|---|---|---|---|---|---|
| `pointmaps_abs_rel` | Aerial-only | 空中视图的逐像素全局点图 | 先将同一 multi-view set 的 GT 和预测分别转换到各自的 $view0$ 参考系，再对整个 multi-view set 联合归一化后计算相对误差 | 否 | 衡量模型恢复的整体三维几何结构是否正确，重点反映全局点图的相对几何质量与多视图自洽性 | 主指标 |
| `z_depth_abs_rel` | Aerial-only | 空中视图的 z-depth | 与 `pointmaps_abs_rel` 相同，先在 $view0$ 参考系下组织，再使用联合归一化后的深度进行比较 | 否 | 衡量深度恢复质量，重点反映每个视图在深度方向上的几何精度 | 主指标 |
| `pose_ate_rmse` | Aerial-only | 多视图相机轨迹 | 在 $view0$ 参考系下比较预测与 GT 位姿；平移量受联合归一化影响，旋转不受绝对尺度影响 | 否 | 衡量相机轨迹恢复质量；如果某些视图的 pose 预测明显错误，该指标会明显恶化 | 主指标 |
| `pose_auc_5` | Aerial-only | 相对位姿误差 | 在 $view0$ 参考系下比较各视图间相对位姿关系，以 5 度阈值统计 AUC | 否 | 衡量多视图之间的相对位姿关系是否正确，适合判断模型是否建立了稳定的视图几何关系 | 主指标 |
| `ray_dirs_err_deg` | Aerial-only | 像素射线方向 | 直接比较 GT 与预测的 ray direction 角度误差，不依赖绝对尺度 | 否 | 衡量像素对应射线方向是否正确，可视作相机几何和局部投影关系质量的指标 | 主指标 |
| `metric_scale_abs_rel` | Aerial-only | 整体尺度因子 | 仅当数据集为 metric 且模型显式输出 `metric_scaling_factor` 时可计算；比较预测尺度因子与 GT 尺度因子的相对误差 | 是 | 衡量模型是否恢复出了正确的绝对尺度量级；它评估的是“尺度因子是否准”，不是点云本身的空间误差 | 主指标 |
| `metric_point_l1` | Aerial-only | 真实尺度下的三维点 | 在真实全局坐标或真实尺度语义下比较预测点与 GT 点的 L1 误差，单位可直接解释为米 | 是 | 衡量真实尺度空间中的点云误差，直接回答“平均每个点偏了多少米”；可和 `metric_scale_abs_rel` 互补使用 | 主指标 |

### 17.2 为什么 Aerial-only 要同时保留 `metric_scale_abs_rel` 和 `metric_point_l1`

| 指标名 | 回答的问题 | 优点 | 局限 |
|---|---|---|---|
| `metric_scale_abs_rel` | 模型恢复出来的整体尺度量级准不准？ | 对“绝对尺度恢复能力”非常直接；适合判断模型是否有真实尺度意识 | 不能反映局部点云形状是否正确，也不能反映位姿或局部几何细节 |
| `metric_point_l1` | 真实尺度空间里，点云平均偏了多少米？ | 结果有清晰物理意义，最适合直接报告 metric 误差 | 同时受到尺度、位姿、局部结构误差影响，不能单独说明问题出在尺度还是几何 |

因此：

- `metric_scale_abs_rel` 评估“尺度是否准”
- `metric_point_l1` 评估“空间位置是否准”
- 两者不互相替代，建议共同保留

### 17.3 为什么不选“不归一化的 `pointmaps_abs_rel`”作为 metric-space 指标

| 方案 | 问题 |
|---|---|
| 不归一化的 `pointmaps_abs_rel` | 依然是相对误差，本质上会被 `||gt||` 归一化，结果不是直接的米制误差 |
| `metric_point_l1` | 直接在真实尺度下比较，单位明确、解释最直接 |

因此当前建议：

- 保留归一化后的 `pointmaps_abs_rel` 用于和现有 MapAnything benchmark 体系对齐
- 另行增加 `metric_point_l1` 作为真正的 metric-space 点云误差指标

### 17.4 RS-only 指标表

| 指标名 | 任务阶段 | 定义对象 | 计算空间 / 对齐方式 | 是否 metric | 指标含义 | 当前建议角色 |
|---|---|---|---|---|---|---|
| `rs_height_mae` | RS-only | 卫星影像逐像素高度 | 在有效像素上先估计每个 scene 的 z-offset，并用 $z^{pr}_{\mathrm{aligned}} = z^{pr} + b$ 做对齐，然后计算 MAE | 是 | 衡量模型在卫星影像输入下恢复高度结构的平均绝对误差；因为允许 z 轴整体偏移，所以更符合当前 RS 任务定义 | 主指标 |
| `rs_height_rmse` | RS-only | 卫星影像逐像素高度 | 与 `rs_height_mae` 相同，先做每个 scene 的 z-offset 对齐，再计算 RMSE | 是 | 对较大高度误差更敏感，适合衡量建筑高度、边缘区域或局部失败区域带来的较大误差 | 主指标 |

### 17.5 为什么 RS-only 不把 `rs_pointmap_abs_rel` 作为主指标

| 原因 | 说明 |
|---|---|
| 卫星影像不是标准 pinhole 投影 | 因此直接复用普通 pointmap 相对误差并不完全符合该视图的几何性质 |
| RS 标签真正有判别意义的是高度结构 | 当前任务更关心建筑、地面、地形的高度恢复，而不是完整 3D 点图每一维都严格服从普通透视几何 |
| z 轴存在整体偏移自由度 | 如果不做 z-offset 对齐，直接比较完整 pointmap 会把不应惩罚的偏移也算进误差 |

因此 RS-only 的正式主指标建议收敛为：

- `rs_height_mae`
- `rs_height_rmse`

### 17.6 Joint aerial + RS 指标表

Joint 阶段需要分成两类问题来评估：

1. 联合输入后，各自任务是否提升
2. 联合输入后，Aerial 与 RS 的结果是否进入了同一个全局空间

#### 17.6.1 Joint 下的“各自任务是否提升”指标

| 指标组 | 具体指标 | 比较方式 | 指标用途 |
|---|---|---|---|
| Aerial 侧 | `pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_ate_rmse`、`pose_auc_5`、`ray_dirs_err_deg`、`metric_scale_abs_rel`、`metric_point_l1` | 将 `Joint aerial + RS` 的结果与 `Aerial-only` 对比 | 判断引入卫星影像后，空中视图重建是否变好 |
| RS 侧 | `rs_height_mae`、`rs_height_rmse` | 将 `Joint aerial + RS` 的结果与 `RS-only` 对比 | 判断引入空中视图后，卫星视图高度恢复是否变好 |

#### 17.6.2 Joint 下的“空间一致性”指标

| 指标名 | 定义对象 | 计算空间 / 对齐方式 | 是否 metric | 指标含义 | 当前建议角色 |
|---|---|---|---|---|---|
| `joint_global_point_l1` | Aerial 与 RS 联合预测结果 | Aerial 预测点云不能只停留在 $view0$ 参考系，而应进一步变换到全局坐标系；RS 标签本身在全局坐标系下；然后与整体 GT 全局点云比较 | 是 | 衡量联合输入后，空中视图与卫星视图的预测结果是否真正进入了同一个全局空间，并在真实尺度下配准准确 | 主指标（待实现） |

### 17.7 当前不建议作为正式主指标的量

| 指标名 | 当前处理建议 | 原因 |
|---|---|---|
| `rs_pointmap_abs_rel` | 可保留为分析量，但不放入正式主表 | 对 RS 视图而言，完整 pointmap 相对误差不如高度误差更符合任务定义 |
| `crossview_pointmap_gap_abs` | 可保留为辅助分析量，但不作为正式目标指标 | 它更像“两个任务误差差距”的分析量，而不是“联合重建是否进入同一空间”的直接证据 |

### 17.8 分辨率协议表

| 项目 | 当前建议 | 处理规则 | 说明 |
|---|---|---|---|
| Aerial 输入分辨率 | `518 x 518` | 按 MapAnything 现有 crop/resize 协议处理，图像高质量缩放、深度最近邻、内参同步更新 | 先与现有 Pi3 / VGGT / MapAnything wrapper 保持兼容 |
| RS 输入分辨率 | `518 x 518` | 图像双线性；`valid_mask` 最近邻；`height_map` 最近邻；`pixel_to_point_map` 最近邻 | 避免对几何标签做连续插值，从而减少产生悬空点或错误高度标签的风险 |
| 是否允许混合分辨率 | 不建议 | 同一个 benchmark 版本中不混用多个分辨率协议 | 否则不同模型结果不可直接比较 |

### 17.9 关于分辨率与标签风险的说明

你担心的风险是成立的：

- 如果对深度、点云、高度图做双线性插值
- 会生成本不存在的中间值
- 对几何标签来说，这些中间值可能没有物理意义

MapAnything 当前的处理原则是：

- 图像可做连续插值
- 几何标签尽量用最近邻
- 相机内参同步更新

这也是仓库里很多原始数据分辨率并不是 `518`，但依然可以统一 benchmark 的原因。

不过对 RS 标签，这个问题更敏感，因此当前 benchmark 文档明确要求：

- `pixel_to_point_map`
- `height_map`
- `valid_mask`

统一按最近邻处理，不使用双线性或 bicubic。


## 18. 指标的对齐方式与公式说明

本节补充每个指标在计算前需要做什么对齐，以及建议采用的公式形式。

需要先强调一个总原则：

- 正式主指标默认不使用后验最优 $\mathrm{SE}(3)$ / $\mathrm{Sim}(3)$ 配准
- 也就是说，不额外给模型一个“把预测整体再拟合到 GT”的机会
- 这样做的目的是保留 benchmark 对模型真实几何恢复能力、位姿恢复能力和尺度恢复能力的约束

因此，当前 benchmark 里的“对齐”主要是以下几种：

1. $view0$ 参考系变换
2. multi-view 联合尺度归一化
3. RS 高度的 per-scene z-offset 对齐
4. Joint 阶段把 aerial 点云恢复到全局坐标系

### 18.1 各指标需要的对齐方式总表

| 指标名 | 是否需要 $view0$ 参考系变换 | 是否需要联合尺度归一化 | 是否需要 z-offset 对齐 | 是否需要恢复到全局坐标系 | 是否使用后验 $\mathrm{SE}(3)$ / $\mathrm{Sim}(3)$ 最优对齐 |
|---|---|---|---|---|---|
| `pointmaps_abs_rel` | 是 | 是 | 否 | 否 | 否 |
| `z_depth_abs_rel` | 是 | 是 | 否 | 否 | 否 |
| `pose_ate_rmse` | 是 | 是（平移部分） | 否 | 否 | 否 |
| `pose_auc_5` | 是 | 否（主要是角度/相对关系） | 否 | 否 | 否 |
| `ray_dirs_err_deg` | 否 | 否 | 否 | 否 | 否 |
| `metric_scale_abs_rel` | 否 | 否（比较的是尺度因子本身） | 否 | 否 | 否 |
| `metric_point_l1` | 否 | 否 | 否 | 是（如果在全局坐标下评测） | 否 |
| `rs_height_mae` | 否 | 否 | 是 | 否 | 否 |
| `rs_height_rmse` | 否 | 否 | 是 | 否 | 否 |
| `joint_global_point_l1` | 否 | 否 | 可选仅对 RS 高度分支做 z-offset 分析版 | 是 | 否 |

### 18.2 $view0$ 参考系变换

在 `dense_n_view` 风格的评测里，Aerial multi-view 的 GT 和 prediction 不会直接在原始世界坐标中逐点比较，而是先分别变换到各自的 $view0$ 参考系中。

设：

- 第 $0$ 个视图的 GT 位姿为 $T_0^{gt}$
- 第 $i$ 个视图的 GT 世界点为 $\mathbf{P}_i^{gt}$
- 第 $0$ 个视图的预测位姿为 $T_0^{pr}$
- 第 $i$ 个视图的预测世界点为 $\mathbf{P}_i^{pr}$

则：

$$
\tilde{\mathbf{P}}_i^{gt} = (T_0^{gt})^{-1} \, \mathbf{P}_i^{gt}
$$

$$
\tilde{\mathbf{P}}_i^{pr} = (T_0^{pr})^{-1} \, \mathbf{P}_i^{pr}
$$

这一步的意义是：

- GT 在 GT 自己的 $view0$ 参考系下组织
- prediction 在 prediction 自己的 $view0$ 参考系下组织
- 不做额外后验刚体拟合

### 18.3 multi-view 联合尺度归一化

对 `pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_ate_rmse` 来说，当前 benchmark 还会对同一个 multi-view set 的所有视图做联合归一化。

设一个 scene 内所有有效点组成集合 $\mathcal{M}$，点为 $\mathbf{P}_k$，则归一化因子为：

$$
s = \frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} \left\lVert \mathbf{P}_k \right\rVert_2
$$

归一化点云为：

$$
\hat{\mathbf{P}}_k = \mathbf{P}_k / s
$$

这样做的作用是：

- 去掉绝对尺度
- 保留多视图内部相对几何结构

因此：

- `pointmaps_abs_rel`
- `z_depth_abs_rel`
- `pose_ate_rmse`

本质上都不是直接米制误差，而是归一化几何误差。

### 18.3.1 符号约定

为避免公式渲染歧义，并尽量与 `MapAnything` 代码中的 `gt` / `pr` 命名保持一致，本文档统一采用以下记号：

- GT 用上标 $^{gt}$ 表示
- Prediction 用上标 $^{pr}$ 表示
- 三维点统一写为 $\mathbf{P}$
- 射线方向统一写为 $\mathbf{r}$
- 相机平移统一写为 $\mathbf{t}$
- 有效像素集合统一写为 $\mathcal{M}$
- Aerial / RS 的有效集合分别写为 $\mathcal{M}_a$、$\mathcal{M}_r$
- 指标名若写入公式左侧，统一用 \mathrm{...} 包裹，避免被误解析为上下标表达式

例如：

- $\mathbf{P}_i^{gt}$ 表示 GT 的第 $i$ 个三维点
- $\mathbf{P}_i^{pr}$ 表示 prediction 的第 $i$ 个三维点
- $\mathrm{metric\_point\_l1}$ 表示指标名，而不是带下标的变量

### 18.4 `pointmaps_abs_rel`

定义对象：

- Aerial multi-view 中每个视图的逐像素全局点图

建议公式：

对每个有效像素 `i`：

$$
e_i = \frac{\left\lVert \hat{\mathbf{P}}_i^{pr} - \hat{\mathbf{P}}_i^{gt} \right\rVert_2}{\left\lVert \hat{\mathbf{P}}_i^{gt} \right\rVert_2 + \epsilon}
$$

然后对有效像素平均：

$$
\mathrm{pointmaps\_abs\_rel} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} e_i
$$

其中：

- $\hat{\mathbf{P}}_i^{gt}$ 和 $\hat{\mathbf{P}}_i^{pr}$ 表示经过 $view0$ 参考系变换和联合尺度归一化后的点
- $\mathcal{M}$ 为有效像素集合

### 18.5 `z_depth_abs_rel`

定义对象：

- Aerial multi-view 中相机坐标系下的 z-depth

建议公式：

$$
e_i = \frac{\left| \hat{z}_i^{pr} - \hat{z}_i^{gt} \right|}{\left| \hat{z}_i^{gt} \right| + \epsilon}
$$

整体指标为：

$$
\mathrm{z\_depth\_abs\_rel} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} e_i
$$

其中 $\hat{z}$ 表示经过联合归一化后的 z-depth。

### 18.6 `pose_ate_rmse`

定义对象：

- 多视图相机轨迹的平移误差

设在 $view0$ 参考系、并经过相同尺度规范后的位姿平移为：

- $\mathbf{t}_k^{gt}$
- $\mathbf{t}_k^{pr}$

则：

$$
\mathrm{pose\_ate\_rmse} = \sqrt{\frac{1}{N} \sum_k \left\lVert \mathbf{t}_k^{pr} - \mathbf{t}_k^{gt} \right\rVert_2^2}
$$

说明：

- 当前不做额外的轨迹后验拟合
- 如果模型位姿本身错误，该项会直接变差

### 18.7 `pose_auc_5`

定义对象：

- 视图间相对位姿关系

对每对视图，先计算：

- 相对旋转误差 $err_r$
- 相对平移方向误差 $err_t$

然后构造阈值 $\tau \in [0, 5^\circ]$ 内的 inlier 曲线，AUC 为：

$$
\mathrm{pose\_auc\_5} = \int_0^{5^\circ} \mathrm{InlierRate}(\tau) \, d\tau
$$

实现上通常会再归一化成百分数。

### 18.8 `ray_dirs_err_deg`

定义对象：

- GT 和 prediction 的单位射线方向

设：

- $\mathbf{r}_i^{gt}$
- $\mathbf{r}_i^{pr}$

则角度误差为：

$$
\theta_i = \arccos\!\left( \mathrm{clamp}(\langle \mathbf{r}_i^{pr}, \mathbf{r}_i^{gt} \rangle, -1, 1) \right)
$$

整体指标：

$$
\mathrm{ray\_dirs\_err\_deg} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \theta_i
$$

该指标：

- 不需要 $\mathrm{SE}(3)$ / $\mathrm{Sim}(3)$ 对齐
- 也基本不依赖绝对尺度

### 18.9 `metric_scale_abs_rel`

定义对象：

- 模型输出的整体 metric scaling factor

设：

- GT 的尺度因子为 $s^{gt}$
- 预测的尺度因子为 $s^{pr}$

则：

$$
\mathrm{metric\_scale\_abs\_rel} = \frac{|s^{pr} - s^{gt}|}{|s^{gt}| + \epsilon}
$$

该指标的前提是：

- 数据集 `is_metric_scale=True`
- 模型显式输出 `metric_scaling_factor`

因此它不是每个模型都一定能提供的指标。

### 18.10 `metric_point_l1`

定义对象：

- 真实尺度下的三维点云

设：

- $\mathbf{P}_i^{gt}$ 为 GT 全局坐标下的 3D 点
- $\mathbf{P}_i^{pr}$ 为 prediction 在真实尺度、真实坐标语义下的 3D 点

则：

$$
\mathrm{metric\_point\_l1} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\lVert \mathbf{P}_i^{pr} - \mathbf{P}_i^{gt} \right\rVert_1
$$

如果实现中更偏好欧氏距离，也可以定义 `metric_point_l2` 或 `metric_point_rmse`，但当前建议主表优先保留 `metric_point_l1`，原因是：

- 它的单位是米
- 对真实尺度误差最直接
- 不会再像相对误差那样被 GT 距离归一化

### 18.11 RS-only 的 z-offset 对齐

RS-only 当前建议只对高度进行评测，并允许每个 scene 在 z 方向上做一个整体偏移对齐。

设：

- GT 高度为 $h_i^{gt}$
- 预测高度为 $h_i^{pr}$
- 有效像素集合为 $\mathcal{M}$

先计算全局 z 偏移：

$$
b = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left( h_i^{gt} - h_i^{pr} \right)
$$

然后得到对齐后的预测高度：

$$
h_{i,\mathrm{aligned}}^{pr} = h_i^{pr} + b
$$

这样做的原因是：

- 对当前卫星视图任务，z 轴存在一个整体偏移自由度
- 我们真正关心的是高度结构，而不是绝对 z 零点

### 18.12 `rs_height_mae`

$$
\mathrm{rs\_height\_mae} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left| h_{i,\mathrm{aligned}}^{pr} - h_i^{gt} \right|
$$

### 18.13 `rs_height_rmse`

$$
\mathrm{rs\_height\_rmse} = \sqrt{\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left( h_{i,\mathrm{aligned}}^{pr} - h_i^{gt} \right)^2}
$$

这两项都是 metric 指标，单位可直接解释为米。

### 18.14 Joint 阶段为什么需要恢复到全局坐标系

在 `Joint aerial + RS` 中：

- Aerial 分支如果继续停留在 $view0$ 参考系，就无法直接和 RS 的全局点图比较
- 但卫星标签 `pixel_to_point_map.npz['xyz']` 本身就在全局坐标系中

因此 joint 空间一致性评测时，Aerial 预测点云必须先从 $view0$ 坐标系恢复到全局坐标系。

设：

- Aerial 在 $view0$ 参考系中的预测点为 $\mathbf{P}_i^{aerial,view0}$
- 预测的 $view0 \to global$ 变换为 $T_{view0 \rightarrow global}^{pr}$

则：

$$
\mathbf{P}_i^{aerial,global} = T_{view0 \rightarrow global}^{pr} \, \mathbf{P}_i^{aerial,view0}
$$

之后才能与：

- RS 的全局点
- 以及整体 GT 全局点

放在同一个空间里比较。

### 18.15 `joint_global_point_l1`

该指标当前尚未实现，但定义建议如下：

设：

- $\mathcal{M}_a$ 为 Aerial 分支有效点集合
- $\mathcal{M}_r$ 为 RS 分支有效点集合

则联合误差可定义为：

$$
\mathrm{joint\_global\_point\_l1} = \frac{1}{|\mathcal{M}_a| + |\mathcal{M}_r|} \left[ \sum_{i \in \mathcal{M}_a} \left\lVert \mathbf{P}_i^{aerial,global,pr} - \mathbf{P}_i^{aerial,global,gt} \right\rVert_1 + \sum_{j \in \mathcal{M}_r} \left\lVert \mathbf{P}_j^{rs,global,pr} - \mathbf{P}_j^{rs,global,gt} \right\rVert_1 \right]
$$

该指标的目标是：

- 评估联合输入后，Aerial 和 RS 是否真正进入了同一个全局 metric 空间
- 并在真实尺度下配准准确

### 18.16 为什么当前不建议对主指标做后验 $\mathrm{SE}(3)$ / $\mathrm{Sim}(3)$ 最优对齐

如果在正式主指标前额外做：

- $\mathrm{SE}(3)$ 对齐
- 或 $\mathrm{Sim}(3)$ 对齐

会带来一个问题：

- 模型本身的位姿错误、尺度错误、空间漂移，会被后验对齐部分抵消

这会削弱 benchmark 的约束力。

因此当前建议：

- 正式主表不做后验最优配准
- 如果后续需要，可以额外报告一组分析指标，例如：
  - $\mathrm{Sim3}$-aligned $\mathrm{metric\_point\_l1}$
  - $\mathrm{Sim3}$-aligned $\mathrm{joint\_global\_point\_l1}$

但它们应作为辅助分析结果，而不是正式主指标。


## 19. 当前统一 benchmark 实现

当前统一 benchmark 已经支持三种评测模式在一次运行中同时输出：

- `aerial_only`
- `rs_only`
- `joint`

其中 `joint` 当前的含义是：

- 将同一 scene 的 aerial multi-view 与一张 RS 图像一起输入模型前向
- 分别读取联合前向下的 aerial 预测结果与 RS 预测结果
- 再与 `aerial_only`、`rs_only` 的 baseline 做差，得到提升量

因此当前统一 benchmark 已经能够回答：

- 加入 RS 输入后，aerial 重建是否提升
- 加入 aerial 输入后，RS 高度恢复是否提升

当前仍未实现的部分是：

- `joint_global_point_l1` 已实现
- 它在真实全局坐标系下联合统计 aerial 点云与 RS 点云的 L1 误差，用于衡量二者是否共同进入同一个空间

当前正式实现已经收敛为统一入口：

- benchmark 入口：`benchmarking/rs_guided_dense_mv/benchmark_unified.py`
- 顶层配置：`configs/rs_aerial_benchmark.yaml`
- 数据配置：`configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml`
- 启动脚本：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified.sh`

当前统一 benchmark 的执行范围为：

- `aerial_only`
  - 输出：`pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_ate_rmse`、`pose_auc_5`、`ray_dirs_err_deg`、`metric_scale_abs_rel`、`metric_point_l1`
- `rs_only`
  - 输出：`rs_height_mae`、`rs_height_rmse`
- `joint`
  - 当前仅保留结果结构占位
  - 目标指标为 `joint_global_point_l1`
  - 当前尚未在正式 benchmark 中实现

统一 benchmark 的结果文件组织为：

- `rs_aerial_benchmark_results.json`
- `rs_aerial_per_scene_results.json`

其中主结果 JSON 结构固定为：

- `metadata`
- `aerial_only`
- `rs_only`
- `joint`
- `improvement`
- `per_scene_results`

当前仍保留的 `benchmark.py`、`benchmark_stage1.py`、`benchmark_stage2.py` 仅作为历史实现与迁移参考，不再作为正式对外入口。


## 20. 结果可视化建议

当前 benchmark 结果建议配套使用：

- `scripts/visualize_rs_aerial_benchmark.py`

输入：

- `rs_aerial_benchmark_results.json`

输出：

- `summary.md`
  - 适合直接阅读平均指标与 improvement 表格
- `per_scene_compact.csv`
  - 适合按 scene 做透视表、排序和散点图
- `scene_ranking.csv`
  - 适合快速查看哪些 scene 在 joint 输入下提升最大或退化最明显

建议的可视化方式：

- 平均指标总表：直接阅读 `summary.md`
- scene 排名条形图：横轴使用 `aerial_gain_pointmaps_abs_rel` 或 `rs_gain_height_mae`
- baseline vs joint 对比散点图：
  - x 轴：`aerial_only.pointmaps_abs_rel`
  - y 轴：`joint.pointmaps_abs_rel`
  - 落在对角线下方表示 joint 更好
- RS 高度对比散点图：
  - x 轴：`rs_only.rs_height_mae`
  - y 轴：`joint.rs_height_mae`
- joint 空间一致性排序图：
  - 使用 `joint_global_point_l1` 做 scene 级排序


## 20. 2026-03-25 统一 benchmark 指标适用范围修订

### 20.1 指标适用模型修订

当前统一 benchmark 中，指标按是否要求 **metric scale** 分为两类：

- 仅对显式输出 `metric_scaling_factor` 的模型报告：
  - `metric_scale_abs_rel`
  - `metric_point_l1`
  - `rs_height_mae`
  - `rs_height_rmse`
  - `joint_global_point_l1`
- 对所有模型都报告：
  - `pointmaps_abs_rel`
  - `z_depth_abs_rel`
  - `pose_ate_rmse`
  - `pose_auc_5`
  - `ray_dirs_err_deg`
  - `rs_height_mae_affine`
  - `rs_height_rmse_affine`
  - `joint_global_pointmaps_abs_rel`

在当前接入的 3 个模型中：

- `MapAnything`：报告全部上述指标
- `Pi3`：不报告 metric-only 指标，metric-only 字段留空
- `VGGT`：不报告 metric-only 指标，metric-only 字段留空

### 20.2 有效集合记号统一

为避免之前文档中 $M$ 含义不清，这里统一使用以下记号：

- $\mathcal{M}_a$：Aerial-only 指标中的有效像素集合；对每个 aerial 视图单独定义，来源于当前视图的 `valid_mask`，并进一步与 $\lVert \mathbf{P}^{gt} Vert_2 > 0$ 的条件求交
- $\mathcal{M}_r$：RS-only 指标中的有效像素集合；来源于 `remote_valid_mask`，并进一步要求 GT 与预测高度都是有限值
- $\mathcal{M}_{global}$：Joint 全局非 metric 指标中的有效像素集合；它由所有 aerial 视图与 RS 视图的有效像素并集组成，但在具体计算时仍按各自 view 的有效像素逐项累积

### 20.3 `pointmaps_abs_rel` 当前到底在评什么

当前 unified benchmark 中的 `pointmaps_abs_rel` **只评测 aerial 视图**，不包含 RS 视图。

其计算过程是：

1. 将同一个 multi-view set 的 GT aerial 点云与预测 aerial 点云分别转换到各自的 $view0$ 参考系；
2. 仅使用 **aerial 视图集合** 做联合尺度归一化；
3. 在每个视图自己的有效集合 $\mathcal{M}_a$ 上计算相对误差；
4. 再对多视图结果取平均。

因此：

- 这里的联合归一化 **不包含 RS 点云**；
- 这里的有效集合也不是“全体像素全集”，而是每个 aerial 视图自己的 $\mathcal{M}_a$。

### 20.4 RS 非 metric 指标修订

由于 Pi3 与 VGGT 当前不提供显式 metric scale 输出，因此对这两个模型：

- 不再把 `rs_height_mae` / `rs_height_rmse` 作为主报告指标；
- 改为报告仿射 z 对齐后的非 metric 指标：
  - `rs_height_mae_affine`
  - `rs_height_rmse_affine`

其对齐形式为：

$$
z^{pr}_{aligned} = a z^{pr} + b
$$

其中 $a, b$ 通过在 $\mathcal{M}_r$ 上做最小二乘拟合得到。

这样做的目的，是把非 metric 模型在 RS 任务上的“整体 z 缩放与偏移自由度”消掉，只评估其恢复相对高度结构的能力。

### 20.5 Joint 非 metric 指标修订

为了给所有模型提供统一的 Joint 全局一致性指标，新增：

- `joint_global_pointmaps_abs_rel`

其定义是：

1. 将 aerial 视图的全局点图与 RS 视图的全局点图一起组成一个联合点云集合；
2. 对 GT 联合集合与预测联合集合分别做一次 **全局联合尺度归一化**；
3. 在 $\mathcal{M}_{global}$ 上累计逐像素相对误差。

因此该指标：

- 不要求模型具有 metric scale；
- 可以同时用于 `Pi3`、`VGGT`、`MapAnything`；
- 比当前 aerial-only 的 `pointmaps_abs_rel` 更接近“联合输入后是否进入同一个全局空间”的问题定义。

### 20.6 可视化规则修订

统一 sweep 可视化现在采用：

- 颜色：区分模型
- 线型：区分是否输入 RS
  - `Aerial+RS`：实线
  - `Aerial-only`：虚线
- 横轴：显式固定为帧数 $\{2,4,8,16,24,32,40\}$

其中：

- `rs_height_mae` / `rs_height_rmse` 只对 `MapAnything` 出现曲线；
- `rs_height_mae_affine` / `rs_height_rmse_affine` 对所有模型都出现曲线；
- `joint_global_point_l1` 只对 `MapAnything` 出现曲线；
- `joint_global_pointmaps_abs_rel` 对所有模型都出现曲线。
