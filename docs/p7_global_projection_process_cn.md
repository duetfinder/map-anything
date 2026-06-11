# P7 Projection-Aux 中的 global 重建（详细说明）

本文面向当前 P7 remote pipeline，说明 `global` 投影重建的完整数学过程、数据流，以及每个参数是**预测值**还是**GT 标签**。  
重点对应以下代码路径：
- 训练重建逻辑：[mapanything/train/losses.py](/root/autodl-tmp/Models/map-anything/mapanything/train/losses.py)
- 推理导出/诊断：[scripts/export_pointcloud_ply.py](/root/autodl-tmp/Models/map-anything/scripts/export_pointcloud_ply.py)
- 标签/数据格式：[mapanything/datasets/wai/vigor_chicago_rs_common.py](/root/autodl-tmp/Models/map-anything/mapanything/datasets/wai/vigor_chicago_rs_common.py)
- 标签几何可视化复现：[scripts/reconstruct_remote_pointcloud_from_projection_aux.py](/root/autodl-tmp/Models/map-anything/scripts/reconstruct_remote_pointcloud_from_projection_aux.py)

## 1）global 的定义（核心思想）

`global` 是一种“低维几何参数化”：  
- 不直接对每个像素回归完整偏移向量（那是 `offset` 分支），  
- 而是把 xy 偏移写成  
  `offset_xy = rel_height * global_slope * global_dir_xy`。  

即每个像素都共享一个方向和坡度（`global_dir_xy/slope`），只有高度 `rel_height` 在像素上变化。

重建公式（训练中的定义）：

1. 先有像素级基准点：
   - `projected = remote_projection_projected_xyz_centered`
   - `center_xy = remote_projection_center_xy`
2. 用方向约束得到位移：
   - `dir_xy = normalize(global_dir_xy)`
   - `offset_xy = rel_height * global_slope * dir_xy`
3. 生成重建点（xy）：
   - `recon[..., :2] = projected[..., :2] + center_xy - offset_xy`
4. z 通常保持基准面：
   - `recon[..., 2] = projected[..., 2]`

对应训练代码里的 `_projection_reconstruct_points`。  

## 2）数据/变量来源（谁是预测，谁是 GT）

`global` 重建用到的量按来源可分三类：

### A. 预测值（模型输出）

- `remote_projection_rel_height_pred`
  - 来自 projection-aux 分支
  - 对应 `losses.py` 中的 `pred['remote_projection_rel_height_pred']`
- `remote_projection_global_dir_xy_pred`
  - 来自 projection-aux 分支
  - 对应 `pred['remote_projection_global_dir_xy_pred']`
- `remote_projection_global_slope_pred`
  - 来自 projection-aux 分支
  - 对应 `pred['remote_projection_global_slope_pred']`
- `pts3d`（或 point head 产物）
  - 来自 remote point head，用作某些导出场景下的 proxy base（非严格 GT 时）
  - 对应 `pred['pts3d']`

### B. GT 标签（projection_aux.npz）

- `remote_projection_projected_xyz_centered`
  - GT 的基准几何平面点（严格定义下推荐用于 gt base 重建）
- `remote_projection_center_xy`
  - GT 的 xy 中心偏移项
- `remote_projection_rel_height`
  - GT 像素级相对高度
- `remote_projection_global_dir_xy`
  - GT 全局方向
- `remote_projection_global_slope`
  - GT 全局坡度
- `valid_mask`
  - 监督掩码

### C. 不直接来自 GT 的可视化对齐量

- `projection_aux_gt_remote_dir` 里读取的 `pixel_to_point_map.npz` 只用于可视化统一尺度（`xyz align`、`gt_height_range`），不参与训练。
- 例如 `gt_height_range`、`gt_pointmap_unit_xy_zrange_flipz` 等模式属于诊断归一化策略，不是训练的真实几何真值。

## 3）三种 global 重建口径（当前对比）

### 口径 1：严格训练定义（推荐诊断）
- base: `projected_xyz_centered + projection_center_xy`（GT）
- 方向/坡度: 可切换为 GT 或 Pred
  - GT：`--projection_aux_use_gt_global_direction` + `--projection_aux_use_gt_global_slope`
  - Pred：不加上述两个开关（默认用预测值）
- 高度: 可按可视化策略做归一化对齐再反投影（`gt_height_range` 等）
- 特点：能还原 `losses.py` 的逻辑，便于验证 `global` head 是否学到了正确方向和坡度。

### 口径 2：纯推理默认口径
- base: 通常用 `pred['pts3d']`
- 方向/坡度: 默认用预测值
- 特点：可覆盖更多像素但会掺入 point head 的偏差，不等价于训练定义。

### 口径 3：self-contained grid global（新增实验）

这个口径对应用户提出的“不要依赖 GT projection base，只用 aux 多任务输出自身反投影”的思路。  
它不使用 `projected_xyz_centered`，也不使用 point head 的 `pts3d` 作为 base，而是直接用 normalized 像素网格作为 xy 基准：

```text
grid_xy = normalized_pixel_grid(x, y) in [-1, 1]
dir_xy = normalize(remote_projection_global_dir_xy_pred)
slope = remote_projection_global_slope_pred
h = remote_projection_rel_height_pred

offset_xy = h * slope * dir_xy
recon_xyz = [grid_xy - offset_xy, h]
```

这里的 `h`、`dir_xy`、`slope` 全部是模型预测值；`grid_xy` 是图像坐标派生值，不是 GT。  
训练时新增 `projection_grid_global_to_gt_loss`，把 `recon_xyz` 和 GT `remote_pointmap/pts3d` 在同一 mask 下做中心化/尺度归一化后计算 L1，因此它仍是尺度无关的点云形状损失，不强迫模型学习米制绝对尺度。

导出时 `scripts/export_pointcloud_ply.py --export_projection_aux_reconstruction` 会额外写：

```text
mapanything_pointcloud_same_aux_grid_global_remote.ply
mapanything_pointcloud_same_aux_reconstruction_summary.json
```

其中 PLY 为了方便肉眼查看，可以再用 `--projection_aux_xyz_align_mode gt_pointmap_unit_xy_zrange_flipz` 做可视化尺度对齐；这个对齐只发生在导出阶段，不参与训练。

## 4）global 方向错时为什么会“建筑整体歪”

`global` 的 xy 位移是 `rel_height * slope * dir`。  
- 如果 `dir` 方向错（角度偏差大），建筑会整体沿错误方向拉伸。
- 如果 `slope` 比 GT 小，起伏不够；比 GT 大，则放大起伏。
- 如果 `rel_height` 放缩不对，建筑高度和边缘形状会扭曲。

这也是你看到“整体倾斜/方向不对”最常见根因。

## 5）命令层面的关键开关（当前可复现路径）

- 只换方向头为预测值（你上一步的“步骤1”）：  
  - 不传 `--projection_aux_use_gt_global_direction`  
  - 不传 `--projection_aux_use_gt_global_slope`  
  - 保持 `--projection_aux_use_gt_projection_base`（用于严格几何口径）
- 用 GT 方向/坡度：加上对应两个开关。
- 其它用于可视化的归一化参数：
  - `--projection_aux_xyz_align_mode gt_pointmap_unit_xy_zrange_flipz`
  - `--projection_aux_rel_height_scale_mode gt_height_range`

## 6）建议你快速自检的三条线

1. `gt_base + pred_dir + pred_slope`
   - 看方向头是否已经能复原形状
2. `gt_base + gt_dir + gt_slope`
   - 看 supervision 上界（理论上限）
3. `pred_base(point-head) + pred_dir + pred_slope`
   - 看 pipeline 端到端可用性（部署式输出）
4. `grid_xy + pred_height + pred_dir + pred_slope`
   - 看 aux head 是否能在不借助 GT base/point-head base 的情况下独立复原 remote 点云形状

如果第 2 条远好于第 1 条，当前瓶颈是 `global_dir/slope` 头；  
如果第 1 条也坏于第 2 条，说明 `rel_height` 或 base 的尺度/归一化路径也有偏差；  
如果两条都可读，但第 3 条掉得快，说明 `point` base 引入了主要误差，需回到 base 解耦。
如果第 4 条失败而 height 单项误差较低，说明问题在 global 投影组合参数化或 direction/slope 训练，而不是 height 标签本身。


## 7）训练时 projection-aux loss 的总体结构

projection-aux loss 在 [losses.py](/root/autodl-tmp/Models/map-anything/mapanything/train/losses.py) 的 `_projection_aux_loss_one` 中计算。  
每个 remote 样本会得到一个 `projection_loss`，最后训练总 loss 是：

```text
total_loss = base_loss + mean(projection_aux_loss over views/samples)
```

其中 `base_loss` 是原有 pointmap/depth/pose 等主任务损失；`projection_aux_loss` 是下面这些子项按权重相加：

```text
projection_aux_loss =
  w_h       * L_rel_height
+ w_h_aff   * L_rel_height_affine
+ w_h_bal   * L_rel_height_balanced
+ w_h_con   * L_rel_height_contrast
+ w_h_bucket* L_rel_height_bucket_mean
+ w_h_low   * L_rel_height_low_overpred
+ w_dense_h * L_dense_rel_height
+ w_off     * L_offset
+ w_off_*   * L_offset_auxiliary_terms
+ w_dir     * L_global_dir
+ w_slope   * L_global_slope
+ w_vec     * L_global_vector
+ w_cons    * L_consistency
+ w_recon   * L_reconstruct_to_gt_or_point
+ w_moge_*  * L_moge_priors
```

实际启用哪些项由训练脚本里的 `LAMBDA_PROJ_*` / `PROJ_*` 参数决定。

## 8）mask 和尺度空间

projection-aux 的基础监督 mask 是：

```text
base_mask =
  remote_projection_valid_mask
& finite(rel_gt, offset_gt, rel_pred, offset_pred)
```

然后不同 loss 还会再套额外 mask：

- `rel_height_mask`: 可选 `tilt_mask`，可选 `rel_height_min`
- `offset_mask`: 可选 `tilt_mask`，可选 `offset_min_magnitude`
- `consistency_mask`: 可选 `tilt_mask`，可选 `offset_min_magnitude`
- `reconstruct_mask`: 还要求 GT pointmap、reconstructed points、remote valid mask 都有效

height 不是直接用米制 GT 数值训练，而是先进入 loss space：

```text
rel_gt_for_loss = rel_gt / rel_height_scale
rel_pred_for_loss = rel_pred
```

`rel_height_scale` 的来源由 `projection_rel_height_scale_mode` 决定：

- `fixed`: 使用固定 `projection_rel_height_scale`
- `valid_quantile`: 使用当前 valid GT height 的指定分位数
- 其它默认路径: 使用 pointmap normalization factor，例如 `gt_pointmap_norm` / `avg_dis` 风格

offset 也有自己的 loss space：

```text
offset_gt_for_loss = offset_gt / projection_offset_scale
offset_pred_for_loss = offset_pred
```

因此训练中 aux head 通常预测的是归一化后的 height/offset，不是原始米制量。

## 9）height loss 怎么算

基础 height loss 是 masked L1：

```text
L_rel_height = mean_masked(|rel_pred_for_loss - rel_gt_for_loss|)
```

可选 target weighting 会让高目标值区域权重更高：

```text
weight = 1 + strength * (|target| / mean(|target|))^gamma
```

其它 height 辅助项：

- `rel_height_balanced`: 按 GT height magnitude 分成 low/mid/high bucket，每个 bucket 分别算 L1 后平均，避免低值背景淹没高建筑。
- `rel_height_contrast`: 约束预测的 high-low bucket 差值接近 GT 的 high-low 差值。
- `rel_height_bucket_mean`: 约束每个 bucket 的预测均值接近 GT 均值。
- `rel_height_low_overpred`: 只惩罚低 height bucket 里的过预测，避免背景被抬高。
- `rel_height_affine`: 先对预测 height 做一维 affine fit，再算 L1，用来评估/约束尺度和偏移无关的形状。

`rel_height_affine` 的拟合形式：

```text
aligned_pred = scale * rel_pred_for_loss + shift
L_rel_height_affine = mean_masked(|aligned_pred - rel_gt_for_loss|)
```

## 10）dense height loss 怎么算

dense height target 从 remote pointmap 派生，不只依赖稀疏 `projection_aux` mask。  
做法是先在 `projection_aux` 和 pointmap 的 common 有效区域估计地面：

```text
ground = median(pointmap_z - rel_gt) over common mask
dense_rel_height = pointmap_z - ground
```

然后在 pointmap 有效区域监督：

```text
dense_gt_for_loss = dense_rel_height / rel_height_scale
L_dense_rel_height = mean_weighted_masked(|rel_pred_for_loss - dense_gt_for_loss|)
```

这里可选：

- 排除原 hard aux mask，只训新增 dense 区域
- 按 height quantile 过滤低结构区域
- 降低低 height 区域权重，避免背景主导

## 11）offset loss 怎么算

基础 offset loss 是每像素 xy L1：

```text
L_offset = mean_masked(mean_xy(|offset_pred_for_loss - offset_gt_for_loss|))
```

其中：

```text
offset_gt_for_loss = offset_gt / projection_offset_scale
```

辅助项包括：

- `offset_balanced`: 按 offset magnitude 分桶后平均 loss
- `offset_contrast`: 约束 high-low offset magnitude gap
- `offset_bucket_mean`: 约束 bucket 内 offset magnitude 均值
- `offset_low_overpred`: 惩罚低 offset 区域的过预测
- `offset_mag`: 只监督 offset magnitude
- `offset_dir`: 监督 offset 方向，公式是 `1 - cosine(pred_offset, gt_offset)`

## 12）global dir / slope / vector loss 怎么算

`global_dir` 是每个样本一个二维方向向量，训练前会 normalize：

```text
dir_pred = normalize(remote_projection_global_dir_xy_pred)
dir_gt = normalize(remote_projection_global_dir_xy)
```

方向 loss 是 cosine loss：

```text
L_global_dir = mean(1 - clamp(dot(dir_pred, dir_gt), -1, 1))
```

`global_slope` 是 L1：

```text
L_global_slope = mean(|slope_pred - slope_gt|)
```

`global_vector` 把方向和坡度合成一个二维向量一起监督：

```text
global_vec_pred = slope_pred * dir_pred
global_vec_gt = slope_gt * dir_gt
L_global_vector = mean(|global_vec_pred - global_vec_gt|)
```

如果开启 `projection_global_target_from_pointmap`，GT 的 `dir_gt/slope_gt` 可以不直接用 `projection_aux.npz` 中的原标签，而是从 GT pointmap 拟合：

```text
offset_xy = projected_xy + center_xy - pointmap_xy
vector = sum(rel_gt * offset_xy) / sum(rel_gt^2)
slope_gt = ||vector||
dir_gt = normalize(vector)
```

这个拟合目标更贴近当前 pointmap 标签，但实验里仍没有稳定修复 global direction 退化。

## 13）height / slope / dir 到 global 重建点云

训练中的 global 反投影是：

```text
rel_pred_world = rel_pred * rel_height_scale
dir_pred = normalize(dir_pred)
offset_from_global = rel_pred_world * slope_pred * dir_pred

recon_global = projected_xyz_centered
recon_global.xy = projected_xyz_centered.xy + center_xy - offset_from_global
recon_global.z = projected_xyz_centered.z
```

注意这里 `z` 保持 `projected_xyz_centered.z`，不是 `ground_z + rel_height`。  
所以 global aux 的核心作用是通过 height 解释 xy 投影偏移，而不是直接用 height 重造 z。

## 14）consistency loss 怎么算

`consistency` 约束 direct offset head 和 global 参数化结果一致：

```text
offset_from_field =
  rel_for_consistency * slope_pred * dir_pred

L_consistency =
  mean_masked(mean_xy(|offset_for_consistency - offset_from_field|))
```

这里的 `rel_for_consistency` 和 `offset_for_consistency` 可以选择用 loss space：

```text
rel_for_consistency = rel_pred_for_loss
offset_for_consistency = offset_pred_for_loss
```

也可以用原预测空间：

```text
rel_for_consistency = rel_pred
offset_for_consistency = offset_pred
```

这个 loss 不直接看 GT pointmap，只要求 aux 的两个分解方式自洽。  
如果 aux 本身不稳，过强的一致性可能只是让两个错误分支彼此贴近。

## 15）aux reconstructed point loss 怎么算

当开启 reconstruction loss 时，会先从 aux 输出重建点云：

```text
recon_offset = reconstruct(projected, center, offset_pred_world)
recon_global = reconstruct(projected, center, rel_pred_world, dir_pred, slope_pred)
```

然后有两类 target。

第一类是对 GT pointmap：

```text
L_reconstruct_global_to_gt =
  mean_masked(mean_xyz(|recon_global - gt_pointmap|))

L_reconstruct_offset_to_gt =
  mean_masked(mean_xyz(|recon_offset - gt_pointmap|))
```

如果开启 `projection_reconstruct_to_gt_use_pointmap_norm`，比较前会对 `recon` 和 `gt_pointmap` 做同一种 pointmap norm 对齐，避免绝对尺度直接支配 loss：

```text
recon_for_loss, gt_for_loss = normalize_pair(recon, gt_pointmap, mask, norm_mode)
```

第二类是对模型 point head 输出，但 point head detach：

```text
point_target = pred['pts3d'].detach()

L_reconstruct_global_to_point_detach =
  mean_masked(mean_xyz(|recon_global - point_target|))
```

这个 loss 会更新 aux head，但不会通过这个项反向拉动 point head。  
它的目的更像是让 aux 重建贴近当前 point head，而不是让坏 aux 直接拖坏 point head。

## 16）哪些 loss 会影响 point head

普通 `rel_height/offset/dir/slope/consistency` 主要作用在 projection-aux head 及其上游共享特征。  
如果 aux head 的输入来自 shared tokens，梯度也可能影响共享 aggregator/token 表征；如果训练参数冻结得多，则主要只更新 aux head。

`reconstruct_*_to_gt` 会通过 aux 参数反投影点云再和 GT pointmap 比较，它不需要 point head 输出作为 target，因此不会直接把 point head 当 target；但如果共享 trunk 未冻结，仍可能通过 shared features 间接影响 point head 表征。

`reconstruct_*_to_point_detach` 使用 `pred['pts3d'].detach()`，target 被 detach，因此这条 loss 不会把梯度传回 point head。  
这也是为了避免 aux 重建还很差时反向破坏 point head。

真正直接训练 remote point head 的仍是主 pointmap loss，例如 remote pointmap L1 / robust top-percent / overlap pointmap 等，不属于 projection-aux loss 本身。
