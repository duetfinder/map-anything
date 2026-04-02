# MapAnything Loss 汇总

说明：本文档中的文件链接统一使用相对于当前 Markdown 文件的相对路径，不使用绝对路径。

本文档整理 `Models/map-anything` 当前训练框架中支持的 loss，包括：

- loss 的实例化方式
- `mapanything/train/losses.py` 中的主要 loss 类
- `configs/loss/*.yaml` 中的 loss 配方
- 哪些 loss 依赖相机模型 / pose / intrinsics，哪些相对独立
- 对 `P2 RS-only` 的 loss 设计建议

## 1. loss 是怎么接入训练的

训练时，loss 不是写死在训练循环里的，而是由 Hydra 配置中的字符串定义，然后在训练阶段直接实例化：

- 训练代码：[mapanything/train/training.py](mapanything/train/training.py)
- loss 实现：[mapanything/train/losses.py](mapanything/train/losses.py)
- loss 配置目录：[configs/loss](configs/loss)

当前训练中关键逻辑是：

- `train_criterion = eval(args.loss.train_criterion)`
- `test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion)`

这意味着：

1. loss 配置本质上是一个可组合的 Python 构造表达式。
2. 新增 loss 有两种方式：
   - 直接新建一个 `configs/loss/*.yaml`，重组已有 loss 类
   - 在 [mapanything/train/losses.py](mapanything/train/losses.py) 中新增类，再在 yaml 中引用

## 2. loss 类总表

当前 [mapanything/train/losses.py](mapanything/train/losses.py) 中可直接用于训练组合的主要类如下。

### 2.1 基础逐元素 / 回归损失

#### `L1Loss`

- 作用：对最后一维做 L1 距离并求和
- 输入：任意同 shape 的 `a` / `b`
- 典型用途：点云、深度、姿态平移等逐元素回归
- 相机依赖：无，是否依赖相机取决于你传进去的目标是什么

#### `L2Loss`

- 作用：对最后一维做欧式距离
- 输入：任意同 shape 的 `a` / `b`
- 典型用途：点云、深度、姿态等回归
- 相机依赖：无，是否依赖相机取决于目标定义

#### `GenericLLoss`

- 作用：统一封装 `l1` / `l2`
- 输入：`a` / `b` + `loss_type`
- 相机依赖：无

#### `FactoredLLoss`

- 作用：对不同 factor 用不同范数
- 支持 factor：
  - `points`
  - `depth`
  - `ray_directions`
  - `pose_quats`
  - `pose_trans`
  - `scale`
- 相机依赖：无，但常被用于依赖相机的几何量

#### `RobustRegressionLoss`

- 作用：Generalized Robust Loss，较抗 outlier
- 输入：任意同 shape 的 `a` / `b`
- 超参数：`alpha`、`scaling_c`
- 典型用途：点云 / 深度回归主损失
- 相机依赖：无

#### `BCELoss`

- 作用：二值交叉熵 with logits
- 输入：
  - 预测：mask logits
  - GT：binary mask
- 典型用途：`non_ambiguous_mask`
- 相机依赖：无

### 2.2 包装器 / 组合器

#### `NonAmbiguousMaskLoss`

- 作用：监督 `pred['non_ambiguous_mask_logits']`
- GT 依赖：`gt['non_ambiguous_mask']`
- 计算：逐 view 做 BCE，再求和 / 记录平均
- 相机依赖：无

#### `ConfLoss`

- 作用：把逐像素回归 loss 用预测 confidence 重加权
- 预测依赖：
  - 下层 pixel loss 的预测字段
  - 额外依赖 `pred['conf']`
- 计算：`raw_loss * conf - alpha * log(conf)`
- 适用前提：底层 loss 必须能以 `reduction='none'` 返回 per-pixel loss
- 相机依赖：间接依赖底层 loss；`conf` 本身不依赖相机

#### `ExcludeTopNPercentPixelLoss`

- 作用：对 per-pixel loss 去掉最高的 top N% 再求平均
- 预测依赖：下层 pixel loss 的预测字段
- GT 依赖：下层 pixel loss 的 GT 字段，另外会读 `gt['is_synthetic']` 区分真实/合成数据
- 计算：
  - synthetic 样本保留全部有效像素
  - real 样本只保留 bottom `(100 - top_n_percent)%` 的 per-pixel loss
- 相机依赖：间接依赖底层 loss

#### `ConfAndExcludeTopNPercentPixelLoss`

- 作用：一部分 loss set 用 confidence weighting，另一部分用 top-N 排除
- 预测依赖：
  - 下层 pixel loss 的预测字段
  - `pred['conf']`
- GT 依赖：下层 pixel loss 的 GT 字段 + `gt['is_synthetic']`
- 相机依赖：间接依赖底层 loss

### 2.3 几何主损失族

#### `Regr3D`

- 作用：世界坐标点图回归，view0 为 anchor
- 预测依赖：`pred['pts3d']`
- GT 依赖：
  - `gt['pts3d']`
  - `gt['camera_pose']`
  - `gt['valid_mask']`
  - `gt['non_ambiguous_mask']`（如果启用 ambiguous 逻辑）
  - `gt['is_metric_scale']`（如果启用 `?avg_dis` 这类模式）
- 计算方式：
  - 先把 GT 点图转换到 view0 frame
  - 再做可选归一化 / log-scale
  - 对有效像素计算回归误差
- 相机依赖：强依赖 `camera_pose`

#### `PointsPlusScaleRegr3D`

- 作用：世界点图 + metric scale 联合回归
- 预测依赖：
  - `pred['pts3d']`
  - 可选 `pred['metric_scaling_factor']`
- GT 依赖：
  - `gt['pts3d']`
  - `gt['camera_pose']`
  - `gt['valid_mask']`
  - `gt['is_metric_scale']`
- 计算方式：
  - 点图部分与 `Regr3D` 类似
  - 额外监督 metric scale
- 相机依赖：强依赖 `camera_pose`

#### `NormalGMLoss`

- 作用：法向 + gradient matching 辅助 loss
- 预测依赖：`pred['pts3d']`
- GT 依赖：
  - `gt['pts3d']`
  - `gt['camera_pose']`
  - `gt['valid_mask']`
  - `gt['is_synthetic']`（若只对 synthetic 启用）
- 计算方式：
  - 先在 local / normalized pointmap 上算 normal loss
  - 再在 `log(depth_z)` 上算多尺度 gradient matching loss
- 限制：当前实现只支持 `1 view`
- 相机依赖：中等，内部会用 `camera_pose` 把 GT 转到局部相机系

#### `FactoredGeometryRegr3D`

- 作用：对“分解几何表示”做联合监督
- 可监督项：
  - 世界点图 `pts3d`
  - 相机系点图 `pts3d_cam`
  - 深度 `depth_along_ray` 或 `depth_z`
  - 射线方向 `ray_directions`
  - 姿态四元数 `cam_quats`
  - 姿态平移 `cam_trans`
- 预测依赖：
  - `pred['pts3d']`
  - `pred['pts3d_cam']`
  - `pred['depth_along_ray']`
  - `pred['ray_directions']`
  - `pred['cam_quats']`
  - `pred['cam_trans']`
- GT 依赖：
  - `gt['pts3d']`
  - `gt['pts3d_cam']`
  - `gt['depth_along_ray']`
  - `gt['camera_pose']`
  - `gt['camera_pose_quats']`
  - `gt['camera_pose_trans']`
  - `gt['ray_directions_cam']`
  - `gt['valid_mask']`
  - `gt['non_ambiguous_mask']`
  - `gt['is_metric_scale']`
- 计算方式：
  - 先把 GT 和预测都规整到 view0 frame / 归一化尺度
  - 再对各个 factor 分别回归
  - 可选 absolute pose 或 pairwise relative pose
- 相机依赖：非常强

#### `FactoredGeometryRegr3DPlusNormalGMLoss`

- 作用：`FactoredGeometryRegr3D` + `NormalGMLoss`
- 额外预测 / GT 依赖：同 `NormalGMLoss`
- 相机依赖：非常强

#### `FactoredGeometryScaleRegr3D`

- 作用：`FactoredGeometryRegr3D` 的 metric-scale 版本，额外监督 scale
- 可监督项：
  - `pts3d`
  - `pts3d_cam`
  - `depth`
  - `ray_directions`
  - `pose_quats`
  - `pose_trans`
  - `scale`
- 预测依赖：上面所有项，外加可选 `pred['metric_scaling_factor']`
- GT 依赖：与 `FactoredGeometryRegr3D` 基本相同，外加 metric scale 约束
- 相机依赖：非常强

#### `FactoredGeometryScaleRegr3DPlusNormalGMLoss`

- 作用：`FactoredGeometryScaleRegr3D` + `NormalGMLoss`
- 相机依赖：非常强

#### `DisentangledFactoredGeometryScaleRegr3D`

- 作用：将分解几何的各个分量更 disentangled 地监督
- 可监督项：同 `FactoredGeometryScaleRegr3D`
- 相机依赖：非常强

#### `DisentangledFactoredGeometryScaleRegr3DPlusNormalGMLoss`

- 作用：`DisentangledFactoredGeometryScaleRegr3D` + `NormalGMLoss`
- 相机依赖：非常强

## 3. 哪些 loss 相对不依赖相机，哪些强依赖相机

### 3.1 相对独立 / 容易改造成 RS-only 的

- `L1Loss`
- `L2Loss`
- `GenericLLoss`
- `FactoredLLoss`
- `RobustRegressionLoss`
- `BCELoss`
- `ConfLoss`（前提是底层 loss 本身不依赖相机）
- `ExcludeTopNPercentPixelLoss`（前提是底层 loss 本身不依赖相机）
- `ConfAndExcludeTopNPercentPixelLoss`（前提是底层 loss 本身不依赖相机）

这些类本身不强绑定 camera pose / intrinsics。真正决定它们是否能用于 RS-only 的，是你喂给它们的预测量和 GT 量是否已经定义在同一个坐标 / 表示空间内。

### 3.2 强依赖相机模型 / 不适合直接拿去做 RS-only 的

- `Regr3D`
- `PointsPlusScaleRegr3D`
- `NormalGMLoss`（当前实现内部仍会利用 `camera_pose`）
- `FactoredGeometryRegr3D`
- `FactoredGeometryRegr3DPlusNormalGMLoss`
- `FactoredGeometryScaleRegr3D`
- `FactoredGeometryScaleRegr3DPlusNormalGMLoss`
- `DisentangledFactoredGeometryScaleRegr3D`
- `DisentangledFactoredGeometryScaleRegr3DPlusNormalGMLoss`

这些 loss 要么直接使用：

- `camera_pose`
- `camera_pose_quats`
- `camera_pose_trans`
- `ray_directions_cam`
- `depth_along_ray`
- `pts3d_cam`

要么默认预测与 GT 可以在同一个相机 / view0 坐标系下比较。因此当前不适合直接用于没有可靠相机模型的 RS-only。

## 4. loss config 配方总表

下面按 `configs/loss/*.yaml` 做汇总。

### 4.1 基础配置

#### [default.yaml](configs/loss/default.yaml)

- 空模板，不定义具体 loss

### 4.2 世界点图 / mask / scale 基础配方

#### [conf_pm_mask_loss.yaml](configs/loss/conf_pm_mask_loss.yaml)

- 训练：`ConfLoss(Regr3D(...)) + 0.3 * NonAmbiguousMaskLoss(BCELoss())`
- 验证：`ExcludeTopNPercentPixelLoss(Regr3D(...)) + mask BCE`
- 核心监督：世界点图 + 非歧义 mask
- 相机依赖：强

#### [conf_pm_mask_scale_loss.yaml](configs/loss/conf_pm_mask_scale_loss.yaml)

- 训练：`ConfLoss(PointsPlusScaleRegr3D(...)) + mask BCE`
- 验证：`ExcludeTopNPercentPixelLoss(PointsPlusScaleRegr3D(...)) + mask BCE`
- 核心监督：世界点图 + scale + mask
- 相机依赖：强

### 4.3 Regr3D / monocular-like 配方

#### [moge2_loss.yaml](configs/loss/moge2_loss.yaml)

- `ExcludeTopNPercentPixelLoss(Regr3D(...)) + 3.0 * NormalGMLoss(...)`
- 核心监督：世界点图 + normal / gradient matching
- 备注：`NormalGMLoss` 当前只支持 1 view
- 相机依赖：强

### 4.4 FactoredGeometryRegr3D 配方

#### [pi3_loss.yaml](configs/loss/pi3_loss.yaml)

- 训练时的完整表达式是：
  - `ExcludeTopNPercentPixelLoss(FactoredGeometryRegr3DPlusNormalGMLoss(RobustRegressionLoss(alpha=0.5, scaling_c=0.05), ...), top_n_percent=5, apply_to_real_data_only=True, loss_set_indices=[0, 1, 2])`
- 这个配方可以拆成两层理解：
  - 外层 `ExcludeTopNPercentPixelLoss` 只对选中的 pixel-level loss sets 做 top 5% 排除
  - 内层 `FactoredGeometryRegr3DPlusNormalGMLoss` 负责真正的几何监督
- 关键设置：
  - `norm_mode='avg_dis'`
  - `depth_type_for_loss='depth_z'`
  - `loss_in_log=True`
  - `flatten_across_image_only=True`
  - `compute_pairwise_relative_pose_loss=True`
  - `convert_predictions_to_view0_frame=True`
  - `compute_world_frame_points_loss=True`
  - `apply_normal_and_gm_loss_to_synthetic_data_only=True`
  - `normal_loss_weight=3.0`
  - `gm_loss_weight=3.0`
- 内层的基础回归核是：
  - `RobustRegressionLoss(alpha=0.5, scaling_c=0.05)`
- `FactoredGeometryRegr3D` 这类 loss 的主要监督顺序是：
  - `pts3d`
  - `pts3d_cam`
  - `depth_z` 或 `depth_along_ray`
  - `ray_directions`
  - `pose_quats`
  - `pose_trans`
- 其中 `FactoredGeometryRegr3DPlusNormalGMLoss` 额外再加：
  - local camera-frame point normals
  - multi-scale gradient matching on `log(depth_z)`
- 预测 / GT 依赖可以概括为：
  - `pts3d`: `pred['pts3d']` vs `gt['pts3d']`
  - `pts3d_cam`: `pred['pts3d_cam']` vs `gt['pts3d_cam']`
  - `depth_z`: `pred['pts3d_cam'][..., 2:]` vs `gt['pts3d_cam'][..., 2:]`
  - `ray_directions`: `pred['ray_directions']` vs `gt['ray_directions_cam']`
  - `pose_quats`: `pred['cam_quats']` vs `gt['camera_pose_quats']`
  - `pose_trans`: `pred['cam_trans']` vs `gt['camera_pose_trans']`
  - `normal / gm`: 基于 `pts3d_cam` 和 `gt['pts3d_cam']` 计算
- `compute_pairwise_relative_pose_loss=True` 的含义是：
  - 除了绝对 pose 外，还会把当前 view 作为 reference，显式监督其他 view 的 pairwise relative pose
- `convert_predictions_to_view0_frame=True` 的含义是：
  - 预测的 world points / pose 先转换到 view0 frame 再比较
- `loss_set_indices=[0, 1, 2]` 的含义是：
  - 外层 top-N 排除只应用在前三个 pixel-level loss set 上
  - 后面的 `ray_directions / pose_quats / pose_trans` 继续按原始均值参与总损失
- 核心监督可以概括为：
  - `world points`
  - `camera-frame points`
  - `depth_z`
  - `ray directions`
  - `absolute + pairwise relative pose`
  - `normal + gradient matching`
- 相机依赖：非常强

#### [vggt_loss.yaml](configs/loss/vggt_loss.yaml)

- `ConfAndExcludeTopNPercentPixelLoss(FactoredGeometryRegr3DPlusNormalGMLoss(...))`
- 与 `pi3_loss` 的关键区别：
  - 带 confidence weighting
  - `compute_pairwise_relative_pose_loss=False`
- 相机依赖：非常强

#### [up_to_scale_loss.yaml](configs/loss/up_to_scale_loss.yaml)

- `ConfAndExcludeTopNPercentPixelLoss(FactoredGeometryRegr3DPlusNormalGMLoss(...)) + mask BCE`
- 关键设置：
  - `depth_type_for_loss='depth_along_ray'`
  - `compute_pairwise_relative_pose_loss=False`
- 相机依赖：非常强

#### [entangled_metric_loss.yaml](configs/loss/entangled_metric_loss.yaml)

- 与 `up_to_scale_loss` 接近，但更偏 metric / entangled 版本
- 相机依赖：非常强

### 4.5 FactoredGeometryScaleRegr3D 主配方

#### [overall_loss.yaml](configs/loss/overall_loss.yaml)

- 训练：`ConfAndExcludeTopNPercentPixelLoss(FactoredGeometryScaleRegr3DPlusNormalGMLoss(...)) + 0.3 * mask BCE`
- 核心监督：
  - world points
  - cam points
  - depth
  - ray directions
  - pose
  - scale
  - normal / gm
- 相机依赖：非常强

#### [overall_disentangled_loss.yaml](configs/loss/overall_disentangled_loss.yaml)

- `DisentangledFactoredGeometryScaleRegr3DPlusNormalGMLoss(...) + mask BCE`
- 与 `overall_loss` 区别：监督结构更 disentangled
- 相机依赖：非常强

#### [overall_loss_weigh_pm_higher.yaml](configs/loss/overall_loss_weigh_pm_higher.yaml)

- `overall_loss` 的权重变体
- world pointmap 权重更高
- 相机依赖：非常强

#### [overall_loss_highpm_plus_rel_pose.yaml](configs/loss/overall_loss_highpm_plus_rel_pose.yaml)

- 在 `overall_loss` 基础上启用：
  - `compute_absolute_pose_loss=True`
  - `compute_pairwise_relative_pose_loss=True`
- 相机依赖：非常强

#### [overall_loss_highpm_plus_rel_pose_no_conf.yaml](configs/loss/overall_loss_highpm_plus_rel_pose_no_conf.yaml)

- 与上面类似，但不使用 confidence weighting
- 相机依赖：非常强

#### [overall_loss_highpm_plus_rel_pose_no_conf_no_ref_view.yaml](configs/loss/overall_loss_highpm_plus_rel_pose_no_conf_no_ref_view.yaml)

- 启用 pairwise relative pose
- `convert_predictions_to_view0_frame=True`
- 相机依赖：非常强

#### [overall_loss_highpm_rel_pose_no_ref_view.yaml](configs/loss/overall_loss_highpm_rel_pose_no_ref_view.yaml)

- 强化 pairwise relative pose 的变体
- 相机依赖：非常强

### 4.6 ablation 配方

这些配置主要是在 `overall_loss` 或相关主配方上把某个分量权重关掉或替换：

- [no_depth_loss.yaml](configs/loss/no_depth_loss.yaml)
  - `depth_loss_weight=0`
- [no_points_loss.yaml](configs/loss/no_points_loss.yaml)
  - `world_frame_points_loss_weight=0` / `cam_frame_points_loss_weight=0`
- [no_pose_loss.yaml](configs/loss/no_pose_loss.yaml)
  - `pose_quats_loss_weight=0` / `pose_trans_loss_weight=0`
- [no_ray_dirs_loss.yaml](configs/loss/no_ray_dirs_loss.yaml)
  - `ray_directions_loss_weight=0`
- [no_log_scaling.yaml](configs/loss/no_log_scaling.yaml)
  - `loss_in_log=False`
- [no_robust_loss.yaml](configs/loss/no_robust_loss.yaml)
  - 用 `L2Loss` 替代 `RobustRegressionLoss`

这些 ablation 仍然大多建立在强相机依赖的主几何 loss 上，不适合直接照搬到 RS-only。

## 5. 对 P2 RS-only 的具体建议

### 5.1 先说结论

对 `P2 RS-only`，不建议直接复用当前任何一个现成 `configs/loss/*.yaml`。

原因不是这些 loss 不强，而是它们大多默认：

- 有可靠 `camera_pose`
- 有 `camera_intrinsics`
- 有 `depth_along_ray` / `ray_directions_cam`
- 可以把 GT 和预测都规整到同一个相机 / view0 坐标系下比较

而当前 RS 数据不满足这个前提。

所以 `P2` 更合理的做法是：

- 不直接用 `pi3_loss`
- 不直接用 `vggt_loss`
- 不直接用 `overall_loss`
- 新建一个 `RS-only` 专用 loss

### 5.2 对 Pi3 而言，P2 最该保留和最该避开的是什么

如果目标是：

- 让 `Pi3` 学会 RS 图像上的几何重建
- 同时尽量不要破坏它原本的多视角能力

那么最重要的原则是：

1. 尽量只监督 dense geometry 输出。
2. 不要在 `P2` 阶段去强行训练 camera / pose 相关输出。
3. 不要在 `P2` 阶段用任何依赖 perspective camera 假设的 loss。
4. 不要直接把 `pi3_loss` 这套完整分解几何监督搬到 RS-only 上。

对 `Pi3Wrapper` 当前可用预测字段而言：

- 可以优先考虑：
  - `pred['pts3d']`
  - 必要时由 `pred['pts3d']` 派生的高度量
- 暂时不要重点训练：
  - `pred['cam_quats']`
  - `pred['cam_trans']`
  - `pred['ray_directions']`
  - `pred['depth_along_ray']`
  - `pred['pts3d_cam']`

原因：

- 这些后一类量都强绑定了相机分解方式
- 对 RS 数据，目前没有可靠相机模型去定义它们的 GT
- 如果用错误 supervision 强压这些头，最容易破坏模型原本的多视角归纳能力

### 5.3 建议的 P2 loss 结构

第一版建议：

```text
total_loss = lambda_pm * rs_pointmap_loss + lambda_h * rs_height_loss
```

其中：

#### `rs_pointmap_loss`

建议：

- 对 `pred['pts3d']` 和 `remote_pointmap` 做 masked regression
- 第一版优先用 `masked L1`
- 如果 outlier 明显，再试 `RobustRegressionLoss`

为什么优先 `L1`：

- 更直接
- 更容易 debug
- 更容易先看出 prediction / GT 是否在同一表示空间里

#### `rs_height_loss`

建议：

- 从预测点图派生高度量，再和 `remote_height_map` 做 masked `L1`
- 作为辅助项，不要一开始权重太大

为什么只做辅助项：

- 你当前真正的主监督仍应该是完整几何
- 高度图更适合当补充约束，而不是唯一目标

### 5.4 更保守的第一版建议

如果你担心一开始就把 RS-only 训练做得太重，建议按下面顺序逐步升级：

#### 版本 A

```text
total_loss = masked_L1(pred_pts3d, gt_remote_pointmap)
```

用途：

- 只验证模型能不能在 RS 数据上学到基本几何
- 最容易 debug

#### 版本 B

```text
total_loss = masked_L1(pred_pts3d, gt_remote_pointmap) + 0.1 * masked_L1(pred_height, gt_height)
```

用途：

- 在版本 A 稳定后再加高度辅助项

#### 版本 C

```text
total_loss = robust_pointmap_loss + 0.1 * masked_L1(pred_height, gt_height)
```

用途：

- 当版本 A/B 已确认有效，但点图里存在较多 outlier 时再尝试

### 5.5 怎样尽量避免破坏 Pi3 原本的多视角能力

如果只从 loss 角度约束，最重要的是：

- 不要在 `P2` 中加入 pose loss
- 不要在 `P2` 中加入 pairwise relative pose loss
- 不要在 `P2` 中加入 ray-direction / perspective depth 的监督
- 不要把 `pi3_loss` 里那套完整分解几何监督直接拿来用在 RS-only

如果允许配合训练策略一起做，建议进一步：

- 优先小学习率
- 优先短 schedule
- 优先只训和 dense geometry 更直接相关的部分
- `camera_decoder / camera_head` 最好少动，甚至冻结

换句话说，`P2` 不应该试图把 Pi3 重新训练成“RS 专用相机几何模型”，而应该只让它学会：

- 从 RS 图像中恢复稳定的几何表面表示

这样更有利于后续进入 `P3 Joint Input` 时，保留它原本的多视角能力。

## 6. 当前最推荐的 P2 起点

如果只给一个最务实的起点，我建议：

1. 新增一个 `RS-only` 专用 loss config。
2. 第一版只做：
   - `masked L1 pointmap`
   - `+ 0.1 * masked L1 height`
3. 不引入任何 pose / ray / depth-along-ray / confidence weighting。
4. 如果第一版稳定，再比较：
   - `L1` vs `RobustRegressionLoss`
   - `pointmap-only` vs `pointmap + height`

一句话总结：

- `P2` 的核心不是复用 MapAnything 现有强几何 loss
- 而是为没有可靠相机模型的 RS 数据，单独设计一个“最小、稳定、尽量不伤原模型能力”的几何回归目标
