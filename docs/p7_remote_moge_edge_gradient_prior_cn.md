# P7 Remote MoGe Edge/Gradient Prior 方案

## 背景

P7 当前希望让 remote/satellite 分支学到更稳定的几何和投影表征。现有 remote 标签主要来自 `exp_005_map_points_generate` 的稀疏/半稀疏 `pixel_to_point_map.npz` 与 `projection_aux.npz`。

问题是 remote 高度标签在很多区域不够稠密，直接用于多任务训练时会出现两个风险：

- 有效监督区域太少，remote 分支容易学到低幅值、塌缩或只拟合局部统计。
- 如果用 MoGe/Poisson 直接补成稠密硬标签，伪标签的 scale/offset 偏差、边界漂移和跨边界扩散会污染主几何监督。

因此本方案不把 MoGe 当作绝对高度 GT，而是把 MoGe 作为边缘和梯度先验，只约束 remote 预测的局部结构。

## 核心判断

MoGe aligned depth 的视觉结构通常比原始稀疏标签更连续，建筑轮廓也更清楚，但它不满足严格 metric 对齐。它更适合回答：

```text
哪里应该有高度变化？
高度变化的大致方向是什么？
哪些区域可能属于同一连续结构？
```

它不适合作为：

```text
每个像素的绝对高度真值
每个像素的严格 world-frame pointmap
projection_aux 的直接替代标签
```

所以更科学的目标是：

```text
原始标签负责绝对几何。
projection_aux 负责投影机制。
MoGe 负责边界、梯度和局部相对结构。
```

## 与直接稠密化标签的比较

| 方法 | 主要假设 | 优点 | 风险 |
| --- | --- | --- | --- |
| 直接 MoGe/Poisson 稠密硬标签 | 每个补全像素的绝对高度都可信 | loss 简单，监督覆盖率高 | 伪标签偏差会被当成真值，可能破坏 metric 几何 |
| 原始 hard 标签 + Poisson soft 标签 | 补全区大体可信，但置信度低于原始标签 | 兼顾覆盖率和保真 | 需要 per-pixel weight；Poisson 跨边界扩散仍需控制 |
| 原始 hard 标签 + MoGe gradient/edge prior | MoGe 的结构边界比绝对高度更可靠 | 不依赖绝对 scale/offset，较少污染主标签 | 信号较弱，需要 mask 和权重调参 |
| hard + soft + MoGe prior | 原始、补全、结构先验各司其职 | 最完整 | 工程复杂度最高，需充分 ablation |

本方案优先实现第三种，再根据实验结果决定是否加入 Poisson soft label。

## 训练目标

推荐总 loss：

```text
L_remote =
    L_pointmap_hard
  + lambda_proj * L_projection_aux
  + lambda_grad * L_moge_gradient
  + lambda_edge * L_moge_edge
  + lambda_rank * L_moge_rank
```

其中：

- `L_pointmap_hard` 使用原始 `remote_pointmap` 和 `remote_valid_mask`，保持绝对几何锚点。
- `L_projection_aux` 使用现有 `projection_aux.npz`，继续监督 `rel_height/offset_xy/global_dir/global_slope`。
- `L_moge_gradient` 只匹配归一化梯度，不匹配绝对高度。
- `L_moge_edge` 监督预测高度边缘响应与 MoGe 边缘一致。
- `L_moge_rank` 可选，只监督边缘两侧的局部高低关系。

第一阶段建议只开 `L_moge_gradient` 和 `L_moge_edge`，暂不开 `L_moge_rank`。

## 数据侧新增字段

每个 remote provider 目录可以新增：

```text
moge_prior.npz
```

建议字段：

```text
moge_aligned_height        # float32[H,W]，仅用于生成 prior 和诊断，不作为硬 GT
moge_grad_xy              # float32[H,W,2]，MoGe aligned height 的归一化梯度
moge_grad_mag             # float32[H,W]，梯度幅值，建议做 robust normalization
moge_edge_mask            # bool[H,W]，高置信 MoGe 高度边缘
moge_prior_weight         # float32[H,W]，MoGe prior 的像素权重，范围 [0,1]
moge_confidence_mask      # bool[H,W]，可选，过滤低可信区域
source_keep_mask          # bool[H,W]，原始可靠标签 mask，便于质检
quality_meta              # 可放在 json 或 npz 标量字段中
```

Dataset 读取后转成 view 字段：

```text
remote_moge_grad_xy
remote_moge_grad_mag
remote_moge_edge_mask
remote_moge_prior_weight
remote_moge_confidence_mask
```

这些 map 字段应跟随 remote crop/resize 变换。建议 resize 策略：

- `moge_grad_xy`: bilinear resize 后重新 normalize。
- `moge_grad_mag`: bilinear 或 area。
- `moge_edge_mask`: nearest。
- `moge_prior_weight`: bilinear。

## MoGe Prior 生成流程

### 1. MoGe 推理与对齐

继续沿用 `exp_011_satelitedepth` 的 MoGe 推理和对齐，但输出只用于 prior。

对齐建议：

- 不采用单纯全图 median affine 作为唯一策略。
- 地面区域单独估 offset。
- 非地面/建筑区域估 scale + offset。
- 记录对齐残差分布，作为质量分。

如果 `align_method=range_match` 或 `robust_irls` 出现极端 scale/offset，应直接降权或丢弃该 provider。

### 2. 梯度计算

对 `moge_aligned_height` 计算 Sobel 或有限差分：

```text
g_moge = gradient(moge_aligned_height)
mag_moge = sqrt(gx^2 + gy^2)
```

做 robust normalization：

```text
mag_norm = clip(mag_moge / percentile(mag_moge, 95), 0, 1)
grad_xy_norm = g_moge / (mag_moge + eps)
```

`grad_xy_norm` 用于方向监督，`mag_norm` 用于边缘/权重。

### 3. 边缘 mask

初版：

```text
moge_edge_mask = mag_norm > edge_percentile_threshold
```

建议阈值：

```text
edge_percentile_threshold = p90 或 p95
```

更稳的版本需要同时过滤纹理边缘和低可信区域：

```text
edge_mask =
    high_moge_gradient
  & optional_rgb_edge
  & not_bad_alignment_region
```

其中 `bad_alignment_region` 可由原始有效标签上的 MoGe 残差扩张得到。

### 4. prior weight

`moge_prior_weight` 不应全图为 1。建议由以下因素相乘：

```text
weight =
    edge_or_gradient_weight
  * alignment_consistency_weight
  * distance_to_hard_label_weight
  * provider_quality_weight
```

其中：

- `edge_or_gradient_weight`: 高梯度区域权重大，平坦区域权重小。
- `alignment_consistency_weight`: 原始有效点附近 MoGe 残差越小，权重越大。
- `distance_to_hard_label_weight`: 离可靠原始标签越远，权重越低。
- `provider_quality_weight`: provider/scene 级质量分。

初版可简化为：

```text
weight = 0.2 * mag_norm
weight[bad_alignment_region] = 0
weight[keep_mask & high_residual] = 0
```

## Loss 设计

### 1. Gradient Direction Loss

只监督方向，不监督绝对梯度幅值：

```text
g_pred = gradient(height_pred)
g_pred_dir = g_pred / (|g_pred| + eps)
g_moge_dir = remote_moge_grad_xy

L_grad_dir = weight * (1 - dot(g_pred_dir, g_moge_dir))
```

适用区域：

```text
remote_moge_prior_weight > 0
```

建议权重：

```text
lambda_grad = 0.05 到 0.2
```

### 2. Gradient Magnitude Shape Loss

可选，只匹配归一化幅值：

```text
mag_pred = |gradient(height_pred)|
mag_pred_norm = mag_pred / percentile(mag_pred, 95)

L_grad_mag = robust(mag_pred_norm - mag_moge_norm)
```

这个 loss 比方向 loss 风险更高，因为预测高度幅值早期可能不稳定。建议第二阶段再开。

### 3. Edge Response Loss

将预测高度梯度转成边缘响应：

```text
edge_pred = sigmoid(k * (mag_pred_norm - tau))
edge_gt = remote_moge_edge_mask

L_edge = BCE(edge_pred, edge_gt)
```

为避免把所有纹理边缘都当高度边缘，建议只在候选区域内计算：

```text
candidate_mask = remote_moge_prior_weight > 0
```

建议权重：

```text
lambda_edge = 0.02 到 0.1
```

### 4. Local Rank Loss

可选，用于 MoGe 不严格对齐但相对高低可信的场景。

在边缘两侧采样像素对 `(p1, p2)`：

```text
s_moge = sign(h_moge[p1] - h_moge[p2])
s_pred = h_pred[p1] - h_pred[p2]

L_rank = softplus(-s_moge * s_pred / temperature)
```

该 loss 对 scale/offset 最不敏感，但实现比 gradient/edge 更复杂。建议等前两个 loss 验证有效后再加入。

## 与现有 P7 的接入点

### Dataset

当前 remote provider 会读取：

```text
pixel_to_point_map.npz
projection_aux.npz
```

建议新增可选读取：

```text
moge_prior.npz
```

如果文件不存在，则不启用 MoGe prior loss，保持向后兼容。

### Loss

在 `RSPointmapHeightProjectionAuxLoss` 中新增可选项：

```text
moge_gradient_loss_weight
moge_edge_loss_weight
moge_rank_loss_weight
moge_prior_min_weight
moge_gradient_use_hard_mask
moge_edge_candidate_mode
```

默认值全部为 0，确保旧实验行为不变。

### 预测高度来源

对 VGGT remote point head，预测高度可取 remote predicted pointmap 的 z 分量：

```text
height_pred = pred_remote_pts3d[..., 2]
```

如果当前 loss 使用 view0/canonical 坐标系，需要确保 MoGe prior 也在同一 remote raster 坐标下，只监督 image-plane 梯度，不涉及 world-frame 旋转。

## 质量控制

每个 provider 应生成 `moge_prior_quality.json` 或写入 `moge_prior.npz` 标量字段：

```text
keep_coverage
moge_overlap_residual_mae
moge_overlap_residual_p95
moge_scale
moge_offset
edge_ratio
prior_weight_mean
bad_region_ratio
quality_score
```

建议过滤规则：

```text
keep_coverage < 0.05       -> 丢弃 provider 或 MoGe prior weight=0
keep_coverage 0.05~0.15    -> 只保留 edge loss，低权重
residual_p95 过大          -> 降权或丢弃
edge_ratio < 0.01          -> prior 太弱，跳过
edge_ratio > 0.35          -> 可能纹理边缘过多，降权
```

对已经观察到 `keep_coverage` 低到约 `0.008` 的样本，应默认不参与 MoGe prior 或 remote hard supervision 的主实验。

## 推荐实验顺序

### A0: Sparse Baseline

当前 p7 remote-head projection_aux 设置，不加入 MoGe prior。

目标：

- 确认现有 remote pointmap/projection_aux 的基线表现。
- 记录 rel_height、offset 和 remote pointmap 指标。

### A1: Hard GT + MoGe Gradient Direction

只开方向梯度：

```text
lambda_grad = 0.05
lambda_edge = 0.0
```

判据：

- remote 边界更清晰。
- ordinary/aerial 指标不退化。
- remote pointmap metric 不明显变差。

### A2: Hard GT + MoGe Gradient + Edge

加入 edge response：

```text
lambda_grad = 0.05
lambda_edge = 0.02
```

如果边缘更清楚且没有过度锐化，再尝试：

```text
lambda_grad = 0.1
lambda_edge = 0.05
```

### A3: Hard GT + Poisson Soft Label

对比直接稠密化方案，但补全区必须低权重：

```text
hard_weight = 1.0
soft_weight = 0.1 到 0.3
```

判据：

- 如果 metric 变差，说明 dense pseudo label 污染较大。
- 如果 coverage 带来稳定收益，再考虑 hard + soft + MoGe prior。

### A4: Hard + Soft + MoGe Prior

完整方案：

```text
L = L_hard_pointmap
  + L_soft_poisson_pointmap
  + L_projection_aux
  + L_moge_gradient
  + L_moge_edge
```

这个实验只有在 A1/A2/A3 单独验证有效后再跑。

## 评估指标

不要只看训练 loss。至少看：

```text
remote pointmap abs_rel / mae
projection rel_height mae
projection offset mae
MoGe edge alignment score
height gradient precision/recall
ordinary_damage_vs_reference
same_gain / blank_gain / shuffled_gain
```

新增诊断建议：

```text
edge_precision = pred_edge 与 moge_edge 的交集 / pred_edge
edge_recall = pred_edge 与 moge_edge 的交集 / moge_edge
hard_region_mae = 原始 hard mask 区域误差
soft_region_mae = soft/pseudo 区域误差
edge_region_mae = MoGe edge 附近误差
```

关键验收标准：

- hard region 误差不能因 MoGe prior 上升。
- ordinary/aerial 重建不能明显退化。
- MoGe edge 附近预测结构更清晰，但不能出现全图噪声边缘。

## 风险与规避

### MoGe 纹理边缘误导

卫星图有道路标线、阴影、树冠等纹理边缘，MoGe 可能把它们误当高度变化。

规避：

- 使用 `moge_prior_weight` 降权平坦/低置信区域。
- 只在高 MoGe 梯度且原始标签残差不过大的区域启用。
- 后续可引入 building/tilt mask 过滤。

### 梯度 loss 导致高度噪声

如果只追边缘，模型可能产生过多高频。

规避：

- `lambda_grad/lambda_edge` 从小值开始。
- edge loss 只在 candidate mask 内算。
- 保持 hard pointmap loss 为主。

### 与 projection_aux 不一致

不能用补全高度直接替代 pointmap 而不重算 projection_aux。

规避：

- 第一阶段 MoGe prior 不改变 `pixel_to_point_map.npz` 和 `projection_aux.npz`。
- 如果后续要生成 dense pointmap，必须同步生成 dense/weighted projection_aux 或仅在原始 `projection_aux_valid_mask` 内监督 projection_aux。

## 推荐实现阶段

### Stage 1: 离线 prior 生成

在 `exp_011_satelitedepth` 或新脚本中输出：

```text
moge_prior.npz
moge_prior_quality.json
```

先只覆盖 chicago 的 Google/Bing provider，抽样可视化。

### Stage 2: Dataset 可选读取

给 remote sample 增加可选字段。文件缺失时跳过，不影响旧实验。

### Stage 3: Loss 增量接入

在 `RSPointmapHeightProjectionAuxLoss` 中新增 MoGe prior loss，所有权重默认为 0。

### Stage 4: 小样本 overfit 验证

先使用 `REMOTE_OVERFIT_NUM_SETS=5` 或类似设置验证：

- loss 是否下降；
- edge 可视化是否合理；
- hard region 是否被破坏。

### Stage 5: 正式 ablation

按 A0 到 A4 顺序跑，并记录训练/验证/可视化结果。

## 当前建议默认参数

初版建议：

```text
edge_percentile = 95
prior_weight_base = 0.2
lambda_grad = 0.05
lambda_edge = 0.02
lambda_rank = 0.0
moge_prior_min_weight = 0.02
bad_residual_percentile = 90
keep_coverage_min = 0.05
```

如果发现信号太弱，再逐步提高：

```text
lambda_grad -> 0.1
lambda_edge -> 0.05
prior_weight_base -> 0.3
```

如果 hard region metric 退化，优先降低：

```text
lambda_edge
prior_weight_base
edge_ratio 上限
```

## 最小可行版本

最小版本只需要：

```text
moge_grad_xy
moge_edge_mask
moge_prior_weight
```

只实现：

```text
L_moge_gradient_direction
L_moge_edge_response
```

不改变：

```text
remote_pointmap
remote_valid_mask
projection_aux
```

这样可以用最低风险验证核心假设：MoGe 的结构先验是否能改善 remote 几何边界，同时不破坏原始标签的绝对准确性。

## 2026-06-06 快速验证记录

已实现最小可行版本：

- `scripts/generate_crossview_moge_priors.py`：使用 `MoGeWrapper(name="moge-2", model_string="Ruicheng/moge-2-vitl")` 生成 `moge_prior.npz`。
- dataset 侧读取并随 remote crop/resize 变换：
  - `remote_moge_grad_xy`
  - `remote_moge_grad_mag`
  - `remote_moge_edge_mask`
  - `remote_moge_prior_weight`
  - `remote_moge_confidence_mask`
- `RSPointmapHeightLoss` 新增：
  - `moge_gradient_loss_weight`
  - `moge_edge_loss_weight`
  - `moge_prior_min_weight`
  - `moge_edge_temperature`
  - `moge_edge_threshold`
- `build_remote_supervision_view()` 已修正，会把 MoGe prior 字段复制到 synthetic remote GT view；修正前训练日志中 `rs_moge_required_present=0`，修正后为 1。

数据与质量门控：

- balanced scene list：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/metadata/p7_moge_balanced_20x4_train_scene_list.npy`
- 4 城每城 20 scene，总计 80 scene；Google/Bing provider。
- 使用 `moge_residual_p95 <= 30` 过滤后保留 126 个 prior，删除 31 个 bad prior，缺失 3 个。
- 保留分布：SanFrancisco 39，Seattle 37，Chicago 35，NewYork 15。
- NewYork 的 MoGe2 对齐失败比例较高，因此不适合直接把 MoGe 当 dense hard label；必须继续用 residual gate。

正式快速验证：

```text
p7_moge2_balanced20x4_private_tokens_raw001_gradz005_mogegrad001_edge0002_h003_warme2_e30_b24_4gpu_fixed
```

关键设置：

```text
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_RAW_PM=0.001
LAMBDA_REMOTE_PM_GRAD=0.05
LAMBDA_REMOTE_MOGE_GRAD=0.01
LAMBDA_REMOTE_MOGE_EDGE=0.002
REMOTE_MOGE_PRIOR_MIN_WEIGHT=0.03
LAMBDA_REMOTE_HIGH_Z=0.03
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
REMOTE_PROJECTION_AUX_SOURCE=tokens
USE_REMOTE_PRIVATE_POINT_HEAD=true
```

训练现象：

- 训练稳定，无 NaN。
- `rs_moge_required_present=1`，说明 prior 确实进入 loss。
- `rs_moge_prior_active_ratio` 约 `0.23-0.36`。
- `rs_moge_prior_loss_weighted` 约 `0.015-0.018`，没有压过主 pointmap loss。
- 这次只有 80 scene 且 global batch 96，因此每个 epoch 只有 1 个训练 step；30 epoch 只是快速方向验证，不是充分训练。

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MoGe2 final | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.97 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| MoGe2 best | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.96 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| oldP7 aggtail2 e2 | 0.0450 | 0.0486 | 95.67 | 0.2958 | 9.72 | 0.0486 | 0.0494 | 0.0497 | 0.0009 |
| oldP7 aggtail2 e4 | 0.0448 | 0.0486 | 95.67 | 0.2954 | 10.07 | 0.0486 | 0.0494 | 0.0497 | 0.0009 |

输出路径：

```text
checkpoint:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_moge2_balanced20x4_private_tokens_raw001_gradz005_mogegrad001_edge0002_h003_warme2_e30_b24_4gpu_fixed/checkpoint-final.pth

benchmark summary:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls/p7_moge2_fixed_summary

PLY:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/p7_moge2_balanced20x4_private_tokens_mogegrad001_edge0002_fixed_final
```

结论：

- MoGe2 prior 路线已经证明工程上可跑，并且作为弱梯度/边缘先验不会明显破坏当前 P7-P5B/private/oldP7 结构。
- 它没有证明能显著降低 remote 高度误差：`RS-only MAE=9.96/9.97`，弱于 oldP7 aggtail2 e2 的 `9.72`，强于 e4 的 `10.07`。
- `joint_global` 有小幅正向，但幅度很小；same-vs-blank delta 没有扩大。
- 因此当前 MoGe2 方案只能算“可行但未验证有效”。若用户查看 `mapanything_pointcloud_same_remote.ply` 发现局部建筑形状更好，可以继续扩大质量门控后的 MoGe prior 数据量；否则不建议把 MoGe2 作为下一阶段主线。

## 2026-06-06 projection-aux MoGe prior 验证

动机：用户指出 MoGe 不应只辅助 pointmap，稀疏的 `projection_aux` 多任务 height/offset 监督也可能需要稠密的局部结构先验。因此新增 `RSPointmapHeightProjectionAuxLoss` 内的 projection-MoGe prior，把 MoGe2 edge/gradient 用在 projection-aux 的 `rel_height` 预测上。

实现：

- `mapanything/train/losses.py`：`RSPointmapHeightProjectionAuxLoss` 新增 `projection_moge_gradient_loss_weight`、`projection_moge_edge_loss_weight`、`projection_moge_prior_min_weight`、`projection_moge_edge_temperature`、`projection_moge_edge_threshold`。
- 训练脚本新增 `LAMBDA_PROJ_MOGE_GRAD`、`LAMBDA_PROJ_MOGE_EDGE`、`PROJ_MOGE_PRIOR_MIN_WEIGHT` 等 override。
- 默认向后兼容，所有 projection-MoGe 权重默认为 0。

正式快速验证：

```text
p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu
```

关键设置：

```text
LAMBDA_REMOTE_MOGE_GRAD=0
LAMBDA_REMOTE_MOGE_EDGE=0
LAMBDA_PROJ_MOGE_GRAD=0.02
LAMBDA_PROJ_MOGE_EDGE=0.005
PROJ_MOGE_PRIOR_MIN_WEIGHT=0.03
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_RAW_PM=0.001
LAMBDA_REMOTE_PM_GRAD=0.05
LAMBDA_REMOTE_HIGH_Z=0.03
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
REMOTE_PROJECTION_AUX_SOURCE=tokens
USE_REMOTE_PRIVATE_POINT_HEAD=true
```

训练现象：

- 训练稳定，无 NaN。
- `rs_projection_moge_required_present=1`，说明 MoGe prior 确实进入 projection-aux loss。
- `rs_projection_moge_prior_active_ratio` 约 `0.25-0.38`。
- weighted projection-MoGe prior 约 `0.002-0.003`，相对主 aux loss 较弱。
- global batch 112，80 scene 快速验证每 epoch 约 1 个训练 step，适合方向筛选，不是充分收敛实验。

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS-only MAE | joint RS MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| projection-MoGe aux final | 0.0448 | 0.0485 | 95.67 | 0.2958 | 10.10 | 16.73 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| projection-MoGe aux best | 0.0448 | 0.0486 | 95.67 | 0.2956 | 10.01 | 16.70 | 0.0486 | 0.0494 | 0.0497 | 0.0009 |
| MoGe2 remote prior final | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.97 | 16.72 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| oldP7 aggtail2 e2 | 0.0450 | 0.0486 | 95.67 | 0.2958 | 9.72 | 16.65 | 0.0486 | 0.0494 | 0.0497 | 0.0009 |

输出路径：

```text
checkpoint:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu/checkpoint-final.pth

PLY:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final
```

结论：

- MoGe2 edge/gradient 辅助 projection-aux height 的机制已经验证可训练，并且不会明显破坏 joint reconstruction。
- 但它没有改善 remote-only 高度，`RS-only MAE=10.01/10.10` 弱于 oldP7 aggtail2 e2 的 `9.72`，也弱于 remote pointmap MoGe prior final 的 `9.97`。
- 当前 projection-MoGe prior 权重偏弱，只能提供小的局部结构梯度；如果可视化仍无明显变化，继续单纯提高权重有风险，因为 MoGe2 对齐质量在 NewYork 上并不稳定。
- 下一步更合理的是两条并行线：一是设计更强但门控严格的 aux dense height/ranking prior；二是把 parallel aux-head 思路移植到 PI3 或 VGGT-Omega 这类更强 base，测试是否是 VGGT P7 结构容量/初始化限制。
