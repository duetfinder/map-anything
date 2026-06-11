# P7 Projection Auxiliary 实验说明

## 目标

P7 的目标是在 p5g split-remote late fusion 基础上，显式监督 remote 正射/倾斜投影机制，让 remote 分支学到更稳定的几何表征，再通过 late/gated fusion 辅助普通视角重建。

当前实现仍保留普通 VGGT 主路径，不把 remote 当作早期普通透视视角混入。projection auxiliary 只作用在 remote view 的辅助输出和 loss 上。

## 数据输入

每个 remote provider 目录需要包含：

```text
projection_aux.npz
```

字段：

```text
valid_mask
rel_height
offset_xy
building_mask
tilt_projected_mask
global_dir_xy
global_slope
```

Dataset 会把它们转成：

```text
remote_projection_valid_mask
remote_projection_rel_height
remote_projection_offset_xy
remote_projection_building_mask
remote_projection_tilt_mask
remote_projection_global_dir_xy
remote_projection_global_slope
```

这些 map 字段会跟随 remote crop/resize 变换，global 字段保持不变。训练 loss 使用 `remote_projection_valid_mask`，不是 height occupancy。

## 模型输出

开启：

```yaml
model.model_config.use_remote_projection_aux_head: true
```

VGGTWrapper 会在 remote `pts3d` 上接一个轻量 auxiliary head，输出：

```text
remote_projection_rel_height_pred
remote_projection_offset_xy_pred
remote_projection_global_dir_xy_pred
remote_projection_global_slope_pred
```

第一版的 auxiliary head 只从 remote point prediction 派生，不改 VGGT aggregator 和普通视角输出结构。

## Loss

新增 `RSPointmapHeightProjectionAuxLoss`，在原 `RSPointmapHeightLoss` 之外增加：

```text
L_rel_height
L_offset_xy
L_global_dir
L_global_slope
L_consistency
```

一致性项：

```text
offset_xy_pred ~= rel_height_pred * global_slope_pred * global_dir_xy_pred
```

默认配置在：

```text
configs/loss/vggt_loss_rs_joint_p7_projection_aux.yaml
```

## 训练脚本

主入口：

```bash
bash bash_scripts/train/Crossview/vggt/p7_vggt_projection_aux_split_late_fusion.sh
```

常用对比：

```bash
bash bash_scripts/train/Crossview/vggt/p7_vggt_projection_aux_no_fusion_split_remote.sh
bash bash_scripts/train/Crossview/vggt/p7_vggt_projection_aux_film_split_remote.sh
bash bash_scripts/train/Crossview/vggt/p7_vggt_projection_aux_crossattn_split_remote.sh
```

建议先跑 no-fusion，确认 projection auxiliary head 本身能学习且普通路径不退化，再跑 film/cross-attention。

## Benchmark

训练完成后使用 p7 benchmark wrapper，确保评估时模型也开启 auxiliary head 以兼容 checkpoint：

```bash
REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p7_projection_aux_no_fusion_unified.sh
REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p7_projection_aux_film_unified.sh
REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p7_projection_aux_crossattn_unified.sh
```

## 评估重点

P7 不是只看 projection auxiliary loss 是否下降。最终仍要看：

```text
same_gain__pointmaps_abs_rel > 0
specific_gain_blank__pointmaps_abs_rel > 0
specific_gain_shuffled__pointmaps_abs_rel > 0
ordinary_damage_vs_reference__pointmaps_abs_rel 接近 0 或为负
```

projection auxiliary 指标主要用于解释 remote branch 是否学到了投影机制。

## 当前限制

1. 第一版 auxiliary head 从 remote predicted pointmap 派生，参数很轻，不是完整 projection field decoder。
2. 目前只同步了 chicago 的 projection auxiliary 标签；训练脚本默认 chicago，可直接跑。多城市训练前需要补齐其他城市。
3. `offset_xy` 和 `rel_height` 是 meter-scale 辅助监督，loss 权重需要通过 mini benchmark 调整。


## P7 Remote-Head Projection-Aux 变体

新增 `p7_vggt_remote_head_projection_aux` 是更接近 p5d 的多任务版本：

- 不启用 `use_split_remote_aggregator`。
- 不启用 late fusion / gated residual / view-type bias。
- 普通视角仍走 VGGT 原始 camera+depth 解码路径。
- remote 视角走独立 `remote_point_head`。
- remote 分支额外预测 `rel_height`、`offset_xy`、`global_dir_xy`、`global_slope`，并使用 projection consistency 约束。
- 不使用 p5d 的 `VGGTBranchConsistencyLoss`，即不再强制普通 depth branch 与 point branch 对齐。
- 默认 train_params 冻结 VGGT 主干和普通解码头，只训练 `remote_point_head`、`remote_projection_aux_pixel_head`、`remote_projection_aux_global_head`。

推荐先跑这个变体作为 projection_aux 的低风险验证，因为它最大限度避免 p5g/p6 split fusion 对普通视角表征的额外扰动。

训练命令：

```bash
bash bash_scripts/train/Crossview/vggt/p7_vggt_remote_head_projection_aux.sh
```

导出命令：

```bash
python scripts/export_pointcloud_ply.py \
  --model vggt \
  --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux/checkpoint-best.pth \
  --image_folder /root/autodl-tmp/test/scence/125 \
  --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/125/vggt_p7_remote_head_projection_aux_mixed \
  --vggt_p7_remote_head_projection_aux_export \
  --remote_view_names image.png \
  --export_remote_control_modes same blank
```

## 2026-06-03 迭代：projection_aux 归一化与动态 height scale

### 已观察结果

固定归一化实验：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_posslope_init01_fielddir_deep4_norm_h40_o32_warm_bs4_static_m01
```

关键现象：

- `offset_xy / 32` 后进入较稳定的学习区间，验证集 `rs_projection_offset_pred_loss_abs_mean` 和 `gt_loss_abs_mean` 已经比较接近。
- `rel_height / 40` 仍然明显低估，location_1 可视化里预测均值约 2.4m，GT 均值约 17.1m。
- `global_dir_from_offset` 的方向指标很好，但这主要说明 offset 方向能学，不代表 height 已解决。

height-boost 对照：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_posslope_init01_fielddir_deep4_norm_h40_o32_heightboost_warm_bs4_static_m01
```

关键现象：

- 单纯提高 `LAMBDA_PROJ_REL_HEIGHT` 没有让 height 幅值起来。
- aux 总 loss 反而变差，height 仍停在低幅值输出。
- 因此当前问题不像是简单 loss 权重不足，更像是 height 标签尺度和主重建归一化空间不一致，或有效监督像素/标签质量/decoder 表达能力仍有限。

### 本轮代码改动

`RSPointmapHeightProjectionAuxLoss` 新增 height 标签尺度模式：

```text
projection_rel_height_scale_mode = fixed | gt_pointmap_norm
projection_rel_height_min
projection_rel_height_use_tilt_mask
```

`gt_pointmap_norm` 模式下，`remote_projection_rel_height` 会除以当前 remote pointmap loss 使用的 GT pointmap normalization factor。这样 height 辅助任务和主重建的归一化空间更一致，不再依赖固定米制常数。

同时新增日志：

```text
rs_projection_rel_height_mask_ratio
rs_projection_rel_height_scale_mean
```

这两个指标用于判断 height 监督是否过稀疏，以及动态 scale 是否处在合理范围。

### 下一轮实验

实验名：

```text
p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_posslope_init01_fielddir_deep4_norm_gtpm_o32_hmin2_warm_bs4_static_m01
```

核心设置：

```text
PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm
PROJ_REL_HEIGHT_MIN=2.0
PROJ_OFFSET_SCALE=32.0
PROJ_CONSISTENCY_USE_LOSS_SPACE=true
LAMBDA_PROJ_REL_HEIGHT=1.0
LAMBDA_PROJ_OFFSET=2.0
LAMBDA_PROJ_GLOBAL_DIR=0.05
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
LAMBDA_PROJ_CONSISTENCY=0.0
```

判据：

- 先看 `rs_projection_rel_height_pred_loss_abs_mean_avg` 是否接近 `gt_loss_abs_mean_avg`。如果二者接近，说明 height 至少在归一化空间学到了幅值。
- 再看 `rs_projection_rel_height_loss_avg` 是否稳定下降，且 `mask_ratio` 不是极低值。
- offset 不能明显退化，否则说明 height 目标或 mask 干扰了当前已经可学的 offset。
- 如果 aux 学会但点云可视化仍差，再转向 remote point head 与普通视角重建的耦合方式；如果 aux 仍学不会，优先排查标签质量和 auxiliary decoder 表达能力。

### 结果：gt_pointmap_norm + hmin2 warm-start

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_posslope_init01_fielddir_deep4_norm_gtpm_o32_hmin2_warm_bs4_static_m01
```

运行完成 3 epochs。关键验证集结果：

```text
epoch1: rs_projection_aux_loss ~= 0.1500
        rel_height pred ~= 0.0275, gt ~= 0.0443
        offset pred ~= 0.0329, gt ~= 0.0424

epoch2: rs_projection_aux_loss ~= 0.1555
        rel_height pred ~= 0.0294, gt ~= 0.0443
        offset pred ~= 0.0398, gt ~= 0.0424

epoch3: rs_projection_aux_loss ~= 0.1451
        rel_height pred ~= 0.0245, gt ~= 0.0443
        offset pred ~= 0.0250, gt ~= 0.0424
```

解读：

- 动态 height scale 是正确方向：训练没有崩，`rs_projection_rel_height_scale_mean` 稳定在约 300-370，和 remote pointmap 的 GT norm factor 一致。
- 相比固定 `rel_height/40`，aux 最终略有改善，但幅度有限。
- height 仍然偏低，说明问题不只是标签尺度。当前 aux head 主要从 predicted pointmap/RGB/coord 解码，容易学成低频/常值幅值，缺少对建筑纹理和边界的直接建模能力。
- offset 能进入可学习区间，但 val 上仍不稳定；继续调权重的收益预计有限。

### 下一步：P7 image-stem auxiliary decoder

新增模型开关：

```text
model.model_config.remote_projection_aux_image_stem_dim
```

当该值大于 0 时，remote RGB 会先经过一个轻量卷积 stem，再与 pointmap/RGB/coord 一起输入 projection aux pixel head。这个改动只增强 auxiliary decoder，不改变 VGGT 主干结构，不改变默认 checkpoint 兼容性。

下一轮实验：

```text
p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_imgstem16_posslope_init01_fielddir_deep4_norm_gtpm_o32_hmin2_warm_bs4_static_m01
```

核心假设：如果 height 学不会主要是 aux decoder 表达能力不足或只看 pointmap 太弱，那么 image stem 应该让 `rel_height pred_loss_abs_mean` 更接近 `gt_loss_abs_mean`，同时降低 `rel_height_loss`；如果仍没有改善，就更应优先怀疑标签质量或 remote pointmap/head 的信息瓶颈。


### 结果：image-stem16 auxiliary decoder

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk_detach_rgb_coord_imgstem16_posslope_init01_fielddir_deep4_norm_gtpm_o32_hmin2_warm_bs4_static_m01
```

运行完成 3 epochs。关键验证集结果：

```text
epoch0: rs_projection_aux_loss ~= 0.1462
        rel_height pred ~= 0.0238, gt ~= 0.0443
        offset pred ~= 0.0235, gt ~= 0.0424

epoch1: rs_projection_aux_loss ~= 0.1446
        rel_height pred ~= 0.0246, gt ~= 0.0443
        offset pred ~= 0.0185, gt ~= 0.0424

epoch3: rs_projection_aux_loss ~= 0.1446
        rel_height pred ~= 0.0239, gt ~= 0.0443
        offset pred ~= 0.0221, gt ~= 0.0424
```

解读：

- image-stem 能正常训练，warm-start 只跳过了输入通道变化导致的 `remote_projection_aux_pixel_head.0.weight`，其余权重正确加载。
- best val aux loss 从上一轮 `0.1451` 到 `0.1446`，改善非常小，不能认为解决了问题。
- height 和 offset 仍然存在系统性低幅值：预测均值大约只有 GT 的一半。
- 训练 batch 中 `global_dir_cosine` 能明显上升，说明全局低维投影参数是可学的；真正困难集中在稀疏 pixel field 回归。

当前判断：

- 问题不像是单纯 warm-start、固定 height scale 或 decoder 缺少 RGB 细节导致。
- 更可能是 sparse label 的监督分布把 L1 回归推向低幅值均值，或者 projection_aux 标签中有效像素/高度/offset 的质量和分布仍需核查。
- 继续盲目加大 decoder 容量收益不高；下一步应该先做小样本 overfit 诊断。如果小样本也不能把 height/offset 幅值拟合上去，优先回查标签和 mask；如果小样本能拟合但 Chicago val 不行，再考虑泛化、正则、采样和 loss balancing。

### 下一步：小样本 overfit 诊断

目标：

- 使用 `overfit_num_sets` 限制训练集到极少数场景。
- 让 projection aux head 在固定少量样本上高强度学习。
- 判断 `rel_height pred_loss_abs_mean` 和 `offset pred_loss_abs_mean` 是否能接近 GT。

判据：

- 如果小样本 overfit 后预测幅值仍明显低于 GT，说明当前 pixel field 目标、mask 或 head 输入存在结构性问题。
- 如果小样本能拟合，但正常训练不能泛化，说明标签并非完全不可学，后续重点转向样本分布、稀疏监督重加权和数据增强。


### 结果：Chicago top16 hard-overfit 诊断

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_top16_auxonly_imgstem16_gtpm_o32_hmin2_google
```

配置要点：

- 只使用 Chicago 中 projection_aux 信号最强的 top16 scenes。
- `remote_pointmap_loss_weight=0`，只诊断 projection aux 是否可学。
- `remote_projection_aux_detach_pointmap=true`，aux head 不反向污染 pointmap head。
- height 使用 `gt_pointmap_norm` 归一化，offset 使用 `offset_scale=32`。
- aux head/image stem/global head LR 为 `3e-4`。

最终 epoch40 验证集关键结果：

```text
rs_projection_aux_loss ~= 0.9321
rel_height pred ~= 0.3075, gt ~= 0.2535
offset pred ~= 0.2612, gt ~= 0.1983
global_dir_cosine ~= 1.0000
```

解读：

- 这个结果推翻了“aux head 完全学不到 projection field”的判断。高信号样本上，height/offset 的预测幅值能追上甚至超过 GT。
- 但 loss 没有被压到很低，说明当前结构学到的是投影字段的全局/低频强度，像素级精确匹配仍然不足。
- top16 后期出现过预测，说明单纯继续增大 aux 权重或 LR 不一定有益，容易从全量训练的低幅值塌缩走到小样本过强幅值。
- 当前核心问题更像是标签/样本分布与稀疏监督训练策略，而不是 projection_aux 标签绝对不可学。

### 标签统计：Chicago vs New York

新增脚本：

```text
scripts/analyze_projection_aux_labels.py
```

统计输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_label_stats/chicago_google_projection_aux_stats.csv
outputs/mapanything_experiments/mapanything/debug/projection_aux_label_stats/newyork_google_projection_aux_stats.csv
```

Google_Satellite 标签摘要：

```text
Chicago: n=500
  valid_ratio mean ~= 0.2087
  tilt_ratio mean ~= 0.0419
  height_mean_tilt mean ~= 23.7352
  offset_mean_tilt mean ~= 4.5142
  score_height_offset mean ~= 1.4537, max ~= 8.9498

New York: n=500
  valid_ratio mean ~= 0.1534
  tilt_ratio mean ~= 0.0852
  height_mean_tilt mean ~= 37.2053
  offset_mean_tilt mean ~= 6.7128
  score_height_offset mean ~= 3.5923, max ~= 13.2318
```

解读：

- New York 的有效倾斜监督比例、height 强度和 offset 强度整体高于 Chicago。
- New York 不是所有 scene 都更好，仍有少量 `tilt_ratio=0` 的样本，但整体监督信号显著更强。
- 因此 New York 是更合理的 city-level 训练对照：它能测试“全量 Chicago 表现差”是否主要由弱标签/弱监督分布造成。

### 进行中：New York city-level projection_aux 对照

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_auxonly_imgstem16_gtpm_o32_hmin2_city
```

配置要点：

- 使用 `dataset.vigor_chicago_joint_rs_aerial.{train,val,test}.cities=[newyork]` 直接指定城市。
- 不使用 `scene_list_path`，避免 topK 选择带来的 hard-overfit 偏差。
- `remote_pointmap_loss_weight=0`，继续先诊断 projection_aux 学习。
- aux head/image stem/global head LR 降为 `2e-4`，因为 Chicago top16 后期出现幅值过预测。
- 保持 image-stem16、RGB+coord、positive slope、4 residual blocks、`gt_pointmap_norm` height 归一化。

启动前修复：

- `scene_list_path='None'` 曾被 dataset 当成真实路径读取，导致 city-level 训练报 `Missing scene list: None`。
- 已在 `VigorChicagoWAI._load_data()` 中将 `None`、`"None"`、`"null"`、空字符串统一视为不启用 scene list。

epoch0 初始观察：

```text
train length = 225
initial rel_height pred ~= 0.0246, gt ~= 0.1302
initial offset pred ~= 0.0218, gt ~= 0.1327
```

后续判据：

- 如果 New York 在 epoch1-5 内 prediction 明显追上 GT，说明标签强度/质量是 Chicago 全量失败的重要因素。
- 如果 New York 仍长期停在低幅值，说明问题更偏向 decoder 输入/监督形式，需要考虑 pixel-level reweight、mask-aware sampling 或更直接的 image-conditioned projection field decoder。
- 如果 New York 很快过预测，则需要降低 aux LR/权重，或对 height/offset 加幅值分桶重加权而不是统一 L1。


### New York city-level epoch5 诊断结果

实验在 epoch5 eval 后停止，`checkpoint-best.pth` 已保存。停止原因是诊断目标已经达成：可以判断 New York 标签是否缓解 projection_aux 低幅值问题，不需要继续跑满 30 epochs。

关键结果：

```text
train epoch0:
  rs_projection_aux_loss ~= 0.6665
  rel_height pred ~= 0.0587, gt ~= 0.1299
  offset pred ~= 0.0599, gt ~= 0.1092

train epoch1:
  rs_projection_aux_loss ~= 0.5213
  rel_height pred ~= 0.0816, gt ~= 0.1149
  offset pred ~= 0.0775, gt ~= 0.1021

train epoch4:
  rs_projection_aux_loss ~= 0.5319
  rel_height pred ~= 0.0998, gt ~= 0.1278
  offset pred ~= 0.0841, gt ~= 0.1054

eval epoch5:
  rs_projection_aux_loss ~= 0.6776
  rel_height pred ~= 0.1821, gt ~= 0.1360
  offset pred ~= 0.1267, gt ~= 0.1230
  global_slope pred ~= 0.1821, gt ~= 0.1744
```

解读：

- New York 确实缓解了 projection_aux 低幅值塌缩。到 epoch5，height/offset 的预测幅值已经能达到甚至超过 GT。
- 这说明 projection_aux 目标和当前 aux decoder 不是完全不可学；之前 Chicago/full 低幅值问题很大程度来自标签信号弱、强弱样本混合和稀疏监督分布。
- 但 eval aux loss 仍然较高，height 出现过预测，offset 只是在幅值上接近。问题已经从“能否学出投影幅值”转移到“能否在像素级位置和局部结构上对齐”。
- 继续单纯跑更久或者继续加大 aux 权重预计收益有限，甚至可能扩大 height 过预测。

下一步改进：

- 引入 projection_aux pixel loss 的 target-magnitude-aware weighting。
- 目的不是继续放大预测幅值，而是让强投影区域提供更稳定梯度，同时避免大量弱样本/弱像素把 L1 训练推向平均场。
- 权重需要归一化到均值约 1，避免整体 loss scale 被改变太多。
- 先做 aux-only 诊断，再决定是否重新打开 remote pointmap 主任务。


### New York target-magnitude weighting 短诊断

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_auxonly_imgstem16_gtpm_o32_hmin2_targetw_h2_lr1e4_e3
```

配置要点：

- city override 使用 `newyork`。
- aux-only：`remote_pointmap_loss_weight=0`，继续只判断 projection_aux。
- 从上一轮 image-stem16/gt_pointmap_norm checkpoint warm-start。
- `LAMBDA_PROJ_REL_HEIGHT=2`，`LAMBDA_PROJ_OFFSET=4`。
- `PROJ_REL_HEIGHT_TARGET_WEIGHT=0.5`，`PROJ_OFFSET_TARGET_WEIGHT=1.0`，按目标幅值归一化加权，权重均值约 1。
- 只跑 3 epochs，用于快速判断 target weighting 是否稳定。

关键验证结果：

```text
eval epoch3:
  rs_projection_aux_loss ~= 0.4923
  rel_height pred ~= 0.1186, gt ~= 0.1360
  offset pred ~= 0.1105, gt ~= 0.1230
  global_slope pred ~= 0.1798, gt ~= 0.1744
  global_dir_cosine ~= 0.9497
```

对比上一轮 New York epoch5：

```text
previous eval epoch5:
  rs_projection_aux_loss ~= 0.6776
  rel_height pred ~= 0.1821, gt ~= 0.1360
  offset pred ~= 0.1267, gt ~= 0.1230
  global_slope pred ~= 0.1821, gt ~= 0.1744
```

解读：

- target weighting 没有引入训练崩溃，验证 aux loss 从 `0.6776` 降到 `0.4923`，但两个实验 epoch 数和 loss 权重不同，只能视为正向诊断信号。
- 新实验避免了上一轮 height 明显过预测，height/offset 均值更接近 GT。
- `global_slope` 已经非常稳定，说明全局投影斜率不是当前瓶颈。
- 第 3 epoch 后学习率已经接近 0，这轮 schedule 太短，不能作为最终训练配置。

### target-weight checkpoint 可视化诊断

可视化输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_vis/newyork_location_121_targetw_h2
outputs/mapanything_experiments/mapanything/debug/projection_aux_vis/newyork_location_1_targetw_h2
```

使用参数：

```text
--preset remote_head
--detach-aux
--use-rgb-aux
--use-coord-aux
--aux-num-blocks 4
--image-stem-dim 16
--positive-slope-aux
--field-dir-from-offset
--pred-normalized
--rel-height-scale 315
--offset-scale 32
```

`location_121` 是高信号样本：

```text
rel_height_gt_mean ~= 88.27m
rel_height_pred_mean ~= 8.63m
offset_gt_abs_mean ~= 15.54px
offset_pred_abs_mean ~= 3.27px
field_dir_from_offset_cosine ~= 0.9994
global_dir_head_cosine ~= -0.0685
```

`location_1` 是普通样本：

```text
rel_height_gt_mean ~= 15.66m
rel_height_pred_mean ~= 7.49m
offset_gt_abs_mean ~= 2.67px
offset_pred_abs_mean ~= 3.15px
field_dir_from_offset_cosine ~= 1.0000
global_dir_head_cosine ~= -0.0307
```

解读：

- offset field 的平均方向和 GT 高度一致，说明模型从像素 offset 里学到了投影方向。
- global direction head 自身仍然不可靠；目前应继续使用 `global_dir_from_offset=true`，并弱化/移除独立 global dir head 的解释权。
- 普通样本上 offset 幅值基本可学；高信号样本上 height/offset 明显低估。
- 因此当前主要问题不是 projection_aux 完全学不会，而是强投影/高建筑样本在混合训练中被平均掉。下一步优先尝试 high-signal curriculum 或更温和的高幅值样本重采样，再考虑打开 remote pointmap 主任务。

### 下一步实验建议

优先级从高到低：

1. New York high-signal curriculum：用 `analyze_projection_aux_labels.py` 的 score 选 New York topK 场景，先训练 aux-only 8-12 epochs，验证高楼样本是否能拟合。
2. New York city-level longer schedule：保留 target weighting，但把训练延长到 8-10 epochs，避免 3 epoch cosine schedule 太快衰减。
3. 若 high-signal curriculum 成功，再混合 full New York，并打开很小的 remote pointmap loss，例如 `LAMBDA_REMOTE_PM=0.05-0.1`。
4. 在 aux 没有稳定学会强投影样本之前，不建议直接做全数据主重建实验，否则很可能继续污染 remote point head 或普通视角表征。


### New York top128 capacity / image-stem-lr 诊断

实验输出：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_capacity_imgstemlr_e6
```

关键设置：

```text
cities=[newyork]
scene_list_path=newyork_google_top128_projection_aux_scenes.npy
Google_Satellite only
aux-only: LAMBDA_REMOTE_PM=0, LAMBDA_REMOTE_H=0
hidden_dim=128, image_stem_dim=32, aux_num_blocks=6
remote_projection_aux_image_stem lr=1e-4
rel_height_scale_mode=gt_pointmap_norm, rel_height_min=2, offset_scale=32
target weighting: rel_height=0.5, offset=1.0
```

本轮同时修复了两个实验配置问题：

- 默认 `vggt_loss_rs_joint_p7_remote_head_projection_aux.yaml` 缺少若干 projection loss 字段，导致脚本传参时 Hydra 严格模式报错；已补齐到和 anticollapse 配置一致的接口。
- 默认 `vggt_p7_remote_head_projection_aux.yaml` 没有给 `remote_projection_aux_image_stem` 配 lr，导致 RGB image stem 虽然存在但不训练；已补上 `1e-4`。

训练/验证结果：

```text
Eval epoch3:
  rs_projection_aux_loss ~= 0.8992
  rel_height pred ~= 0.2446, gt ~= 0.2304
  offset pred ~= 0.1933, gt ~= 0.1169
  global_slope pred ~= 0.1534, gt ~= 0.1754

Eval epoch6:
  rs_projection_aux_loss ~= 0.8836
  rel_height pred ~= 0.2988, gt ~= 0.2304
  offset pred ~= 0.1910, gt ~= 0.1169
  global_slope pred ~= 0.1760, gt ~= 0.1754
```

对比上一轮 `hidden=64/image_stem=16/blocks=4/top128/e4`：上一轮最终 `rs_projection_aux_loss ~= 0.8222`，因此单纯增大 aux head 容量没有带来验证提升。

可视化输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_vis/newyork_location_121_capacity_imgstemlr_e6
outputs/mapanything_experiments/mapanything/debug/projection_aux_vis/newyork_location_1_capacity_imgstemlr_e6
```

`location_121` 高信号样本：

```text
rel_height_gt_mean ~= 88.27m
rel_height_pred_mean ~= 23.22m
offset_gt_abs_mean ~= 15.54px
offset_pred_abs_mean ~= 4.50px
rel_height_mae ~= 78.43m
offset_mae ~= 6.94px
field_dir_from_offset_cosine ~= 1.0000
global_slope_pred ~= 0.1767, gt ~= 0.1763
```

`location_1` 普通样本：

```text
rel_height_gt_mean ~= 15.66m
rel_height_pred_mean ~= 23.99m
offset_gt_abs_mean ~= 2.67px
offset_pred_abs_mean ~= 4.58px
rel_height_mae ~= 17.23m
offset_mae ~= 1.51px
field_dir_from_offset_cosine ~= 0.9995
global_slope_pred ~= 0.1767, gt ~= 0.1799
```

解读：

- 更大 aux head 可以稳定学到投影方向和全局 slope，但没有显著改善高楼局部幅值；`location_121` 的 height 只从上一轮约 `20.86m` 到 `23.22m`，仍远低于 `88.27m`。
- 普通样本上 height/offset 明显过预测，说明模型仍倾向输出一个偏平滑的全局投影场，而不是按建筑局部结构拟合。
- 因此当前瓶颈不太像简单解码容量不足，更像监督/采样分布在稀疏标签下驱动模型学均值：高信号样本被局部稀疏、crop、mask 分布和 L1 平均目标稀释。

下一步优先方向：

1. 不继续盲目加大 head；保留 `hidden=64/image_stem=16/blocks=4` 作为更稳的 baseline。
2. 把评估从“batch 平均 loss”拆成 high/low projection 分桶指标，至少按 GT offset/height 分位数输出 MAE，确认高楼区域是否持续被低估。
3. 训练上改为 hard-pixel / high-offset reweight，而不是只按样本级 topK；当前 target-weight 在 mask 内均值为 1，未真正拉开高楼像素权重。
4. 如果继续改结构，优先做多尺度/边界感知的局部 decoder 或加入 building footprint/valid dense proxy，而不是继续增大普通 conv head。

### New York top128 balanced / contrast / bucket calibration 追加实验

本轮围绕“高值建筑投影被平均掉、低值区域被整体抬高”的问题，新增并测试了三类 loss：

- balanced bucket loss：按 GT 幅值分成 low/mid/high 三桶，对每桶 L1 等权平均，减少高值像素被数量更多的低值像素淹没。
- bucket contrast loss：约束 high-low 的预测均值差接近 GT 均值差，直接鼓励空间起伏。
- bucket mean calibration：分别约束 low/mid/high 三桶的预测均值接近 GT，试图避免只拉高 high 时把 low 一起抬高。

#### balanced high-value loss

实验输出：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_balanced_high_e4
```

epoch4 验证关键结果：

```text
rel_height_high20_gt/pred ~= 0.5509 / 0.1543
rel_height_low80_gt/pred  ~= 0.1503 / 0.1442
offset_high20_gt/pred     ~= 0.6383 / 0.3196
offset_low80_gt/pred      ~= 0.1304 / 0.2781
```

`location_121` 可视化：

```text
rel_height_pred_mean/std ~= 42.16 / 3.26, gt ~= 88.27 / 105.10
offset_pred_abs_mean ~= 8.09, gt ~= 15.54
```

解读：balanced loss 能改善幅值塌缩，但空间起伏仍很弱，height 几乎是平滑场。

#### balanced + contrast loss

实验输出：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_balanced_contrast_e4
```

epoch4 验证关键结果：

```text
rel_height_high20_gt/pred ~= 0.5509 / 0.3951
rel_height_low80_gt/pred  ~= 0.1503 / 0.2514
rel_height_gap_gt/pred    ~= 0.4592 / 0.1559
offset_high20_gt/pred     ~= 0.6383 / 0.6749
offset_low80_gt/pred      ~= 0.1304 / 0.3672
offset_gap_gt/pred        ~= 0.5839 / 0.3445
```

`location_121` 可视化：

```text
rel_height_pred_mean/std ~= 41.98 / 23.27, gt ~= 88.27 / 105.10
offset_pred_abs_mean ~= 5.34, gt ~= 15.54
```

解读：contrast loss 明显提高了空间起伏，是目前最有用的方向；但它通过整体抬高 low 区域来换取 high 区域改善，低值区域过预测变成主要误差。

#### bucket mean calibration

本轮新增代码：

```text
RSPointmapHeightProjectionAuxLoss.projection_*_bucket_mean_loss_weight
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT
PROJ_OFFSET_BUCKET_MEAN_WEIGHT
```

实验输出：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_bucketmean_e4
```

epoch4 验证关键结果：

```text
rel_height_high20_gt/pred ~= 0.5509 / 0.2640
rel_height_low80_gt/pred  ~= 0.1503 / 0.1952
offset_high20_gt/pred     ~= 0.6383 / 0.4363
offset_low80_gt/pred      ~= 0.1304 / 0.2844
```

`location_121` 可视化：

```text
rel_height_pred_mean/std ~= 40.96 / 17.02, gt ~= 88.27 / 105.10
offset_pred_abs_mean ~= 5.66, gt ~= 15.54
```

解读：bucket mean calibration 没有优于 contrast-only。它缓和了部分 high 过冲，但单场景空间起伏反而变弱，说明只锚定桶均值仍不足以让 decoder 学会局部建筑/非建筑分离。

#### low-overpred 反例

本轮新增代码：

```text
RSPointmapHeightProjectionAuxLoss.projection_*_low_overpred_loss_weight
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT
PROJ_OFFSET_LOW_OVERPRED_WEIGHT
```

实验输出：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_lowover_e4
```

该实验在 epoch2 后中断，因为现象已经明确：固定大权重 low-overpred 会把 offset 整体压到接近 0。epoch2 附近典型指标：

```text
offset_high20_gt/pred ~= 0.6383 / 0.0243
offset_low80_gt/pred  ~= 0.1304 / 0.0165
```

解读：低桶过预测惩罚方向是必要的，但不能从训练一开始以大权重固定启用；否则模型选择“全部预测接近 0”来规避 low 区域惩罚，high 区域完全学不起来。

### 当前结论

1. 标签分布不均衡确实是 projection_aux 学不稳的重要原因：普通 L1/均值 loss 会倾向输出平滑常数场。
2. high-value balanced 和 contrast 能显著改善高值区域，但会引入 low 区域过预测。
3. bucket mean 不能单独解决 low 过预测，且可能降低单场景空间起伏。
4. low-overpred 必须做成调度项或弱约束项，不能固定大权重从头启用。
5. 当前 aux head 已能学到全局投影方向和 slope，但“局部哪里该有 offset/height”仍是主要瓶颈。

### 下一步实验

优先做 warmup/gated loss schedule，而不是继续堆 loss 权重：

```text
阶段1: 先训练 balanced + contrast，允许模型学 high 区域和方向。
阶段2: 从 epoch2 或 epoch3 开始逐步启用 low-overpred，小权重线性升高。
阶段3: low-overpred 只作用于 offset，height 先保留 contrast/balanced，避免两者同时压塌。
```

建议初始参数：

```text
PROJ_REL_HEIGHT_BALANCED_WEIGHT=1.0
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=1.0
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=0.0
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.0

PROJ_OFFSET_BALANCED_WEIGHT=1.5
PROJ_OFFSET_CONTRAST_WEIGHT=0.8
PROJ_OFFSET_BUCKET_MEAN_WEIGHT=0.5
PROJ_OFFSET_LOW_OVERPRED_WEIGHT: epoch0-1 为 0，epoch2 后线性升到 1.0 或 1.5
```

如果调度后仍然不能提升单场景 `pred_std`，下一步应改 decoder 输入，而不是继续调 loss：加入更强的局部图像分支、多尺度 satellite feature，或显式 building/edge proxy。

## 2026-06-03 Newyork Top128 Aux-Only 对照

### 实验目的

这一轮只在 Newyork top128 / Google remote 上做 aux-only 诊断，目标不是提升主重建，而是判断 projection_aux 是否真的可学习：

1. 如果 aux 仍然学不动，说明标签、任务定义或 remote 表征本身还有问题。
2. 如果 aux 能学动，再把它作为 remote 成像机制监督接回主重建联训。

### 对照组

#### scheduled_lowover_e6

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_scheduled_lowover_e6
```

epoch6 验证结果：

```text
rel_height high20 gt/pred ~= 0.5509 / 0.6184
rel_height low80  gt/pred ~= 0.1503 / 0.2953
rel_height contrast gt/pred ~= 0.4592 / 0.3569

offset high20 gt/pred ~= 0.6383 / 0.6525
offset low80  gt/pred ~= 0.1304 / 0.2937
offset low bucket gt/pred ~= 0.0544 / 0.2502
offset contrast gt/pred ~= 0.5839 / 0.4022
```

解读：模型能把 high 区域抬起来，但 low/background 过预测严重。调度 low-overpred 只缓解了一部分，不能解决局部区域分离。

#### strong_bucket_low_e6

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_strong_bucket_low_e6
```

epoch6 验证结果：

```text
rel_height high20 gt/pred ~= 0.5509 / 0.4676
rel_height low80  gt/pred ~= 0.1503 / 0.2291
rel_height contrast gt/pred ~= 0.4592 / 0.2637

offset high20 gt/pred ~= 0.6383 / 0.5647
offset low80  gt/pred ~= 0.1304 / 0.2620
offset low bucket gt/pred ~= 0.0544 / 0.2250
offset contrast gt/pred ~= 0.5839 / 0.3397
```

解读：加强 bucket/low 约束会压低一部分低值预测，但同时削弱 high 响应，空间起伏更弱。说明继续堆 loss 权重不是主要突破口。

#### capacity_grad_e6

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diag_newyork_top128_auxonly_capacity_grad_e6
```

关键设置：

```text
remote_projection_aux_hidden_dim=96
remote_projection_aux_image_stem_dim=32
remote_projection_aux_num_blocks=6
remote_projection_aux_detach_pointmap=false
```

epoch6 验证结果：

```text
rel_height high20 gt/pred ~= 0.5509 / 0.5337
rel_height low80  gt/pred ~= 0.1503 / 0.1152
rel_height low bucket gt/pred ~= 0.0917 / 0.0729
rel_height high bucket gt/pred ~= 0.5509 / 0.5337
rel_height contrast gt/pred ~= 0.4592 / 0.4608

offset high20 gt/pred ~= 0.6383 / 0.6947
offset low80  gt/pred ~= 0.1304 / 0.1328
offset low bucket gt/pred ~= 0.0544 / 0.0545
offset high bucket gt/pred ~= 0.6383 / 0.6947
offset contrast gt/pred ~= 0.5839 / 0.6402
```

`newyork__location_121` 可视化 summary：

```text
rel_height_mae ~= 59.28
offset_mae ~= 4.77
rel_height_norm_mae ~= 0.1882
offset_norm_mae ~= 0.1490
rel_height pred mean/std ~= 38.91 / 36.08, gt ~= 88.27 / 105.10
offset pred abs mean ~= 7.26, gt ~= 15.54
field_dir_from_offset_cosine ~= 0.99998
global_dir_head_cosine ~= 0.7454
```

可视化输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_vis/newyork_location_121_capacity_grad_e6/projection_aux_gt_pred_grid.png
```

解读：这是目前第一组能比较稳定学到 projection_aux 分布结构的实验。它不是通过更大的 low-overpred 权重得到的，而是通过更强 aux decoder 和允许 aux loss 回传到 remote point features 得到的。由此看，早期失败的主因更像是“aux 头输入表征/解码能力不足 + detach 阻断了 remote 表征适配”，而不只是标签或 loss 权重。

### 当前判断

1. projection_aux 是可学习的，至少在 Newyork top128 上已经能学到 height/offset 的 high-low 分布、contrast 和 offset 方向。
2. offset 已经明显不再是主要瓶颈；height 仍更难，单样本可视化中 pred mean/std 仍低于 GT，说明高楼/高值区域幅度还没有完全恢复。
3. `global_dir_head` 仍不如直接从 offset field 聚合出的方向稳定，后续可以降低或取消 global_dir_head 的权重，把方向一致性更多交给 offset field。
4. 下一步应进入主重建联训验证：保留 `capacity_grad` 的 aux 结构，让 projection_aux 作为 remote 成像机制监督，同时小心保护普通视角重建，避免 remote 梯度污染 ordinary branch。

### 下一步训练建议

优先从以下配置开始：

```text
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false

LAMBDA_REMOTE_PM 先设小值或 0
LAMBDA_PROJ_REL_HEIGHT 保持 0.5 左右
LAMBDA_PROJ_OFFSET 保持 1.0-1.5
PROJ_GLOBAL_DIR_WEIGHT 降低或关闭
PROJ_GLOBAL_SLOPE_WEIGHT 保留小权重
```

验证顺序：

1. 先用 `newyork` 小规模联训，确认 projection_aux 不退化。
2. 再跑 mini benchmark，比较 ordinary-only、remote-only、joint remote 的重建变化。
3. 如果 ordinary 仍被拉差，打开 ordinary branch 保护或把 remote fusion 推迟到更 late/gated 的位置。

## 2026-06-03 P7 融入 P5B Shared-Norm 主联训进度

### 已完成

已经把 projection_aux 多任务监督接入最基础的 P5B shared-norm 训练路线，形成新的主联训脚本：

```text
bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh
configs/train_params/vggt_p7_p5b_shared_norm_projection_aux.yaml
```

这个实验不再是 aux-only 诊断，而是同时包含：

```text
普通视角 aerial reconstruction loss
remote pointmap shared-normalization loss
remote projection_aux 成像机制多任务 loss
```

核心默认设置：

```text
REMOTE_POINTMAP_NORM_MODE=aerial_avg_dis
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_H=0.0

REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
REMOTE_PROJECTION_AUX_USE_RGB=true
REMOTE_PROJECTION_AUX_USE_COORD=true

PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm
PROJ_OFFSET_SCALE=32.0
LAMBDA_PROJ_REL_HEIGHT=0.5
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
LAMBDA_PROJ_GLOBAL_DIR=0.0
LAMBDA_PROJ_CONSISTENCY=0.0
```

`LAMBDA_PROJ_CONSISTENCY` 暂时关闭，因为当前 height 使用 pointmap norm 尺度，而 offset 使用固定像素尺度，现有 consistency 公式不是 scale-aware；直接打开可能把两个不同尺度硬绑在一起。后续如果要开 consistency，应先改成显式 scale-aware 版本。

### Smoke Run 结果

测试命令采用 Newyork、2 GPU、1 epoch smoke，输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_p5b_projection_aux_smoke_newyork_2v
```

#### 发现并修复的问题 1：DDP ready-twice

最初使用 P5B shared point head 时，2 卡 backward 报错：

```text
model.point_head.scratch.output_conv2.2.weight has been marked as ready twice
```

原因：shared point head 在 VGGT checkpointed/reentrant backward 下被 remote pointmap 和 projection_aux 路径复用，DDP 默认动态图模式会重复标记同一参数。

修复：在专用 train_params 中启用固定图 DDP：

```yaml
ddp_static_graph: true
ddp_find_unused_parameters: false
```

修复后，ready-twice 错误消失。

#### 发现的问题 2：4 views / batch 4 在 2 卡上 OOM

`NUM_VIEWS=4, BATCH_SIZE=4` 在 2 张 48G 卡上会 OOM。原因是主干联训 + shared point head + projection_aux head 同时反传，显存明显高于 aux-only 诊断实验。

已将脚本默认值改为 2 卡可运行设置：

```bash
NUM_VIEWS=${NUM_VIEWS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
```

如果要恢复 4 views，需要更多显存、梯度累积、冻结部分模块，或降低 aux decoder 容量。

#### 已跑通的路径

使用 `NUM_VIEWS=2, BATCH_SIZE=2` 已成功跑过：

```text
model build
projection_aux head build
train/test criterion build
dataset remote projection labels read
forward
loss compute
backward
optimizer step
projection_aux metrics logging
```

第一步训练日志已经打印完整的主重建和 projection_aux 指标，例如：

```text
aerial_loss
remote_loss
rs_pointmap_loss
rs_projection_aux_loss
rs_projection_rel_height_high20_gt_mean / pred_mean
rs_projection_rel_height_low80_gt_mean / pred_mean
rs_projection_rel_height_contrast_gt_gap / pred_gap
rs_projection_offset_high20_gt_mean / pred_mean
rs_projection_offset_low80_gt_mean / pred_mean
rs_projection_offset_contrast_gt_gap / pred_gap
rs_projection_offset_bucket_mean_low_gt / pred
```

这说明当前脚本已经不是纯配置可解析，而是实际训练第一步可运行。

### 当前阶段结论

1. projection_aux 多任务路径已经从 aux-only 诊断成功接入 P5B shared-norm 主联训。
2. 当前训练路径的主要运行 bug 已处理：DDP shared-head backward 通过 `static_graph` 修复。
3. 2 GPU 下推荐从 `NUM_VIEWS=2, BATCH_SIZE=2` 开始，不建议直接用 4 views。
4. 当前还没有证明它能提升主重建，只证明了主联训代码路径已经跑通。
5. 下一步重点不再是继续写结构，而是跑完整小规模实验，看 projection_aux 在主联训中是否保持可学习，并检查 ordinary reconstruction 是否被破坏。

### 下一步目标

#### 目标 1：完整跑 Newyork 小规模主联训

推荐先跑：

```bash
cd /root/autodl-tmp/Models/map-anything
TRAIN_CITIES=[newyork] VAL_CITIES=[newyork] TEST_CITIES=[newyork] \
NUM_GPUS=2 CUDA_DEVICES=0,1 NUM_VIEWS=2 BATCH_SIZE=2 \
OUTPUT_DIR='/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_p5b_projection_aux_newyork_2v' \
bash bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh
```

优先观察 6 个信号：

```text
aerial_loss 是否稳定，不能明显劣于 P5B
rs_pointmap_loss 是否稳定
rs_projection_aux_loss 是否下降
rel_height contrast_pred_gap 是否接近 contrast_gt_gap
offset contrast_pred_gap 是否接近 contrast_gt_gap
offset low bucket pred 是否接近 low bucket gt
```

#### 目标 2：可视化 projection_aux 输出

训练到若干 epoch 后，用当前 checkpoint 可视化 projection_aux：

```text
scripts/visualize_projection_aux_outputs.py
```

重点比较：

```text
rel_height_gt / pred
offset_gt / pred
rel_height_norm_mae
offset_norm_mae
field_dir_from_offset_cosine
```

如果主联训后 aux 输出重新塌缩，说明 remote pointmap 主监督和 projection_aux 之间仍存在优化冲突，需要降低 `LAMBDA_REMOTE_PM` 或阶段式训练。

#### 目标 3：mini benchmark 评估主重建

aux 学起来后再跑 mini benchmark，评估真正关心的问题：

```text
ordinary-only 是否保持或提升
remote-only 是否稳定
ordinary + remote joint 是否优于 ordinary-only
```

如果 projection_aux 指标好，但 joint reconstruction 仍差，问题就不在成像机制是否可学习，而在 remote token 进入 VGGT 后如何影响 ordinary 表征。

### 后续可能分支

如果 Newyork 2v 主联训结果好：

```text
扩大到更多城市
尝试 NUM_VIEWS=4 但降低 batch 或使用梯度累积
跑 mini benchmark 和 export_pointcloud_ply 可视化重建
```

如果 projection_aux 在主联训中退化：

```text
降低 LAMBDA_REMOTE_PM
先 warmstart capacity_grad aux checkpoint
前若干 epoch 冻结主干只训 remote/aux，再解冻主干
尝试 USE_REMOTE_PRIVATE_POINT_HEAD=true 避免 shared point head 梯度冲突
```

如果 projection_aux 学得好但主重建变差：

```text
保护 ordinary heads 或 ordinary branch
尝试 late/gated remote fusion
降低 remote loss 对 shared trunk 的影响
对 ordinary-only / same remote / shuffled remote 做控制评估
```

## 2026-06-04 迭代：scene matching projection head 与控制评估

### P7N2 结论

实验：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7n2_newyork_2v_e6_crossattn_aux_scene_match_gather05_t007_gate001_b4_t128
```

mini controls：

```text
outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p7n2_newyork_2v_e6_crossattn_aux_scene_match_gather05_t007_gate001_b4_t128_mini_controls_t128
```

关键结果：

```text
aerial-only pointmaps_abs_rel 0.204832, z_depth_abs_rel 0.144921
same        pointmaps_abs_rel 0.184356, z_depth_abs_rel 0.111799
blank       pointmaps_abs_rel 0.190598, z_depth_abs_rel 0.124701
shuffled    pointmaps_abs_rel 0.184351, z_depth_abs_rel 0.111628
```

结论：projection_aux 能学，same/blank 均优于 aerial-only，但 same 和 shuffled 几乎完全一样。distributed negatives 的 pooled-token scene matching 没有让模型使用匹配 remote 的特异信息。

### P7O 实验

P7O 在 P7N2 上加入一个很小的 trainable scene-matching projection head。工程上需要注意：projection head 必须在 VGGT forward 内完成 descriptor finalize，且 DDP 需要 `train_params.ddp_static_graph=true`，否则会触发 checkpoint/reentrant backward 的 `mark variable ready twice`。

实验：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7o_newyork_2v_e6_crossattn_aux_scene_projhead05_t007_gate001_b4_t128
```

训练设置：

```text
NUM_GPUS=2, NUM_VIEWS=2, BATCH_SIZE=4, EPOCHS=6
remote_scene_matching_loss_weight=0.5
remote_scene_matching_temperature=0.07
remote_scene_matching_gather_distributed=true
remote_control_ranking_loss_weight=0.0
```

训练内现象：

```text
remote_scene_matching_diag_prob 从约 0.25 随机水平提升到约 0.55-0.60
remote_scene_matching_top1 后期约 0.75-0.80
final val rs_projection_rel_height_contrast_pred_gap / gt_gap ~= 0.1575 / 0.2337
final val rs_projection_offset_contrast_pred_gap / gt_gap ~= 0.2968 / 0.3080
```

这说明 projection_aux 仍能学习 remote 成像机制，scene-matching projection head 也确实能在训练内形成可分 descriptor。

mini controls：

```text
outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p7o_newyork_2v_e6_crossattn_aux_scene_projhead05_t007_gate001_b4_t128_mini_controls_t128
```

关键结果：

```text
aerial-only pointmaps_abs_rel 0.204832, z_depth_abs_rel 0.144921
same        pointmaps_abs_rel 0.183035, z_depth_abs_rel 0.111984, pose_ate_rmse 0.127186
blank       pointmaps_abs_rel 0.190395, z_depth_abs_rel 0.124443, pose_ate_rmse 0.124869
shuffled    pointmaps_abs_rel 0.183078, z_depth_abs_rel 0.111959, pose_ate_rmse 0.127370
```

结论：P7O 比 P7N2 的训练内 matching 指标强很多，但 benchmark 仍然是 same ~= shuffled。它证明“匹配 descriptor 可分”不等价于“joint reconstruction 使用了匹配 remote 的特异增益”。当前更可能的问题是 remote 通过 cross-attention 进入 ordinary 表征时，仍学成了泛化的 remote adapter，而不是 scene-conditioned evidence。

### 下一步 P7Q

不再继续增加复杂结构，先做一个直接目标：在 P7O 的 aux/matching 基础上加入实际输出层面的 shuffled ranking，让 same aerial reconstruction loss 必须低于 shuffled remote reconstruction loss。

建议短跑：

```text
REMOTE_CONTROL_RANKING_LOSS_WEIGHT=0.2
REMOTE_CONTROL_RANKING_MODES=[shuffled]
REMOTE_CONTROL_RANKING_MARGIN=0.0
REMOTE_CONTROL_RANKING_SAME_LOSS=aerial_grad
EPOCHS=4 或 6
```

判据：

```text
训练内 remote_control_ranking_shuffled_loss 是否不再长期为 0
mini benchmark 中 same 是否明显优于 shuffled
如果 same 仍约等于 shuffled，说明当前 fusion path 对 remote 内容的可控性不足，下一步应改 fusion 位置或让 aux geometry field 直接参与 fusion，而不是继续增强 scene descriptor loss。
```

## 2026-06-04 迭代：projection_aux 高值区校准与高显存训练

### 新增 one-sided high-underpred loss

前面多组实验说明，projection_aux 已经可以学习方向、slope 和一定的 height/offset 分布，但强投影/高楼区域容易被低估；直接加强 balanced/contrast 又会把 low/background 区域一起抬高。

因此新增了一个更直接的单边校准项：

```text
projection_rel_height_high_underpred_loss_weight
projection_offset_high_underpred_loss_weight
```

它只在 high bucket 的预测均值低于 GT 均值时产生梯度：

```text
loss = max(gt_high_mean - pred_high_mean, 0)
```

目标不是继续惩罚所有 high bucket 像素的 L1，而是给“高值区整体幅值不足”一个明确的校准梯度。对应日志：

```text
rs_projection_rel_height_high_underpred_pred_mean / gt_mean / loss
rs_projection_offset_high_underpred_pred_mean / gt_mean / loss
```

### 显存利用结论

在 2 张 48GB GPU 上做 batch 探测：

```text
B8:  max mem ~= 27GB / GPU，吞吐较高但显存未满
B12: max mem ~= 35GB / GPU，稳定
B16: max mem ~= 44-46GB / GPU，nvidia-smi 约 47-48GB / GPU，util 约 100%
```

当前按用户要求优先最大化显存，因此后续短跑实验使用：

```text
NUM_GPUS=2
NUM_VIEWS=2
BATCH_SIZE=16
```

如果后续目标改成单位时间吞吐最大化，B8/B12 可能更划算；但 B16 是目前最接近显存上限且稳定的设置。

### P7Y：强 high-underpred 诊断

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7y_auxwarm_p7u_height_underpred_b16_e4
```

配置要点：

```text
warmstart = p7u_auxfirst_ny_chicago_offset_strong_weak_consistency_e6_b4/checkpoint-best.pth
PROJ_REL_HEIGHT_HIGH_UNDERPRED_WEIGHT=3.0
PROJ_OFFSET_HIGH_UNDERPRED_WEIGHT=0.0
BATCH_SIZE=16, EPOCHS=4
```

验证结果：

```text
epoch1:
  rel_height high20 gt/pred ~= 0.1878 / 0.1375
  rel_height low80  gt/pred ~= 0.0157 / 0.0174
  offset high20     gt/pred ~= 0.1681 / 0.0772

epoch3:
  rel_height high20 gt/pred ~= 0.1878 / 0.2006
  rel_height low80  gt/pred ~= 0.0157 / 0.0257
  offset high20     gt/pred ~= 0.1681 / 0.1677

epoch4:
  rel_height high20 gt/pred ~= 0.1878 / 0.1994
  rel_height low80  gt/pred ~= 0.0157 / 0.0223
  offset high20     gt/pred ~= 0.1681 / 0.1789
```

解读：

- one-sided high-underpred loss 是有效的，能把长期欠估的 height high20 推到 GT 附近甚至略高。
- 但 `weight=3.0` 偏强，low80 也被整体抬高，offset 后期也有过冲迹象。
- 这轮证明了“高值欠估”可以通过明确的单边校准梯度修正，但还没有得到稳定的 high/low 分离。

存储处理：该实验只保留 `checkpoint-best.pth`，删除了重复的 `checkpoint-last.pth` 和 `checkpoint-final.pth`；同时删除了已判定失败的 P7W 权重。

### P7Z：较温和 high-underpred + offset 保持

当前运行实验：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7z_auxwarm_p7u_balanced_underpred_b16_e4
```

配置要点：

```text
warmstart = p7u best
PROJ_REL_HEIGHT_BALANCED_WEIGHT=1.0
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=1.0
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=1.2
PROJ_REL_HEIGHT_HIGH_UNDERPRED_WEIGHT=1.2

PROJ_OFFSET_BALANCED_WEIGHT=1.2
PROJ_OFFSET_CONTRAST_WEIGHT=1.0
PROJ_OFFSET_BUCKET_MEAN_WEIGHT=2.0
PROJ_OFFSET_HIGH_UNDERPRED_WEIGHT=0.8

LAMBDA_PROJ_CONSISTENCY=0.05
BATCH_SIZE=16, EPOCHS=4
```

设计目的：

- 相比 P7Y，降低 height high-underpred，避免把 low 区域一起抬高。
- 给 offset 也加一个弱 high-underpred，避免 warm-start 后 offset high bucket 在验证早期偏弱。
- 继续保持 no-fusion/protected ordinary 路线，先把 projection_aux 本身训练稳定，再考虑重新接入主重建和 remote-to-ordinary 增益。

判据：

```text
rel_height_high20_pred_mean 接近 gt_mean
rel_height_low80_pred_mean 不明显高于 gt_mean
offset_high20_pred_mean 接近 gt_mean
offset_low80_pred_mean 不明显高于 gt_mean
rs_projection_aux_loss 不升高
```

如果 P7Z 仍然 low 区域过预测，下一步不应继续加 high 权重，而应尝试更明确的空间分离监督或弱低值约束调度；如果 P7Z 能稳定 high/low，则再进入主重建联训和 mini controls。

### P7AA/P7AB：low-overpred 修复与低值区校准

P7Z 说明单纯降低 high-underpred 权重并不能解决问题：验证 aux loss 虽低，但 height low80 被严重抬高。

随后从 P7Y best warm-start 做低值区校准。过程中发现一个实现问题：

```text
_low_bucket_overpred_loss 使用 target < q0
```

offset 的低分位数经常是 `q0=0`，因此 `target < 0` 没有像素，导致 offset low-overpred 实际没有生效。已修复为：当 `q0` 接近 0 时使用 `target <= q0 + eps`，把 GT 为 0 的 offset 区域纳入低桶约束。

P7AA 是修复前的中断实验，仅用于暴露该问题，没有保留权重。

P7AB 实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ab_auxwarm_p7y_lowover_fix_b16_e3
```

配置要点：

```text
warmstart = P7Y best
PROJ_REL_HEIGHT_HIGH_UNDERPRED_WEIGHT=0.3
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.6, start_epoch=0
PROJ_OFFSET_HIGH_UNDERPRED_WEIGHT=0.5
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=0.3, start_epoch=0
BATCH_SIZE=16, EPOCHS=3
```

验证结果：

```text
P7Y epoch4 baseline:
  rel_height high20 gt/pred ~= 0.1878 / 0.1994
  rel_height low80  gt/pred ~= 0.0157 / 0.0223
  offset high20     gt/pred ~= 0.1681 / 0.1789
  offset low80      gt/pred ~= 0.0042 / 0.0198

P7AB epoch1:
  rel_height high20 gt/pred ~= 0.1878 / 0.1625
  rel_height low80  gt/pred ~= 0.0157 / 0.0207
  offset high20     gt/pred ~= 0.1681 / 0.1509
  offset low80      gt/pred ~= 0.0042 / 0.0148

P7AB epoch2:
  rel_height high20 gt/pred ~= 0.1878 / 0.1309
  rel_height low80  gt/pred ~= 0.0157 / 0.0171
  offset high20     gt/pred ~= 0.1681 / 0.1314
  offset low80      gt/pred ~= 0.0042 / 0.0140

P7AB epoch3:
  rel_height high20 gt/pred ~= 0.1878 / 0.1608
  rel_height low80  gt/pred ~= 0.0157 / 0.0190
  offset high20     gt/pred ~= 0.1681 / 0.1799
  offset low80      gt/pred ~= 0.0042 / 0.0162
```

解读：

- 修复后的 offset low-overpred 确实生效，`rs_projection_offset_low_overpred_loss` 不再恒为 0。
- low 区域相比 P7Y 有改善，尤其 offset low80 从约 `0.0198` 降到 `0.014-0.016`。
- 但从 epoch0 立即启用 low-overpred 会压低 height high，epoch2 最明显。
- 下一轮不应继续增加 low-overpred，而应延迟启用，并提高 high-underpred 保护高值区。

下一轮 P7AC：

```text
warmstart = P7Y best
low-overpred start_epoch=1.0, ramp=1.0
rel_height high_underpred 提高到 1.0，low_overpred 降到 0.3
offset high_underpred 提高到 0.8，low_overpred 降到 0.2
```



## 2026-06-04 进度更新：P7AC 与 metadata-clean topK

### 当前实验进度

- 已把 projection_aux 的高值区一侧 under-pred 校准加入 loss，并修复了 offset zero-heavy 情况下 `q0=0` 导致 low-overpred loss 失效的问题。
- P7Y 证明高值区可以被明显拉起来，但会带来低值/背景区泄漏；P7AB/P7AC 进一步验证 low-overpred 约束可以压低低值区，但 high/low 之间仍有权衡。
- P7AC 使用 B16 双卡，显存基本打满；训练已结束并只保留 `checkpoint-best.pth`，删除了重复的 `checkpoint-last.pth` 和 `checkpoint-final.pth`。

P7AC final val 关键指标：

| 指标 | pred | gt | 备注 |
| --- | ---: | ---: | --- |
| rel_height high bucket mean | 0.1972 | 0.1878 | 高值区已能追上，略偏高 |
| rel_height low bucket mean | 0.0085 | 0.0065 | 低值区仍略偏高，但可控 |
| offset high bucket mean | 0.1902 | 0.1681 | 高值区略偏高 |
| offset low/mid bucket mean | 0.0189 | 0.0042 | 仍是主要问题，低 offset 区泄漏明显 |
| offset zero-bucket pred | 0.0051 | 0.0000 | zero 区已经受约束，但还没完全压下去 |
| global dir cosine | 0.5232 | - | field-derived 方向中等 |
| global dir head cosine | 0.6362 | - | 显式 global head 更稳定 |

当前判断：projection_aux 不是完全学不动，高值建筑投影信号已经可以被拉起来；主要未解决的是 offset/height 的低值区泄漏，以及高值保护和低值抑制之间的平衡。下一步不应急着接主重建，而应先做 offset 低值区更精细的约束，例如 low80/zero 双低值约束、分阶段增加低值权重，避免把高值一起压掉。

### topK 与 metadata 过滤

已确认常规 `VigorChicagoJointRSAerial` 训练流程先从 `traindata/mapanything_metadata` 的 split scene list 构造 `self.scenes`，再按 `cities` 和 remote 标签可用性过滤。因此只用 `cities=[newyork]` / `cities=[chicago]` 时，不会采到 metadata 中已经剔除的 location。

但旧的 `scripts/analyze_projection_aux_labels.py` 默认直接扫描 `traindata/Crossview_rs/**/projection_aux.npz`，不读 metadata。旧的 `newyork_google_top128_projection_aux_scenes.npy` 中有 14 个 scene 不在当前 train metadata 中；如果某些 remote-only 或手写 topK 流程直接使用这份列表，就有采到低质量标签的风险。

已更新 `scripts/analyze_projection_aux_labels.py`，新增：

- `--metadata-dir` / `--metadata-kind` / `--split`：按 `traindata/mapanything_metadata` 过滤 scene。
- `--scene-list-path`：也可以显式用已有 scene list 过滤。
- `--output-top-scenes-npy` / `--output-top-scenes-txt`：直接导出 topK scene list。

已生成并校验 metadata-clean topK：

- `outputs/mapanything_experiments/mapanything/debug/projection_aux_label_stats/newyork_google_metadata_train_top128_projection_aux_scenes.npy`
- `outputs/mapanything_experiments/mapanything/debug/projection_aux_label_stats/newyork_google_metadata_train_top256_projection_aux_scenes.npy`

校验结果：top128/top256 对 `Crossview_rs_aerial/train/Crossview_rs_aerial_scene_list_train.npy` 的 `missing_from_metadata=0`。后续 topK 实验应使用带 `metadata_train` 的新列表，不再使用旧的 `newyork_google_top128_projection_aux_scenes.npy`。


## 2026-06-04 进度更新：P7AD4 best 与 zero-overpred 负结果

### 代码更新

新增了独立于 balanced quantiles 的 near-zero overprediction loss：

- `remote_projection_rel_height_zero_overpred_loss_weight`
- `remote_projection_rel_height_zero_overpred_quantile`
- `remote_projection_rel_height_zero_overpred_margin`
- `remote_projection_offset_zero_overpred_loss_weight`
- `remote_projection_offset_zero_overpred_quantile`
- `remote_projection_offset_zero_overpred_margin`

实现位置：`mapanything/train/losses.py`，配置入口：`configs/loss/vggt_loss_rs_joint_p7h_projection_aux_robust_branch.yaml`，训练脚本入口：`bash_scripts/train/Crossview/vggt/p7h_vggt_protected_film_projection_aux.sh`。

这个 loss 的目的不是替代 balanced q50/q80，而是单独选择 zero/near-zero 像素压低 offset/height 的低值假阳性，避免像 P7AE 那样把主分桶改成 q70/q90 后牺牲 high bucket。

已通过：

```text
python -m py_compile mapanything/train/losses.py
```

并用小张量实例化测试确认 `rs_projection_*_zero_overpred_*` 日志字段正常输出。

### 当前 best：P7AD4

目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ad4_auxwarm_p7ac_offset_lowmid_b8_e4
```

保留：`checkpoint-best.pth`。

P7AD4 是目前 projection_aux 校准阶段最稳的结果。epoch3/epoch4 附近的验证指标大致为：

| 指标 | pred | gt | 备注 |
| --- | ---: | ---: | --- |
| rel_height high20 | 0.3096-0.3156 | 0.3407 | 高值略低估，但接近 |
| rel_height low bucket | 0.0275-0.0290 | 0.0147 | 低值仍偏高 |
| offset high20 | 0.4290-0.4344 | 0.4222 | 高值区基本对齐 |
| offset low bucket | 0.0439-0.0478 | 0.0157 | low bucket 假阳性仍明显 |
| offset low80 | 0.0574-0.0626 | 0.0480 | low80 偏高但可接受 |
| global dir cosine | 0.9573-0.9580 | - | 方向场较稳定 |
| global slope | 0.1534-0.1575 | 0.1768 | slope 略低估 |

结论：P7AD4 可以作为后续“是否接回 remote pointmap/普通视角增益”的基线，但 projection_aux 本身还没有完全解决低值泄漏。

### P7AE：q70/q90 失败

目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ae_auxwarm_p7ad4_q70_lowover_b8_e3
```

处理：已停止并删除 `checkpoint-best.pth`，只保留日志。

目的：把 balanced quantiles 改为 q70/q90，试图更强压低低值区。

结果：

- offset q90 high 可以保持，但 height high 明显变难。
- low80 没有比 P7AD4 更好，height high underprediction 变严重。

结论：不要再用 q70/q90 改主分桶。它把“低值抑制”和“高值定义”绑在一起，副作用过大。

### P7AF：q25 near-zero 失败

目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7af_auxwarm_p7ad4_zeroq25_b8_e3
```

处理：首轮验证后停止，删除 `checkpoint-best.pth`，只保留日志。

配置核心：

```text
PROJ_OFFSET_ZERO_OVERPRED_WEIGHT=0.8
PROJ_OFFSET_ZERO_OVERPRED_QUANTILE=0.25
PROJ_REL_HEIGHT_ZERO_OVERPRED_WEIGHT=0.15
PROJ_REL_HEIGHT_ZERO_OVERPRED_QUANTILE=0.25
```

首轮验证关键指标：

| 指标 | pred | gt | 对比 |
| --- | ---: | ---: | --- |
| offset high20 | 0.4179 | 0.4222 | 高值区很好 |
| offset low80 | 0.0656 | 0.0480 | 比 P7AD4 差 |
| offset zero bucket | 0.0317 | 0.0000 | 假阳性仍高 |
| rel_height high20 | 0.2752 | 0.3407 | height high 低估 |
| rel_height low80 | 0.0508 | 0.0353 | 低值偏高 |

结论：q25 near-zero 区域太宽，验证上没有压低 low80/zero 假阳性，反而不如 P7AD4。

### P7AG：exact-zero 仍未超过 P7AD4

目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ag_auxwarm_p7ad4_zeroexact_b8_e2
```

处理：训练完成后删除 `checkpoint-best.pth`、`checkpoint-last.pth`、`checkpoint-final.pth`，只保留日志。

配置核心：

```text
PROJ_OFFSET_ZERO_OVERPRED_WEIGHT=3.0
PROJ_OFFSET_ZERO_OVERPRED_QUANTILE=0.0
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=0.2
PROJ_OFFSET_BUCKET_MEAN_WEIGHT=2.0
PROJ_OFFSET_HIGH_UNDERPRED_WEIGHT=1.5
```

验证结果：

| 指标 | epoch1 pred/gt | epoch2 pred/gt | 备注 |
| --- | ---: | ---: | --- |
| offset high20 | 0.4414 / 0.4222 | 0.4166 / 0.4222 | high 可保持，但波动 |
| offset low80 | 0.0545 / 0.0480 | 0.0586 / 0.0480 | 比 P7AF 好，但仍差于 P7AD4 |
| offset zero bucket | 0.0247 / 0.0000 | 0.0254 / 0.0000 | exact-zero loss 未能泛化压到足够低 |
| rel_height high20 | 0.3048 / 0.3407 | 0.2824 / 0.3407 | 第 2 epoch 变差 |
| rel_height low80 | 0.0450 / 0.0353 | 0.0476 / 0.0353 | 仍偏高 |
| global dir cosine | 0.9579 | 0.9577 | 方向稳定 |

结论：exact-zero 训练集上能降低 zero/low，但验证集仍有明显泄漏。单纯加大 zero loss 不是主要解法。

### 当前判断

1. projection_aux 的 offset high、global direction、slope 是可学习的；P7AD4 已经能稳定学到主要投影方向和高值 offset。
2. 低值泄漏仍是核心难点，尤其是 validation 上 zero/low 区域。这个问题不是简单 loss 权重不足；P7AF/P7AG 说明更强 zero 约束会改善训练集，但验证提升有限。
3. 继续盲目加大 low/zero 权重收益不高，容易在 height high 或 offset high 上引入副作用。
4. 下一步应先做诊断，而不是立刻接主重建：按 scene/location 统计 zero bucket 和 low80 指标，找出是否由少数标签质量差、mask 稀疏或城市/场景分布导致。如果低值泄漏集中在少数 scene，应回到 metadata-clean/topK 或标签质量过滤；如果普遍存在，再考虑结构上增加“background/zero offset confidence”或分段头。

### 下一步建议

- 编写/扩展 projection_aux validation analyzer：从 checkpoint 对每个 scene 输出 `offset_zero_pred_mean`、`offset_low80_pred_mean`、`offset_high20_pred_mean`、`rel_height_low80/high20`，并和标签统计合并。
- 在 P7AD4 checkpoint 上跑 per-scene 分析，确定 bad scenes 是否集中。
- 暂不启动新的大训练；下一轮实验应基于 per-scene 结论选择：
  - 如果坏样本集中：metadata/topK 过滤训练。
  - 如果坏样本普遍：尝试显式 zero/background 分类头或 two-stage offset magnitude calibration。


## 2026-06-05 进度更新：projection_aux per-scene 预测诊断

新增脚本：

```text
scripts/analyze_projection_aux_predictions.py
```

目的：不再只看 epoch-level validation 均值，而是逐 scene/provider 对比 projection_aux 预测和 GT，定位哪些样本存在低值假阳性、高值低估或方向偏差。脚本默认读取 `traindata/mapanything_metadata/Crossview_rs_aerial/{split}` 的 scene list；也可以显式传入 metadata-filtered topK scene list，因此不会采到已经从 metadata 中剔除的低质量 location。

关键实现：

- 使用 `export_pointcloud_ply.py` 的同一套 VGGT/P7 checkpoint 初始化逻辑。
- 支持 `remote_head`/`split` 两种 P7 projection_aux export preset。
- 对 offset 使用训练一致的 loss-space 统计：`offset_gt / 32` 对齐预测输出。
- 对 height 不再使用固定 scale；从 `projection_aux.npz` 的 `projected_xyz_centered` resize 到推理分辨率后，用有效点云范数 median 作为 scene-level 动态 scale，使 height 指标回到训练归一化空间。
- 输出 CSV，并按 bounded `badness_score` 输出 bad scene txt。`badness_score` 由 `offset_low80_excess`、`offset_high20_under_gt` 和 direction penalty 组成，避免 GT 接近 0 时 ratio 爆炸。

验证命令：

```text
python -m py_compile scripts/analyze_projection_aux_predictions.py
```

已通过。

P7AD4 / New York / Google / metadata-filtered top128 诊断命令：

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_projection_aux_predictions.py \
  --checkpoint-path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ad4_auxwarm_p7ac_offset_lowmid_b8_e4/checkpoint-best.pth \
  --root /root/autodl-tmp/traindata/Crossview_rs \
  --provider Google_Satellite \
  --scene-list-path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/projection_aux_label_stats/newyork_google_metadata_train_top128_projection_aux_scenes.npy \
  --output-csv /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/p7ad4_newyork_google_metadata_top128.csv \
  --bad-scenes-txt /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/p7ad4_newyork_google_metadata_top128_bad.txt \
  --preset remote_head \
  --hidden-dim 96 \
  --image-stem-dim 32 \
  --aux-num-blocks 6 \
  --offset-scale 32 \
  --rel-height-scale-mode pointmap_norm_median
```

输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/p7ad4_newyork_google_metadata_top128.csv
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/p7ad4_newyork_google_metadata_top128_bad.txt
```

### P7AD4 top128 诊断结论

样本：New York / Google / metadata-filtered top128，128/128 有效。

核心均值：

| 指标 | pred | gt | 备注 |
| --- | ---: | ---: | --- |
| rel_height low80 | 0.1214 | 0.1013 | 低值区略偏高 |
| rel_height high20 | 0.4440 | 0.4105 | high height 不再塌缩，略偏高 |
| offset low80 | 0.1633 | 0.1498 | 低值区仍有泄漏，但幅度不大 |
| offset high20 | 0.5944 | 0.6363 | 高值区系统性低估 |
| offset low80 excess | 0.0168 | - | 低值假阳性的平均超额 |
| offset high20 under | 0.0580 | - | 当前更主要的问题 |
| global dir cosine | 0.8096 | 1.0 | 方向 head 偏差明显，比 epoch-level val 低 |
| global slope | 0.1618 | 0.1779 | slope 略低估 |

结论：P7AD4 已经能学到 projection_aux 的主要幅值结构，不能再简单判定为“aux 没学会”。但它还没有达到“可以放心反哺主重建”的程度，主要短板从早期的整体塌缩，转为：

1. high20 offset 在强投影样本上仍低估，尤其 `newyork__location_121/393/338/234` 等高楼 scene。
2. global direction cosine 只有约 0.81，说明方向头或方向监督还不够可靠；这会影响 offset vector 的全局一致性。
3. low80 offset 仍有轻度 over-pred，但平均超额只有 0.0168，优先级低于 high20 under 和 direction 偏差。

下一步实验不应继续单纯提高 low/zero penalty；更合理的是围绕两点改：

- 强化 direction/field-direction 一致性：让 global_dir_head 与 offset field 的高值区域方向互相约束，或直接用 offset 高值区域估计方向作为辅助监督。
- 对 high20 offset 做更温和的一侧 under-pred 校准，同时保留 low80 bounded excess 约束，避免回到全场抬高。

如果后续主重建联训，建议先以 P7AD4 为 warm start，使用 protected/no-fusion 或 remote-head-only 的简洁结构，先验证 projection_aux 在联训中不退化，再看 ordinary-view reconstruction 是否获得增益。


## 2026-06-05 进度更新：p7ad5/p7ad6 有效训练修复

### 发现的问题

`p7ad5` 和 `p7ad5b` 表面完成训练并打印了 loss，但 checkpoint 与 `p7ad4` 完全一致，optimizer state 为空。单步梯度探针确认原因不是保存或 warm-start，而是 `projection_offset_dir_loss` 产生 NaN 梯度，GradScaler 每步跳过 optimizer step。

已加入两个默认关闭的训练诊断开关：

- `+train_params.debug_optimizer_grads_once=true`：单步打印每个 optimizer 参数组的梯度情况。
- `+train_params.debug_autograd_anomaly=true`：启用 autograd anomaly 定位 NaN 梯度来源。

### 代码修复

在 `mapanything/train/losses.py` 中加入安全方向归一化，并对 offset direction loss 做近零预测保护。实际后续训练中先关闭 `LAMBDA_PROJ_OFFSET_DIR`，因为该项不是主约束，且在当前阶段会阻断有效更新。

### p7ad6：p7ad4 warm-start，关闭 offset_dir，训练 direct global_dir

实验：`p7ad6_auxwarm_p7ad4_globaldir_no_offsetdir_top128_e3`

关键设置：

- `PRETRAINED_CKPT=p7ad4/checkpoint-best.pth`
- `LAMBDA_PROJ_OFFSET_DIR=0.0`
- `LAMBDA_PROJ_GLOBAL_DIR=0.2`
- `LAMBDA_PROJ_GLOBAL_SLOPE=0.1`
- `LAMBDA_PROJ_CONSISTENCY=0.05`
- `PROJ_GLOBAL_DIR_FROM_OFFSET=false`
- New York / Google / metadata top128

有效性检查：

- `checkpoint-best.pth` optimizer state entries = 107，step = 16。
- `checkpoint-last.pth` optimizer state entries = 107，step = 48。
- 关键 projection aux head 参数相对 p7ad4 已真实变化。

### p7ad6 结果解读

训练日志验证集：

| checkpoint/epoch | global_dir_cosine | aux_loss_avg | offset high20 pred/gt | rel_height high20 pred/gt |
|---|---:|---:|---:|---:|
| p7ad4 baseline | 0.7636(head) / 0.958(offset-derived) | 0.3504 | 0.5789 / 0.5851 | 0.3156 / 0.3407 |
| p7ad6 epoch1 best | 0.8516 | 0.3966 | 0.5502 / 0.5851 | 0.2642 / 0.3407 |
| p7ad6 epoch2 | 0.8708 | 0.3322 | 0.6086 / 0.5851 | 0.2682 / 0.3407 |
| p7ad6 epoch3 | 0.8687 | 0.4681 | 0.5241 / 0.5851 | 0.2319 / 0.3407 |

per-scene top128 诊断：

| model | global_dir_cosine | offset low80 excess | offset high20 under | offset MAE | rel_height MAE | badness |
|---|---:|---:|---:|---:|---:|---:|
| p7ad4 | 0.8096 | 0.0168 | 0.0580 | 0.0717 | 0.0552 | 0.5483 |
| p7ad6 best | 0.9309 | 0.0188 | 0.0669 | 0.0732 | 0.0485 | 0.5235 |
| p7ad6 last | 0.9633 | 0.0085 | 0.0777 | 0.0769 | 0.0514 | 0.4749 |

结论：

1. direct global_dir head 已经可以被有效训练，从 0.81 提升到 0.93-0.96。这证明此前“学不动”主要是 NaN 梯度导致的无效 step，而不是 head 完全不可学。
2. low80 offset 泄漏在 `p7ad6_last` 明显改善，但 high20 offset 欠估变严重，说明模型开始变保守，幅度被压低。
3. rel_height 高值也被压低，说明高值区域仍需更强的幅度约束。
4. 当前 `projection_consistency` 可能存在尺度不一致：`rel_height` 是 pointmap-norm 空间，而 `offset` 是 `/32` 的像素位移空间，`offset ~= rel_height * slope * dir` 可能会把 offset 幅度往小压。下一步应关闭或重标定 consistency。

### 下一步实验

从 `p7ad6_last` warm-start：

- 关闭 `LAMBDA_PROJ_CONSISTENCY`。
- 保留 direct `global_dir` 和 `global_slope` 监督。
- 增强 `offset_high_underpred`，降低 `offset_low_overpred`。
- 给 `rel_height_high_underpred` 一定权重，避免高值高度继续被压低。
- 继续禁用 `offset_dir`，直到设计出更稳定的方向损失。



## 2026-06-05 进度更新：p7ad7/p7ad8 aux 训练进入可用状态

### p7ad7：关闭 consistency，强化高值幅度

实验：`p7ad7_p7ad6_no_consistency_highamp_top128_e2`

关键设置：

- 从 `p7ad6/checkpoint-last.pth` warm-start。
- `LAMBDA_PROJ_CONSISTENCY=0.0`
- `LAMBDA_PROJ_OFFSET_DIR=0.0`
- `LAMBDA_PROJ_GLOBAL_DIR=0.2`
- `LAMBDA_PROJ_GLOBAL_SLOPE=0.1`
- `projection_offset_high_underpred_loss_weight=1.5`
- `projection_rel_height_high_underpred_loss_weight=0.8`
- New York / Google / metadata-filtered top128。

结果说明：

- 关闭 consistency 后，offset 高值区不再被系统性压低。
- direct global_dir 在 top128 逐场景诊断中达到 `0.9918`。
- 代价是 rel_height 和 offset 低值区有一定过预测，尤其 rel_height high20 也偏高。

### p7ad8：降低高值推动，提高低值过预测惩罚

实验：`p7ad8_p7ad7_rebalance_low_overpred_top128_e2`

关键设置：

- 从 `p7ad7/checkpoint-best.pth` warm-start。
- 保持 `LAMBDA_PROJ_CONSISTENCY=0.0` 和 `LAMBDA_PROJ_OFFSET_DIR=0.0`。
- `projection_offset_high_underpred_loss_weight=0.8`
- `projection_rel_height_high_underpred_loss_weight=0.3`
- `PROJ_OFFSET_LOW_OVERPRED_WEIGHT=0.5`
- `PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.5`
- 训练侧 rel-height 归一化使用 `gt_pointmap_norm`；诊断脚本等价使用 `pointmap_norm_median`。

有效性检查：

- `checkpoint-best.pth` optimizer state entries = 450，权重真实更新。
- `checkpoint-final.pth` 和 `checkpoint-last.pth` 与 best 模型参数重复，已删除以节省空间。
- 保留：`outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ad8_p7ad7_rebalance_low_overpred_top128_e2/checkpoint-best.pth`

### top128 逐场景诊断汇总

CSV 汇总输出：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top128_summary_p7ad4_p7ad8.csv
```

核心指标：

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad4 | 0.4440 / 0.4105 | 0.1214 / 0.1013 | 0.0552 | 0.5944 / 0.6363 | 0.0580 | 0.1633 / 0.1498 | 0.0168 | 0.0717 | 0.8096 | 0.5483 |
| p7ad6 best | 0.3852 / 0.4105 | 0.1069 / 0.1013 | 0.0485 | 0.5814 / 0.6363 | 0.0669 | 0.1641 / 0.1498 | 0.0188 | 0.0732 | 0.9309 | 0.5235 |
| p7ad6 last | 0.3605 / 0.4105 | 0.0876 / 0.1013 | 0.0514 | 0.5683 / 0.6363 | 0.0777 | 0.1445 / 0.1498 | 0.0085 | 0.0769 | 0.9633 | 0.4749 |
| p7ad7 best | 0.5134 / 0.4105 | 0.1237 / 0.1013 | 0.0654 | 0.6438 / 0.6363 | 0.0315 | 0.1736 / 0.1498 | 0.0264 | 0.0794 | 0.9918 | 0.4214 |
| p7ad8 best | 0.3586 / 0.4105 | 0.0872 / 0.1013 | 0.0462 | 0.6272 / 0.6363 | 0.0297 | 0.1696 / 0.1498 | 0.0217 | 0.0749 | 0.9962 | 0.3656 |

### 当前结论

1. projection_aux 已经证明可学习，且不是简单记均值：global direction、offset 高低值结构、rel-height 分布都能被优化到稳定区域。
2. 之前“aux 学不动”的核心原因之一是 `offset_dir` 的 NaN 梯度导致 optimizer step 被跳过；关闭该项后训练正常。
3. `projection_consistency` 当前不适合直接使用，因为 rel_height 与 offset 的损失空间尺度不一致，会压低 offset/height 幅度。
4. p7ad8 是目前最稳的 aux checkpoint：badness 最低，global_dir 最好，offset high20 欠估最小；但 rel_height high20 仍略低估，offset low80 仍有轻度泄漏。

### 下一步目标

- 将 p7ad8 作为 aux warm-start，先扩大到 New York metadata top256 或更多高质量 scene，确认不是 top128 过拟合。
- 若 top256 仍稳定，再把 projection_aux 作为 remote 多任务监督接回主训练，先保护 ordinary head，不急着打开强 remote-to-aerial fusion。
- 主训练评估优先看：projection_aux 是否退化、remote pointmap 是否稳定、ordinary-only 与 ordinary+remote 的 rs guided mini benchmark 是否出现正向差异。


## 2026-06-05 进度更新：top256 扩展与 split-head 试验

### 新增代码

- `VGGTWrapper` 增加 `remote_projection_aux_split_pixel_heads`，默认关闭。
- 打开后，projection_aux pixel trunk 共享，但 final output 拆成：
  - `remote_projection_aux_rel_height_head`: 输出 rel_height。
  - `remote_projection_aux_offset_head`: 输出 offset_xy。
- 增加旧 checkpoint 迁移逻辑：旧 `remote_projection_aux_pixel_head` 最后 3 通道卷积会自动拆到 height/offset 两个新 head，避免 split 实验从随机输出层开始。
- 迁移逻辑已接入：
  - `VGGTWrapper` 自身 custom checkpoint 加载。
  - `mapanything/train/training.py` warm-start 加载。
  - `mapanything/utils/hf_utils/hf_helpers.py` 本地 export/diagnostic 加载。
- `scripts/export_pointcloud_ply.py` 和 `scripts/analyze_projection_aux_predictions.py` 增加 split-head 配置开关。
- `configs/train_params/vggt_p7_remote_head_projection_aux.yaml` 增加 split heads 参数组，确保两个新 head 有 1e-4 学习率。

### top256 对照结果

统一评估集：New York / Google / metadata-filtered top256。
汇总 CSV：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top256_summary_p7ad8_p7ad11.csv
```

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad8 top256 | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7ad9 top256 | 0.2279 / 0.2967 | 0.0591 / 0.0717 | 0.0379 | 0.4554 / 0.4732 | 0.0293 | 0.1355 / 0.1070 | 0.0294 | 0.0636 | 0.9996 | 0.4406 |
| p7ad10 top256 | 0.2585 / 0.2967 | 0.0710 / 0.0717 | 0.0343 | 0.4398 / 0.4732 | 0.0426 | 0.1353 / 0.1070 | 0.0296 | 0.0629 | 0.9997 | 0.5096 |
| p7ad11 top256 | 0.2531 / 0.2967 | 0.0639 / 0.0717 | 0.0364 | 0.4753 / 0.4732 | 0.0212 | 0.1407 / 0.1070 | 0.0339 | 0.0662 | 0.9996 | 0.4454 |

### 结论

1. p7ad8 仍是当前最佳 aux checkpoint。top256 上 badness 仍最低，offset low80 泄漏也最小。
2. p7ad9 说明：直接用 p7ad8 配方扩大到 top256 会压低 rel_height high20，并增加 offset low80 泄漏。
3. p7ad10 说明：强推 height 的配方会牺牲 offset 高值，badness 明显变差。
4. p7ad11 说明：split pixel heads 能保护 offset high20，但仍会带来 offset low80 泄漏，整体不如 p7ad8。
5. 当前主要瓶颈不是“最后 3 通道输出互相抢容量”本身，而是 height 高值与 offset 低值泄漏之间的损失平衡/标签分布问题。

### 存储清理

- 已删除 p7ad9/p7ad10/p7ad11 的 checkpoint，保留日志与诊断 CSV。
- 继续保留当前有效基线：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7ad8_p7ad7_rebalance_low_overpred_top128_e2/checkpoint-best.pth
```

### 下一步

- 不继续从 p7ad9/p7ad10/p7ad11 warm-start。
- 以 p7ad8 为有效 aux 基线。
- 下一轮若继续优化 aux，应优先尝试“只训练 height head/height 相关参数、冻结 offset 相关输出”的 split-head 小步实验，或者直接在 p7ad8 上进入主训练集成，同时把 aux 指标作为 regularizer/监控项，而不是继续在 aux-only 上追求单一 badness 最优。

## 2026-06-05 进度更新：p7ad12 height-only split-head 对照

### 实验目的

p7ad12 从 p7ad8 warm-start，打开 `remote_projection_aux_split_pixel_heads=true`，冻结 offset head / pixel trunk / image stem / global head / remote point head，仅训练 `remote_projection_aux_rel_height_head`。目标是验证：是否能只校准 rel_height 高值区域，同时不破坏 offset。

为了排除结构迁移误差，额外补跑了 `p7ad8_split_migrated`：直接将 p7ad8 的旧 3-channel pixel head 自动迁移成 split heads，不做训练，再用同一 top256 评估。

### top256 对照结果

统一评估集：New York / Google / metadata-filtered top256。
汇总 CSV：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top256_summary_p7ad8split_p7ad12.csv
```

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad8 orig | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7ad8 split migrated | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7ad11 split trained | 0.2531 / 0.2967 | 0.0639 / 0.0717 | 0.0364 | 0.4753 / 0.4732 | 0.0212 | 0.1407 / 0.1070 | 0.0339 | 0.0662 | 0.9996 | 0.4454 |
| p7ad12 height only | 0.2364 / 0.2967 | 0.0569 / 0.0717 | 0.0386 | 0.4197 / 0.4732 | 0.0595 | 0.1205 / 0.1070 | 0.0174 | 0.0635 | 0.9891 | 0.4716 |

### 结论

1. split-head 迁移是无损的：`p7ad8_orig` 与 `p7ad8_split_migrated` 指标完全一致，说明旧 checkpoint 拆分到 height/offset heads 的代码没有引入误差。
2. p7ad12 不是有效改进：rel_height high20 从 `0.2658/0.2967` 退到 `0.2364/0.2967`，badness 从 `0.3637` 退到 `0.4716`。
3. 只训练 height head 没有解决高值欠估，反而让 offset high20 欠估更严重。这说明当前 shared feature / 主干表征仍然参与并影响 offset，不能只靠最后一层 height head 做可靠校准。
4. 当前最好基线仍是 p7ad8；后续不保留 p7ad12 checkpoint。

### 存储清理

- 已删除 p7ad12 的 `checkpoint-best.pth`、`checkpoint-last.pth`、`checkpoint-final.pth`，释放约 20G。
- 保留 p7ad12 日志与诊断 CSV，便于复盘。

### 下一步判断

aux-only 路线已经给出一个稳定可学习基线，但继续小幅调最后输出层收益很低。下一步更值得做的是：以 p7ad8 作为 projection_aux regularizer/warm-start，把它接回 remote reconstruction 主训练；训练中持续监控 projection_aux 是否退化、remote pointmap 是否稳定，以及 ordinary-only / ordinary+remote benchmark 是否出现正向差异。



## 2026-06-05 进度更新：p7main1/p7main2 主训练接入对照

### 实验目的

将 p7ad8 作为 projection_aux warm-start 接回 p5b/shared-norm 主训练，验证 remote pointmap 主损失与 projection_aux 显式投影机制能否共存。

- p7main1：remote pointmap 权重较强，主干按 p5b 方式训练。
- p7main2：降低 remote pointmap 权重，近似冻结主干，只让 remote point head 与 aux heads 小步更新。

统一离线评估集：New York / Google / metadata-filtered top256。
汇总 CSV：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top256_summary_p7ad8_p7main1_p7main2.csv
```

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad8 aux only | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7main1 strong PM | 0.1743 / 0.2967 | 0.0309 / 0.0717 | 0.0679 | 0.2850 / 0.4732 | 0.1882 | 0.0539 / 0.1070 | 0.0017 | 0.1067 | 0.9892 | 0.9578 |
| p7main2 low PM/frozen | 0.2178 / 0.2967 | 0.0531 / 0.0717 | 0.0425 | 0.3938 / 0.4732 | 0.0814 | 0.0975 / 0.1070 | 0.0034 | 0.0621 | 0.9826 | 0.4408 |

### 结论

1. p7main1 证明：强 remote pointmap loss + 主干训练会明显破坏 p7ad8 学到的显式投影机制，表现为 height/offset 高值区域被压低。rs pointmap 收敛不能说明模型保留了 remote 正射投影机制。
2. p7main2 明显优于 p7main1：降低 PM 权重并冻结大部分主干后，height/offset 不再严重塌缩，offset MAE 甚至略优于 p7ad8。
3. p7main2 仍不如 p7ad8：rel_height high20 从 0.2658 退到 0.2178，offset high20 从 0.4651 退到 0.3938，说明 aux heads 在主训练中继续更新时会适应新的 pointmap 分布，而不是稳稳充当投影机制锚点。
4. 下一步应把 p7ad8 的 aux heads 冻结，用 projection_aux loss 反向约束 remote point head，而不是继续让 aux heads 自己漂移。

### 存储清理

- p7main1 checkpoint 已删除，保留日志与诊断 CSV。
- p7main2 仅保留 `checkpoint-best.pth`；已删除 `checkpoint-last.pth` 和 `checkpoint-final.pth`，释放约 12G。

### 下一步实验

p7main3：从 p7ad8 warm-start，冻结主干、patch embed、projection_aux pixel/image/global heads，只训练 remote point head 的低学习率更新；projection_aux 不 detach pointmap，使 frozen aux heads 作为几何 regularizer 约束 remote pointmap。


## 2026-06-05 进度更新：p7main3c frozen-aux 对照

### 实验目的

p7main3c 从 p7ad8 warm-start，冻结 projection_aux pixel/image/global heads，仅训练 remote point head，并保留 `patch_embed=1e-7` 作为 checkpointed point head 的梯度通路。目标是验证：能否把 p7ad8 的 aux heads 固定为投影机制 regularizer，从而约束 remote pointmap 学习。

修正过程：完全冻结 VGGT 特征流会让 checkpointed remote point head 的输入不带 grad，导致 `loss.backward()` 报 `element 0 of tensors does not require grad`。因此最终有效配置保留了极小 patch_embed LR。

### top256 对照结果

统一评估集：New York / Google / metadata-filtered top256。
汇总 CSV：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top256_summary_p7ad8_p7main1_p7main2_p7main3c.csv
```

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad8 aux only | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7main2 low PM/frozen | 0.2178 / 0.2967 | 0.0531 / 0.0717 | 0.0425 | 0.3938 / 0.4732 | 0.0814 | 0.0975 / 0.1070 | 0.0034 | 0.0621 | 0.9826 | 0.4408 |
| p7main3c frozen aux | 0.1901 / 0.2967 | 0.0391 / 0.0717 | 0.0550 | 0.3357 / 0.4732 | 0.1378 | 0.0947 / 0.1070 | 0.0064 | 0.0827 | 0.9857 | 0.7531 |

### 结论

1. frozen aux 不是更好的集成方式：p7main3c 比 p7main2 明显退化，height/offset high20 都被进一步压低。
2. 这说明 p7ad8 aux heads 不能简单当作固定 teacher 去约束 remote point head；pointmap 一旦被更新，固定 aux heads 对新 pointmap 分布的约束会变成过强/偏置的优化目标。
3. 当前主训练接入的最好方向仍是 p7main2：低 remote PM、主干近似冻结、aux heads 允许小步适配。
4. 下一步不再尝试 frozen aux；改为更保守的 p7main2 变体：降低 remote PM、remote point head LR、aux head LR，目标是减少 p7ad8 投影机制退化。

### 存储清理

- 已删除 p7main3c 的 `checkpoint-best.pth`、`checkpoint-last.pth`、`checkpoint-final.pth`。
- 保留 p7main3c 日志与诊断 CSV，作为 frozen-aux 负结果记录。

## 2026-06-05 进度更新：p7main4 protected-lowpm 对照

### 实验目的

p7main4 从 p7ad8 warm-start，沿用 p7main2 的“低 remote PM + 主干近似冻结 + aux heads 小步适配”方向，但进一步保守：

- `LAMBDA_REMOTE_PM=0.1`，低于 p7main2。
- `remote_point_head lr=1e-6`，低于 p7main2。
- `remote_projection_aux_* lr=1e-5`，低于 p7main2。
- `patch_embed lr=1e-7`，仅保留极小梯度通路。
- New York / Google / metadata-filtered top256 训练 2 epoch。

目标是验证：更弱的主重建更新是否能减少 p7ad8 投影机制退化，同时仍让 remote pointmap 参与主训练。

### top256 对照结果

统一评估集：New York / Google / metadata-filtered top256。
汇总 CSV：

```text
outputs/mapanything_experiments/mapanything/debug/projection_aux_pred_diagnostics/projection_aux_top256_summary_p7ad8_p7main1_p7main2_p7main3c_p7main4.csv
```

| model | rel_height high20 pred/gt | rel_height low80 pred/gt | rel_height MAE | offset high20 pred/gt | offset high20 under | offset low80 pred/gt | offset low80 excess | offset MAE | global_dir_cosine | badness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p7ad8 aux only | 0.2658 / 0.2967 | 0.0614 / 0.0717 | 0.0357 | 0.4651 / 0.4732 | 0.0276 | 0.1286 / 0.1070 | 0.0225 | 0.0636 | 0.9898 | 0.3637 |
| p7main1 strong PM | 0.1743 / 0.2967 | 0.0309 / 0.0717 | 0.0679 | 0.2850 / 0.4732 | 0.1882 | 0.0539 / 0.1070 | 0.0017 | 0.1067 | 0.9892 | 0.9578 |
| p7main2 low PM/frozen | 0.2178 / 0.2967 | 0.0531 / 0.0717 | 0.0425 | 0.3938 / 0.4732 | 0.0814 | 0.0975 / 0.1070 | 0.0034 | 0.0621 | 0.9826 | 0.4408 |
| p7main3c frozen aux | 0.1901 / 0.2967 | 0.0391 / 0.0717 | 0.0550 | 0.3357 / 0.4732 | 0.1378 | 0.0947 / 0.1070 | 0.0064 | 0.0827 | 0.9857 | 0.7531 |
| p7main4 protected low PM | 0.2094 / 0.2967 | 0.0545 / 0.0717 | 0.0421 | 0.3777 / 0.4732 | 0.0969 | 0.0929 / 0.1070 | 0.0018 | 0.0575 | 0.9857 | 0.5027 |

### 结论

1. p7main4 没有超过 p7main2。虽然 `offset_mae` 从 `0.0621` 降到 `0.0575`，但 high20 offset 欠估从 `0.0814` 增到 `0.0969`，综合 `badness` 从 `0.4408` 退到 `0.5027`。
2. 更低 PM 和更低 LR 能防止严重崩溃，但不能解决核心问题：主训练接入后，height/offset 高值区域仍会被压低。
3. p7main2 仍是当前主训练接入的最好 checkpoint；p7ad8 仍是 projection_aux 学习能力的最好 aux-only 证据。
4. 下一步不宜继续只靠“更小 LR / 更低 PM”微调。更有价值的方向是改变 aux 与 pointmap 的耦合方式，例如让 aux head 更少依赖已经被主损失改变的 predicted pointmap，或者做 staged/distillation：先锁定 projection_aux 学习，再单独验证 remote pointmap 是否能在不破坏高值投影机制的情况下适配。

### 存储清理

- p7main4 的 `checkpoint-best.pth`、`checkpoint-last.pth`、`checkpoint-final.pth` 已删除。
- 保留 p7main4 日志、训练配置记录、诊断 CSV 与坏例列表。

