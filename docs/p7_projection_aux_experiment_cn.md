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

