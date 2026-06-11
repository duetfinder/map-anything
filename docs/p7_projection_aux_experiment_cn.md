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

## 2026-06-04 P5B Newyork top128 主联训结果

这一轮使用单卡 80G，在 Newyork / Google_Satellite / top128 high-signal 场景上跑完整 6 epoch。固定设置：

```text
NUM_GPUS=1
NUM_VIEWS=4
BATCH_SIZE=8
REMOTE_TRAIN_CROP_MODE=random_scale_offset
REMOTE_VAL_CROP_MODE=none
REMOTE_TEST_CROP_MODE=none
REMOTE_CROP_SCALE_MIN=0.6
REMOTE_CROP_SCALE_MAX=1.0

REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
REMOTE_PROJECTION_AUX_USE_RGB=true
REMOTE_PROJECTION_AUX_USE_COORD=true

LAMBDA_PROJ_REL_HEIGHT=0.5
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
```

### 关键对照

#### PM=1 joint

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top128_p5b_joint_pm1_aux_capacity_e6
```

epoch6 固定评估：

```text
loss=2.2981
aerial_loss=2.3371
remote_loss=0.5648
rs_pointmap_loss=0.1989
rs_projection_aux_loss=0.3658

height high gt/pred = 0.4483 / 0.4624
height low  gt/pred = 0.0601 / 0.0635
height contrast gt/pred = 0.4309 / 0.4355

offset high gt/pred = 0.6409 / 0.6384
offset low  gt/pred = 0.0900 / 0.0987
offset contrast gt/pred = 0.6473 / 0.6096
global slope gt/pred = 0.1776 / 0.1804
```

解读：这是主联训中最明确的“显式投影机制可学习”证据。height、offset、contrast、global slope 都能贴近 GT，不是常数场或随机拟合。但 remote pointmap loss 较高，说明 PM=1 对主重建约束偏弱。

#### PM=4 joint baseline

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top128_p5b_joint_pm4_aux_capacity_e6
```

epoch6 固定评估：

```text
loss=3.0488
aerial_loss=2.2051
remote_loss=0.9731
rs_pointmap_loss=0.0926
rs_projection_aux_loss=0.6028

height high gt/pred = 0.4483 / 0.4987
height low  gt/pred = 0.0601 / 0.0513
height contrast gt/pred = 0.4309 / 0.4657

offset high gt/pred = 0.6409 / 0.5751
offset low  gt/pred = 0.0900 / 0.0832
offset contrast gt/pred = 0.6473 / 0.5378
global slope gt/pred = 0.1776 / 0.1801
```

解读：PM=4 明显改善 remote pointmap，且 projection_aux 仍能学到结构；但 offset high 和 contrast 有欠拟合。当前它是主重建与机制监督之间更均衡的默认配置。

#### PM=1 warmstart 后切 PM=4

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top128_p5b_stage_pm1_to_pm4_aux_capacity_e4
```

epoch4 固定评估：

```text
loss=2.9898
aerial_loss=2.3059
remote_loss=0.9184
rs_pointmap_loss=0.0910
rs_projection_aux_loss=0.5545

height high gt/pred = 0.4483 / 0.4598
height low  gt/pred = 0.0601 / 0.0745
height contrast gt/pred = 0.4309 / 0.4084

offset high gt/pred = 0.6409 / 0.6642
offset low  gt/pred = 0.0900 / 0.1405
offset contrast gt/pred = 0.6473 / 0.5909
global slope gt/pred = 0.1776 / 0.1770
```

解读：warmstart 能保留 high 区域幅值，但 low 区域过预测更明显；它不比 PM=4 scratch 更干净。

#### PM=4 + offset weight 2.0

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top128_p5b_joint_pm4_aux_offset2_e6
```

epoch6 固定评估：

```text
loss=3.5359
aerial_loss=2.5315
remote_loss=1.1350
rs_pointmap_loss=0.0631
rs_projection_aux_loss=0.8827

height high gt/pred = 0.4483 / 0.4006
height low  gt/pred = 0.0601 / 0.0877
height contrast gt/pred = 0.4309 / 0.3315

offset high gt/pred = 0.6409 / 0.5370
offset low  gt/pred = 0.0900 / 0.1418
offset contrast gt/pred = 0.6473 / 0.4356
global slope gt/pred = 0.1776 / 0.1223
```

解读：提高 offset 权重没有解决显式投影校准，反而带来 low overpred、contrast 下降和 slope 欠拟合；它只让 pointmap loss 更低，不应作为下一步默认。

#### PM=2 joint

输出目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top128_p5b_joint_pm2_aux_capacity_e6
```

epoch6 固定评估：

```text
loss=3.2723
aerial_loss=2.6405
remote_loss=0.9760
rs_pointmap_loss=0.0895
rs_projection_aux_loss=0.7971

height high gt/pred = 0.4483 / 0.4850
height low  gt/pred = 0.0601 / 0.0922
height contrast gt/pred = 0.4309 / 0.4154

offset high gt/pred = 0.6409 / 0.6760
offset low  gt/pred = 0.0900 / 0.1559
offset contrast gt/pred = 0.6473 / 0.5724
global slope gt/pred = 0.1776 / 0.1216
```

解读：PM=2 没有成为稳定折中。它能恢复 high 区域幅值，但 low 区域过预测，global slope 明显偏低，pointmap 也没有优于 PM=4 baseline。

### 当前判断

1. aux 分支已经证明 remote 显式投影机制可学习：PM=1 主联训中 height/offset 的 high-low 结构、contrast 和 slope 都能接近 GT。
2. rs pointmap loss 确实包含 remote 高度几何监督，且 PM=4 下能收敛到更低 pointmap loss；但这不保证显式 height/offset head 会自动得到良好校准。
3. 当前主要矛盾是主 pointmap 监督和显式 projection_aux 监督之间存在优化取舍。PM 权重从 1 到 4 会改变偏好，但简单取 PM=2 或提高 offset 权重都没有消除冲突。
4. 后续不应继续做大量标量权重微调。更有价值的验证是扩大 high-signal 样本覆盖，检查 PM=4 baseline 结论是否能从 top128 扩展到 top256 或 city-level；如果扩展后 aux 退化，才需要考虑更明确的阶段训练或 head/梯度隔离。

## 2026-06-04 Newyork top256 扩展验证

为了避免 top128 hard-overfit，只扩大 high-signal 覆盖，不改模型结构。生成新的 top256 场景列表：

```text
/root/autodl-tmp/traindata/mapanything_metadata/Crossview/train/newyork_google_top256_projection_aux_scenes.npy
```

注意：旧 top128 列表并不完全等价于当前脚本按 Google score 直接排序出的前 128。为保持和既有实验可比，top256 列表采用“保留旧 top128 顺序，再按当前 score 追加 128 个未包含场景”的方式生成。

2026-06-04 复核：这个 top256 列表生成时扫描的是 raw `traindata/Crossview_rs` 下的 `projection_aux.npz`，不是先用 `traindata/mapanything_metadata/Crossview/train/Crossview_scene_list_train.npy` 过滤后的 New York train 集合。因此它包含 28 个不在 metadata train 中的 scene：

```text
top128: 128/128 in metadata NewYork train
top256(raw): 228/256 in metadata NewYork train, missing=28
```

所以本节结果只能作为 raw-label 扩容压力测试，不能作为 clean top256 结论。正式 clean top256 应使用：

```text
/root/autodl-tmp/traindata/mapanything_metadata/Crossview/train/newyork_google_metadata_top256_projection_aux_scenes.npy
```

该列表保留原 top128 前缀，再从 metadata New York train 中按 `score_height_offset` 追加，校验结果：

```text
selected=256
unique=256
in metadata NewYork train=256/256
prefix_top128=True
```

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_top256_p5b_joint_pm4_aux_capacity_e6
```

配置和 top128 PM=4 baseline 保持一致：

```text
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.5
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
```

### 固定评估结果

epoch2：

```text
loss=4.8538
aerial_loss=4.0636
remote_loss=1.4110
rs_pointmap_loss=0.0791
rs_projection_aux_loss=1.0948

height high gt/pred = 0.3356 / 0.3595
height low  gt/pred = 0.0618 / 0.0970
height contrast gt/pred = 0.3108 / 0.2891

offset high gt/pred = 0.5526 / 0.5593
offset low  gt/pred = 0.1004 / 0.1783
offset contrast gt/pred = 0.5132 / 0.4181
global slope gt/pred = 0.1768 / 0.1181
```

epoch4：

```text
loss=4.1583
aerial_loss=3.7525
remote_loss=1.1410
rs_pointmap_loss=0.0686
rs_projection_aux_loss=0.8667

height high gt/pred = 0.3356 / 0.3409
height low  gt/pred = 0.0618 / 0.0988
height contrast gt/pred = 0.3108 / 0.2737

offset high gt/pred = 0.5526 / 0.3829
offset low  gt/pred = 0.1004 / 0.1295
offset contrast gt/pred = 0.5132 / 0.2810
global slope gt/pred = 0.1768 / 0.1400
```

epoch6：

```text
loss=3.3849
aerial_loss=2.4498
remote_loss=1.0800
rs_pointmap_loss=0.0621
rs_projection_aux_loss=0.8315

height high gt/pred = 0.3356 / 0.3201
height low  gt/pred = 0.0618 / 0.0906
height contrast gt/pred = 0.3108 / 0.2581

offset high gt/pred = 0.5526 / 0.3789
offset low  gt/pred = 0.1004 / 0.1233
offset contrast gt/pred = 0.5132 / 0.2823
global slope gt/pred = 0.1768 / 0.1451
```

### raw top256 解读

1. PM=4 在 top256 上仍能稳定收敛 remote pointmap，epoch6 `rs_pointmap_loss=0.0621`，比 top128 PM=4 的 `0.0926` 更低，说明 rs pointmap 主监督本身没有问题。
2. projection_aux 没有塌缩：height high/low/contrast 结构能学出来，epoch6 high `0.3356/0.3201`，contrast `0.3108/0.2581`。
3. offset 在 top256 上明显更保守。low-overpred ramp 后 low 从 epoch2 的 `0.1783` 降到 epoch6 的 `0.1233`，但 high 也从 `0.5593` 降到 `0.3789`，contrast 从 `0.4181` 降到 `0.2823`。
4. 这说明当前 loss 不是单纯“学不会 offset”，而是在 larger high-signal set 上存在 high 幅值恢复和 low 区域抑制之间的校准冲突。
5. 由于该 top256 混入 metadata 已剔除 location，不能用来证明 clean top256 上的最终趋势；只能说明 raw 扩容下仍没有出现显式 aux 塌缩。

## 2026-06-04 Newyork metadata-clean top256 PM=4 复验

使用 metadata-filtered top256 重新训练，确认 top256 不是从 raw `Crossview_rs` 任意采样：

```text
scene_list_path=/root/autodl-tmp/traindata/mapanything_metadata/Crossview/train/newyork_google_metadata_top256_projection_aux_scenes.npy
selected=256
unique=256
in metadata NewYork train=256/256
prefix_top128=True
```

训练目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_metadata_top256_p5b_joint_pm4_aux_capacity_e6_b8
```

配置仍保持 PM=4 + capacity_grad，不改模型结构：

```text
NUM_GPUS=2
NUM_VIEWS=4
BATCH_SIZE=8
EPOCHS=6
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.5
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
```

固定评估结果：

```text
epoch2:
loss=4.6655
aerial_loss=4.2927
remote_loss=1.2596
rs_pointmap_loss=0.0612
rs_projection_aux_loss=1.0150
height high gt/pred = 0.3208 / 0.0802
height low  gt/pred = 0.0394 / 0.0666
height contrast gt/pred = 0.3053 / 0.0154
offset high gt/pred = 0.4246 / 0.1524
offset low  gt/pred = 0.0584 / 0.1320
offset contrast gt/pred = 0.4932 / 0.0263
global slope gt/pred = 0.1744 / 0.1104
global dir cosine avg = 0.9495

epoch4:
loss=3.4966
aerial_loss=3.3353
remote_loss=0.9145
rs_pointmap_loss=0.0623
rs_projection_aux_loss=0.6655
height high gt/pred = 0.3208 / 0.2498
height low  gt/pred = 0.0394 / 0.0648
height contrast gt/pred = 0.3053 / 0.1919
offset high gt/pred = 0.4246 / 0.3684
offset low  gt/pred = 0.0584 / 0.1027
offset contrast gt/pred = 0.4932 / 0.3341
global slope gt/pred = 0.1744 / 0.1208
global dir cosine avg = 0.9499

epoch6:
loss=3.2111
aerial_loss=2.7744
remote_loss=0.9120
rs_pointmap_loss=0.0553
rs_projection_aux_loss=0.6909
height high gt/pred = 0.3208 / 0.3599
height low  gt/pred = 0.0394 / 0.0843
height contrast gt/pred = 0.3053 / 0.2856
offset high gt/pred = 0.4246 / 0.4400
offset low  gt/pred = 0.0584 / 0.1147
offset contrast gt/pred = 0.4932 / 0.4112
global slope gt/pred = 0.1744 / 0.1232
global dir cosine avg = 0.9494
```

### metadata-clean top256 解读

1. clean top256 明显改变了 raw top256 的判断：aux 分支可以学到显式 remote 投影机制。到 epoch6，height high/contrast 接近 GT，offset high 也已经从 epoch2 的 `0.1524` 追到 `0.4400`，并不再表现为 high 幅值学不上去。
2. PM=4 下 remote pointmap 也稳定收敛，epoch6 `rs_pointmap_loss=0.0553`，优于 raw top256 的 `0.0621` 和 top128 PM=4 的 `0.0926`。
3. 剩余问题集中在低区过预测：height low `0.0394/0.0843`，offset low `0.0584/0.1147`。因此当前不是“projection_aux 学不会”，而是 clean 数据上 high/contrast 已可恢复，但 low/background 校准仍偏高。
4. epoch4 的 aux loss 更低、low 区更稳；epoch6 的 high/contrast 更强但 low 偏高。这说明后续最小改动应先调低区 overpred 惩罚，而不是继续增加结构复杂度。

## 2026-06-04 metadata-clean top256 low-overpred 小改动

只改 low-overpred 权重，验证是否能降低低区偏高，同时保留 PM=4 clean top256 已经学出的 high/contrast：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_newyork_metadata_top256_p5b_joint_pm4_aux_lowover15_e6_b8

PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.75  # baseline 0.5
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=1.5       # baseline 1.0
```

固定评估结果：

```text
epoch2:
loss=4.6909
aerial_loss=4.4036
remote_loss=1.2446
rs_pointmap_loss=0.0610
rs_projection_aux_loss=1.0006
height high gt/pred = 0.3208 / 0.0845
height low  gt/pred = 0.0394 / 0.0678
height contrast gt/pred = 0.3053 / 0.0189
offset high gt/pred = 0.4246 / 0.1663
offset low  gt/pred = 0.0584 / 0.1398
offset contrast gt/pred = 0.4932 / 0.0343
low-overpred effective weight height/offset = 0.0 / 0.0

epoch4:
loss=3.4935
aerial_loss=3.3106
remote_loss=0.9191
rs_pointmap_loss=0.0594
rs_projection_aux_loss=0.6813
height high gt/pred = 0.3208 / 0.2279
height low  gt/pred = 0.0394 / 0.0567
height contrast gt/pred = 0.3053 / 0.1778
offset high gt/pred = 0.4246 / 0.3302
offset low  gt/pred = 0.0584 / 0.0858
offset contrast gt/pred = 0.4932 / 0.3060
low-overpred effective weight height/offset = 0.75 / 1.5

epoch6:
loss=3.1891
aerial_loss=2.7569
remote_loss=0.9053
rs_pointmap_loss=0.0565
rs_projection_aux_loss=0.6792
height high gt/pred = 0.3208 / 0.3382
height low  gt/pred = 0.0394 / 0.0778
height contrast gt/pred = 0.3053 / 0.2700
offset high gt/pred = 0.4246 / 0.4170
offset low  gt/pred = 0.0584 / 0.1018
offset contrast gt/pred = 0.4932 / 0.3997
global slope gt/pred = 0.1744 / 0.1228
global dir cosine avg = 0.9494
```

与 metadata-clean baseline 的 epoch6 对比：

```text
loss:     3.2111 -> 3.1891
pointmap: 0.0553 -> 0.0565
aux:      0.6909 -> 0.6792

height high pred:    0.3599 -> 0.3382, gt 0.3208
height low pred:     0.0843 -> 0.0778, gt 0.0394
height contrast pred:0.2856 -> 0.2700, gt 0.3053

offset high pred:    0.4400 -> 0.4170, gt 0.4246
offset low pred:     0.1147 -> 0.1018, gt 0.0584
offset contrast pred:0.4112 -> 0.3997, gt 0.4932
```

解读：

1. 低区偏高确实可以通过更强 low-overpred 惩罚压下来，且没有破坏 PM=4 pointmap 收敛。
2. high 并没有塌缩：height high 更接近 GT，offset high 也更接近 GT；说明 clean top256 的主要问题不是 high 学不上去。
3. 代价是 contrast 略低，尤其 offset contrast 从 `0.4112` 到 `0.3997`。因此不建议继续大幅加大 low-overpred 权重；当前 `0.75/1.5` 更像是一个温和、可用的校准点。

### 当前实验结论

目前不建议继续在 `LAMBDA_REMOTE_PM`、`LAMBDA_PROJ_OFFSET` 这类全局标量上做密集搜索。已经看到：

```text
PM=1: 显式投影最好，但 pointmap 弱。
PM=4: pointmap 最好，显式投影可学但 offset 保守。
PM=2: 不是稳定折中。
offset weight 2.0: 没有改善显式投影校准。
raw top256 PM=4: 结论能扩展，但混入 28 个 metadata 已剔除 scene，不能作为 clean 结论。
metadata-clean top256 PM=4: pointmap 和显式 high/contrast 都可学习，主要残留 low/background 过预测。
metadata-clean top256 low-overpred 0.75/1.5: 低区校准改善，pointmap 基本保持，contrast 略降。
```

当前已经可以较可靠地回答核心问题：aux 分支证明了 remote 显式投影机制可学习；rs pointmap 也能收敛；剩余优化点是 low/background calibration，而不是 projection mechanism 本身学不出来。

下一步若继续，应优先做 clean city-level 或更多 metadata-filtered scene 的扩展验证，而不是继续堆结构或做大规模标量搜索。若只在 top256 内微调，`PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.75`、`PROJ_OFFSET_LOW_OVERPRED_WEIGHT=1.5` 是当前更稳的默认。

### 下一步目标

#### 目标 1：扩展 clean data 覆盖

优先用 metadata-filtered scene 扩到更大 New York clean subset，或按 city 指定走 metadata 列表，避免再次从 raw `Crossview_rs` 混入已剔除 location。

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

## 2026-06-04 完整两城 clean metadata 训练

### 实验配置

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu
```

数据：

```text
TRAIN_CITIES=[chicago,newyork]
VAL_CITIES=[chicago,newyork]
TEST_CITIES=[chicago,newyork]
RS_PROVIDER=Google_Satellite,Bing_Satellite
REMOTE_PROVIDER_SAMPLING_MODE=random
```

该设置按 `traindata/mapanything_metadata` 的 city split 取样，不从 raw `Crossview_rs` top list 直接取样，因此不会混入 metadata 中已经剔除的低质量 location。训练集为 Chicago+NewYork 共 900 scenes，val/test 各 50 scenes。

训练设置：

```text
NUM_GPUS=2
NUM_VIEWS=4
BATCH_SIZE=8
EPOCHS=50
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.5
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.75
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=1.5
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
```

完整训练耗时 2:29:32。训练中双卡显存约 83.7G/97.9G，GPU 利用率接近满载。为节省存储，训练结束后删除了额外的 `checkpoint-last.pth`，当前仅保留：

```text
checkpoint-best.pth
checkpoint-final.pth
log.txt
train.log
events.out.tfevents...
```

### 固定 eval 关键结果

固定 eval GT 常量：

```text
height high/low/gap GT = 0.1886 / 0.0197 / 0.1814
offset high/low/gap GT = 0.1583 / 0.0071 / 0.4312
```

后半程关键 epoch：

```text
epoch34: loss=1.2317 remote=0.4069 pm=0.0400 aux=0.2471
         height pred high/low/gap = 0.1802 / 0.0113 / 0.1711
         offset pred high/low/gap = 0.1536 / 0.0093 / 0.3566

epoch38: loss=1.1697 remote=0.3933 pm=0.0370 aux=0.2455
         height pred high/low/gap = 0.1504 / 0.0091 / 0.1431
         offset pred high/low/gap = 0.1548 / 0.0091 / 0.3827

epoch44: loss=1.1567 remote=0.3943 pm=0.0370 aux=0.2464
         height pred high/low/gap = 0.1578 / 0.0115 / 0.1482
         offset pred high/low/gap = 0.1607 / 0.0102 / 0.3773

epoch48: loss=1.1437 remote=0.3931 pm=0.0366 aux=0.2468
         height pred high/low/gap = 0.1530 / 0.0107 / 0.1442
         offset pred high/low/gap = 0.1609 / 0.0099 / 0.3808

epoch50: loss=1.1415 remote=0.3928 pm=0.0368 aux=0.2456
         height pred high/low/gap = 0.1537 / 0.0110 / 0.1447
         offset pred high/low/gap = 0.1627 / 0.0100 / 0.3839
```

按固定 eval 聚合指标：

```text
best loss: epoch50, 1.1415
best remote_loss: epoch50, 0.3928
best rs_pointmap_loss: epoch40, 0.0360
best rs_projection_aux_loss: epoch42, 0.2454
```

### 结论

完整两城、多 provider、4-view 联训下，remote pointmap 可以稳定收敛，显式 projection aux 也没有退化。offset 是目前最强的正向证据：后半程 `offset_high_pred` 基本贴近 GT，`offset_low_pred` 只略高于 GT，`offset_gap_pred` 稳定在 0.36-0.38，接近 GT 0.4312。

height 分支也学到了高低结构，但比 offset 更保守。epoch34 的 height 幅度最接近 GT，后续稳定在 high 0.15 左右、gap 0.145 左右。也就是说问题不是 height 完全学不出，而是 full run 下 height amplitude 仍有轻微低估。

当前判断：

```text
aux 分支已经证明 remote 投影机制可学习。
rs pointmap 主监督可以收敛，且与 aux 多任务目标没有出现严重冲突。
low/background calibration 基本被 low-overpred 权重控制住。
剩余问题是 height 高区幅度偏保守，以及 global direction 指标波动较大。
```

### 5-set mini benchmark sanity check

使用 `checkpoint-final.pth` 跑了一个很小的 NewYork 5-set 控制评估：

```text
outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p7_chicago_newyork_full_p5b_aux_lowover15_e50_final_mini5_retry
```

配置：

```text
NUM_VIEWS=4
REMOTE_OVERFIT_NUM_SETS=5
REMOTE_CONTROL_MODES=[same,blank,shuffled]
CITY=newyork
REMOTE_PROVIDER=Google_Satellite
vggt_export_mode=mixed
use_remote_projection_aux_head=true
remote_projection_aux_hidden_dim=96
remote_projection_aux_use_rgb=true
remote_projection_aux_use_coord=true
remote_projection_aux_image_stem_dim=32
remote_projection_aux_positive_slope=true
remote_projection_aux_num_blocks=6
```

结果：

```text
aerial-only pointmaps_abs_rel = 0.0452367
same remote pointmaps_abs_rel = 0.0448126
blank remote pointmaps_abs_rel = 0.0452226
shuffled remote pointmaps_abs_rel = 0.0455939

same - aerial = -0.0004241
blank - aerial = -0.0000141
shuffled - aerial = +0.0003572
```

误差指标越低越好。这个 5-set sanity check 的方向是合理的：正确 remote 有轻微收益，blank 基本持平，shuffled remote 变差。样本量太小，不能替代正式 benchmark，但它说明 full run 的 final checkpoint 能正常用于 remote-control joint inference，且 remote 信号不是完全被忽略。

下一步不建议立刻增加复杂结构。更高效的后续实验是基于本轮 full run 做轻量 ablation：

```text
1. 保持完整两城配置，略降低 height low-overpred 或略提高 LAMBDA_PROJ_REL_HEIGHT，验证 height high/gap 是否恢复。
2. 保持 aux 配置，扩展正式 benchmark / 可视化，确认 projection 指标提升是否转化为更稳定的 remote-guided reconstruction 收益。
3. 若 benchmark 收益不足，再考虑 late/gated remote fusion 或 ordinary branch 保护，而不是继续扩大 aux head。
```

## 2026-06-04 全城市 clean metadata warm-start 训练

### 实验配置

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun
```

从完整两城 checkpoint warm-start：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu/checkpoint-final.pth
```

数据：

```text
TRAIN_CITIES=[chicago,newyork,sanfrancisco,seattle]
VAL_CITIES=[chicago,newyork,sanfrancisco,seattle]
TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
RS_PROVIDER=Google_Satellite,Bing_Satellite
REMOTE_PROVIDER_SAMPLING_MODE=random
```

该设置仍按 `traindata/mapanything_metadata` 的 city split 取样，不直接扫描 raw `traindata/Crossview_rs`，因此会避开 metadata 中已经剔除的低质量 location。训练集 1800 scenes，val/test 各 100 scenes。

训练设置：

```text
NUM_GPUS=2
NUM_VIEWS=4
BATCH_SIZE=8
EPOCHS=30
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.75
LAMBDA_PROJ_OFFSET=1.5
LAMBDA_PROJ_GLOBAL_SLOPE=0.1
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.75
PROJ_OFFSET_LOW_OVERPRED_WEIGHT=1.5
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_IMAGE_STEM_DIM=32
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_DETACH_POINTMAP=false
```

完整训练耗时 2:54:46。双卡显存约 83.7G/97.9G，GPU 利用率接近满载。训练结束后删除冗余 `checkpoint-last.pth`，只保留 `checkpoint-best.pth` 和 `checkpoint-final.pth`。

### 固定 eval 关键结果

固定 eval GT 常量：

```text
height high/low GT = 0.1226 / 0.0111
offset high/low GT = 0.1063 / 0.00035
```

关键 epoch：

```text
epoch20: loss=1.1696 remote=0.2955 pm=0.0300 aux=0.1756
         height pred high/low = 0.1130 / 0.0117
         offset pred high/low = 0.1111 / 0.0033

epoch22: loss=1.0731 remote=0.2857 pm=0.0296 aux=0.1674
         height pred high/low = 0.1096 / 0.0110
         offset pred high/low = 0.1012 / 0.0021

epoch28: loss=1.0531 remote=0.2963 pm=0.0289 aux=0.1808
         height pred high/low = 0.1020 / 0.0125
         offset pred high/low = 0.1191 / 0.0030

epoch30: loss=1.0313 remote=0.2883 pm=0.0287 aux=0.1734
         height pred high/low = 0.1039 / 0.0120
         offset pred high/low = 0.1107 / 0.0028
```

按固定 eval 聚合指标：

```text
best loss: epoch30, 1.0313
best remote_loss: epoch22, 0.2857
best rs_pointmap_loss: epoch30, 0.0287
best rs_projection_aux_loss: epoch22, 0.1674
```

相对两城 final：

```text
remote_loss: 0.3928 -> 0.2883
rs_pointmap_loss: 0.0368 -> 0.0287
rs_projection_aux_loss: 0.2456 -> 0.1734
```

全城市训练没有破坏 projection aux。height high 仍略低估，但 low 区校准更接近 GT；offset high/low 也保持可学习，low/background 过预测比早期实验明显小。

### NewYork 10-scene benchmark

使用统一 benchmark：

```text
scripts/evaluate_crossview_all_models.py --only \
  vggt_p7_p5b_shared_norm_projection_aux_allcities_best \
  vggt_p7_p5b_shared_norm_projection_aux_allcities_final
```

结果：

| checkpoint | aerial point | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allcities best | 0.0490 | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 |
| allcities final | 0.0493 | 0.0474 | 0.0486 | 95.67 | 0.2948 | 9.68 | 0.0486 | 0.0494 | 0.0503 | 0.0008 |
| full 2city final | 0.0506 | 0.0485 | 0.0491 | 92.33 | 0.2961 | 12.60 | 0.0491 | 0.0507 | 0.0524 | 0.0016 |

解读：

```text
全城市训练提升了绝对重建和 pose：joint_global、joint_point、AUC 都优于两城。
best/final 很接近；best 的 point/ray 略好，final 的 AUC 略好。
same < blank < shuffled 仍成立，说明真实卫星内容没有被忽略。
但 same-vs-blank delta 下降到 0.0007-0.0008，说明全城市收益主要来自更强主重建和泛化，卫星内容敏感性没有同步增强。
```

### 当前结论

1. `rs pointmap` 主监督在全城市 clean metadata 上可以稳定收敛，而且比两城显著更好。
2. 显式 projection aux 在全城市上仍可学习；height/offset 的 high/low 结构没有崩。
3. `LAMBDA_PROJ_REL_HEIGHT=0.75` 没有造成明显冲突，但也没有完全解决 height high 幅度偏保守的问题。
4. 当前 P7-P5B shared-norm 路线的瓶颈已经不再是 aux head 是否可学，而是 remote 信息如何更强、更稳定地影响 ordinary multi-view reconstruction。
5. 下一步更应该尝试 p5e/p5h 风格的 remote private head / viewtype / late fusion 与 projection aux 结合，而不是继续单独加大 aux head。

## 2026-06-05 P7-P5E private-head projection-aux 全城市短跑

### 实验配置

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static
```

从 P7-P5B allcities final warm-start：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun/checkpoint-final.pth
```

关键差异：

```text
EPOCHS=12
TRAIN_CITIES=[chicago,newyork,sanfrancisco,seattle]
USE_REMOTE_PRIVATE_POINT_HEAD=true
model.model_config.use_view_type_bias=true
model.model_config.remote_output_head=point
model.model_config.output_point_head_for_consistency=false
train_params=vggt_p7_p5e_projection_aux
```

新增 train params：

```text
configs/train_params/vggt_p7_p5e_projection_aux.yaml
```

该配置冻结 ordinary camera/depth/point heads，训练 shared trunk、viewtype embedding、remote private point head 和 projection aux heads。首次尝试 `ddp_find_unused_parameters=true` 会触发 DDP ready-twice，参数为 `remote_point_head.scratch.output_conv2.2.weight`；改回 `ddp_static_graph=true` 后训练正常。

### 固定 eval 关键结果

固定 eval GT 常量同全城市 P7-P5B：

```text
height high/low GT = 0.1226 / 0.0111
offset high/low GT = 0.1063 / 0.00035
```

关键 epoch：

```text
epoch2:  loss=1.2565 remote=0.3324 pm=0.0358 aux=0.1893
         height pred high/low = 0.1087 / 0.0134
         offset pred high/low = 0.1042 / 0.0032

epoch6:  loss=1.1230 remote=0.2940 pm=0.0297 aux=0.1753
         height pred high/low = 0.1236 / 0.0139
         offset pred high/low = 0.1155 / 0.0043

epoch8:  loss=1.0150 remote=0.2501 pm=0.0249 aux=0.1504
         height pred high/low = 0.1127 / 0.0115
         offset pred high/low = 0.1216 / 0.0045

epoch10: loss=0.9523 remote=0.2386 pm=0.0241 aux=0.1421
         height pred high/low = 0.1170 / 0.0105
         offset pred high/low = 0.1222 / 0.0051

epoch12: loss=0.9043 remote=0.2335 pm=0.0227 aux=0.1427
         height pred high/low = 0.1080 / 0.0102
         offset pred high/low = 0.1210 / 0.0050
```

按固定 eval 聚合指标：

```text
best loss: epoch12, 0.9043
best remote_loss: epoch12, 0.2335
best rs_pointmap_loss: epoch12, 0.0227
best rs_projection_aux_loss: epoch10, 0.1421
```

相对 P7-P5B allcities final：

```text
remote_loss: 0.2883 -> 0.2335
rs_pointmap_loss: 0.0287 -> 0.0227
rs_projection_aux_loss: 0.1734 -> 0.1427
```

训练集/固定 eval 层面，P5E private head + viewtype 明显优于 shared-norm P5B：remote pointmap 和 aux 都更低，height high/low 也更接近 GT。

### NewYork 10-scene benchmark

benchmark best/final 完全一致：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 |
| P7-P5B aux allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 |
| original p5e baseline | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 0.0501 | 0.0533 | 0.0517 | 0.0032 |

解读：

```text
P7-P5E aux 改善了 P7 路线的 joint_global，接近原始 p5e/p5h baseline。
但 joint_point 和 AUC 不如 P7-P5B allcities，satellite same-vs-blank delta 也更小。
训练指标很强，不等于 benchmark 中真实卫星内容依赖更强。
```

### 当前判断

1. P5E private head + viewtype 与 projection aux 兼容，训练可以稳定收敛。
2. 它显著提升 remote pointmap 和显式 projection aux，证明 private remote head 能更高效拟合 remote 重建。
3. benchmark 上它主要改善 `joint_global`，但没有改善 `joint_point` 和 satellite delta；remote 内容利用仍是瓶颈。
4. 如果继续 p5h，应该把目标明确为“增强 same-vs-blank / same-vs-shuffled”，而不是继续追训练集 remote pointmap。

## 2026-06-05 P7-P5H film protected weak-ranking 短跑

### 实验配置

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank005_e8_b8_2gpu
```

从 P7-P5E aux allcities final warm-start：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static/checkpoint-final.pth
```

关键差异：

```text
EPOCHS=8
TRAIN_CITIES=[chicago,newyork,sanfrancisco,seattle]
FUSION_TYPE=film
LATE_GATE_INIT=0.02
PROTECT_ORDINARY_HEADS=true
remote_control_ranking_loss_weight=0.05
remote_control_ranking_margin=0.0
remote_control_ranking_modes=[blank,shuffled]
remote_control_blank_value=0.5
```

训练只更新 late remote-to-aerial gate/film adapter，主干、ordinary heads 和 remote point head 都保持冻结。`checkpoint-last.pth` 已删除以节省空间，保留 `checkpoint-best.pth` 和 `checkpoint-final.pth`。

### 训练和固定 eval

固定 eval 基本没有变化：

```text
epoch2: loss=0.5123 aerial=1.0246 pts3d=0.1142
epoch4: loss=0.5139 aerial=1.0277 pts3d=0.1144
epoch6: loss=0.5139 aerial=1.0277 pts3d=0.1145
epoch8: loss=0.5122 aerial=1.0245 rs_pointmap=0.9163
```

训练时 remote-control ranking 信号很弱：

```text
late_gate_abs ~= 0.028
late_delta_l2 ~= 0.42
remote_control_ranking_loss_weighted ~= 0.0009
```

same 的 aerial loss 通常略低于 blank/shuffled，但差异很小，无法明显驱动 adapter 改变 ordinary reconstruction。

### NewYork 10-scene benchmark

新增 benchmark job：

```text
vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final
```

关键结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5H film rank005 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |

精确对比 P7-P5E：

```text
joint_global: 0.046629 -> 0.046639
joint_point:  0.049535 -> 0.049543
joint_auc5:   93.00 -> 93.00
same-blank:   0.000412 -> 0.000401
same-shuffle: 0.001481 -> 0.001474
```

### 当前判断

1. P7-P5H film protected weak-ranking 基本等价于 P7-P5E final，没有增强 satellite content delta。
2. adapter-only + 很弱 ranking loss 的路径信号不足，继续拉长训练大概率只是耗时，不会解决 remote 内容依赖问题。
3. 当前较可靠的结论是：显式 projection_aux 和 rs pointmap 都可学习；P5E private head 能提升 remote 重建和全局对齐；但让正确卫星图更强地影响 ordinary multi-view reconstruction 仍没有解决。
4. 下一步如果继续，应避免只训练小 gate adapter，优先考虑更直接的 paired remote-to-aerial 监督、更强 fusion 反传，或把 satellite-control objective 以更高权重接入主联训，而不是单独追求更低 remote pointmap/aux loss。

## 2026-06-05 P7-P5H film protected strong-ranking 复核

### 实验配置

实验目录：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank05_gate005_e6_b8_2gpu
```

仍从 P7-P5E aux allcities final warm-start，复核 weak-ranking 是否只是权重太小：

```text
EPOCHS=6
FUSION_TYPE=film
LATE_GATE_INIT=0.05
PROTECT_ORDINARY_HEADS=true
remote_control_ranking_loss_weight=0.5
remote_control_ranking_margin=0.0
remote_control_ranking_modes=[blank,shuffled]
```

这轮仍只训练 `remote_to_aerial_late_gate` 和 `remote_to_aerial_late_film`，因此显存峰值约 32GB/GPU。`checkpoint-last.pth` 已删除，保留 `checkpoint-best.pth` 和 `checkpoint-final.pth`。

### 训练现象

相对 weak-ranking，strong-ranking 的训练信号确实更强：

```text
weak final late_gate_abs ~= 0.028
strong final late_gate_abs ~= 0.044
strong final remote_control_ranking_loss_weighted ~= 0.0115
strong final same/blank/shuffled train aerial loss ~= 0.6921 / 0.7335 / 0.6997
```

但固定 eval 没有改善：

```text
epoch2: loss=0.5127 aerial=1.0254
epoch4: loss=0.5134 aerial=1.0268
epoch6: loss=0.5138 aerial=1.0275
```

训练中 blank gap 能被拉开一些，但 shuffled gap 小且不稳定；late gate 从初值 0.05 降到约 0.044，说明模型没有主动增强 remote-to-aerial 融合。

### NewYork 10-scene benchmark

新增 benchmark job：

```text
vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final
```

关键结果：

| model | joint global | joint point | joint AUC5 | joint ray | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5H rank05 gate005 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5H rank005 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |

精确值：

```text
joint_global=0.046639
joint_point=0.049543
same=0.049543
blank=0.049944
shuffled=0.051017
blank_delta=0.000401
shuffled_delta=0.001474
```

checkpoint 参数确实不同：

```text
weak gate = 0.02797
strong gate = 0.04403
gate diff = 0.01605
remote_to_aerial_late_film.0.weight mean abs diff ~= 0.00045
```

因此 benchmark 完全一致不是因为没读到 checkpoint，而是 adapter-only 改动幅度太小，对普通视角输出没有可测影响。

### 当前判断

1. P5H adapter-only ranking 路线已经用 weak/strong 两档权重复核，均不能改善 satellite content delta。
2. 继续简单拉长这类训练不划算，因为 final gate 仍很小，fixed eval 不改善，benchmark 与 P7-P5E 基本相同。
3. 可靠结论进一步收敛：projection_aux 和 remote pointmap 可学；P5E private head 可提升 remote 重建/全局对齐；当前缺口是“如何让 remote 内容真正进入 ordinary reconstruction”。
4. 下一步不建议再做 adapter-only ranking。更合理的是回到主联训路径，设计一个直接作用于 ordinary 输出的 remote-control objective，或者解除更多 fusion/aggregator 参数冻结，让控制损失有足够容量改变输出。

## 2026-06-05 P7-P5B allcities 2-view curriculum 复核

### 实验配置

本轮回到当前最稳的 P7-P5B shared-norm 主联训结构，不启用 private remote head，也不启用 late fusion。训练是全量微调 VGGT 主路径；`model.aggregator.patch_embed` 使用较小 lr，projection aux heads 使用较大 lr。

```text
train_params=vggt_p7_p5b_shared_norm_projection_aux
use_remote_private_point_head=false
use_remote_projection_aux_head=true
remote_projection_aux_hidden_dim=96
remote_projection_aux_use_rgb=true
remote_projection_aux_use_coord=true
remote_projection_aux_image_stem_dim=32
remote_projection_aux_num_blocks=6
remote_projection_aux_positive_slope=true

trunk / camera / depth / shared point head lr = 1e-5
model.aggregator.patch_embed lr = 5e-7
projection aux heads lr = 1e-4
```

curriculum 分两段：

```text
Stage A:
  output=p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_warmbest_e4_b8_2gpu
  warmstart=P7-P5B allcities best
  NUM_VIEWS=2
  EPOCHS=4

Stage B:
  output=p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_to4v_e4_b8_2gpu
  warmstart=Stage A checkpoint-best.pth
  NUM_VIEWS=4
  EPOCHS=4
```

B12 曾 OOM；B8 在 2 张 80G 卡上稳定，显存峰值约 81GB/GPU。`checkpoint-last.pth` 已删除，只保留 best/final。

### 训练验证结果

Stage A 的 2-view 预热没有优于原 allcities best：

```text
epoch2 best val:
  loss=1.1644
  rs_pointmap_loss=0.0380
  rs_projection_aux_loss=0.1658

final val:
  loss=1.2468
  rs_pointmap_loss=0.0399
  rs_projection_aux_loss=0.1748
```

Stage B 切回 4-view 后，final validation 比 epoch2 更好：

```text
epoch4 final val:
  loss=1.0544
  aerial_loss=0.9726
  remote_loss=0.2841
  rs_pointmap_loss=0.0291
  rs_projection_aux_loss=0.1678
  global_dir_cosine=0.4768
  global_dir_head_cosine=0.4369
```

### NewYork 10-scene benchmark

新增 benchmark job：

```text
vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final
```

结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5B curric2v->4v final | 0.0483 | 0.0506 | 93.00 | 0.2941 | 10.10 | 0.0506 | 0.0527 | 0.0533 | 0.0021 | 0.0027 |
| P7-P5B allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 | 0.0025 |
| P7-P5B allcities final | 0.0474 | 0.0486 | 95.67 | 0.2948 | 9.68 | 0.0486 | 0.0494 | 0.0503 | 0.0008 | 0.0016 |

### 当前判断

1. 2-view curriculum 会提高真实卫星内容依赖：same-vs-blank delta 从 `0.0007-0.0008` 增到 `0.0021`，same-vs-shuffled 仍为正。
2. 但 absolute reconstruction 明显退化：`joint_point 0.0485 -> 0.0506`，`joint_global 0.0474 -> 0.0483`，AUC 从 95+ 降到 93。
3. 这说明低 view 预热可能迫使模型更多看 remote，但也破坏了当前 4-view 主重建边界；不能把 curriculum final 当新 best。
4. 下一步应保留 4-view 主联训，从数据对齐和 remote crop 策略上做更小的改动。优先尝试取消训练时 remote 随机 crop，减少 projection 标签和 benchmark 口径之间的分布差异。

## 2026-06-05 P7-P5B allcities nocrop warmstart

### 目的

验证训练时取消 remote random crop 是否能减少 projection 标签和 benchmark 固定 satellite 输入之间的分布差异，从而提高真实卫星图对 joint reconstruction 的贡献。

结构仍为 P7-P5B shared-norm projection-aux，不启用 private remote head / late fusion。训练是全量微调：VGGT trunk、aggregator、camera/depth/shared point head 使用 `1e-5`，`model.aggregator.patch_embed` 使用 `5e-7`，projection aux heads 使用 `1e-4`。

### 训练设置

```text
output=p7_allcities_p5b_joint_pm4_aux_h075_lowover15_nocrop_warmbest_e8_b8_2gpu
warmstart=P7-P5B allcities best
TRAIN/VAL/TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
NUM_VIEWS=4
BATCH_SIZE=8
EPOCHS=8
REMOTE_TRAIN_CROP_MODE=none
REMOTE_VAL_CROP_MODE=none
REMOTE_TEST_CROP_MODE=none
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.75
LAMBDA_PROJ_OFFSET=1.5
```

B8 在 2 张 GPU 上稳定，显存约 `81GB/GPU` 训练峰值，`nvidia-smi` 约 `87GB/GPU`。`checkpoint-last.pth` 已删除，保留 best/final。

### 训练验证结果

| epoch | val loss | aerial | remote | rs pointmap | aux | global dir cosine | high20 height gt/pred | high20 offset gt/pred |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 1.4133 | 1.5305 | 0.3240 | 0.0349 | 0.1846 | 0.2817 | 0.1226 / 0.1058 | 0.1063 / 0.0981 |
| 4 | 1.1785 | 1.0195 | 0.3344 | 0.0352 | 0.1934 | 0.4416 | 0.1226 / 0.0912 | 0.1063 / 0.1143 |
| 6 | 1.1032 | 0.9911 | 0.3038 | 0.0288 | 0.1886 | 0.4823 | 0.1226 / 0.0950 | 0.1063 / 0.1169 |
| 8 | 1.0322 | 0.8955 | 0.2922 | 0.0284 | 0.1787 | 0.4799 | 0.1226 / 0.1040 | 0.1063 / 0.1116 |

final/best 比 epoch6 稍好，rs pointmap 已接近旧 allcities best，但 aux 显式 height 仍欠预测。

### NewYork 10-scene benchmark

新增 benchmark job：

```text
vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best
```

结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5B nocrop warmbest best | 0.0482 | 0.0495 | 92.33 | 0.3247 | 9.14 | 0.0495 | 0.0510 | 0.0516 | 0.0015 | 0.0021 |
| P7-P5B curric2v->4v final | 0.0483 | 0.0506 | 93.00 | 0.2941 | 10.10 | 0.0506 | 0.0527 | 0.0533 | 0.0021 | 0.0027 |
| P7-P5B allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 | 0.0025 |
| P7-P5B allcities final | 0.0474 | 0.0486 | 95.67 | 0.2948 | 9.68 | 0.0486 | 0.0494 | 0.0503 | 0.0008 | 0.0016 |

### 当前判断

1. nocrop 确实提高了真实卫星图相对 blank 的收益：`0.0007 -> 0.0015`，但不如 curriculum 的 `0.0021`。
2. 代价仍明显：`joint_point 0.0485 -> 0.0495`，AUC `95.33 -> 92.33`，ray `0.2893 -> 0.3247`。
3. 因此当前最好模型仍是 `vggt_p7_p5b_shared_norm_projection_aux_allcities_best`；nocrop 只证明固定对齐能增加一点 remote 内容依赖，不能解决显式 projection height 欠预测。
4. 下一步不宜继续沿 nocrop 单独加长训练；更合理的是回到 allcities best，用更小幅度的 aux 权重/校准策略调整，目标是在保住 `joint_point/AUC` 的同时提高 blank delta。

## 2026-06-05 P7-P5B h1/off1 early-stop

### 目的

测试单纯提高 height 权重并降低 offset 权重是否能修正显式 projection height 欠预测。

### 设置

```text
output=p7_allcities_p5b_joint_pm4_aux_h10_off10_lowover15_warmbest_e8_b8_2gpu
warmstart=P7-P5B allcities best
TRAIN/VAL/TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
NUM_VIEWS=4
BATCH_SIZE=8
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=1.0
LAMBDA_PROJ_OFFSET=1.0
REMOTE_TRAIN_CROP_MODE=random_scale_offset
```

### early-stop 结果

epoch2 validation：

```text
loss=1.5290
aerial_loss=1.6740
remote_loss=0.3460
rs_pointmap_loss=0.0363
rs_projection_aux_loss=0.2010
global_dir_cosine=0.4788
high20 height gt/pred=0.1226 / 0.0760
high20 offset gt/pred=0.1063 / 0.0915
```

对比 allcities best epoch30：

```text
rs_pointmap_loss=0.0287
rs_projection_aux_loss=0.1734
high20 height gt/pred=0.1226 / 0.1039
high20 offset gt/pred=0.1063 / 0.1107
```

### 判断

1. h1/off1 同时恶化主重建、remote pointmap 和 aux projection，height high20 pred 从 `0.1039` 掉到 `0.0760`。
2. 这说明问题不是简单的 height 权重不足；提高 height 权重会改变优化平衡，反而压低显式高层预测。
3. 已中止训练并删除失败 `checkpoint-best.pth`，仅保留日志。下一步改用低主干 LR 的全量微调，保留原 h0.75/off1.5，避免再大幅扰动已收敛的主重建。

## 2026-06-05 P7-P5B lowtrunklr2e6 warmstart

### 目的

测试更保守的全量微调是否能在不破坏 P7-P5B allcities best 主重建的前提下，让 aux 头继续校准，并提高 satellite control delta。

### 设置

新增 train_params：

```text
configs/train_params/vggt_p7_p5b_shared_norm_projection_aux_lowtrunklr.yaml
```

相对标准 P7-P5B shared-norm projection-aux，只改学习率：

```text
trunk / camera / depth / shared point head lr = 2e-6
model.aggregator.patch_embed lr = 1e-7
projection aux heads lr = 1e-4
```

训练命令核心参数：

```text
output=p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu
warmstart=P7-P5B allcities best
TRAIN/VAL/TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
NUM_VIEWS=4
BATCH_SIZE=8
EPOCHS=8
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.75
LAMBDA_PROJ_OFFSET=1.5
REMOTE_TRAIN_CROP_MODE=random_scale_offset
```

`checkpoint-last.pth` 已删除，保留 best/final。

### 训练验证结果

| epoch | val loss | aerial | remote | rs pointmap | aux | global dir cosine | high20 height gt/pred | high20 offset gt/pred |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 1.0800 | 0.9328 | 0.3068 | 0.0304 | 0.1854 | 0.3131 | 0.1226 / 0.0949 | 0.1063 / 0.1076 |
| 4 | 1.0432 | 0.8808 | 0.3014 | 0.0283 | 0.1882 | 0.5466 | 0.1226 / 0.0947 | 0.1063 / 0.1146 |
| 6 | 1.0288 | 0.8859 | 0.2929 | 0.0286 | 0.1784 | 0.4805 | 0.1226 / 0.0992 | 0.1063 / 0.1127 |
| 8 | 1.0347 | 0.9077 | 0.2904 | 0.0290 | 0.1744 | 0.4760 | 0.1226 / 0.1022 | 0.1063 / 0.1096 |

best checkpoint 对应最低 val loss；final 的 aux 更接近旧 allcities best，但 pointmap 略差。

### NewYork 10-scene benchmark

新增 benchmark jobs：

```text
vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best
vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final
```

结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 0.0484 | 0.0495 | 0.0499 | 0.0011 | 0.0014 |
| lowtrunklr best | 0.0472 | 0.0487 | 96.00 | 0.2997 | 9.79 | 0.0487 | 0.0499 | 0.0502 | 0.0012 | 0.0015 |
| P7-P5B allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 | 0.0025 |
| P7-P5B allcities final | 0.0474 | 0.0486 | 95.67 | 0.2948 | 9.68 | 0.0486 | 0.0494 | 0.0503 | 0.0008 | 0.0016 |
| nocrop warmbest best | 0.0482 | 0.0495 | 92.33 | 0.3247 | 9.14 | 0.0495 | 0.0510 | 0.0516 | 0.0015 | 0.0021 |
| curric2v->4v final | 0.0483 | 0.0506 | 93.00 | 0.2941 | 10.10 | 0.0506 | 0.0527 | 0.0533 | 0.0021 | 0.0027 |

### 当前判断

1. lowtrunklr final 是当前 P7-P5B shared-norm 路线的综合 best：`joint_global=0.0471`、`joint_point=0.0484`，均优于旧 allcities best。
2. satellite blank delta 也从旧 best 的 `0.0007` 提高到 `0.0011-0.0012`，但仍低于 nocrop/curriculum；same-vs-shuffled delta 反而小于旧 best。
3. 显式 height 欠预测没有根本解决：final high20 height 仍是 `0.1226 / 0.1022`，只比旧 allcities best `0.1226 / 0.1039` 持平或略差。
4. 结论是：低主干 LR 可以作为稳定微调策略改善全局/point 指标和部分 blank sensitivity，但不能证明 projection height 已显式学好。下一步如果继续攻显式机制，应优先改 aux 输入/目标定义，而不是继续调大 height 权重。

## 2026-06-05 P7-P5E lowtrunkfull warmstart

### 目的

P7-P5E private-viewtype projection-aux 原始版本的显式 projection/remote 指标强于 P7-P5B，但 benchmark 上 `joint_point/AUC/satellite delta` 不如 P7-P5B。这个实验从 P7-P5E final warmstart，保持 private remote point head + viewtype + projection aux，同时把普通 camera/depth/point heads 以低学习率解冻，测试是否能保留 P5E 的 remote 机制并改善 ordinary reconstruction。

### 结构与冻结状态

这组不是冻结头训练，也不是完全同学习率全量微调：

```text
use_view_type_bias=true
use_remote_private_point_head=true
remote_output_head=point
output_point_head_for_consistency=false
use_remote_projection_aux_head=true
ordinary camera/depth/point heads: trainable, lr=2e-6
VGGT trunk/aggregator: trainable, base lr=2e-6
aggregator.patch_embed: trainable, lr=1e-7
remote_point_head: trainable, lr=1e-5
projection aux heads: trainable, lr=5e-5
```

对比：原始 `configs/train_params/vggt_p7_p5e_projection_aux.yaml` 冻结普通 camera/depth/point heads（lr=0），但 shared trunk 仍可训练。

### 设置

新增 train_params：

```text
configs/train_params/vggt_p7_p5e_projection_aux_lowtrunk_full.yaml
```

训练命令核心参数：

```text
output=p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu
warmstart=P7-P5E private-viewtype projection-aux final
TRAIN/VAL/TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
NUM_VIEWS=4
BATCH_SIZE=9
EPOCHS=6
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.75
LAMBDA_PROJ_OFFSET=1.5
REMOTE_TRAIN_CROP_MODE=random_scale_offset
```

`checkpoint-last.pth` 已删除，保留 best/final。

### 训练验证结果

| epoch | val loss | aerial | remote | rs pointmap | aux | global dir cosine | high20 height gt/pred | low80 height gt/pred | high20 offset gt/pred | low80 offset gt/pred |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 2 | 0.9459 | 0.9648 | 0.2317 | 0.0240 | 0.1356 | 0.477 | 0.1226 / 0.1047 | 0.0111 / 0.0126 | 0.1063 / 0.1074 | 0.0004 / 0.0064 |
| 4 | 0.9581 | 0.9457 | 0.2427 | 0.0247 | 0.1438 | 0.479 | 0.1226 / 0.1159 | 0.0111 / 0.0123 | 0.1063 / 0.1234 | 0.0004 / 0.0057 |
| 6 | 0.8941 | 0.9117 | 0.2191 | 0.0227 | 0.1285 | 0.480 | 0.1226 / 0.1105 | 0.0111 / 0.0112 | 0.1063 / 0.1093 | 0.0004 / 0.0052 |

epoch6 保存为 best；final 和 best 在 benchmark 上完全一致。

### NewYork 10-scene benchmark

新增 benchmark jobs：

```text
vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_best
vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final
```

结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5E lowtrunkfull best/final | 0.0460 | 0.0491 | 93.67 | 0.2990 | 9.60 | 0.0491 | 0.0500 | 0.0507 | 0.0008 | 0.0016 |
| P7-P5E aux allcities final | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5B lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 0.0484 | 0.0495 | 0.0499 | 0.0011 | 0.0014 |

### 当前判断

1. 低 LR 解冻普通 heads 是有效的：相对原始 P7-P5E，`joint_global 0.0466 -> 0.0460`，`joint_point 0.0495 -> 0.0491`，RS-only MAE `10.11 -> 9.60`。
2. 显式 projection 指标也更强：val aux `0.1427 -> 0.1285`，height high20 `0.1226 / 0.1105`，offset high20 `0.1063 / 0.1093`，说明 P5E private head 路线确实更适合学习 remote 投影机制。
3. 但综合 ordinary reconstruction 仍不是最好：P7-P5B lowtrunklr final 的 `joint_point=0.0484`、`AUC=95.33` 优于这组 `0.0491/93.67`。
4. satellite blank delta 从旧 P7-P5E 的 0.0004 提升到 0.0008，但仍低于 P7-P5B lowtrunklr final 的 0.0011；remote 内容利用有所改善但没有根本解决。
5. 因此 export 默认暂时保留 P7-P5B lowtrunklr final；`scripts/export_pointcloud_ply.py` 已支持自动识别 P7-P5E private-viewtype projection-aux checkpoint，可手动指定这组 checkpoint 做机制/remote 重建可视化。

## 2026-06-05 P7-P5E midtrunkfull 与 PLY 导出复核

### midtrunkfull 训练

为了测试更大幅度解冻 trunk/普通 heads 是否能改善 P5E lowtrunkfull 的可视化和 `joint_point`，从 P5E lowtrunkfull final warm-start 继续短跑：

```text
output=p7_allcities_p5e_private_viewtype_projection_aux_midtrunkfull_warmp5elowfull_e4_b9_2gpu
warmstart=p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-final.pth
train_params=configs/train_params/vggt_p7_p5e_projection_aux_midtrunk_full.yaml
ordinary camera/depth/point heads: lr=5e-6
VGGT trunk/aggregator: lr=5e-6
aggregator.patch_embed: lr=2e-7
remote_point_head: lr=1e-5
projection aux heads: lr=3e-5
BATCH_SIZE=9, NUM_GPUS=2, EPOCHS=4
```

固定验证：

| epoch | val loss | aerial | remote | rs pointmap | aux | high20 height gt/pred | low80 height gt/pred | high20 offset gt/pred | low80 offset gt/pred |
|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 2 | 1.0037 | 1.1142 | 0.2233 | 0.0252 | 0.1226 | 0.1226 / 0.1085 | 0.0111 / 0.0118 | 0.1063 / 0.0984 | 0.0004 / 0.0053 |
| 4 | 0.9167 | 0.9479 | 0.2214 | 0.0230 | 0.1295 | 0.1226 / 0.1152 | 0.0111 / 0.0121 | 0.1063 / 0.1133 | 0.0004 / 0.0051 |

结论：epoch4 比 epoch2 恢复，但仍差于 lowtrunkfull epoch6 的 `val loss=0.8941`、`aerial=0.9117`、`rs_pointmap=0.0227`。更大 trunk/普通头 LR 没有解决可视化或 remote 利用问题，暂不做 benchmark，也不作为新 best。`checkpoint-last.pth` 已删除，保留 best/final。

### PLY 导出复核

用户用 `scripts/export_pointcloud_ply.py` 查看 P7-P5B lowtrunklr 和 P7-P5E lowtrunkfull 时反馈可视化效果不好。复核脚本发现一个会显著误导 mixed export 的问题：

```text
旧逻辑：P5/P7 joint remote export 如果没有显式传 --remote_view_indices/--remote_view_names，
       会把所有输入图都标记为 remote。
```

对 `/root/autodl-tmp/test/scence/461_1` 这类目录，输入为：

```text
image.png
location_461_21.jpg
location_461_26.jpg
```

旧逻辑会让 `location_*.jpg` 也走 remote point head，而不是 ordinary camera+depth head，PLY 会非常容易看起来坏。已更新 `scripts/export_pointcloud_ply.py`：

1. mixed export 不再默认全 remote。
2. 若未显式指定 remote，会自动识别常见 satellite/remote 文件名，如 `image.png`、`zimage.png`、`sate*.png`、`*Satellite*`。
3. 识别不到 remote 时打印警告，并保持普通视角逻辑；remote-only 调试需要显式传 `--force_remote_instance`。

smoke test：

```text
P7-P5B lowtrunklr, /root/autodl-tmp/test/scence/461_1:
Auto-detected remote view names: image.png
View 0: head=point
View 1: head=depth
View 2: head=depth

P7-P5E lowtrunkfull, /root/autodl-tmp/test/scence/461_1:
Auto-detected remote view names: image.png
View 0: head=point
View 1: head=depth
View 2: head=depth
```

新导出路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_p5b_shared_norm_projection_aux_mixed_autoremote_fix
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_mixed_autoremote_fix
```

因此，之前 PLY “看起来不好”至少有一部分可能来自导出分支错配。benchmark 结果不受该问题影响，因为 dataset/eval 中 remote instance 是显式设置的。需要用修正后的 export 重新做可视化判断，再决定是否继续训练结构。

### remote-only PLY 导出

为直接比较 remote 点云，`scripts/export_pointcloud_ply.py` 新增：

```text
--export_view_filter all|remote|ordinary
```

默认 `all` 保持旧行为；`remote` 只导出被标记为 remote 的 satellite/remote view 点云；`ordinary` 只导出普通视角点云，但 remote 仍可作为条件输入参与推理。

448 场景 remote-only smoke test：

```text
P5B original:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p5b_remote_only/mapanything_pointcloud.ply

P7-P5B:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_p5b_shared_norm_projection_aux_remote_only/mapanything_pointcloud_same.ply
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_p5b_shared_norm_projection_aux_remote_only/mapanything_pointcloud_blank.ply

P7-P5E:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_remote_only/mapanything_pointcloud_same.ply
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_remote_only/mapanything_pointcloud_blank.ply
```

日志确认每个 remote-only PLY 只包含 view0 `image.png` 的 `point` head，两个 street views 均被 `view_filter=remote` 跳过。

448 remote-only 统计：

| model | z min/max | z mean/std | 备注 |
|---|---:|---:|---|
| P5B original | 0.682 / 0.978 | 0.906 / 0.046 | remote 高度更高、更集中 |
| P7-P5B same | 0.654 / 0.925 | 0.854 / 0.046 | 相比原始 P5B 高度整体压低，y 方向拉宽 |
| P7-P5B blank | 0.195 / 0.657 | 0.438 / 0.123 | same/blank 差异大，说明 satellite 内容确实影响 remote head |
| P7-P5E same | 0.650 / 0.947 | 0.864 / 0.045 | 数值接近 P7-P5B，但可视化局部有条纹 |
| P7-P5E blank | 0.203 / 0.768 | 0.482 / 0.158 | blank 下分布更散 |

当前判断：P7-P5B 可视化强于 P7-P5E，但相对原始 P5B remote 并没有明确变好；P7 的 projection aux 让模型更依赖 satellite 内容，但没有把 remote pointmap 恢复质量提升到可视化可用的程度。

## 2026-06-05 projection-aux 标签反投诊断

用户观察到 `newyork__location_448` 较高区域 remote 点云在 P7-P5B 中飞掉，怀疑多任务 projection 标签本身可能和 remote pointmap 标签不一致。为此新增诊断脚本：

```text
scripts/reconstruct_remote_pointcloud_from_projection_aux.py
```

用法：

```bash
python scripts/reconstruct_remote_pointcloud_from_projection_aux.py \
  --remote_dir /root/autodl-tmp/traindata/Crossview_rs/newyork__location_448/Google_Satellite \
  --output_dir /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/projection_aux_reconstruct/newyork_448_google
```

脚本输出：

```text
pixel_to_point_map_xyz.ply
aux_original_xyz_world_common.ply
aux_reconstructed_from_offset_common.ply
aux_reconstructed_from_rel_global_common.ply
aux_projected_xyz_centered.ply
summary.json
```

反投关系验证：

```text
original_xy = projected_xyz_centered.xy + projection_center_xy - offset_xy
offset_xy ~= rel_height * global_slope * global_dir_xy
original_z = projected_xyz_centered.z
```

因此可以用 projection aux 标签直接重建一个 remote 点云，再和 `pixel_to_point_map.npz` 的原始 pointmap 标签比较。

### 448 结果

| scene/provider | common mean L2 | common p95 | common p99 | high mean L2 | high p95 | high p99 | recon offset -> aux original |
|---|---:|---:|---:|---:|---:|---:|---:|
| newyork_448 Google | 1.397 | 4.750 | 13.608 | 2.090 | 6.074 | 35.680 | 7e-7 |
| newyork_448 Bing | 1.163 | 3.739 | 12.978 | 1.730 | 5.989 | 16.987 | 9e-7 |
| newyork_448 ESRI | 2.930 | 8.814 | 33.185 | 4.824 | 21.640 | 42.002 | 1e-6 |
| newyork_448 Yandex | 10.644 | 29.660 | 49.015 | 14.526 | 36.413 | 50.270 | 1e-6 |
| chicago_130 Google | 0.007 | 0.000 | 0.000 | 0.012 | 0.000 | 0.247 | 1e-6 |

结论：

1. projection aux 标签内部是自洽的：用 `projected + center - offset` 可以几乎精确还原 `projection_aux.npz/original_xyz_world`。
2. 但 `newyork__location_448` 的 `projection_aux/original_xyz_world` 和 `pixel_to_point_map.npz/xyz` 在同一像素 common mask 上不一致，高区域更明显；Google/Bing 这两个训练常用 provider 也有 p95 约 6m 的 high 区域偏差。
3. 这种不一致足以解释多任务冲突：remote pointmap loss 监督模型贴 `pixel_to_point_map`，projection aux loss 通过 non-detached aux head 又推动共享 point features 表达另一个几何目标。
4. 因此 P7-P5B 较高区域飞点不应只归因于模型结构；至少部分 location/provider 的 projection aux 标签与原始 pointmap 标签存在可测冲突。下一步训练前应先过滤或修正这类 label-inconsistent 样本，或者让 aux head 对 pointmap detach，避免错误 aux 标签反向破坏 remote pointmap。

## 2026-06-06 P7 remote pointmap 可视化复核与 P5B-anchor 实验

### 背景

用户在 448 场景观察到 P7-P5B / P7-P5E remote 点云较高区域飞掉或整体不好。修正 `export_pointcloud_ply.py` 的 mixed remote 标记后，又用 `--export_view_filter remote` 单独导出 remote 点云，确认问题不只是 mixed export 错配：

```text
P5B same:       z_mean=0.9058, z_std=0.0456, q95=0.9637
P7 oldP7 same:  z_mean≈0.854,  z_std≈0.043,  q95≈0.909
```

即 oldP7 warmstart/private-head 系列在 448 remote-only 上把高度分布整体压低了约 0.052。

### pmgrad05 复核

实验：

```text
p7_allcities_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_e4_b9_4gpu
```

配置：

```text
warmstart=P7-P5B lowtrunklr final
remote_private_point_head initialized from original P5B point_head
remote_pointmap_gradient_loss_weight=0.5
remote_pointmap_gradient_channels=z
remote_pointmap_gradient_scales=4
4 GPUs, batch=9, max mem≈83GB/GPU
```

训练中 `rs_pointmap_gradient_loss` 稳定在约 `0.004-0.006`。最终 validation 的 rel-height 显式指标很好：

```text
rel_height high20 pred/gt ≈ 0.1553 / 0.1510
rel_height low80  pred/gt ≈ 0.0159 / 0.0121
```

但 448 remote-only PLY 与 oldP7 private E2/E6 几乎一致：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| P5B original | 0.9058 / 0.0456 | 0.9637 | 0.5571 / 0.0972 |
| oldP7 private pmgrad05 | 0.8540 / 0.0428 | 0.9088 | 0.4383 / 0.1088 |
| oldP7 private e2 | 0.8533 / 0.0431 | 0.9084 | 0.4379 / 0.1073 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oldP7 private pmgrad05 | 0.0446 | 0.0485 | 94.67 | 0.2995 | 10.47 | 0.0485 | 0.0495 | 0.0498 |
| oldP7 private h035 best | 0.0446 | 0.0485 | 95.33 | 0.2957 | 10.53 | 0.0485 | 0.0494 | 0.0497 |

结论：pointmap z-gradient 辅助损失没有把 remote pointmap 的高度分布拉回 P5B；它主要改善/稳定了显式 aux 指标，不能解决当前可视化问题。

### P5B-anchor 诊断实验

实验：

```text
p7_allcities_p5b_parallel_token_aux_p5b_anchor_h035_e4_b10_4gpu
```

新增训练参数：

```text
configs/train_params/vggt_p7_p5b_parallel_token_aux_p5b_anchor.yaml
```

设计：

```text
warmstart=P5B original checkpoint
remote_point_head lr=0
model.aggregator.patch_embed lr=0
base trunk lr=1e-8
parallel-token aux heads lr=5e-5 / 1e-4
LAMBDA_REMOTE_PM=4.0
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
4 GPUs, batch=10, max mem≈92GB/GPU
```

目标不是直接追 benchmark，而是验证：在不破坏 P5B remote point head 的前提下，parallel token aux head 是否能学到显式投影高度。

训练现象：

- epoch0 aux 从随机常数预测快速下降，`rs_projection_aux_loss` 从约 0.44 降到 0.12。
- epoch1/2 后 rel-height high/low 已经稳定贴近 GT；最终 validation 约：

```text
rel_height high20 pred/gt ≈ 0.1553 / 0.1510
rel_height low80  pred/gt ≈ 0.0159 / 0.0121
rs_pointmap_loss ≈ 0.0297
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std | 备注 |
|---|---:|---:|---:|---|
| P5B original | 0.9058 / 0.0456 | 0.9637 | 0.5571 / 0.0972 | 原始 remote pointmap |
| P7 p5b-anchor | 0.9053 / 0.0457 | 0.9635 | 0.5565 / 0.0977 | 几乎完整保住 P5B |
| oldP7 pmgrad05 | 0.8540 / 0.0428 | 0.9088 | 0.4383 / 0.1088 | 仍整体压低 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P7 p5b-anchor h035 | 0.0523 | 0.0560 | 86.33 | 0.4257 | 17.62 | 0.0560 | 0.0567 | 0.0576 |
| oldP7 private h035 best | 0.0446 | 0.0485 | 95.33 | 0.2957 | 10.53 | 0.0485 | 0.0494 | 0.0497 |
| P7-P5B lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 0.0484 | 0.0495 | 0.0499 |

结论：

1. `p5b-anchor` 成功证明 parallel-token aux head 可以在 P5B remote 几何不被破坏的条件下学习显式 projection height。
2. 但它的 ordinary/joint benchmark 明显退化，说明“冻结/近冻结主路径 + 只训 aux”不能作为最终重建模型。
3. 当前存在一个清晰 tradeoff：
   - oldP7/private 路线 benchmark 好，但 remote-only 视觉高度整体压低；
   - P5B-anchor 视觉 remote 点云好，但 benchmark 差。
4. 下一步不应继续单纯加 pm-gradient 或小幅 aux 权重；更合理的是做受控蒸馏/约束：从 oldP7/private 的 benchmark 好状态出发，引入 P5B remote pointmap teacher 或更明确的绝对高度保持项，只约束 remote point head 的 absolute z/scale，而不把普通重建能力退回 P5B-anchor。

## 2026-06-06 freeze-remote-head 与 parallel-token aux-only 复核

在 P5B-anchor 诊断后，继续验证两个问题：

1. 从 oldP7 trunk + P5B remote head 出发，如果冻结 `remote_point_head`，能否在不完全退回 P5B-anchor 的情况下恢复 benchmark。
2. 如果 trunk/remote head 全冻结，只训练 parallel-token aux head，是否能证明固定 tokens 中已经包含可解码的 projection 信息，同时不破坏 remote 点云。

### freeze remote head

实验：

```text
p7_allcities_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_e3_b9_4gpu
```

结构：

```text
oldP7 trunk/aux warmstart
remote_point_head 使用原始 P5B point_head 权重初始化
remote_point_head lr=0
trunk 极低 lr=2e-7
parallel-token projection aux heads lr=3e-5 / 6e-5
4 GPUs, batch=9
```

448 remote-only PLY：

| model | same z mean/std | same q95 | 备注 |
|---|---:|---:|---|
| diagnostic oldP7 trunk + P5B head | 0.8980 / 0.0479 | 0.9557 | 仅替换 remote head，不训练 |
| freeze remote head final | 0.8913 / 0.0466 | 0.9470 | 比 oldP7 private 的 0.854 明显恢复，但低于 diagnostic |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| freeze remote head final | 0.0466 | 0.0501 | 95.67 | 0.2856 | 10.1601 | 0.0501 | 0.0509 | 0.0514 |

解读：

- 冻结 P5B remote head 后，trunk 低 LR 训练仍会把 448 remote-only 高度从 `0.8980` 压到 `0.8913`，但远好于 oldP7 private 系列的 `~0.854`。
- benchmark 的 `joint_global/ray/RS MAE` 比 diagnostic 更好，但 `joint_point` 变差；它是一个折中点，不是明确全面 best。

### aux-only from freeze final

实验：

```text
p7_allcities_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b16_4gpu
```

结构：

```text
warmstart=freeze remote head final
冻结 aggregator / camera_head / point_head / depth_head / remote_point_head
只训练 remote_projection_aux_token_* / image_stem
4 GPUs, batch=16
```

结果：

```text
final val rs_projection_aux_loss = 0.0899
remote-only same z = 0.8913 / 0.0466, q95=0.9470
benchmark 与 freeze remote head final 完全一致
```

结论：固定 trunk/remote head 上，parallel-token aux head 可以继续校准 projection 输出，但不会改变点云或 benchmark。这证明 projection 机制可由 fixed tokens 解码，但它本身不能修复 remote pointmap。

### diagnostic aux-only from best geometry

实验：

```text
p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b32_4gpu
```

结构：

```text
warmstart=diagnostic oldP7 trunk + P5B remote head
冻结 aggregator / camera_head / point_head / depth_head / remote_point_head
只训练 parallel-token projection aux head
4 GPUs, batch=32, max mem≈70GB/GPU
```

训练/验证：

```text
epoch1 best val rs_projection_aux_loss = 0.0800
epoch4 final val rs_projection_aux_loss = 0.0834
best high20 rel_height pred/gt = 0.1384 / 0.1453
final high20 rel_height pred/gt = 0.1544 / 0.1453
```

保留 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_e4_b32_4gpu/checkpoint-best.pth
```

`checkpoint-final.pth` 和 `checkpoint-last.pth` 已删除，节省存储。

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std | 备注 |
|---|---:|---:|---:|---|
| diagnostic oldP7 trunk + P5B head | 0.8980 / 0.0479 | 0.9557 | - | 未训练诊断基线 |
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 | 完整保持 diagnostic geometry |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |

结论：

1. 目前视觉 remote-only 点云最稳的 P7 候选是 `diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best`，它保住了 P5B-head diagnostic 的高度分布，并有可用的 projection aux 输出。
2. 但它没有 benchmark 增益；几何主路径被冻结，所以点云/benchmark 等同 diagnostic。
3. `aux-only` 的价值是证明 fixed tokens 中有 projection 可学习信号，不是最终修复路径。
4. 下一步应测试 absolute height / teacher-style 约束能否在允许 trunk 低 LR 改动时保住 `z_mean≈0.898`，同时保留 freeze-remote-head 带来的 `joint_global/ray` 改善。

## 2026-06-06 diagnostic P5B-head + height001 短跑

目标：验证现有 `remote_height_loss_weight` 是否能在允许 trunk 极低 LR 更新时，缓解 freeze-remote-head 训练造成的 remote-only 高度下降。

实验：

```text
p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_e3_b10_4gpu
```

配置：

```text
warmstart=diagnostic oldP7 trunk + P5B remote head
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_freeze_remotehead_parallel_token_aux
remote_point_head lr=0
trunk lr=2e-7
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_H=0.001
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
4 GPUs, batch=10, max mem≈93.4GB/GPU
```

训练现象：

```text
epoch1 val rs_pointmap_loss=0.0534, rs_projection_aux_loss=0.1637
epoch2 val rs_pointmap_loss=0.0410, rs_projection_aux_loss=0.1262
epoch3 val rs_pointmap_loss=0.0390, rs_projection_aux_loss=0.1262
train rs_height_loss ≈ 291-293, weighted≈0.291-0.293
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.8980 / 0.0479 | 0.9557 | - |
| freeze remote head final | 0.8913 / 0.0466 | 0.9470 | 0.4414 / 0.1137 |
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 final | 0.8928 / 0.0471 | 0.9492 | 0.4376 / 0.1139 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| freeze remote head final | 0.0466 | 0.0501 | 95.67 | 0.2856 | 10.1601 | 0.0501 | 0.0509 | 0.0514 |
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 final | 0.0467 | 0.0497 | 95.33 | 0.2862 | 10.1588 | 0.0497 | 0.0506 | 0.0507 |

结论：

1. `height001` 相比 freeze-remote-head 是小幅正向：remote-only z mean `0.8913 -> 0.8928`，`joint_point 0.0501 -> 0.0497`，RS MAE 也略好。
2. 但它仍未恢复到 diagnostic/aux-only 的 `z_mean=0.8980` 和 `joint_point=0.0485`。
3. 训练中 height loss weighted 约 `0.29`，已经和 pointmap weighted loss 同量级；继续大幅增大 height 权重有风险，但可以短跑 `0.002-0.003` 验证是否存在更好的折中。
4. `checkpoint-best.pth` 为 12GB optimizer/full-state 权重，已删除；仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height003 短跑

目标：复核把 `remote_height_loss_weight` 从 `0.001` 提高到 `0.003` 是否能进一步保住 remote-only 高度。

实验：

```text
p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_e3_b10_4gpu
```

配置与 `height001` 相同，只改：

```text
LAMBDA_REMOTE_H=0.003
4 GPUs, batch=10, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0535, rs_projection_aux_loss=0.1624
epoch2 val rs_pointmap_loss=0.0410, rs_projection_aux_loss=0.1236
epoch3 val rs_pointmap_loss=0.0390, rs_projection_aux_loss=0.1246
epoch3 val high20 rel_height pred/gt = 0.1506 / 0.1510
train rs_height_loss_weighted≈0.874
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.8980 / 0.0479 | 0.9557 | - |
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 final | 0.8928 / 0.0471 | 0.9492 | 0.4376 / 0.1139 |
| height003 final | 0.8927 / 0.0472 | 0.9492 | 0.4371 / 0.1137 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 final | 0.0467 | 0.0497 | 95.33 | 0.2862 | 10.1588 | 0.0497 | 0.0506 | 0.0507 |
| height003 final | 0.0467 | 0.0497 | 95.67 | 0.2891 | 10.1568 | 0.0497 | 0.0507 | 0.0507 |

结论：

1. `height003` 没有进一步提升 remote-only 高度，same z mean `0.8927` 基本等于 `height001` 的 `0.8928`。
2. benchmark 也基本持平：`joint_global/joint_point=0.0467/0.0497`，AUC5 略回升但 ray 略差。
3. 单纯提高 `remote_height_loss_weight` 已经接近收益平台；下一步应减少 trunk 漂移，优先尝试更低 trunk LR，而不是继续加大 height loss。
4. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height001 + trunklr5e8 短跑

目标：复核把 trunk LR 从 `2e-7` 进一步降到 `5e-8`，是否能在保留 height001/aux 训练的同时减少 remote-only 高度漂移。

实验：

```text
p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e3_b10_4gpu
```

配置与 `height001` 相同，只改：

```text
train_params.lr=5e-8
train_params.min_lr=5e-10
LAMBDA_REMOTE_H=0.001
4 GPUs, batch=10, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0615, rs_projection_aux_loss=0.1649
epoch2 val rs_pointmap_loss=0.0570, rs_projection_aux_loss=0.1232
epoch3 val rs_pointmap_loss=0.0562, rs_projection_aux_loss=0.1216
epoch3 val high20 rel_height pred/gt = 0.1497 / 0.1510
train rs_height_loss_weighted≈0.291-0.292
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 final | 0.8928 / 0.0471 | 0.9492 | 0.4376 / 0.1139 |
| height003 final | 0.8927 / 0.0472 | 0.9492 | 0.4371 / 0.1137 |
| height001 trunklr5e8 final | 0.8967 / 0.0477 | 0.9543 | 0.4439 / 0.1140 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic oldP7 trunk + P5B head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 final | 0.0467 | 0.0497 | 95.33 | 0.2862 | 10.1588 | 0.0497 | 0.0506 | 0.0507 |
| height003 final | 0.0467 | 0.0497 | 95.67 | 0.2891 | 10.1568 | 0.0497 | 0.0507 | 0.0507 |
| height001 trunklr5e8 final | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.2399 | 0.0489 | 0.0499 | 0.0501 |

结论：

1. 降低 trunk LR 明显比继续增大 height loss 更能保住 remote-only z：same z mean 从 `0.8928/0.8927` 回升到 `0.8967`，接近 diagnostic/aux-only 的 `0.8980`。
2. 但训练验证 `rs_pointmap_loss=0.0562` 明显差于 `height001/003` 的 `0.0390`，说明 trunk 更新太小会限制 remote pointmap 收敛。
3. benchmark 折中结果优于 height001/003 的 `joint_point=0.0497`，达到 `0.0489`，但仍低于 diagnostic/aux-only 的 `0.0485`。
4. 当前最合理的下一步不是继续加 height 权重，而是加入更直接的 teacher-style z/point anchor 或蒸馏项，约束 trunk 更新不要破坏 P5B-head diagnostic 的 remote 几何，同时保留 pointmap 训练能力。
5. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height001 + trunklr1e7 短跑

目标：在 `2e-7` 和 `5e-8` 之间取中间 trunk LR，验证是否能同时保住 remote-only 高度和更快 pointmap 收敛。

实验：

```text
p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_e3_b10_4gpu
```

配置与 `height001 trunklr5e8` 相同，只改：

```text
train_params.lr=1e-7
train_params.min_lr=1e-9
4 GPUs, batch=10, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0585, rs_projection_aux_loss=0.1672
epoch2 val rs_pointmap_loss=0.0510, rs_projection_aux_loss=0.1283
epoch3 val rs_pointmap_loss=0.0496, rs_projection_aux_loss=0.1249
epoch3 val high20 rel_height pred/gt = 0.1517 / 0.1510
train rs_height_loss_weighted≈0.291-0.292
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 final | 0.8928 / 0.0471 | 0.9492 | 0.4376 / 0.1139 |
| height001 trunklr5e8 final | 0.8967 / 0.0477 | 0.9543 | 0.4439 / 0.1140 |
| height001 trunklr1e7 final | 0.8953 / 0.0476 | 0.9526 | 0.4420 / 0.1139 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 final | 0.0467 | 0.0497 | 95.33 | 0.2862 | 10.1588 | 0.0497 | 0.0506 | 0.0507 |
| height001 trunklr5e8 final | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.2399 | 0.0489 | 0.0499 | 0.0501 |
| height001 trunklr1e7 final | 0.0472 | 0.0494 | 94.33 | 0.2903 | 10.2080 | 0.0494 | 0.0502 | 0.0504 |

结论：

1. `trunklr1e7` 的 pointmap 验证收敛比 `trunklr5e8` 快：final val `0.0496` vs `0.0562`，但仍明显慢于 `height001/2e-7` 的 `0.0390`。
2. remote-only 高度位于中间：`0.8953` 高于 `height001` 的 `0.8928`，低于 `trunklr5e8` 的 `0.8967`。
3. benchmark 没有超过 `trunklr5e8`：`joint_point=0.0494`、AUC5 `94.33` 均更差，虽然 `joint_global/RS MAE` 略好。
4. 因此 `1e-7` 不是当前最佳折中；下一步优先延长 `5e-8` 训练到 6 epoch，看低 LR 是否能在保持高度的同时补足 pointmap 收敛。
5. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height001 + trunklr5e8 e6

目标：复核 `5e-8` trunk LR 加长到 6 epoch 后，是否能在保住 remote-only 高度的同时补足 e3 的 pointmap 收敛不足。

输出目录：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_b10_4gpu
```

核心配置：

```text
warmstart = p7_diagnostic_oldp7_trunk_p5b_remote_head/checkpoint-final.pth
remote private P5B point head frozen
projection_aux_source=tokens
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_H=0.001
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
train_params.lr=5e-8
train_params.min_lr=5e-10
EPOCHS=6, BATCH_SIZE=10, 4 GPUs, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0615, aux=0.1631, high20 pred/gt=0.1888/0.1510
epoch2 val rs_pointmap_loss=0.0563, aux=0.1281, high20 pred/gt=0.1549/0.1510
epoch3 val rs_pointmap_loss=0.0527, aux=0.1286, high20 pred/gt=0.1578/0.1510
epoch4 val rs_pointmap_loss=0.0505, aux=0.1346, high20 pred/gt=0.1576/0.1510
epoch5 val rs_pointmap_loss=0.0497, aux=0.1212, high20 pred/gt=0.1510/0.1510
epoch6 val rs_pointmap_loss=0.0495, aux=0.1184, high20 pred/gt=0.1516/0.1510
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 trunklr5e8 e3 | 0.8967 / 0.0477 | 0.9543 | 0.4439 / 0.1140 |
| height001 trunklr1e7 e3 | 0.8953 / 0.0476 | 0.9526 | 0.4420 / 0.1139 |
| height001 trunklr5e8 e6 | 0.8953 / 0.0476 | 0.9525 | 0.4423 / 0.1140 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 trunklr5e8 e3 | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.2399 | 0.0489 | 0.0499 | 0.0501 |
| height001 trunklr1e7 e3 | 0.0472 | 0.0494 | 94.33 | 0.2903 | 10.2080 | 0.0494 | 0.0502 | 0.0504 |
| height001 trunklr5e8 e6 | 0.0474 | 0.0495 | 94.67 | 0.2912 | 10.2020 | 0.0495 | 0.0504 | 0.0505 |

结论：

1. 加长到 6 epoch 确实把 val `rs_pointmap_loss` 从 e3 的 `0.0562` 降到 `0.0495`，aux 高区均值也从 e3 的 `0.1497/0.1510` 稳到 `0.1516/0.1510`。
2. 但跨场景 benchmark 未提升：`joint_point=0.0495`、AUC5 `94.67`，比 e3 的 `0.0489/95.33` 更差。
3. remote-only 高度也轻微回落：same z mean `0.8967 -> 0.8953`，接近 `trunklr1e7`，说明低 LR 长训仍会慢慢改变 remote 几何。
4. 因此不要继续单纯延长 `5e-8` 或提高 height 权重。下一步应转向更直接的 teacher-style z/point anchor，约束 remote point head 的 absolute geometry；MoGE/edge/gradient prior 更适合解决局部纹理/边界，不是当前全局 z 漂移的第一优先级。
5. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height001 + zdist2 + trunklr5e8

目标：验证一个非常轻量的 remote pointmap z 分布约束是否能压住高区域 remote 点云飞掉。该约束只比较有效像素上的 normalized z mean/std：

```text
loss_zdist = |mean(pred_z)-mean(gt_z)| + |std(pred_z)-std(gt_z)|
LAMBDA_REMOTE_Z_DIST=2.0
```

输出目录：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_e3_b10_4gpu
```

核心配置：

```text
warmstart = p7_diagnostic_oldp7_trunk_p5b_remote_head/checkpoint-final.pth
remote private P5B point head frozen
projection_aux_source=tokens
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_H=0.001
LAMBDA_REMOTE_Z_DIST=2.0
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
train_params.lr=5e-8
train_params.min_lr=5e-10
EPOCHS=3, BATCH_SIZE=10, 4 GPUs, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0610, zdist=0.0468, pred/gt z mean=0.6643/0.6259, pred/gt z std=0.1283/0.1220, aux=0.1587, high20 pred/gt=0.1895/0.1510
epoch2 val rs_pointmap_loss=0.0555, zdist=0.0421, pred/gt z mean=0.6593/0.6259, pred/gt z std=0.1277/0.1220, aux=0.1275, high20 pred/gt=0.1434/0.1510
epoch3 val rs_pointmap_loss=0.0545, zdist=0.0411, pred/gt z mean=0.6583/0.6259, pred/gt z std=0.1276/0.1220, aux=0.1233, high20 pred/gt=0.1511/0.1510
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 trunklr5e8 e3 | 0.8967 / 0.0477 | 0.9543 | 0.4439 / 0.1140 |
| height001 trunklr5e8 e6 | 0.8953 / 0.0476 | 0.9525 | 0.4423 / 0.1140 |
| height001 zdist2 trunklr5e8 | 0.8965 / 0.0476 | 0.9538 | 0.4436 / 0.1139 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 trunklr5e8 e3 | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.2399 | 0.0489 | 0.0499 | 0.0501 |
| height001 trunklr5e8 e6 | 0.0474 | 0.0495 | 94.67 | 0.2912 | 10.2020 | 0.0495 | 0.0504 | 0.0505 |
| height001 zdist2 trunklr5e8 | 0.0475 | 0.0492 | 95.33 | 0.2913 | 10.2150 | 0.0492 | 0.0502 | 0.0505 |

结论：

1. `zdist2` 让 aux 日志明显更好看：epoch3 high20 rel-height 几乎完全对齐到 `0.1511/0.1510`。
2. 但实际导出的 448 remote-only PLY 几乎等同 `trunklr5e8 e3`，same z mean `0.8967 -> 0.8965`，blank 也仍然塌到 `0.4436`。
3. benchmark 未超过 e3：`joint_point=0.0492` 高于 e3 的 `0.0489`，RS-only MAE `10.2150` 也未超过 e3 的 `10.2399` 到足以抵消 point/AUC 差距。
4. 因此 mean/std 型全局 z 分布约束太粗，不足以修复高区域 remote 点云飞掉。下一步不应继续放大该项，而应尝试更直接、更局部的 teacher-style z/point anchor；如果后续主要问题转为局部条纹/边缘，再考虑 MoGE edge/gradient prior。
5. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。

## 2026-06-06 diagnostic P5B-head + height001 + zhigh2q80 + trunklr5e8

目标：验证 “只在 GT z top 20% 区域额外监督 pointmap z” 是否比全局 mean/std 更直接地修复高区域 remote 点云飞掉。

新增 loss：

```text
LAMBDA_REMOTE_HIGH_Z=2.0
REMOTE_HIGH_Z_QUANTILE=0.8
REMOTE_HIGH_Z_MIN_PIXELS=16
```

输出目录：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_e3_b10_4gpu
```

核心配置：

```text
warmstart = p7_diagnostic_oldp7_trunk_p5b_remote_head/checkpoint-final.pth
remote private P5B point head frozen
projection_aux_source=tokens
LAMBDA_REMOTE_PM=4.0
LAMBDA_REMOTE_H=0.001
LAMBDA_REMOTE_HIGH_Z=2.0
LAMBDA_PROJ_REL_HEIGHT=0.35
LAMBDA_PROJ_OFFSET=0.75
LAMBDA_PROJ_GLOBAL_SLOPE=0.05
train_params.lr=5e-8
train_params.min_lr=5e-10
EPOCHS=3, BATCH_SIZE=10, 4 GPUs, max mem≈93.4GB/GPU
```

训练/验证：

```text
epoch1 val rs_pointmap_loss=0.0609, high_z=0.0506, pred/gt high_z mean=0.8351/0.7929, aux=0.1727, high20 pred/gt=0.1975/0.1510
epoch2 val rs_pointmap_loss=0.0556, high_z=0.0458, pred/gt high_z mean=0.8293/0.7929, aux=0.1254, high20 pred/gt=0.1474/0.1510
epoch3 val rs_pointmap_loss=0.0545, high_z=0.0447, pred/gt high_z mean=0.8280/0.7929, aux=0.1258, high20 pred/gt=0.1543/0.1510
```

导出结果：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final_mixed
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final_remote_only
```

448 remote-only PLY：

| model | same z mean/std | same q95 | blank z mean/std |
|---|---:|---:|---:|
| diagnostic aux-only best | 0.8980 / 0.0479 | 0.9557 | 0.4461 / 0.1137 |
| height001 trunklr5e8 e3 | 0.8967 / 0.0477 | 0.9543 | 0.4439 / 0.1140 |
| height001 zdist2 trunklr5e8 | 0.8965 / 0.0476 | 0.9538 | 0.4436 / 0.1139 |
| height001 zhigh2q80 trunklr5e8 | 0.8965 / 0.0476 | 0.9537 | 0.4440 / 0.1140 |

New York 10-scene mini benchmark：

| model | joint global | joint point | AUC5 | ray | RS MAE | same | blank | shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic aux-only best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.2691 | 0.0485 | 0.0494 | 0.0497 |
| height001 trunklr5e8 e3 | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.2399 | 0.0489 | 0.0499 | 0.0501 |
| height001 zdist2 trunklr5e8 | 0.0475 | 0.0492 | 95.33 | 0.2913 | 10.2150 | 0.0492 | 0.0502 | 0.0505 |
| height001 zhigh2q80 trunklr5e8 | 0.0475 | 0.0492 | 94.33 | 0.2895 | 10.2083 | 0.0492 | 0.0503 | 0.0505 |

结论：

1. high-z loss 本身可优化：val `0.0506 -> 0.0447`，但 high-z pred mean 仍高于 GT：`0.8280/0.7929`。
2. 实际 remote-only PLY 与 `zdist2` 和 `trunklr5e8 e3` 基本相同，未改善高区域点云飞掉。
3. benchmark 未超过 e3：`joint_point=0.0492`，AUC5 反而降到 `94.33`。
4. 这说明问题不是 pointmap loss 在高 z 子集上的平均权重不足；更可能是 trunk/features 对 frozen P5B remote head 的输入分布发生了漂移。下一步应优先考虑 teacher-style output/feature anchor 或更强的 remote-head 输入保持，而不是继续对 GT pointmap 做子集重加权。
5. `checkpoint-best.pth` 和 `checkpoint-last.pth` 已删除，仅保留 4.9GB `checkpoint-final.pth`。
