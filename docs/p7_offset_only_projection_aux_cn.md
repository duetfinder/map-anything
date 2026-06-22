# P7 Offset-Only Projection Aux 实验

## 目标

当前 `height + slope + dir` 的 global 投影分解在可视化上容易出现方向/尺度耦合：方向可能看似合理，但 height 幅度一错，`grid_global` 的倾角会被明显放大。新实验先去掉显式 height/global 物理分解，只学习一个更直接的 dense `offset_xy` 场。

目标是验证：模型能否稳定学出 remote 图像中“投影位置到真实点云位置”的逐像素 xy 偏移。

## 结构

- 基础模型：VGGT P7-P5B remote private point head。
- aux 输入：remote shared tokens + remote RGB + 坐标。
- aux 输出仍沿用现有三通道 pixel head：
  - `remote_projection_offset_xy_pred` 是本实验主输出；
  - `rel_height/global_dir/global_slope` 仍会被模型算出，但 loss 权重置零，不作为训练目标。

## Loss

主任务保留：

```text
L_remote_pointmap = 4.0 * rs_pointmap_loss
```

offset 监督：

```text
offset_gt = projected_xy + center_xy - gt_pointmap_xy
offset_gt_loss = offset_gt / PROJ_OFFSET_SCALE
L_offset = |offset_pred - offset_gt_loss|
```

其中当前 `PROJ_OFFSET_SCALE=32`，使 Chicago/NewYork 常见 offset 落在较稳定的归一化 loss 区间。

point head detach 一致性：

```text
offset_from_point = projected_xy + center_xy - stopgrad(point_head_xy)
offset_from_point_loss = offset_from_point / PROJ_OFFSET_SCALE
L_offset_to_point_detach = |offset_pred - offset_from_point_loss|
```

这个 loss 只训练 aux offset，不反向拖坏 point head。

offset 反投影到 GT 的尺度无关点云损失：

```text
recon_xy = projected_xy + center_xy - offset_pred * PROJ_OFFSET_SCALE
recon_z = projected_z
L_reconstruct_offset_to_gt = normalized_point_loss(recon, gt_pointmap)
```

## 第一阶段配置

```text
loss: configs/loss/vggt_loss_rs_joint_p7_remote_head_projection_offset_only.yaml
train_params: configs/train_params/vggt_p7_p5b_offset_only_token_aux.yaml
```

训练策略：

- warmstart 当前较好的 P7 remote point checkpoint；
- 排除旧 `remote_projection_aux_` 权重，重新初始化 offset aux；
- 先跑短实验确认 loss 下降和可视化方向；
- 若有效，再全量城市长 epoch 微调。

## 观测指标

- `rs_projection_offset_loss`
- `rs_projection_offset_to_point_detach_loss`
- `rs_projection_reconstruct_offset_to_gt_loss`
- `rs_projection_offset_pred_loss_abs_mean` vs `rs_projection_offset_gt_loss_abs_mean`
- Seattle 493 / NewYork 461 / Chicago 51 可视化：
  - remote point head 点云；
  - `aux_offset_remote.ply`；
  - offset field / height compare 图。

## 当前结论

### 2026-06-12 smoke

已完成 Chicago-only 1 epoch smoke：

- 配置：单卡，`BATCH_SIZE=4`，`EPOCHS=1`。
- 结果：训练可正常前向/反向，新加入的 `rs_projection_offset_to_point_detach_*` 日志正常出现，无 NaN。
- 显存：约 42.8GB。
- 现象：
  - `rs_projection_offset_loss` 约 0.064-0.069，1 epoch 内尚未明显下降；
  - `rs_projection_reconstruct_offset_to_gt_loss` 约 0.0045-0.005，权重后较小；
  - `rs_projection_offset_to_point_detach_loss` 约 1.5，明显大于直接 GT offset loss。

`offset_to_point_detach` 的 target 来自当前 point head 输出，和 GT projected base 的空间尺度可能仍存在错配风险。为了避免它压过直接 GT offset 监督，正式短训先把权重从 0.25 降到 0.05；如果日志显示仍然主导或拖坏 point head，应跑一个 `PROJ_OFFSET_TO_POINT_DETACH=0.0` 对照。

### 2026-06-12 B16 OOM

第一次正式短训使用 2 GPU、`BATCH_SIZE=16` 时首轮 OOM。这里脚本里的 `BATCH_SIZE` 对应 `train_params.max_num_of_imgs_per_gpu`，是每卡输入 view 数，不是全局 batch。两张 95GB 卡在 B16 下每卡接近满显存并额外申请 1.67GB 失败。

下一步改用 2 GPU、每卡 B8：

```text
OUTPUT_DIR=p7_offset_only_tokenaux_from_stagea_e6_b8_2gpu
PROJ_OFFSET_TO_POINT_DETACH=0.05
```

### 2026-06-12 B8 formal short run

已完成：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_only_tokenaux_from_stagea_e6_b8_2gpu
checkpoint: checkpoint-final.pth
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 8 per GPU
训练时长: 0:15:36
峰值显存: 约 74GB / GPU
```

已删除 `checkpoint-last.pth`，仅保留 `checkpoint-final.pth` 节省空间。

可视化输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_offset_only_tokenaux_from_stagea_e6_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_offset_only_tokenaux_from_stagea_e6_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/chicago_51/vggt_p7_offset_only_tokenaux_from_stagea_e6_final_train_chicago51_n6
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/chicago_51/vggt_p7_offset_only_tokenaux_from_stagea_e6_final_train_chicago51_n12
```

每个目录里重点看：

- `mapanything_pointcloud_same.ply`：mixed 点云；
- `mapanything_pointcloud_same_remote.ply`：point head 的 remote-only 点云；
- `mapanything_pointcloud_same_aux_offset_remote.ply`：aux offset 反投影点云；
- `mapanything_pointcloud_same_aux_reconstruction_summary.json`：aux 反投影统计。

Chicago 51 的 `n6/n12` 导出额外使用：

```text
--projection_aux_gt_remote_dir /root/autodl-tmp/traindata/Crossview_rs/chicago__location_51/Google_Satellite
--projection_aux_use_gt_projection_base
--projection_aux_xyz_align_mode gt_pointmap_unit_xy_zrange_flipz
```

这个设置用于隔离检查 aux offset 本身，不让 point head 预测的 projection base 混入误差。

#### 关键日志

第 0 个 epoch 末：

```text
rs_projection_offset_pred_norm_mean: 0.0897
rs_projection_offset_gt_norm_mean: 1.3949
rs_projection_offset_low80_gt_mean: 0.0139
rs_projection_offset_low80_pred_mean: 0.0918
rs_projection_offset_loss: 0.0730
rs_projection_offset_to_point_detach_target_norm_mean: 2.2820
rs_projection_offset_to_point_detach_loss_weighted: 0.0738
```

第 5 个 epoch 末：

```text
rs_projection_offset_pred_norm_mean: 0.0890
rs_projection_offset_gt_norm_mean: 1.2869
rs_projection_offset_low80_gt_mean: 0.0139
rs_projection_offset_low80_pred_mean: 0.0912
rs_projection_offset_loss: 0.0713
rs_projection_offset_to_point_detach_target_norm_mean: 2.3076
rs_projection_offset_to_point_detach_loss_weighted: 0.0747
```

#### 结论

这次 offset-only 训练可以稳定运行，但没有证明 aux offset 已学到有效投影机制。主要问题：

1. `offset_pred_norm_mean` 基本固定在 0.09 左右，没有随 GT offset 的高低区域自适应变化。
2. 低 offset/background 区域持续过预测，`low80_pred_mean` 约 0.09，而 `low80_gt_mean` 约 0.01。
3. 高 offset 区域经常欠预测，说明输出更像近似常量场，而不是有效 dense offset。
4. `offset_to_point_detach` 的 target norm 约 2.3，明显大于直接 GT offset 的 typical scale；即使权重降到 0.05，weighted loss 仍约 0.07，和直接 offset loss 同量级，可能反向干扰 aux offset 学习。
5. Chicago 51 训练场景内，height comparison 的原始 normalized MAE 约 0.105，affine 后 MAE 约 0.015，说明形状仍主要依赖仿射校正，原始尺度/幅度没有稳定学好。

因此这个 checkpoint 不建议作为长训基础，也不建议跑全量长 epoch。

### 下一步控制实验

优先跑一个更干净的 offset-only 对照：

- 关闭 `offset_to_point_detach`，避免 point head 尺度错配干扰 aux；
- 从 epoch 0 开始启用更强 low-overpred；
- 保留直接 GT offset loss、balanced/bucket mean 和尺度无关 reconstruct loss；
- 如果 aux offset 仍然输出近似常量场，再判断问题主要在 head/任务形式，而不是 point-detach 冲突。

建议输出名：

```text
p7_offset_only_tokenaux_nopointdetach_low1_e8_b8_2gpu
```

### 2026-06-12 nopointdetach + low-overpred 修正

先跑了一个 `nopointdetach_low1_e8`，但中途发现 low/background 惩罚实现有问题：当
`q0=0` 时，旧实现使用 `< q0` 选择低区间，会把大量 offset 正好为 0 的背景像素排除掉，
导致 `rs_projection_offset_low_overpred_loss` 基本不起作用。已修正为 `<= q0`。

修正后重新跑短训：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_only_tokenaux_nopointdetach_fixedlow_e4_b8_2gpu
checkpoint: checkpoint-final.pth
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 8 per GPU
EPOCHS: 4
PROJ_OFFSET_TO_POINT_DETACH: 0.0
PROJ_OFFSET_LOW_OVERPRED_WEIGHT: 1.0
训练时长: 0:10:26
峰值显存: 约 74GB / GPU
```

已删除 `checkpoint-last.pth`，仅保留 final。

可视化输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_offset_only_tokenaux_nopointdetach_fixedlow_e4_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_offset_only_tokenaux_nopointdetach_fixedlow_e4_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/chicago_51/vggt_p7_offset_only_tokenaux_nopointdetach_fixedlow_e4_final_train_chicago51_n12
```

Chicago 51 的输入由训练集内 `location_51` 的 Google remote 图和 n12 普通视角列表构成，普通视角使用软链接目录：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/train_scene_inputs/chicago_51/n12_links
```

同时给 `scripts/export_pointcloud_ply.py` 补了可复现采样参数：

```text
--max_images
--random_sample
--random_seed
```

#### 关键日志

第 3 个 epoch 末：

```text
rs_pointmap_loss: 0.1213
rs_projection_aux_loss: 0.2621
rs_projection_offset_pred_norm_mean: 0.0898
rs_projection_offset_gt_norm_mean: 1.3802
rs_projection_offset_high20_gt_mean: 0.1681
rs_projection_offset_high20_pred_mean: 0.0817
rs_projection_offset_low80_gt_mean: 0.0118
rs_projection_offset_low80_pred_mean: 0.0922
rs_projection_offset_low_overpred_loss_weighted: 0.0946
rs_projection_offset_loss: 0.0729
```

修正后 low-overpred 确实生效了，weighted loss 已经和直接 offset loss 同量级；但它仍没有把背景/低 offset 区域压下来。

Chicago 51 训练内导出 summary：

```text
offset_pred_norm_mean: 0.0934
offset_world_norm_mean: 2.9903
height rel_height_norm_mae: 0.1056
height rel_height_norm_affine_mae: 0.0143
```

#### 结论

这次实验进一步说明：直接 dense `offset_xy` 的想法在几何定义上是合理的，但当前独立 aux token head
仍然倾向输出近常量 offset 场。关闭 point-detach 一致性、修正低区间惩罚后，现象没有根本变化。

因此现在不建议把这个配置直接扩到全量长 epoch。更合理的下一步不是继续堆 epoch，而是改训练形式：

1. 把 offset 作为 point head 的残差/蒸馏目标，而不是完全独立的 aux 头；这样复用 point head 已经学到的 remote 几何。
2. 或者用 point head / depth head 的 decoder 权重初始化 offset head，避免 aux pixel head 从零开始学 dense 几何。
3. 增加直接的 `xy` 分量监督和方向监督，而不仅看 offset magnitude/bucket；当前 magnitude loss 对常量场的约束不够强。
4. 在确认 aux 不会拖坏 point head 前，所有一致性 loss 都应保持 `stopgrad(point_head)`，不能反向更新 point head。

### 2026-06-12: point-head distill / decoder-init / residual 探针

#### centered point-head distill

短训配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_token_centerdistill_w010_e4_b8_1gpu_rerun
训练数据: chicago + newyork
GPU: 1 x RTX PRO 6000 95GB
BATCH_SIZE: 8
EPOCHS: 4 planned, early stop at epoch 0
PROJ_OFFSET_TO_POINT_CENTERED_DETACH: 0.10
```

实现上先发现一个 NaN bug：invalid 像素处的 `point_offset_for_loss` 可能是 NaN，直接乘 mask 仍会传播 NaN。
已改成 `torch.where(mask, value, 0)` 后再求 mean/RMS。

修正后训练可运行，但早期日志仍显示 aux 输出近常量：

```text
rs_projection_offset_pred_norm_mean: 约 0.09
rs_projection_offset_low80_gt_mean: 约 0.01
rs_projection_offset_low80_pred_mean: 约 0.09
rs_projection_offset_to_point_centered_detach_target_rms_mean: 约 2.4
rs_projection_offset_to_point_centered_detach_pred_rms_mean: 约 0.035
```

结论：centered distill 只约束形状归一化后的相似性，不能阻止 aux head 保持低方差/近常量输出。

#### point/depth decoder 初始化 offset head

短训配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_dptinit_frompoint_e4_b4_1gpu_rerun
训练数据: chicago + newyork
GPU: 1 x RTX PRO 6000 95GB
BATCH_SIZE: 4
EPOCHS: 4 planned, early stop at epoch 0
REMOTE_PROJECTION_AUX_SOURCE: dpt_init
```

现象：

```text
rs_projection_offset_pred_norm_mean: 约 0.17-0.18
rs_projection_offset_low80_gt_mean: 约 0.02-0.03
rs_projection_offset_low80_pred_mean: 约 0.17
```

结论：直接复用 point head decoder 的输出尺度和当前 offset loss space 不匹配，初始 over-pred 比 token aux 更严重。

#### point-base residual offset

新增 loss 开关：

```text
loss.remote_projection_offset_residual_from_point_detach=true
```

目标是把 aux 输出解释为 residual：

```text
effective_offset = detached_point_base_offset + residual_pred
loss = |effective_offset - gt_offset|
```

第一版直接用 raw `pred['pts3d']` 和 GT 投影坐标相减作为 point base，发现监督域错误：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_residual_pointbase_token_e4_b8_2gpu
rs_projection_offset_point_base_loss_norm_mean: 约 2.2
rs_projection_offset_gt_loss_abs_mean: 约 0.01-0.03
```

原因是 remote pointmap loss 本身是尺度归一化比较的，raw point head 输出不能直接和 GT 投影坐标相减。
该探针已中断并删除中间权重。

第二版改为在 pointmap 归一化坐标中计算 point base：

```text
point_base = projected_xy / gt_norm_factor - pred_xy.detach() / pred_norm_factor
gt_offset = offset_xy / gt_norm_factor
world_reconstruct_offset = effective_offset * gt_norm_factor
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_offset_residual_pointbase_normbase_token_e4_b8_2gpu
```

step 0/20 观察：

```text
rs_projection_offset_point_base_loss_norm_mean: 约 0.27-0.31
rs_projection_offset_gt_loss_abs_mean: 约 0.001-0.004
rs_projection_offset_low80_pred_mean: 约 0.27-0.31
rs_projection_offset_low80_gt_mean: 约 0.0003-0.0036
```

4 epoch 完整结果：

```text
rs_projection_offset_effective_loss_norm_mean: 约 0.33
rs_projection_offset_low80_gt_mean: 约 0.0016
rs_projection_offset_low80_pred_mean: 约 0.33
rs_projection_offset_point_base_loss_norm_mean: 约 0.31
```

结论：norm-base residual 仍失败。虽然比 raw-base 的坐标域正确，但 point head 反推出的 pixel projection offset
仍比真正 GT offset 大两个数量级左右，aux residual 必须先学会抵消一个很大的错误 base，优化反而更难。
因此该方向不建议扩到长 epoch。

#### point head 直接投影 offset 监督

新增 loss 开关：

```text
loss.remote_projection_point_offset_to_gt_loss_weight=1.0
```

训练配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_pointhead_pointoffset_gt_e4_b8_2gpu
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 8/GPU
EPOCHS: 4
LAMBDA_PROJ_OFFSET: 0.0
PROJ_POINT_OFFSET_TO_GT: 1.0
```

这个实验不训练独立 offset aux head，而是直接从 point head 输出反推出 normalized projection offset：

```text
point_offset = projected_xy / gt_norm_factor - pred_xy / pred_norm_factor
gt_offset = offset_xy / gt_norm_factor
loss = |point_offset - gt_offset|
```

关键日志：

```text
step 0:
  rs_projection_point_offset_to_gt_pred_norm_mean: 0.2660
  rs_projection_point_offset_to_gt_gt_norm_mean: 0.0068
  rs_projection_point_offset_to_gt_loss: 0.1451

epoch 3:
  rs_pointmap_loss: 约 0.02
  rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.32
  rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004
  rs_projection_point_offset_to_gt_loss: 约 0.20
  rs_projection_point_offset_to_gt_low80_pred_mean: 约 0.31
  rs_projection_point_offset_to_gt_low80_gt_mean: 约 0.0016
```

结论：pointmap loss 可以快速降到较低，但从 point head 几何反推出的 projection offset 没有变正确。
这说明当前 normalized pointmap loss 的收敛，并不会自动保证 remote 投影机制正确；单独加 weight=1 的
point-offset 监督也没有明显拉动 point head。下一步需要做强约束/高 LR 诊断，确认是梯度权重不足，还是这个
offset 监督表达本身和 pointmap 的尺度/gauge 存在优化冲突。

#### point head 直接投影 offset 强约束诊断

训练配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_pointhead_pointoffset_gt_w10_headlr1e4_e4_b9_2gpu
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 4
train_params: vggt_p7_p5b_offset_only_pointhead_highlr
PROJ_POINT_OFFSET_TO_GT: 10.0
remote_point_head lr: 1e-4
aggregator frame/global lr: 1e-6
checkpoint: checkpoint-final.pth
```

核心观察：

```text
step 0:
  rs_projection_point_offset_to_gt_pred_norm_mean: 0.2660
  rs_projection_point_offset_to_gt_gt_norm_mean: 0.0068
  rs_projection_point_offset_to_gt_loss_weighted: 1.4507
  rs_pointmap_loss: 0.1098

epoch 1 前段:
  rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.20
  rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004
  rs_pointmap_loss: 约 0.15

epoch 3 末:
  rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.15
  rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004
  rs_projection_point_offset_to_gt_loss: 约 0.093
  rs_projection_point_offset_to_gt_loss_weighted: 约 0.93
  rs_pointmap_loss: 约 0.20
```

结论：强权重和高 remote point head LR 能明显拉动 point-derived projection offset，
从约 0.30 降到约 0.15；这证明该监督不是完全无梯度，也不是不可优化。
但它仍比 GT offset 大约两个数量级，并且 `rs_pointmap_loss` 从正常短训可到的约 0.02 明显退化到约 0.20。
因此当前“直接把 point head 反推出的 offset 对齐 GT”的方式会和 pointmap 重建产生强拉扯，不适合作为最终训练目标。

下一步不应继续简单加权重或加 epoch。更合理的方向是：

1. 把 point head 作为主目标保持稳定，projection offset 作为低权重/后期 warmup 的辅助诊断。
2. 或者改为 `stopgrad(point_head)` 的 offset 蒸馏，让 aux head 学 point head 的几何残差形状，不反向破坏 point head。
3. 对 offset 使用更局部、尺度更稳定的表达，例如局部方向/梯度/高区增量，而不是直接约束全局 normalized offset。

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_pointoffset_gt_w10_headlr1e4_e4_final

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_pointoffset_gt_w10_headlr1e4_e4_final
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
```

#### point head 直接投影 offset 中等权重保护 pointmap

训练配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_pointhead_pointoffset_gt_w3_pm8_headlr1e4_e4_b9_2gpu
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 4
train_params: vggt_p7_p5b_offset_only_pointhead_highlr
PROJ_POINT_OFFSET_TO_GT: 3.0
LAMBDA_REMOTE_PM: 8.0
```

结果：

```text
rs_pointmap_loss: 约 0.025
rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.31
rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004
rs_projection_point_offset_to_gt_loss: 约 0.19
```

结论：把 pointmap 权重提高并把 point-offset 权重降到 3，可以保住 pointmap 收敛，
但 point-derived projection offset 基本没有被拉动。和 weight=10 的结果放在一起看，
目前不是简单调一个静态权重就能解决：强权重能动 offset 但破坏 pointmap，中等权重能保 pointmap 但 offset 被忽略。

#### point head 直接投影 offset warmup/ramp 诊断

新增调度参数：

```text
loss.remote_projection_point_offset_to_gt_start_epoch
loss.remote_projection_point_offset_to_gt_ramp_epochs
```

训练配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_pointhead_pointoffset_gt_w10_pm8_pure_start2_ramp2_e6_b9_2gpu
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 6
train_params: vggt_p7_p5b_offset_only_pointhead_highlr
LAMBDA_REMOTE_PM: 8.0
PROJ_POINT_OFFSET_TO_GT: 10.0
PROJ_POINT_OFFSET_TO_GT_START_EPOCH: 2.0
PROJ_POINT_OFFSET_TO_GT_RAMP_EPOCHS: 2.0
其他 projection aux 子损失: 0.0
```

目的：先用 2 个 epoch 让 remote pointmap 稳定，再线性打开 point-offset-to-GT，
判断前面 weight=10 的冲突是否主要来自训练初期过早施加约束。

启动检查：

```text
epoch 0 step 0:
  rs_pointmap_loss: 0.1098
  rs_projection_aux_loss: 0.0000
  rs_projection_point_offset_to_gt_effective_weight: 0.0000
```

最终日志：

```text
epoch 1.9:
  rs_pointmap_loss: 约 0.0275
  rs_projection_aux_loss: 0.0
  rs_projection_point_offset_to_gt_effective_weight: 0.0

epoch 3.9:
  rs_pointmap_loss: 约 0.0289
  rs_projection_point_offset_to_gt_effective_weight: 约 9.5
  rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.31
  rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004

epoch 5.9:
  rs_pointmap_loss: 约 0.0272
  rs_projection_point_offset_to_gt_effective_weight: 10.0
  rs_projection_point_offset_to_gt_pred_norm_mean: 约 0.326
  rs_projection_point_offset_to_gt_gt_norm_mean: 约 0.004
  rs_projection_point_offset_to_gt_loss: 约 0.201
```

结论：warmup/ramp 能避免 weight=10 直接训练时 pointmap 明显恶化，最终
`rs_pointmap_loss` 仍保持在约 0.02-0.03；但 point-derived projection offset 基本没有被拉到 GT，
`pred_norm` 仍比 GT 大两个数量级左右。也就是说，问题不是单纯“太早加 offset loss”，而是当前
point head 表达和 normalized projection offset 目标之间仍存在很强的 gauge/参数化不匹配。

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_pointoffset_gt_w10_pm8_pure_start2_ramp2_e6_final

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_pointoffset_gt_w10_pm8_pure_start2_ramp2_e6_final
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
```

下一步不建议继续沿着“直接约束 point head 反推出 offset = GT offset”加权或加 epoch。
更合理的方向是让 aux offset 作为 point head 的残差校正量，直接对校正后的 remote pointmap 做尺度无关点云损失；
这样优化目标仍然是最终 remote 点云，而不是要求 point head 本身的内部规范必须等价于显式投影 offset。

#### aux offset 作为 point head 的 normalized xy 残差

新增 loss 开关：

```text
loss.remote_projection_point_residual_offset_to_gt_loss_weight=10.0
```

这个实验不再要求 point head 自身反推出的 projection offset 等于 GT projection offset，而是把
`remote_projection_offset_xy_pred` 解释成 point head 输出到 GT pointmap 的 normalized xy residual：

```text
point_norm = normalize(pred_pts3d.detach(), pointmap_norm_mode)
gt_norm = normalize(gt_remote_pointmap, pointmap_norm_mode)
target_residual_xy = gt_norm.xy - point_norm.xy
corrected_xy = point_norm.xy + remote_projection_offset_xy_pred
loss = |corrected_xy - gt_norm.xy|
```

这样 residual head 只负责修正最终 remote 点云，不需要学习旧 projection offset 的规范。
`pred_pts3d.detach()` 用来避免差的 aux residual 反向拖坏 point head；point head 仍由正常
`rs_pointmap_loss` 训练。

快速验证配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_e4_b9_2gpu
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 4
train_params: vggt_p7_p5b_offset_only_pointhead_highlr
WARMSTART_CKPT: p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b8_2gpu/checkpoint-best.pth
LAMBDA_REMOTE_PM: 8.0
PROJ_POINT_RESIDUAL_OFFSET_TO_GT: 10.0
其他 projection aux 子损失: 0.0
checkpoint: checkpoint-final.pth
```

关键日志：

```text
epoch 0 step 0:
  rs_pointmap_loss: 0.1098
  rs_projection_point_residual_offset_to_gt_loss: 0.1623
  base_mae_norm_mean: 0.2662
  corrected_mae_norm_mean: 0.2789

epoch 0.9:
  rs_pointmap_loss: 0.0224
  residual loss: 0.0871
  base_mae_norm_mean: 0.2684
  corrected_mae_norm_mean: 0.1374

epoch 2.9:
  rs_pointmap_loss: 0.0214
  residual loss: 0.0625
  base_mae_norm_mean: 0.2829
  corrected_mae_norm_mean: 0.1035

epoch 3.9:
  rs_pointmap_loss: 0.0171
  residual loss: 0.0502
  pred_norm_mean: 0.2922
  gt_norm_mean: 0.2875
  base_mae_norm_mean: 0.2875
  corrected_mae_norm_mean: 0.0797
```

结论：这是目前 offset-only 系列里第一个明确可训练的 aux offset 目标。它把训练内 normalized xy
误差从约 `0.29-0.31` 降到约 `0.08-0.09`，同时 pointmap loss 没有被拖坏。它比“point head
直接投影 offset = GT offset”更合理，因为优化目标直接对齐最终 remote pointmap 的尺度无关 xy 形状。

但这还不是最终结论：当前只是 2-city / 4 epoch 快速验证，且 residual 只修正 xy，z 仍来自 point head。
下一步必须看 PLY 和 `remote_pointmetric20`，确认 residual 修正不是只在 loss 空间里有效。

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_e4_final

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_e4_final
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
mapanything_pointcloud_same_aux_point_residual_remote.ply
mapanything_pointcloud_same_aux_point_residual_norm_remote.ply
mapanything_pointcloud_same_aux_point_residual_summary.json
```

其中 `*_aux_point_residual_remote.ply` 是把 normalized residual 乘回 point head 当前尺度后的 remote
点云，适合和 `*_same_remote.ply` 做形状对比；`*_aux_point_residual_norm_remote.ply` 是训练 loss
里的 normalized 坐标点云，主要用于诊断 residual 是否在 loss 空间里形状合理。

#### aux residual e12 continuation：保留 aux 权重继续训练

上一节 4 epoch 验证后，继续从该 `checkpoint-final.pth` warm-start 训练到 12 epoch。关键点是显式覆盖：

```text
train_params.warmstart_exclude_prefixes=[]
```

否则默认 warm-start 会排除 `remote_projection_aux_`，导致 aux residual head 被重新初始化，长训结果不可解释。

配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_b9_2gpu
WARMSTART_CKPT: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_e4_b9_2gpu/checkpoint-final.pth
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 12
LAMBDA_REMOTE_PM: 8.0
PROJ_POINT_RESIDUAL_OFFSET_TO_GT: 10.0
其他 projection aux 子损失: 0.0
checkpoint: checkpoint-final.pth
```

关键日志：

```text
epoch 5.9:
  rs_pointmap_loss: 0.0285
  residual loss: 0.0546
  base_mae_norm_mean: 0.3299
  corrected_mae_norm_mean: 0.0855

epoch 8.9:
  rs_pointmap_loss: 0.0285
  residual loss: 0.0437
  base_mae_norm_mean: 0.3179
  corrected_mae_norm_mean: 0.0686

epoch 9.9:
  rs_pointmap_loss: 0.0234
  residual loss: 0.0398
  base_mae_norm_mean: 0.3285
  corrected_mae_norm_mean: 0.0623

epoch 11.9:
  rs_pointmap_loss: 0.0219
  residual loss: 0.0365
  base_mae_norm_mean: 0.3100
  corrected_mae_norm_mean: 0.0573
```

训练结论：继续训练后 residual 目标仍稳定下降，`corrected_mae_norm_mean` 从 4 epoch 的约 `0.080`
进一步到约 `0.057`，同时 `rs_pointmap_loss` 保持在约 `0.022`，没有看到 aux residual 反向拖坏 point head。
这说明“offset 作为 point head 的 normalized xy residual”在训练度量上是可学习且稳定的。

但它仍不是最终 remote 点云结论。这个 residual 只修正 point head 的 `xy`，`z` 仍来自 point head；
如果 remote 点云主要问题是高度/局部起伏，PLY 改善可能有限。下一步判断应优先看
`same_remote.ply` 与 `same_aux_point_residual_remote.ply` 的差异，并跑 `remote_pointmetric20`。

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_final

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_final
```

summary：

```text
Seattle 493:
  norm_factor: 0.651965
  residual_norm_mean: 0.259340
  residual_norm_p95: 0.510448

NewYork 461_1:
  norm_factor: 0.842344
  residual_norm_mean: 0.255598
  residual_norm_p95: 0.524260
```

benchmark 补充：

```text
standard point head metric:
  RS-only rs_point_l1=292.540
  RS-only rs_point_l1_centered=95.380
  RS-only rs_point_l1_scale_aligned=95.362
  RS-only rs_height_mae_affine=9.944
  Joint rs_point_l1=292.129
  Joint rs_point_l1_centered=95.397
  Joint rs_point_l1_scale_aligned=95.348
  Joint rs_height_mae_affine=15.974

aux xy residual metric:
  RS-only rs_point_l1=292.482
  RS-only rs_point_l1_centered=95.228
  RS-only rs_point_l1_scale_aligned=43.629
  Joint rs_point_l1=292.065
  Joint rs_point_l1_centered=95.230
  Joint rs_point_l1_scale_aligned=68.629
```

benchmark 结论：xy residual 几乎不改变 raw/centered 指标，但能明显改善 `rs_point_l1_scale_aligned`。
这说明 residual head 学到的是尺度无关的局部 xy 形状修正；绝对尺度、中心和 z 仍主要由 point head 决定。

#### aux xyz residual：同时修正 normalized xy 和 z

动机：xy-only residual 的 scale-aligned 改善明显，但 z 仍来自 point head。如果 remote 点云主要坏在高度/局部起伏，
需要测试 aux head 是否能直接预测 `GT_norm - point_norm` 的 xyz residual。

实现：

```text
point_base = pred['pts3d'].detach()
point_norm, gt_norm = normalize_pair(point_base, gt_pointmap, mode='aerial_avg_dis')
target_residual_xyz = gt_norm - point_norm
xyz_residual_pred = concat(remote_projection_offset_xy_pred, remote_projection_rel_height_pred)
corrected_xyz = point_norm + xyz_residual_pred
loss = L1(corrected_xyz, gt_norm) over valid remote mask
```

关键点：

- `point_base.detach()`，避免差的 aux residual 反向拖坏 point head。
- 所有旧 projection aux 子损失显式设为 `0.0`，只保留 `rs_pointmap_loss` 和 `projection_point_residual_xyz_to_gt`。
- 从 e12 xy residual final warm-start，并设置 `train_params.warmstart_exclude_prefixes=[]` 保留 aux 权重。

配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_b9_2gpu
WARMSTART_CKPT: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_b9_2gpu/checkpoint-final.pth
训练数据: chicago + newyork
GPU: 2 x RTX PRO 6000 95GB
BATCH_SIZE: 9/GPU
EPOCHS: 6
LAMBDA_REMOTE_PM: 8.0
PROJ_POINT_RESIDUAL_XYZ_TO_GT: 10.0
其他 projection aux 子损失: 0.0
checkpoint: checkpoint-best-slim.pth / checkpoint-final.pth
```

训练结果：

```text
epoch 0.0:
  rs_pointmap_loss: 0.0213
  xyz residual loss: 0.5596
  base_mae_norm_mean: 1.7994
  corrected_mae_norm_mean: 1.6312
  base_z_mae_norm_mean: 1.7698
  corrected_z_mae_norm_mean: 1.6305

epoch 0.9:
  rs_pointmap_loss: 0.0195
  xyz residual loss: 0.0984
  corrected_mae_norm_mean: 0.2166
  corrected_z_mae_norm_mean: 0.1675

best val around epoch 5:
  val rs_pointmap_loss_avg: 0.0205
  val xyz residual loss avg: 0.0546
  val corrected_mae_norm_mean_avg: 0.1245
  val corrected_z_mae_norm_mean_avg: 0.1088

final val epoch 6:
  val rs_pointmap_loss_avg: 0.0200
  val xyz residual loss avg: 0.0656
  val corrected_mae_norm_mean_avg: 0.1571
  val corrected_z_mae_norm_mean_avg: 0.1460
```

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best
```

新增 PLY：

```text
mapanything_pointcloud_same_aux_point_residual_remote.ply
mapanything_pointcloud_same_aux_point_residual_norm_remote.ply
mapanything_pointcloud_same_aux_point_residual_xyz_remote.ply
mapanything_pointcloud_same_aux_point_residual_xyz_norm_remote.ply
mapanything_pointcloud_same_aux_point_residual_summary.json
```

summary：

```text
Seattle 493:
  norm_factor: 0.640125
  residual_norm_mean: 0.271948
  residual_norm_p95: 0.549319
  z_residual_abs_norm_mean: 1.272684
  z_residual_abs_norm_p95: 1.391170

NewYork 461_1:
  norm_factor: 0.839597
  residual_norm_mean: 0.282109
  residual_norm_p95: 0.586142
  z_residual_abs_norm_mean: 1.382940
  z_residual_abs_norm_p95: 1.524009
```

aux xyz benchmark：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/remote_pointmetric20/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best_auxxyzmetric

RS-only:
  rs_point_l1: 291.204
  rs_point_l1_centered: 95.193
  rs_point_l1_scale_aligned: 30.039
  rs_height_mae_affine: 21.697

Joint:
  rs_point_l1: 291.177
  rs_point_l1_centered: 95.188
  rs_point_l1_scale_aligned: 47.349
  rs_height_mae_affine: 18.506
```

结论：xyz residual 是目前 residual 系列里最明确的 scale-free remote shape 改善：
RS-only scale-aligned 从 xy-only 的 `43.629` 进一步降到 `30.039`。但 raw/centered 仍约 `291/95`，
说明它仍没有解决绝对尺度和中心；height affine 反而偏差较大，说明 z residual 在归一化点云空间可学习，
但不能直接等价为真实高度恢复。下一步不应简单把该 residual 当最终输出，而应考虑把它作为 teacher/repair
目标，继续研究尺度恢复或让 point head 内部吸收这类尺度无关几何修正。

#### aux xyz residual all-cities continuation

目的：验证 2-city xyz residual 的 `scale_aligned ~= 30` 是否只是小数据偶然结果，还是能在全城市数据上复现。

配置：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_b9_2gpu
WARMSTART_CKPT: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_b9_2gpu/checkpoint-best-slim.pth
TRAIN_CITIES/VAL_CITIES/TEST_CITIES: [chicago,newyork,sanfrancisco,seattle]
EPOCHS: 8
BATCH_SIZE: 9
NUM_GPUS: 2
LAMBDA_REMOTE_PM: 8
PROJ_POINT_RESIDUAL_XYZ_TO_GT: 10
all old projection aux losses: 0
train_params.warmstart_exclude_prefixes: []
```

训练后清理：

```text
保留:
  checkpoint-best-slim.pth: 4.9G
  checkpoint-final.pth: 4.9G

已删除:
  checkpoint-last.pth
  14G checkpoint-best.pth
```

关键训练/验证日志：

```text
epoch 0 val:
  xyz residual loss avg: 0.1094
  corrected_mae_norm_mean_avg: 0.2239
  corrected_z_mae_norm_mean_avg: 0.1751

epoch 1 val:
  xyz residual loss avg: 0.0727
  corrected_mae_norm_mean_avg: 0.1513
  corrected_z_mae_norm_mean_avg: 0.1105

epoch 3 val:
  xyz residual loss avg: 0.0564
  corrected_mae_norm_mean_avg: 0.1186
  corrected_z_mae_norm_mean_avg: 0.0936

best val around epoch 4:
  xyz residual loss avg: 0.0537
  corrected_mae_norm_mean_avg: 0.1145
  corrected_z_mae_norm_mean_avg: 0.0876

final val epoch 8:
  xyz residual loss avg: 0.0539
  corrected_mae_norm_mean_avg: 0.1221
  corrected_z_mae_norm_mean_avg: 0.1095
```

可视化输出：

```text
Seattle 493:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best

NewYork 461_1:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best
```

summary：

```text
Seattle 493:
  norm_factor: 0.648779
  residual_norm_mean: 0.270168
  residual_norm_p95: 0.522968
  z_residual_abs_norm_mean: 1.152900
  z_residual_abs_norm_p95: 1.282079

NewYork 461_1:
  norm_factor: 0.845912
  residual_norm_mean: 0.264359
  residual_norm_p95: 0.531734
  z_residual_abs_norm_mean: 1.310771
  z_residual_abs_norm_p95: 1.453615
```

aux xyz benchmark：

```text
output: /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/remote_pointmetric20/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best_auxxyzmetric

RS-only:
  rs_point_l1: 291.241
  rs_point_l1_centered: 95.206
  rs_point_l1_scale_aligned: 30.108
  rs_height_mae_affine: 16.588

Joint:
  rs_point_l1: 291.189
  rs_point_l1_centered: 95.193
  rs_point_l1_scale_aligned: 50.417
  rs_height_mae_affine: 18.162
```

结论：全城市训练没有把 `scale_aligned` 从 2-city xyz residual 的 `30.039` 继续压低，但稳定复现到 `30.108`，
并且 best val residual loss 更低。这说明 `aux xyz residual` 不是只在 chicago/newyork 上偶然过拟合；
它确实是目前最可靠的尺度无关 remote 形状修正目标。问题仍然是 raw/centered 和真实高度尺度没有解决，
所以下一步应转向“让 point head 吸收 residual teacher”或新增尺度/中心恢复，而不是继续单独训练 aux residual。
