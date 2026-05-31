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
