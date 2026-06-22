# P7 Stage A: GT 投影参数下的 Height 机制训练计划

更新时间：2026-06-11

本文档描述下一轮 P7 projection-aux 实验：先固定/使用 GT 投影角度参数，只训练 height 机制，验证模型是否能学到稳定的 normalized relative height，并且该 height 能否通过投影公式还原出合理 remote 点云。

## 0. 当前实验记录

### 0.1 代码改动

本轮加入了 `grid_global_to_gt` 的 GT teacher 开关：

```text
projection_grid_global_to_gt_use_gt_dir
projection_grid_global_to_gt_use_gt_slope
projection_grid_global_to_gt_detach_dir_slope
```

对应文件：

```text
mapanything/train/losses.py
configs/loss/vggt_loss_rs_joint_p7_remote_head_projection_aux.yaml
bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh
```

Stage A 训练时，`grid_global_to_gt` 重建项使用 GT `global_dir/global_slope`，只让 height head 承担主要几何重建压力。由于 DDP static graph 要求所有训练参数有梯度，本轮保留了极小的 `global_dir/slope` 监督权重，仅用于保持角度头参与反传；真正的 grid 重建仍使用 GT 角度。

### 0.2 训练配置

输出目录：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_stageA_height_gtdir_frozentrunk_e6_b32_2gpu
```

关键配置：

```text
NUM_GPUS=2
BATCH_SIZE=32
NUM_VIEWS=4
NUM_EPOCHS=6
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_train_remotehead_aux
REMOTE_PROJECTION_AUX_SOURCE=tokens
REMOTE_PROJECTION_AUX_HIDDEN_DIM=96
REMOTE_PROJECTION_AUX_NUM_BLOCKS=6
REMOTE_PROJECTION_AUX_USE_RGB=true
REMOTE_PROJECTION_AUX_USE_COORD=true
REMOTE_PROJECTION_AUX_POSITIVE_SLOPE=true

LAMBDA_REMOTE_PM=0
LAMBDA_REMOTE_RAW_PM=0
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.8
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.2
PROJ_GRID_GLOBAL_TO_GT=0.2
PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE=0.8
PROJ_GRID_GLOBAL_TO_GT_USE_GT_DIR=true
PROJ_GRID_GLOBAL_TO_GT_USE_GT_SLOPE=true
PROJ_GRID_GLOBAL_TO_GT_DETACH_DIR_SLOPE=true
LAMBDA_PROJ_GLOBAL_DIR=0.001
LAMBDA_PROJ_GLOBAL_SLOPE=0.001

SAVE_FREQ=0
KEEP_FREQ=0
```

显存：

- 2 张 RTX PRO 6000 Blackwell Server Edition；
- `BATCH_SIZE=48/40` 都 OOM；
- `BATCH_SIZE=32` 稳定，`nvidia-smi` 观察约 88GB/卡，GPU 利用率接近满载。

存储：

- 已清理所有历史 `checkpoint-[0-9]*.pth` 中间权重；
- 本实验完成后删除 `checkpoint-last.pth`；
- 当前仅保留 `checkpoint-best.pth` 和 `checkpoint-final.pth`。

### 0.3 训练结果

训练成功完成 6 epoch。日志中的关键指标：

```text
rs_projection_dense_rel_height_loss: final test avg ~= 0.0117
rs_projection_rel_height_high20_mae: final test avg ~= 0.0321
rs_projection_rel_height_contrast_loss: final test avg ~= 0.0033
rs_projection_grid_global_to_gt_loss: final test avg ~= 0.7628
```

结论：

- dense normalized height 监督是可学习的，loss 从训练初期的约 `0.09~0.11` 快速降到约 `0.01`；
- height 的高结构分桶/contrast 指标也能下降，说明不是完全学成平滑背景；
- 但 `grid_global_to_gt_loss` 基本没有有效下降，即使用 GT `dir/slope`，当前 grid-global 点云重建仍不可靠。

### 0.4 可视化输出

严格匹配 Stage A 条件的导出目录如下，导出时使用 GT `global_dir/global_slope`，并对 aux PLY 做了 `gt_pointmap_unit_xy_zrange` 尺度无关对齐：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_stageA_height_gtdir_frozentrunk_e6_final_gtdir
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_stageA_height_gtdir_frozentrunk_e6_final_gtdir
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
mapanything_pointcloud_same_aux_offset_remote.ply
mapanything_pointcloud_same_aux_global_remote.ply
mapanything_pointcloud_same_aux_grid_global_remote.ply
mapanything_pointcloud_same_aux_reconstruction_summary.json
mapanything_pointcloud_same_aux_height_compare/rel_height_gt_pred_grid.png
mapanything_pointcloud_same_aux_height_compare/rel_height_summary.json
```

高度图指标：

```text
Seattle 493:
  rel_height_norm_mae        = 0.0681
  rel_height_norm_affine_mae = 0.0243
  GT mean/std                = 0.1138 / 0.1137
  Pred mean/std              = 0.0475 / 0.0592
  affine scale/shift         = 1.777 / 0.029

New York 461_1:
  rel_height_norm_mae        = 0.0763
  rel_height_norm_affine_mae = 0.0373
  GT mean/std                = 0.1389 / 0.1967
  Pred mean/std              = 0.0680 / 0.0940
  affine scale/shift         = 1.953 / 0.006
```

可视化判断：

- 预测 height 图已经能看到建筑区域和部分高低结构；
- 直接 normalized height 的幅度约为 GT 的一半；
- 仿射对齐后误差明显降低，说明主要问题不是“完全没有形状”，而是 height 幅度/尺度被压缩；
- 当前 aux-grid 点云仍不能作为可靠 remote 点云输出。

### 0.5 当前结论和下一步

Stage A 的结论是：`dense height` 作为单独任务可学，但仅靠当前 normalized height + grid-global 公式还不能稳定还原 remote 点云。下一步不应直接进入预测 `dir/slope` 的 Stage B，而应先解决两个更基础的问题：

1. 训练和导出的 height scale 定义需要统一。训练中的 `gt_pointmap_norm` 使用 GT pointmap 归一化因子；导出默认 `pred_avg_dis` 使用模型 point head 平均距离，容易造成可视化尺度偏差。
2. `grid_global_to_gt_loss` 在 GT `dir/slope` 下仍高，说明 grid 坐标、GT pointmap 对齐、height 尺度、或 high-z mask 选择仍有错位。需要做一个 oracle 诊断：用 GT dense height + GT `dir/slope` 直接构造 `grid_global`，如果仍不能接近 GT pointmap，问题就在重建公式/坐标系，而不是模型 head。

### 0.6 Stage A2: 改用坐标一致的 reconstruct-global 监督

Stage A 后检查发现，旧 `grid_global_to_gt` 路径存在监督域错位：

```text
grid_recon = [grid_xy - height * slope * dir, height]
```

这里的 `grid_xy` 是 `[-1, 1]` 像素网格，`z` 是 normalized relative height；但 loss 直接和 `gt['remote_pointmap']` 比较，而 GT pointmap 是原始世界坐标。当前 `_normalize_pair` 只做尺度除法，不做平移、坐标基底、z 方向或 relative-height 到 world-z 的转换。因此即使用 GT `dir/slope`，`grid_global_to_gt_loss ~= 0.76` 也不说明 height 学不好，而是 loss 本身在比较不同坐标系。

Stage A2 改为使用已有的 `reconstruct_global_to_gt` 路径：用 GT remote 投影基底里的 `projected_xyz_centered` 作为 base，height 只负责沿 GT `global_dir/global_slope` 产生投影位移，重建结果再和 GT `remote_pointmap` 做尺度无关点云监督。这一路径和原始 pointmap 在同一坐标域内。

新增开关：

```text
projection_reconstruct_global_to_gt_use_gt_dir
projection_reconstruct_global_to_gt_use_gt_slope
projection_reconstruct_global_to_gt_detach_dir_slope
```

对应文件：

```text
mapanything/train/losses.py
configs/loss/vggt_loss_rs_joint_p7_remote_head_projection_aux.yaml
bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh
```

训练输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_stageA2_height_gtglobalrecon_frozentrunk_fromA_e6_b36_2gpu
```

关键配置：

```text
WARMSTART_CKPT=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_stageA_height_gtdir_frozentrunk_e6_b32_2gpu/checkpoint-final.pth
NUM_GPUS=2
BATCH_SIZE=36
NUM_EPOCHS=6
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_train_remotehead_aux
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.8
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.2
PROJ_GRID_GLOBAL_TO_GT=0
PROJ_RECON_GLOBAL_TO_GT=0.2
PROJ_RECON_TO_GT_USE_POINTMAP_NORM=true
PROJ_RECON_TO_GT_HIGH_Z_QUANTILE=0.8
PROJ_RECON_GLOBAL_TO_GT_USE_GT_DIR=true
PROJ_RECON_GLOBAL_TO_GT_USE_GT_SLOPE=true
PROJ_RECON_GLOBAL_TO_GT_DETACH_DIR_SLOPE=true
LAMBDA_REMOTE_PM=0
LAMBDA_REMOTE_RAW_PM=0
LAMBDA_PROJ_GLOBAL_DIR=0.001
LAMBDA_PROJ_GLOBAL_SLOPE=0.001
SAVE_FREQ=0
KEEP_FREQ=0
```

资源和存储：

- 2 张 96GB GPU；`BATCH_SIZE=48/40` OOM，`BATCH_SIZE=36` 稳定；
- 训练日志峰值显存约 `88376 MiB/卡`；
- 已删除该实验的 `checkpoint-last.pth`，只保留 `checkpoint-best.pth` 和 `checkpoint-final.pth`。

训练结果：

```text
final test:
  rs_projection_dense_rel_height_loss ~= 0.0136
  rs_projection_rel_height_high20_mae ~= 0.0397
  rs_projection_reconstruct_global_to_gt_loss ~= 0.0018
```

对比 Stage A 的结论：

- `reconstruct_global_to_gt_loss` 从第一轮开始就在 `0.001~0.003` 量级，说明坐标一致的 GT-dir/slope 重建监督是良性的；
- 旧 `grid_global_to_gt_loss ~= 0.76` 是坐标/监督域设计问题，不能再作为判断 height 能否学习的依据；
- height 图形状仍能学到，但 normalized 幅度偏小，说明下一步要解决的是尺度恢复和 `dir/slope` 预测，而不是继续加大旧 grid-global loss。

可视化输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_stageA2_height_gtglobalrecon_frozentrunk_e6_b36_final_gtdir
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_stageA2_height_gtglobalrecon_frozentrunk_e6_b36_final_gtdir
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
mapanything_pointcloud_same_aux_offset_remote.ply
mapanything_pointcloud_same_aux_global_remote.ply
mapanything_pointcloud_same_aux_grid_global_remote.ply
mapanything_pointcloud_same_aux_reconstruction_summary.json
mapanything_pointcloud_same_aux_height_compare/rel_height_gt_pred_grid.png
```

高度图指标：

```text
Seattle 493:
  valid_pixels                = 33550
  rel_height_norm_mae         = 0.0681
  rel_height_norm_affine_mae  = 0.0243
  GT mean/std                 = 0.1138 / 0.1137
  Pred mean/std               = 0.0475 / 0.0592
  affine scale/shift          = 1.777 / 0.029

New York 461_1:
  valid_pixels                = 41357
  rel_height_norm_mae         = 0.0763
  rel_height_norm_affine_mae  = 0.0373
  GT mean/std                 = 0.1389 / 0.1967
  Pred mean/std               = 0.0680 / 0.0940
  affine scale/shift          = 1.953 / 0.006
```

可视化判断：

- `Pred height norm` 能定位建筑区域和主要高度变化；
- 原始 normalized 幅度明显偏小，大约只有 GT 的一半；
- affine 对齐后误差明显降低，说明模型学到了相对形状，但没有学好绝对 normalized height 幅度；
- summary 中导出使用了 GT `global_dir/global_slope`，因此这些 aux 点云是 Stage A 上界诊断，不是最终“全预测投影”效果。

### 0.7 Stage A2 长训 lowtrunk/full finetune 上限测试

日期：2026-06-12。

本实验用于回答一个直接问题：Stage A2 的 height + GT `dir/slope` reconstruct-global 目标，如果开放更多参数并拉长 epoch，是否只是训练不够，还是已经到达当前配方的上限。

训练输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_stageA2_height_gtglobalrecon_fullfinetune_fromA2_e80_b8_2gpu
```

关键配置：

```text
WARMSTART_CKPT=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_stageA2_height_gtglobalrecon_frozentrunk_fromA_e6_b36_2gpu/checkpoint-final.pth
TRAIN_PARAMS=vggt_p7_p5b_shared_norm_projection_aux_lowtrunklr
NUM_GPUS=2
BATCH_SIZE=8
EPOCHS=80 planned, stopped after epoch 25
TRAIN_CITIES=[chicago,newyork]
VAL_CITIES=[chicago,newyork]
TEST_CITIES=[chicago,newyork]
LAMBDA_REMOTE_PM=0
LAMBDA_REMOTE_RAW_PM=0
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.8
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.2
PROJ_RECON_GLOBAL_TO_GT=0.2
PROJ_RECON_TO_GT_USE_POINTMAP_NORM=true
PROJ_RECON_TO_GT_HIGH_Z_QUANTILE=0.8
PROJ_RECON_GLOBAL_TO_GT_USE_GT_DIR=true
PROJ_RECON_GLOBAL_TO_GT_USE_GT_SLOPE=true
PROJ_RECON_GLOBAL_TO_GT_DETACH_DIR_SLOPE=true
PROJ_GRID_GLOBAL_TO_GT=0
SAVE_FREQ=0
KEEP_FREQ=0
```

资源和存储：

- 当前机器只暴露 2 张 96GB GPU；
- `BATCH_SIZE=16/14/13/12` 均 OOM，`BATCH_SIZE=8` 稳定；
- `BATCH_SIZE=8` 训练时每卡约占用 `81GB/96GB`，日志中 PyTorch max mem 约 `75956 MiB/卡`；
- 只保存 `checkpoint-best.pth`，没有周期性中间权重，训练目录约 `4.7GB`。

验证结果：

```text
epoch 5/15/25 validation metrics were effectively identical:
  loss                                      ~= 0.5061
  remote_loss                               ~= 0.0920
  rs_pointmap_loss                          ~= 0.0486
  rs_projection_aux_loss                    ~= 0.0920
  rs_projection_rel_height_loss             ~= 0.0194
  rs_projection_rel_height_high20_gt_mean   ~= 0.1886
  rs_projection_rel_height_high20_pred_mean ~= 0.1779
  rs_projection_rel_height_high20_mae       ~= 0.0508
  rs_projection_rel_height_low80_mae        ~= 0.0104
  rs_projection_rel_height_affine_loss      ~= 0.0202
  rs_projection_dense_rel_height_loss       ~= 0.0164
  rs_projection_reconstruct_global_to_gt_loss ~= 0.0050
  rs_projection_global_dir_cosine           ~= 0.4499
```

`checkpoint-best.pth` 只在 epoch 5 产生，epoch 15 和 epoch 25 都没有刷新 best。因此继续同一配方长训的收益很低，epoch 不足不是当前主要瓶颈。

可视化输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_stageA2_fullfinetune_fromA2_e80_b8_best_gtdir_flipz
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_stageA2_fullfinetune_fromA2_e80_b8_best_gtdir_flipz
```

每个目录包含：

```text
mapanything_pointcloud_same.ply
mapanything_pointcloud_same_remote.ply
mapanything_pointcloud_same_aux_offset_remote.ply
mapanything_pointcloud_same_aux_global_remote.ply
mapanything_pointcloud_same_aux_grid_global_remote.ply
mapanything_pointcloud_same_aux_reconstruction_summary.json
mapanything_pointcloud_same_aux_height_compare/rel_height_gt_pred_grid.png
```

阶段结论：

- Stage A2 的坐标一致 reconstruct-global 监督是良性的，但当前 lowtrunk/full-finetune 配方在 2-city 上很快平台化；
- 继续单纯增加 epoch 不值得，应切换到更直接服务最终 remote 点云的实验；
- 下一步应在保留尺度无关 height/reconstruct 约束的同时，重新引入 remote point head 损失，并用 `remote_pointmetric20` 中的尺度无关点云指标和 Seattle 493 / New York 461_1 可视化筛选。

### 0.8 全城市 robust-overlap + Stage A2 teacher 联训

日期：2026-06-12。

目的：验证 Stage A2 的尺度无关 dense height / GT `dir/slope` reconstruct teacher，能否在不破坏当前最好 remote point head 的前提下，提升最终 remote 点云。这个实验直接面向最终目标，不再只是 aux 上界诊断。

训练输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_fullfinetune_e60_b8_2gpu
```

关键配置：

```text
WARMSTART_CKPT=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu/checkpoint-final.pth
TRAIN_PARAMS=vggt_p7_p5b_shared_norm_projection_aux
NUM_GPUS=2
BATCH_SIZE=8
EPOCHS=60
EVAL_FREQ=5
SAVE_FREQ=0
KEEP_FREQ=0
TRAIN_CITIES=[chicago,newyork,sanfrancisco,seattle]
VAL_CITIES=[chicago,newyork,sanfrancisco,seattle]
TEST_CITIES=[chicago,newyork,sanfrancisco,seattle]
LAMBDA_REMOTE_PM=4
REMOTE_POINTMAP_TOP_N_PERCENT=20
LAMBDA_REMOTE_OVERLAP_PM=6
REMOTE_POINTMAP_NORM_MODE=aerial_avg_dis
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.6
PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK=false
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.2
PROJ_RECON_GLOBAL_TO_GT=0.1
PROJ_RECON_TO_GT_USE_POINTMAP_NORM=true
PROJ_RECON_TO_GT_HIGH_Z_QUANTILE=0.8
PROJ_RECON_GLOBAL_TO_GT_USE_GT_DIR=true
PROJ_RECON_GLOBAL_TO_GT_USE_GT_SLOPE=true
PROJ_RECON_GLOBAL_TO_GT_DETACH_DIR_SLOPE=true
```

已确认事项：

- 虽然脚本初始化提示 `Starting from random initialization`，但训练器日志明确显示：

```text
Warm-starting model weights from:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu/checkpoint-final.pth
```

- 当前机器只暴露 2 张 96GB GPU；`BATCH_SIZE=8` 训练稳定，占用约 `80~81GB/卡`，GPU 利用率接近满载；
- 训练数据按 `traindata/mapanything_metadata` 的 Crossview/Crossview_rs_aerial metadata 和城市过滤读取，不直接绕过 metadata 使用未过滤的 raw Crossview_rs；
- epoch 0 初始 remote pointmap loss 仍在正常量级，aux height 明显 over-pred，正是本轮 teacher 要校正的对象。

结果：

- 原计划 `EPOCHS=60`，实际在 epoch 10 后停止。原因是 epoch 5 和 epoch 10 的 validation 基本完全平台化，且 height raw normalized 幅度明显错误，继续堆 epoch 的收益很低；
- best checkpoint 来自 epoch 5：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_fullfinetune_e60_b8_2gpu/checkpoint-best.pth
```

关键 validation：

| 指标 | epoch 5/10 量级 | 解释 |
|---|---:|---|
| `rs_pointmap_loss` | `~0.0413` | point head 没被明显拖坏 |
| `remote_loss` | `~1.7393` | 主要由 projection aux 拉高 |
| `rs_projection_aux_loss` | `~1.3408` | aux 仍未校准 |
| `rs_projection_rel_height_high20_gt_mean` | `~0.1226` | GT 高区 normalized height 均值 |
| `rs_projection_rel_height_high20_pred_mean` | `~1.0150` | 预测高度幅度严重偏大 |
| `rs_projection_rel_height_high20_mae` | `~0.8979` | 高区 raw normalized height 失败 |
| `rs_projection_dense_rel_height_loss` | `~0.3290` | dense height raw 监督未收敛 |
| `rs_projection_rel_height_affine_loss` | `~0.0274` | 形状经 affine 后还能对齐一部分 |

可视化输出：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_e10best
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_e10best
```

height 可视化摘要：

| 场景 | `rel_height_norm_mae` | `rel_height_norm_affine_mae` | GT mean/std | Pred mean/std |
|---|---:|---:|---:|---:|
| Seattle 493 | `0.2955` | `0.0323` | `0.1138 / 0.1137` | `0.3775 / 0.4964` |
| New York 461_1 | `0.1957` | `0.0420` | `0.1389 / 0.1967` | `0.3093 / 0.4645` |

`rel_height_norm_affine_mae` 低于 raw MAE，说明模型能学到一部分空间形状，但 raw normalized 幅度没有学对；这会直接破坏由 height 反投影得到的 aux 点云。

New York `remote_pointmetric20` benchmark：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/remote_pointmetric20/vggt_p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_e10best/rs_aerial_benchmark_results.json
```

| 分支 | `rs_point_l1` | `rs_point_l1_centered` | `rs_point_l1_scale_aligned` | `rs_height_mae_affine` | `rs_height_rmse_affine` |
|---|---:|---:|---:|---:|---:|
| RS-only | `294.2624` | `95.5059` | `95.5042` | `20.1120` | `26.9596` |
| Joint | `294.2624` | `95.5059` | `95.5042` | `20.0443` | `26.7570` |

结论：

- 这轮全城市 + full finetune + Stage A2 teacher 没有突破当前 remote point head 边界；
- point head 没被明显破坏，但 aux height 的 raw normalized scale 明显过大，导致 projection aux 反投影点云仍不可用；
- 单纯在同一配方上拉长 epoch 不是主要矛盾，下一步应该先把 height head 的 normalized 幅度校准到可用，再重新引入 point head 联训。

## 1. 背景和核心问题

当前 self-contained grid global 实验使用：

```text
grid_xy = normalized pixel grid in [-1, 1]
offset_xy = pred_height_norm * pred_slope * pred_dir
recon_xyz = [grid_xy - offset_xy, pred_height_norm]
```

训练结果显示：

- remote point head 没被拖坏；
- height 有一定形状信号，但幅度偏低；
- `grid_global_to_gt` loss 基本卡住；
- aux 反投影点云仍不稳定。

因此当前失败不是简单的“标签完全错”或“模型完全学不动”，而是 height、dir、slope、reconstruction 同时训练时耦合过强。Stage A 的目标是先把问题拆开：在 `dir/slope` 正确时，只问 height head 能不能学好。

## 2. Stage A 要回答的问题

Stage A 只回答两个问题：

1. 在 GT `global_dir/global_slope` 下，预测 height 能否稳定接近 GT normalized height？
2. 用预测 height + GT `global_dir/global_slope` 反投影出的 grid-global 点云，是否能接近 GT remote pointmap？

如果答案是 yes，说明 height 机制成立，可以进入 Stage B 训练预测 `dir/slope`。  
如果答案是 no，说明 height 定义、grid 坐标、尺度归一化、crop 尺度或标签生成仍有问题，不应继续训练角度头。

## 3. Height 的定义

Stage A 的 height 不是每张图 min-max 到 `[0, 1]`。

推荐定义：

```text
rel_height_gt_raw = remote pointmap z - ground_z
scale_gt = avg_dis(GT remote pointmap) 或同类 pointmap norm factor
rel_height_gt_norm = rel_height_gt_raw / scale_gt
rel_height_pred_norm = model output
```

也就是说，模型直接输出 `rel_height_pred_norm`。GT 用 GT pointmap 的尺度归一化：

```text
loss_height(pred, gt) = loss(rel_height_pred_norm, rel_height_gt_norm)
```

这样保留了“建筑高度相对场景 xy 尺度”的比例，不会像 per-image min-max `[0,1]` 那样丢掉尺度关系。

## 4. Height loss 设计

Stage A 的 height loss 分三层。

### 4.1 主损失：normalized dense height

主项使用 pointmap-derived dense height：

```text
L_dense_height =
  weighted Huber / L1(rel_height_pred_norm, rel_height_gt_norm)
```

推荐初始配置：

```text
PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.5 ~ 1.0
PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK=false
PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT=0.15 ~ 0.30
PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE=0.5
PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE=0.0 或 0.2
```

低 height/background 权重要低一些，避免模型只优化大片低矮背景。

### 4.2 辅助项：affine shape loss

affine loss 只用于稳定形状，不作为主监督：

```text
fit: rel_height_pred_affine = a * rel_height_pred_norm + b
L_affine = L1(rel_height_pred_affine, rel_height_gt_norm)
```

推荐权重：

```text
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.1 ~ 0.25
```

解释：

- 主 normalized loss 约束高度比例；
- affine loss 只问“形状像不像”，能缓解训练早期尺度偏差；
- 不能只用 affine loss，否则反投影所需的 height 幅度会丢失。

### 4.3 高结构/分桶辅助项

可保留少量 bucket/contrast/high-z 项，避免高楼被背景淹没：

```text
PROJ_REL_HEIGHT_BALANCED_WEIGHT=0.5 ~ 1.0
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=0.2 ~ 0.6
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=0.5 ~ 1.0
```

如果训练不稳，优先降低这些辅助项，而不是降低 dense height 主项。

## 5. GT dir/slope teacher reconstruction

Stage A 的关键是重建点云时不使用预测 `dir/slope`：

```text
dir_xy = GT remote_projection_global_dir_xy
slope = GT remote_projection_global_slope
h = rel_height_pred_norm

grid_recon = [
  grid_xy - h * slope * dir_xy,
  h
]
```

然后和 normalized GT pointmap 算重建损失：

```text
gt_pointmap_norm = normalize(GT remote pointmap)
L_grid_global_gt_dir =
  Huber / L1(normalize_pair(grid_recon, gt_pointmap_norm))
```

这个损失只应该回传到 height head。GT `dir/slope` 不是可学习量；预测 `dir/slope` 不参与 Stage A 的重建项。

### 需要新增的代码开关

当前已有 `projection_grid_global_to_gt_loss_weight`，但它默认用预测 `dir/slope`。Stage A 需要新增：

```text
projection_grid_global_to_gt_use_gt_dir: bool
projection_grid_global_to_gt_use_gt_slope: bool
projection_grid_global_to_gt_detach_dir_slope: bool
projection_grid_global_to_gt_detach_non_height: bool
```

最小实现可以是：

```text
if use_gt_dir:
    dir_for_grid = gt["remote_projection_global_dir_xy"]
else:
    dir_for_grid = pred["remote_projection_global_dir_xy_pred"]

if use_gt_slope:
    slope_for_grid = gt["remote_projection_global_slope"]
else:
    slope_for_grid = pred["remote_projection_global_slope_pred"]

grid_recon = reconstruct_grid(rel_pred_for_loss, dir_for_grid, slope_for_grid)
```

Stage A 配置：

```text
use_gt_dir=true
use_gt_slope=true
projection_global_dir_loss_weight=0
projection_global_slope_loss_weight=0
projection_global_vector_loss_weight=0
projection_offset_loss_weight=0
projection_consistency_loss_weight=0
```

## 6. 是否冻结主干

Stage A 建议做两个对照，而不是一上来全量微调。

### A1: frozen trunk / aux-only

目的：测试 aux height head 在现有 tokens 上的可学习上界。

训练参数：

```text
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_train_remotehead_aux
REMOTE_PROJECTION_AUX_SOURCE=tokens
USE_REMOTE_PRIVATE_POINT_HEAD=true
REMOTE_OUTPUT_HEAD=point
```

但需要注意：如果这个 train params 同时训练 remote point head，应把 remote pointmap loss 降低或关闭，避免 point head 变化干扰 height-only 判断。

推荐：

```text
LAMBDA_REMOTE_PM=0 或 1
LAMBDA_REMOTE_RAW_PM=0
LAMBDA_REMOTE_OVERLAP_PM=0
```

A1 的成功标准是：height 图和 aux grid-global PLY 显著变好。

### A2: low-lr trunk / aux + 少量 remote branch

目的：如果 A1 学不好，判断是不是 tokens 表征不足。

训练参数可以使用低 trunk LR 配置，例如只开放 remote point head、projection aux、少量 aggregator tail：

```text
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_aggtail2_lowlr_train_remotehead_aux
```

或现有 midtrunk/lowtrunklr 配置。原则是 trunk LR 小，aux LR 高。

推荐：

```text
remote_projection_aux lr: 1e-4 级
remote_point_head lr: 1e-5 ~ 2e-5
aggregator lr: 5e-7 ~ 2e-6
```

### A3: full fine-tune 上界测试

只有 A1/A2 显示 height 机制可学但上界不足时，再跑短程全量微调。  
全量微调容易影响普通视角重建，因此只作为性能上界测试，不作为第一选择。

## 7. 普通视角损失是否同步计算

建议同步保留普通视角 forward 和 aerial loss 日志，但分阶段控制其训练作用。

### 监控模式

保持 joint batch，记录：

```text
aerial_loss
FactoredGeometryRegr3DPlusNormalGMLoss_pts3d_avg
FactoredGeometryRegr3DPlusNormalGMLoss_depth_z_avg
pose/ray metrics
```

如果冻结 trunk，普通视角 loss 理论上不应恶化明显；如果恶化，说明 shared params 仍被 remote branch 更新影响，需检查 train params。

### 共同下降模式

如果 A2/A3 中 remote height 下降但 aerial loss 明显升高，再尝试把 aerial loss 保持为训练项，目标是：

```text
remote height / aux grid-global loss 下降
aerial ordinary loss 不升或同步下降
```

这时不要关闭 aerial branch；用较小 LR 保护 shared trunk。

## 8. Remote crop 和尺度

Stage A 第一轮必须使用：

```text
REMOTE_TRAIN_CROP_MODE=none
REMOTE_VAL_CROP_MODE=none
REMOTE_TEST_CROP_MODE=none
```

原因：self-contained `grid_xy in [-1,1]` 的物理尺度随 remote crop 改变。如果开 crop，而 height target 仍按原 pointmap scale 定义，`grid_xy` 和 height 的比例会不一致。

后续若要支持 crop，必须加入 crop-aware scale：

```text
xy = scale_xy * grid_xy - scale_h * height_norm * slope * dir
z  = scale_h * height_norm
```

优先用 metadata/crop box 直接计算 `scale_xy`；只有元数据不够时才考虑 scale head。

## 9. 导出与判定

每个 Stage A checkpoint 必须导出：

- Seattle 493
- New York 461_1

固定命令应包含：

```text
--export_projection_aux_reconstruction
--projection_aux_gt_remote_dir <对应 GT dir>
--projection_aux_xyz_align_mode gt_pointmap_unit_xy_zrange_flipz
--export_remote_control_modes same
```

重点文件：

```text
mapanything_pointcloud_same_remote.ply
mapanything_pointcloud_same_aux_grid_global_remote.ply
mapanything_pointcloud_same_aux_height_compare/rel_height_gt_pred_grid.png
mapanything_pointcloud_same_aux_height_compare/rel_height_summary.json
mapanything_pointcloud_same_aux_reconstruction_summary.json
```

成功标准：

1. `rel_height_norm_mae` 明显下降；
2. `rel_height_norm_affine_mae` 下降，并且 `pred mean/std` 接近 GT；
3. `aux_grid_global_remote.ply` 在 GT dir/slope 下形成可读建筑结构；
4. 普通视角 aerial loss 没有明显恶化；
5. 如果 point head 参与训练，remote pointmap benchmark 不劣化。

## 10. 初始实验配置草案

### Stage A1: frozen trunk height teacher

```bash
OUTPUT_DIR='${root_experiments_dir}/mapanything/training/Crossview/vggt/p7_stageA_height_gtdir_frozentrunk_e6_b16_2gpu' \
NUM_GPUS=2 CUDA_DEVICES=0,1 MASTER_PORT=29561 NUM_WORKERS=8 NUM_VIEWS=4 BATCH_SIZE=16 \
EPOCHS=6 WARMUP_EPOCHS=1 EVAL_FREQ=1 SAVE_FREQ=0 KEEP_FREQ=0 PRINT_FREQ=20 \
TRAIN_CITIES='[chicago,newyork,sanfrancisco,seattle]' VAL_CITIES='[chicago,newyork,sanfrancisco,seattle]' TEST_CITIES='[chicago,newyork,sanfrancisco,seattle]' \
REMOTE_TRAIN_CROP_MODE=none REMOTE_VAL_CROP_MODE=none REMOTE_TEST_CROP_MODE=none \
RS_PROVIDER=Google_Satellite,Bing_Satellite REMOTE_PROVIDER_SAMPLING_MODE=random \
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_train_remotehead_aux \
USE_REMOTE_PRIVATE_POINT_HEAD=true REMOTE_OUTPUT_HEAD=point REMOTE_PROJECTION_AUX_SOURCE=tokens \
LAMBDA_REMOTE_PM=0.0 LAMBDA_REMOTE_RAW_PM=0.0 LAMBDA_REMOTE_OVERLAP_PM=0.0 \
LAMBDA_PROJ_REL_HEIGHT=0.05 \
PROJ_DENSE_REL_HEIGHT_WEIGHT=0.8 PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK=false PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT=0.2 PROJ_DENSE_REL_HEIGHT_LOW_QUANTILE=0.5 \
PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm \
PROJ_REL_HEIGHT_AFFINE_WEIGHT=0.2 PROJ_REL_HEIGHT_BALANCED_WEIGHT=0.8 PROJ_REL_HEIGHT_CONTRAST_WEIGHT=0.3 PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=0.8 \
LAMBDA_PROJ_OFFSET=0.0 LAMBDA_PROJ_GLOBAL_DIR=0.0 LAMBDA_PROJ_GLOBAL_SLOPE=0.0 LAMBDA_PROJ_GLOBAL_VECTOR=0.0 LAMBDA_PROJ_CONSISTENCY=0.0 \
PROJ_GRID_GLOBAL_TO_GT=0.2 PROJ_GRID_GLOBAL_TO_GT_USE_GT_DIR=true PROJ_GRID_GLOBAL_TO_GT_USE_GT_SLOPE=true \
WARMSTART_CKPT=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu/checkpoint-final.pth \
bash bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh 2
```

注：`PROJ_GRID_GLOBAL_TO_GT_USE_GT_DIR/SLOPE` 是待新增开关。

### Stage A2: low-trunk height teacher

同 A1，但：

```text
TRAIN_PARAMS=vggt_p7_p5b_private_oldp7_p5bhead_aggtail2_lowlr_train_remotehead_aux
LAMBDA_REMOTE_PM=2.0
LAMBDA_REMOTE_OVERLAP_PM=3.0
```

用于判断 height 学习是否需要轻微更新 shared/remote 表征。

## 11. 进入 Stage B 的条件

只有满足以下条件才进入 Stage B：

- Stage A 的 `rel_height_norm_mae` 和 `rel_height_norm_affine_mae` 明显优于当前 gridglobal self-contained 模型；
- `aux_grid_global_remote.ply` 在 493/461_1 中能看出稳定建筑结构；
- ordinary/aerial loss 没有明显恶化；
- benchmark remote point head 不显著退化。

Stage B 再逐步把 GT `dir/slope` 替换为预测 `dir/slope`，例如 teacher forcing schedule：

```text
epoch 0-2: 100% GT dir/slope
epoch 3-5: 50% GT, 50% pred
epoch 6+: pred dir/slope
```

但 Stage B 不属于本文档第一轮执行范围。

## 12. 2026-06-12 执行记录

### 12.1 reset aux + simple dense height full finetune

实验：

```text
p7_fullfinetune_fromrobust_resetaux_pm4_overlap6_stageA2simple_e60_b8_2gpu_ddpfix
```

路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_fullfinetune_fromrobust_resetaux_pm4_overlap6_stageA2simple_e60_b8_2gpu_ddpfix
```

设置：

- 从 robust remote checkpoint warmstart；
- 排除 `remote_projection_aux_` 前缀，重新初始化 aux head；
- 全量微调，remote pointmap loss `pm4`，overlap pointmap loss `overlap6`；
- dense height 使用 simple L1，GT global dir/slope 用于 grid-global reconstruction；
- `SAVE_FREQ=0`、`KEEP_FREQ=0`，只保留 best/final，减少存储。

中止点：

```text
epoch 5.22
checkpoint-best.pth 已写出
```

关键现象：

```text
epoch 0: high20_pred ~= 0.076, low80_pred ~= 0.076
epoch 5: high20_pred ~= 0.076, low80_pred ~= 0.076
```

`low_overpred` 权重从 0 ramp 到 1 后，低区预测仍没有被压低；高区预测也没有随 GT 高度升高。remote point head 没有立刻崩坏，但 aux height 明显落入“常数高度/均值解”。

结论：

simple dense height L1 即使配合 reset aux 和全量微调，也不足以训练出可用的显式 height 分层。继续跑满 60 epoch 的收益很低，因此该实验在 epoch 5.22 中止。下一步转向 anti-collapse：增加 height balanced/contrast/bucket mean，目标是先验证 aux height 是否能稳定区分高/低区域。

### 12.2 no-crop + anti-collapse, gt-pointmap-norm height

实验：

```text
p7_fullfinetune_fromrobust_resetaux_nocrop_anticollapse_pm3_overlap4_e30_b8_2gpu
```

路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_fullfinetune_fromrobust_resetaux_nocrop_anticollapse_pm3_overlap4_e30_b8_2gpu
```

设置：

- 修正前一轮遗漏，训练/验证/测试均使用 `remote_crop_mode=none`；
- 从 robust remote checkpoint warmstart，但排除 `remote_projection_aux_`，重新初始化 aux head；
- 全量微调，`LAMBDA_REMOTE_PM=3`，`LAMBDA_REMOTE_OVERLAP_PM=4`；
- height 仍使用 `PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm`；
- 加入 `balanced / contrast / bucket mean / low-overpred` 反塌缩项；
- 使用 GT global dir/slope 计算 reconstruction 辅助损失，只验证 height 是否能学出有效形状。

中止点：

```text
epoch 2.0 附近
```

关键现象：

```text
high20_pred_mean ~= 0.0758
low80_pred_mean  ~= 0.0761
contrast_pred_gap ~= 0
contrast_gt_gap   ~= 0.10-0.12
```

结论：

no-crop 和反塌缩项没有解决常数 height 问题。日志里 `high20_gt_mean` 通常只有 `0.08-0.14`，`low80_gt_mean` 约 `0.01-0.02`，而预测常数约 `0.076`。这说明 `gt_pointmap_norm` 下的 height 目标幅度太小，L1/均值解仍然很容易成立。继续长跑该分支收益低，因此中止。

### 12.3 no-crop + valid-quantile normalized height

实验：

```text
p7_fullfinetune_fromrobust_resetaux_nocrop_validqheight_pm3_overlap4_e30_b8_2gpu
```

路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_fullfinetune_fromrobust_resetaux_nocrop_validqheight_pm3_overlap4_e30_b8_2gpu
```

设置：

- 与 12.2 相同的全量城市、no-crop、reset aux、full finetune；
- 将 height target 改为 `PROJ_REL_HEIGHT_SCALE_MODE=valid_quantile`，`PROJ_REL_HEIGHT_SCALE_QUANTILE=0.9`，`PROJ_REL_HEIGHT_CLIP=1.5`；
- 目标是让 dense height 更接近尺度无关的 `0-1` 相对高度，而不是被 pointmap norm 压到很小幅值；
- 反塌缩项保留但降低部分权重，避免过强 regularization 主导 point head。

阶段判断标准：

```text
high20_pred_mean - low80_pred_mean 必须从接近 0 变为明显正值；
high20/low80 MAE 要下降；
Seattle 493 和 New York 461_1 的 aux height compare 图需要能看出建筑高度结构；
GT dir/slope 下的 aux global/grid-global remote 点云需要比当前常数 height 分支更接近 GT。
```

该实验已启动，后续根据 epoch 2-5 的 high/low gap 决定是否继续长跑到 30 epoch，并在关键 checkpoint 上做可视化和 remote pointmetric benchmark。

实际结果：

```text
epoch 1.7:
high20_pred_mean ~= 0.0756
low80_pred_mean  ~= 0.0762
contrast_pred_gap ~= 0
```

虽然 `valid_quantile` 让 GT height 目标从 `0.1` 量级放大到 `0-1` 相对高度量级，但输出仍停在常数附近。进一步检查发现，`REMOTE_PROJECTION_AUX_SOURCE=tokens` 时实际模块名是：

```text
remote_projection_aux_token_norm
remote_projection_aux_token_proj
remote_projection_aux_token_pixel_head
remote_projection_aux_token_global_head
```

而基础 `vggt_p7_p5b_shared_norm_projection_aux` 配置只给 pointmap-source 的
`remote_projection_aux_pixel_head/global_head` 设置了高学习率。因此 token aux 主体此前没有吃到预期的 aux LR。这一轮在 epoch 1.7 中止，转向修正 token aux 参数分组。

### 12.4 no-crop + valid-quantile height + token aux LR fix

新增配置：

```text
configs/train_params/vggt_p7_p5b_shared_norm_projection_aux_tokenlr.yaml
```

实验：

```text
p7_fullfinetune_fromrobust_resetaux_nocrop_validqheight_tokenlr_pm3_overlap4_e30_b8_2gpu
```

路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_fullfinetune_fromrobust_resetaux_nocrop_validqheight_tokenlr_pm3_overlap4_e30_b8_2gpu
```

关键变化：

- 继续使用全量城市、no-crop、reset aux、`valid_quantile` normalized height；
- 显式给 `remote_projection_aux_token_norm/proj/pixel_head` 设置 `lr=5e-4`；
- `remote_projection_aux_token_global_head` 设置 `lr=1e-4`；
- 保持 remote point head `lr=2e-5`，避免过强 aux 训练拖坏 point head；
- 继续只保留 best/final checkpoint，避免存储膨胀。

阶段判断：

如果该分支在 epoch 1-2 仍没有正的 `contrast_pred_gap`，则可以基本排除“只是 token aux 学习率太低”的解释，下一步优先切到 `REMOTE_PROJECTION_AUX_SOURCE=dpt_init`，用 VGGT 原 depth decoder 初始化 height head 来测试表达能力上界。

实际结果：

```text
epoch 1.2:
high20_pred_mean ~= 0.0757
low80_pred_mean  ~= 0.0762
contrast_pred_gap ~= 0
```

修正 token aux 学习率后，`token_norm/proj/pixel_head` 已经达到 `lr=5e-4`，但 height 仍停在常数附近。结论是轻量 token Conv aux head 当前不是稳定的 height decoder；问题不只是学习率分组。

### 12.5 DPT-init height decoder upper-bound probe

新增代码开关：

```text
train_params.reinitialize_dpt_projection_aux_from_shared_after_warmstart=true
```

作用：

- warmstart robust checkpoint 后，将 `remote_projection_aux_height_head` 从已加载的 shared `model.depth_head` 重新拷贝；
- 将 `remote_projection_aux_offset_head` 从已加载的 shared `model.point_head` 重新拷贝；
- 避免 DPT aux 在模型构造阶段从随机/未 warmstart 的 head 初始化。

当前实验：

```text
p7_dptinit_fromrobust_reinit_nocrop_validqheight_pm3_overlap4_e30_b24_2gpu
```

路径：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_dptinit_fromrobust_reinit_nocrop_validqheight_pm3_overlap4_e30_b24_2gpu
```

设置：

- `REMOTE_PROJECTION_AUX_SOURCE=dpt_init`；
- `BATCH_SIZE=24`，两张可见 GPU，约 75 iter/epoch；
- `remote_projection_aux_height_head.lr=5e-5`；
- remote point head 保持 `pm3 + overlap4`；
- no-crop + `valid_quantile` normalized height；
- 使用 GT dir/slope 做 reconstruction 辅助损失，先验证 height 表达/优化上界。

初始现象：

```text
epoch 0.07:
high20_pred_mean ~= 0.159
low80_pred_mean  ~= 0.153
contrast_pred_gap ~= 0.007
gt_gap ~= 0.96
```

DPT-init 不再是 0.076 常数解，说明 decoder 初始化确实改变了输出分布；但初始高低分离仍非常弱。

实际中止点：

```text
epoch 2.4
```

关键现象：

```text
epoch 1.85:
high20_pred_mean ~= 0.2005
low80_pred_mean  ~= 0.1962
contrast_pred_gap ~= 0.0033
contrast_gt_gap   ~= 0.9745

epoch 2.4:
high20_pred_mean ~= 0.1971
low80_pred_mean  ~= 0.1983
contrast_pred_gap ~= -0.0038
contrast_gt_gap   ~= 0.9751
```

该分支没有形成有效高低分层。更重要的是，该配置来自 `aggtail2`，日志中 `model.aggregator.frame_blocks/global_blocks` 的学习率显示为 `0.000000` 量级，实际更接近“冻结主干 + 训练 aux/remote head”的探针，不满足后续要测试上界性能的 full-finetune 要求。因此该分支中止。

### 12.6 DPT-init Stage A full-finetune long run

新增配置：

```text
configs/train_params/vggt_p7_p5b_dptinit_fullfinetune.yaml
```

目的：

- 按 Stage A 目标测试当前方法的上界，而不是只训练轻量 aux head；
- 从 robust remote checkpoint warmstart，但排除旧 `remote_projection_aux_` 权重；
- warmstart 后用 shared `depth_head` 初始化 `remote_projection_aux_height_head`，用 shared `point_head` 初始化 `remote_projection_aux_offset_head`；
- shared VGGT aggregator/head 使用 full finetune 基础 LR，projection aux DPT height/global head 使用较高 LR；
- 继续使用 no-crop + `valid_quantile` 尺度无关 normalized height；
- reconstruction 辅助损失中先使用 GT global dir/slope，隔离测试 height 是否能学稳；
- `SAVE_FREQ=0`、`KEEP_FREQ=0`，只保留 best/final，控制存储。

实验：

```text
p7_dptinit_fullfinetune_fromrobust_nocrop_validqheight_gtglobal_e60_b16_2gpu
```

核心判断：

```text
rs_projection_rel_height_contrast_pred_gap 是否持续变为正且明显增大；
high20_pred_mean 是否能接近 high20_gt_mean 的相对幅度；
Seattle 493 / New York 461_1 的 height_compare 是否显示建筑高度结构；
GT dir/slope 下 aux global/grid-global remote 点云是否接近 pointmap GT；
remote pointmetric 是否不比 robust remote baseline 明显退化。
```

实际结果：

```text
p7_dptinit_fullfinetune_fromrobust_nocrop_validqheight_gtglobal_e60_b16_2gpu:
  BATCH_SIZE=16 / GPU，2 GPU，启动后 OOM。

p7_dptinit_fullfinetune_fromrobust_nocrop_validqheight_gtglobal_e60_b8_2gpu:
  BATCH_SIZE=8 / GPU，2 GPU，稳定，显存约 72GB/卡；
  在 epoch 5.7 左右提前中止。
```

关键日志：

```text
epoch ~= 5.36:
  rs_projection_rel_height_high20_gt_mean   ~= 1.06
  rs_projection_rel_height_high20_pred_mean ~= 0.18
  rs_projection_rel_height_low80_gt_mean    ~= 0.27
  rs_projection_rel_height_low80_pred_mean  ~= 0.18
  rs_projection_rel_height_contrast_pred_gap ~= -0.005
  rs_projection_rel_height_contrast_gt_gap   ~= 0.96
  rs_projection_rel_height_high20_mae        ~= 0.88
  rs_pointmap_loss                           ~= 0.016
```

结论：

- 该配置确实是 full finetune：`submodule_configs` 没覆盖的参数会走默认 `lr=1e-5`，因此 shared VGGT aggregator/block 参与训练，`patch_embed` 单独使用 `5e-7`；
- DPT 初始化 + full finetune + GT dir/slope reconstruction 仍没有让 height 形成高低分层，输出基本收敛到 `0.18~0.20` 的均值附近；
- 这说明当前瓶颈不只是 decoder 初始化或主干冻结，而是 height 监督在当前像素分布/权重下仍被低值区域和均值解吸收；
- 下一步应保持结构简单，先做高 height 区域重权重的 long run，而不是立刻改复杂结构。

### 12.7 DPT-init full-finetune high-height reweight long run

目的：

- 用 4 GPU 跑更长的上限测试；
- 保持 `dpt_init` aux 结构和 GT `dir/slope` teacher 不变；
- 明确提高 high-height 区域对 loss 的影响，避免 normalized height 学成全局均值；
- 同时保留 remote point head 的 `pm3 + overlap4`，监控 remote point head 不被 aux 训练拖坏。

计划输出：

```text
p7_dptinit_fullfinetune_highweight_fromrobust_nocrop_validqheight_gtglobal_e40_b10_4gpu
```

关键设置：

```text
NUM_GPUS=4
BATCH_SIZE=10
EPOCHS=40
LAMBDA_PROJ_REL_HEIGHT=1.5
PROJ_DENSE_REL_HEIGHT_WEIGHT=2.0
PROJ_DENSE_REL_HEIGHT_LOW_WEIGHT=0.05
PROJ_REL_HEIGHT_TARGET_WEIGHT=6.0
PROJ_REL_HEIGHT_TARGET_WEIGHT_GAMMA=1.5
PROJ_REL_HEIGHT_BALANCED_WEIGHT=2.0
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=1.5
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=2.0
PROJ_REL_HEIGHT_LOW_OVERPRED_WEIGHT=0.15
PROJ_RECON_GLOBAL_TO_GT=0.2
PROJ_RECON_GLOBAL_TO_GT_USE_GT_DIR=true
PROJ_RECON_GLOBAL_TO_GT_USE_GT_SLOPE=true
```

阶段判断：

- 如果 epoch `3~6` 后 `high20_pred_mean` 仍长期停在 `0.2` 附近，且 `contrast_pred_gap` 仍接近 0，则说明仅靠连续 L1/分桶重权重还不够，需要进入更强的 height 训练形式，例如 binned/ordinal height 或显式高建筑采样；
- 如果 high/low gap 明显上升，再继续跑长 epoch，并导出 Seattle 493 / New York 461_1 的 height_compare、aux global remote 点云和 remote-only point head 点云。

实际执行：

```text
p7_dptinit_fullfinetune_highweight_fromrobust_nocrop_validqheight_gtglobal_e40_b10_4gpu:
  当前系统只暴露 GPU 0/1，4 GPU 启动失败，rank 2/3 报 invalid device ordinal。

p7_dptinit_fullfinetune_highweight_fromrobust_nocrop_validqheight_gtglobal_e40_b10_2gpu:
  BATCH_SIZE=10 / GPU，2 GPU，稳定；
  显存约 91~93GB/卡，GPU 利用率 100%；
  epoch 2.5 提前中止。
```

关键日志：

```text
epoch ~= 2.5:
  rs_projection_rel_height_high20_gt_mean   ~= 1.06
  rs_projection_rel_height_high20_pred_mean ~= 0.19
  rs_projection_rel_height_low80_gt_mean    ~= 0.27
  rs_projection_rel_height_low80_pred_mean  ~= 0.19
  rs_projection_rel_height_contrast_pred_gap ~= 0.00
  rs_projection_rel_height_contrast_gt_gap   ~= 0.96
  rs_projection_rel_height_high20_mae        ~= 0.87
```

结论：

- 高 height 区域重权重、balanced bucket、contrast、dense height 加权都没有打破均值解；
- 这组负结果比 12.6 更强：full finetune + DPT-init + 高值重权重 + GT dir/slope reconstruction 仍然没有形成 normalized height 的高低分层；
- 下一步不应继续在同一 DPT-init 联合训练上拉长 epoch。需要隔离变量：去掉 remote point/reconstruction 的干扰，使用之前 Stage A 中能学习 height 的 token aux 路径，做 height-only upper-bound 检查。

### 12.8 Token aux height-only full-finetune isolation

目的：

- 判断“height 任务本身在全量数据上是否仍可学”；
- 去掉 remote point head loss 和 reconstruction loss，避免 point/reconstruction 对 height head 的优化方向产生干扰；
- 使用 `REMOTE_PROJECTION_AUX_SOURCE=tokens`，因为早期 Stage A 中 token aux 在 frozen-trunk 条件下已经能学出 height 形状；
- 仍使用 full finetune 默认 LR，使该实验能给出更接近上限的判断。

计划输出：

```text
p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b10_2gpu
```

关键设置：

```text
REMOTE_PROJECTION_AUX_SOURCE=tokens
LAMBDA_REMOTE_PM=0
LAMBDA_REMOTE_OVERLAP_PM=0
PROJ_RECON_GLOBAL_TO_GT=0
LAMBDA_PROJ_REL_HEIGHT=1.5
PROJ_DENSE_REL_HEIGHT_WEIGHT=2.0
PROJ_REL_HEIGHT_TARGET_WEIGHT=6.0
PROJ_REL_HEIGHT_BALANCED_WEIGHT=2.0
PROJ_REL_HEIGHT_CONTRAST_WEIGHT=1.5
PROJ_REL_HEIGHT_BUCKET_MEAN_WEIGHT=2.0
```

判断：

- 如果该实验 high/low gap 能快速上升，说明 height label/loss 基本可用，DPT-init head 或联合训练是主要问题；
- 如果该实验仍停在均值解，则需要重新设计 height 训练形式，例如二阶段分类/ordinal height 或更强的高建筑采样，而不是继续增加连续 L1 权重。

实际执行：

```text
p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b10_2gpu:
  BATCH_SIZE=10 / GPU，2 GPU；
  在 token aux pixel head 处 OOM，中止。

p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b8_2gpu:
  BATCH_SIZE=8 / GPU，2 GPU；
  显存约 86GB/卡，GPU 利用率接近满载；
  SAVE_FREQ=0，KEEP_FREQ=0；epoch 5 后只保存 checkpoint-best.pth。
```

阶段结果：

```text
epoch 0.0:
  high20_gt_mean    ~= 1.07
  high20_pred_mean  ~= 0.075
  low80_gt_mean     ~= 0.24
  low80_pred_mean   ~= 0.075
  contrast_pred_gap ~= 0.0

epoch 2.0:
  high20_gt_mean    ~= 1.12
  high20_pred_mean  ~= 1.12
  high20_mae        ~= 0.15
  low80_gt_mean     ~= 0.31
  low80_pred_mean   ~= 0.33
  low80_mae         ~= 0.17
  contrast_pred_gap ~= 0.94
  contrast_gt_gap   ~= 0.99

epoch 5.1:
  high20_gt_mean    ~= 1.06
  high20_pred_mean  ~= 1.02
  high20_mae        ~= 0.20
  low80_gt_mean     ~= 0.27
  low80_pred_mean   ~= 0.30
  low80_mae         ~= 0.18
  contrast_pred_gap ~= 0.87
  contrast_gt_gap   ~= 0.96

epoch 10.3:
  high20_pred_mean  ~= 1.05
  high20_mae        ~= 0.14
  low80_pred_mean   ~= 0.27
  low80_mae         ~= 0.13
  contrast_pred_gap ~= 0.94

epoch 20.0:
  high20_pred_mean  ~= 1.05
  high20_mae        ~= 0.13
  low80_pred_mean   ~= 0.28
  low80_mae         ~= 0.14
  contrast_pred_gap ~= 0.95
```

阶段结论：

- 与 12.6/12.7 的 DPT-init 结果不同，token aux height-only 在 full finetune 下没有停在均值解，high/low height 分层能在 warmup 结束前快速形成；
- 这说明 dense normalized height 标签和当前 height loss 不是根本不可学，前面失败更可能来自 DPT-init aux 结构、joint/reconstruction 耦合、或 remote point head 与 aux 任务同时优化时的梯度竞争；
- 长训到 epoch 20 后训练侧基本进入平台期，normalized height 的 high/low MAE 大致在 `0.12~0.15`；继续训练的目的主要是确认稳定性和获得 final 对照，而不是期待数量级提升；
- `checkpoint-best.pth` 在 epoch 15 附近刷新，epoch 20 没有刷新 best，后续可视化需要同时比较 best 与 final；
- 当前仍有低区域偏高和平滑化问题，不能只凭训练日志判断最终可用性。下一步需要在 Seattle 493、New York 461_1 导出 height compare 与 GT dir/slope 条件下的 aux global remote 点云；
- 如果可视化正确，再进入 Stage B：保留该 token height 路径，逐步加入 global dir/slope teacher、self-contained global reconstruction、remote point head loss，并用 remote scale-aligned pointmetric 判断是否改善 remote 点云。

完成结果：

```text
训练输出:
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b8_2gpu

保留权重:
  checkpoint-best.pth
  checkpoint-final.pth

清理:
  checkpoint-last.pth 已删除，避免额外占用 14GB+ 存储。
```

最终训练侧日志：

```text
epoch 29.6 avg:
  high20_gt_mean    ~= 1.06
  high20_pred_mean  ~= 1.05
  high20_mae        ~= 0.125
  low80_gt_mean     ~= 0.27
  low80_pred_mean   ~= 0.28
  low80_mae         ~= 0.130
  contrast_pred_gap ~= 0.94
  contrast_gt_gap   ~= 0.96
  dense_height_loss ~= 0.091

test epoch 30 avg:
  high20_gt_mean    ~= 1.05
  high20_pred_mean  ~= 1.05
  high20_mae        ~= 0.138
  low80_gt_mean     ~= 0.22
  low80_pred_mean   ~= 0.25
  low80_mae         ~= 0.136
  contrast_pred_gap ~= 0.93
  contrast_gt_gap   ~= 0.98
  dense_height_loss ~= 0.091
```

可视化输出：

```text
Seattle 493:
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_stagea_token_heightonly_e30_best
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_stagea_token_heightonly_e30_final

New York 461_1:
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_stagea_token_heightonly_e30_best
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_stagea_token_heightonly_e30_final
```

可视化检查：

- `best` 和 `final` 的导出结果一致或几乎一致；后续可直接使用 `checkpoint-best.pth` 作为 Stage B warmstart；
- New York 461_1 的 height 形状和幅度都较好：`rel_height_norm_mae ~= 0.144`，`affine_mae ~= 0.030`；
- Seattle 493 的 height 空间形状正确，但原始幅度/偏置偏大：`rel_height_norm_mae ~= 0.251`，`affine_mae ~= 0.0225`。这说明 token height 已经学到建筑/道路结构，但 normalized height 的全局 scale/shift 仍不稳定；
- 该结果支持进入 Stage B：在 token height checkpoint 上加入 GT `global_dir/slope` 条件下的 scale-independent global reconstruction loss，用点云重建约束 height 的幅度和空间投影，而不是继续单独调 height loss。
