# VGGT-Omega Crossview 训练记录

本文记录将 `VGGT-Omega` 接入 `map-anything` 后，在 Crossview 数据集上启动训练的方式、当前成功配置和日志查看方法。

## 代码入口

- 训练脚本：`bash_scripts/train/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512.sh`
- 模型配置：`configs/model/vggt_omega.yaml`
- 数据配置：`configs/dataset/vigor_chicago_rs_joint_512.yaml`
- 训练参数：`configs/train_params/vggt_omega_finetune.yaml`
- 模型 wrapper：`mapanything/models/external/vggt_omega/__init__.py`

## 权重路径

当前使用的 VGGT-Omega 512 checkpoint：

```bash
/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt
```

训练启动时日志应出现：

```text
Loading VGGT-Omega checkpoint from /root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt ...
<All keys matched successfully>
```

这表示 checkpoint key 与当前 `VGGTOmegaWrapper` 内部模型结构匹配。

## 当前已验证可运行命令

当前机器只有 1 张 RTX 3090，因此先使用 `NUM_VIEWS=2`、`BATCH_SIZE=2` 的保守单卡配置。该配置已经成功进入训练，并跑到 epoch 1。

```bash
cd /root/autodl-tmp/Models/map-anything

PRETRAINED_CKPT=/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt \
LOAD_CUSTOM_CKPT=true \
NUM_GPUS=1 \
CUDA_DEVICES=0 \
NUM_WORKERS=4 \
NUM_VIEWS=2 \
BATCH_SIZE=2 \
EPOCHS=50 \
WARMUP_EPOCHS=1 \
EVAL_FREQ=1 \
SAVE_FREQ=10 \
KEEP_FREQ=10 \
PRINT_FREQ=10 \
OUTPUT_DIR=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v \
bash bash_scripts/train/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512.sh 1
```

如果后续有更多显存或多卡，可以逐步调大：

- `NUM_VIEWS=4`：更接近原 Crossview/VGGT 实验。
- `BATCH_SIZE>=NUM_VIEWS`：脚本会在 batch 小于视角数时自动提升 batch。
- `NUM_GPUS=4 CUDA_DEVICES=0,1,2,3`：多卡训练形式与原 VGGT Crossview 脚本一致。

## 日志与输出

当前输出目录：

```bash
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v
```

实时查看训练日志：

```bash
tail -f /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v/train.log
```

TensorBoard 事件文件也写在同一目录，可用：

```bash
tensorboard --logdir /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v
```

## 当前训练状态示例

成功训练日志中的关键字段示例：

```text
Epoch: [1]  [240/450]
loss: 0.8080
aerial_loss: 0.4710
remote_loss: 0.2077
rs_pointmap_loss: 0.0519
max mem: 30925
```

字段含义：

- `loss`：总训练损失，包含 aerial 分支和 remote 分支。
- `aerial_loss`：普通地面/街景多视角监督损失。
- `remote_loss`：遥感 remote view 的监督损失；当前由 `RSPointmapHeightLoss` 计算。
- `rs_pointmap_loss`：remote pointmap 归一化后的点图损失。
- `rs_pointmap_loss_raw_metric`：remote pointmap 在原始 metric 空间下的 L1 量级，适合观察趋势，但尺度受场景影响较大。
- `rs_pointmap_pred_norm_factor` / `rs_pointmap_gt_norm_factor`：remote pointmap 损失归一化因子。
- `max mem`：训练器记录的峰值显存，当前单卡 512/2-view 约 `30925 MB`。

## 注意事项

1. `VGGT-Omega` 没有原版 VGGT 的 native `point_head`，当前 wrapper 使用 `camera + depth` 反投影得到全局 `pts3d`，然后复用 `vggt_loss_rs_joint` 的 remote pointmap 监督。
2. `VGGT-Omega` 默认 `patch_size=16`，所以使用 `vigor_chicago_rs_joint_512`，不要直接套原 VGGT 的 518 配置。
3. 不加载 checkpoint、随机初始化时会很快出现 NaN；正式训练必须设置 `LOAD_CUSTOM_CKPT=true` 和有效的 `PRETRAINED_CKPT`。
4. 当前单卡 3090 配置已经接近显存上限；调大 `NUM_VIEWS` 或 `BATCH_SIZE` 前应先确认显存。
5. `lr_model.aggregator.patch_embed` 日志早期显示为 `0.000000` 是因为真实值很小，默认格式四舍五入后显示为 0；配置里实际是 `5e-07`，且 warmup 阶段会更小。

## 恢复训练

如果训练中断，可使用同一输出目录并设置：

```bash
RESUME=true \
OUTPUT_DIR=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_1gpu_2v \
... bash bash_scripts/train/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512.sh 1
```

恢复依赖输出目录下的 checkpoint 文件。当前脚本默认 `RESUME=false`，避免误续旧实验。
