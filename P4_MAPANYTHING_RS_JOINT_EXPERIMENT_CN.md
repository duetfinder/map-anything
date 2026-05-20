# P4 MapAnything RS-joint 实验总结

本文档总结当前 `P4 MapAnything RS-joint` 实验的目标、模型结构、训练入口、默认配置、已完成验证和下一步建议。

相关文档：

- 模型设计：[MAPANYTHING_RS_JOINT_MODEL_CN.md](MAPANYTHING_RS_JOINT_MODEL_CN.md)
- loss 说明：[MAPANYTHING_LOSSES_CN.md](MAPANYTHING_LOSSES_CN.md)
- 训练总记录：[VIGOR_CHICAGO_TRAINING_CN.md](VIGOR_CHICAGO_TRAINING_CN.md)
- 脚本目录：[bash_scripts/train/Crossview/mapanything](bash_scripts/train/Crossview/mapanything)

## 1. 实验目标

`P4` 的目标是把前面在 `pi3` 上验证过的 RS joint supervision 迁移到 `MapAnything` 主体上，并进一步利用 `MapAnything` 的原生多视角 transformer 和 DPT head 结构。

核心问题是：

- aerial view 有标准 perspective 相机几何，可以继续使用原始 `MapAnything` 的 factorized geometry
- remote / satellite view 没有可靠相机模型，不适合强行输出 `raydirs + depth + pose`
- remote 侧最稳定的监督是 `remote_pointmap`

因此当前实验采用：

- aerial 分支保留原始 `MapAnything`
- remote 分支新增 direct pointmap head
- 中间 shared transformer 共享
- remote loss 第一版只强调 pointmap

## 2. 模型结构

当前模型是：

- `model=mapanything_rs_joint`
- 配置：[configs/model/mapanything_rs_joint.yaml](configs/model/mapanything_rs_joint.yaml)
- 实现：[mapanything/models/mapanything/rs_joint.py](mapanything/models/mapanything/rs_joint.py)

结构上分为三部分：

- aerial branch  
  复用原始 `MapAnything`，输出 `raydirs + depth + pose + confidence + mask`，再恢复 `pts3d / pts3d_cam`。

- remote branch  
  新增 `remote_encoder + remote_dpt_feature_head + remote_dpt_regressor_head`，直接输出 `pointmap + confidence + mask`。

- shared transformer  
  继续使用原始 `info_sharing` transformer，不改 block 结构，只在 forward 中做 aerial / remote 分流和合流。

remote view 通过：

```text
view["instance"] == "remote"
```

进行识别。

## 3. 初始化策略

当前实验是微调，不是从零训练。

初始化方式：

- `aerial encoder / aerial dense head / pose head / scale head / info_sharing` 从 `MapAnything` checkpoint 加载
- `remote_encoder` 从 `aerial encoder` 镜像初始化
- `remote_dpt_feature_head` 从 aerial DPT feature head 镜像初始化
- `remote_dpt_regressor_head` 只拷贝 shape 对齐的中间层
- remote 最终 `5-channel` 输出层新初始化

当前默认 checkpoint：

```text
/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth
```

如果需要换 checkpoint，可以通过 `PRETRAINED_CKPT` 覆盖。

## 4. 训练参数策略

微调参数配置：

- [configs/train_params/mapanything_rs_joint_finetune.yaml](configs/train_params/mapanything_rs_joint_finetune.yaml)

默认冻结：

- aerial encoder
- 原几何输入 encoder
- 原 `dpt_feature_head / dpt_regressor_head`
- `pose_head`
- `scale_head`

默认训练：

- `remote_encoder`
- `remote_dpt_feature_head`
- `remote_dpt_regressor_head`
- `info_sharing`
- `aerial_view_type_embedding`
- `remote_view_type_embedding`

这样做的目的：

- 保留原始 `MapAnything` 的 aerial 几何能力
- 让 remote 分支学习 satellite 域
- 让 shared transformer 适配 aerial + remote 的联合输入

## 5. Loss 设计

当前 loss 配置：

- [configs/loss/pi3_loss_rs_joint.yaml](configs/loss/pi3_loss_rs_joint.yaml)

虽然文件名仍然带 `pi3`，但当前 `P4` 中它作为通用 joint loss 容器使用。

loss 组成：

- aerial loss  
  使用原始 `MapAnything` 的 `FactoredGeometryRegr3DPlusNormalGMLoss`。

- remote loss  
  使用 `RSPointmapHeightLoss`，当前主项是 pointmap。

正式脚本当前默认：

```text
LAMBDA_REMOTE_PM=6
LAMBDA_REMOTE_H=0.0
SCALE_REMOTE_BY_NUM_VIEWS=true
REMOTE_COMPARE_IN_VIEW0=true
REMOTE_DETACH_POSE_ALIGN=false
```

`SCALE_REMOTE_BY_NUM_VIEWS=true` 表示 remote loss 会按 aerial view 数放大，避免 remote 只有单 view 时在 joint loss 中被相对稀释。

## 6. 脚本入口

当前所有 `P4 MapAnything` 训练脚本已经迁移到：

- [bash_scripts/train/Crossview/mapanything](bash_scripts/train/Crossview/mapanything)

### 6.1 1 GPU debug / smoke

脚本：

- [bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_debug_1gpu.sh](bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_debug_1gpu.sh)

用途：

- 快速检查模型 forward
- 检查 joint loss
- 检查 backward
- 检查 checkpoint 保存
- 检查 val/test 评估链路

默认设置：

```text
NUM_GPUS=1
CUDA_DEVICES=0
NUM_VIEWS=2
BATCH_SIZE=2
TRAIN_SETS=8
VAL_SETS=4
TEST_SETS=4
LAMBDA_REMOTE_PM=0.2
EPOCHS=1
```

运行示例：

```bash
cd /root/autodl-tmp/Models/map-anything
bash bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_debug_1gpu.sh
```

### 6.2 4 GPU formal

脚本：

- [bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all.sh](bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all.sh)

用途：

- 当前正式 `P4 MapAnything RS-joint` 训练入口

默认设置：

```text
NUM_GPUS=4
CUDA_DEVICES=0,1,2,3
NUM_WORKERS=4
NUM_VIEWS=4
BATCH_SIZE=12
EPOCHS=50
WARMUP_EPOCHS=1
LR=5e-06
MIN_LR=5e-08
WEIGHT_DECAY=0.05
SAVE_FREQ=10
KEEP_FREQ=10
PRINT_FREQ=20
```

数据与 remote 设置：

```text
dataset=vigor_chicago_rs_joint_518
cities=[chicago]
RS_PROVIDER=Google_Satellite,Bing_Satellite
REMOTE_PROVIDER_SAMPLING_MODE=random
REMOTE_TRAIN_CROP_MODE=random_scale_offset
REMOTE_VAL_CROP_MODE=random_scale_offset
REMOTE_TEST_CROP_MODE=none
REMOTE_CROP_SCALE_MIN=0.6
REMOTE_CROP_SCALE_MAX=1.0
REMOTE_IMAGE_RESIZE_MODE=nearest
REMOTE_LABEL_RESIZE_MODE=nearest
```

运行示例：

```bash
cd /root/autodl-tmp/Models/map-anything
PRETRAINED_CKPT=/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
CUDA_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all.sh
```

## 7. 当前验证状态

已经完成：

- `mapanything_rs_joint` 模型注册
- 新模型配置解析
- 新训练参数配置解析
- 1 GPU smoke 训练完整跑通
- val/test 评估链路跑通
- checkpoint 保存链路跑通
- 正式 4 GPU 脚本完成静态语法检查

历史 smoke 结果：

```text
loss=23.67
aerial_loss=23.07
remote_loss=0.30
remote_loss_weight_effective=2.00
rs_pointmap_loss=1.50
```

历史 smoke 输出目录：

```text
../../outputs/mapanything_experiments/mapanything/training/vigor_chicago/p4_mapanything_rs_joint_debug_1gpu
```

当前脚本默认 smoke 输出目录：

```text
../../outputs/mapanything_experiments/mapanything/training/Crossview/mapanything/p4_mapanything_rs_joint_debug_1gpu
```

## 8. 已修复的工程问题

为跑通 `P4` smoke，已经修复：

1. 脚本工作目录问题  
   训练脚本现在会自动 `cd` 到仓库根目录，再调用 `scripts/train.py`。

2. torchvision/Pillow hue jitter 溢出问题  
   在 [mapanything/datasets/base/base_dataset.py](mapanything/datasets/base/base_dataset.py) 中将 `ColorJitter` 的 hue 抖动关闭，避免当前环境里负 hue 偏移触发 `uint8` overflow。

## 9. 当前距离正式长跑的状态

当前代码链路已经可以起正式实验。剩余风险主要不在工程接线，而在训练配方：

- `LAMBDA_REMOTE_PM=6` 是否过强
- `info_sharing` 是否应该全量微调
- `NUM_VIEWS=4, BATCH_SIZE=12` 的显存和吞吐是否稳定
- remote loss 乘 `num_aerial_views` 后是否需要重新定标
- Google/Bing random provider sampling 是否会带来更强的数据噪声

建议正式长跑前先做一次接近正式配置的短跑：

```bash
cd /root/autodl-tmp/Models/map-anything
EPOCHS=3 \
SAVE_FREQ=1 \
KEEP_FREQ=1 \
PRINT_FREQ=1 \
PRETRAINED_CKPT=/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
CUDA_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash bash_scripts/train/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all.sh
```

观察重点：

- `aerial_loss`
- `remote_loss`
- `remote_loss_weight_effective`
- `rs_pointmap_loss`
- 显存峰值
- checkpoint 是否正常保存

如果短跑稳定，再进入 50 epoch 正式训练。
