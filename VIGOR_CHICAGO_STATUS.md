# VIGOR Chicago 接入与训练状态

本文档记录当前将 VIGOR Chicago 重建数据接入 `Models/map-anything` 的工作状态，目标是让 MapAnything 框架可以在该数据上进行多视角训练，并验证训练链路已经打通。

说明：本文档中的文件链接统一使用相对于当前 Markdown 文件的相对路径，不使用绝对路径。

训练相关的详细设计、当前 loss 与后续 joint training 规划，单独整理在：[VIGOR_CHICAGO_TRAINING_CN.md](VIGOR_CHICAGO_TRAINING_CN.md)。

当前相关路径：

- 原始数据：`/root/autodl-tmp/outputs/experiments/exp_001_reconstrc/vigor_chicago_processed`
- 转换后的 WAI 数据：`/root/autodl-tmp/outputs/dataset/vigor_chicago_wai`
- split / metadata：`/root/autodl-tmp/outputs/dataset/mapanything_metadata/vigor_chicago`
- 训练输出：`/root/autodl-tmp/outputs/mapanything_experiments`

## 1. 当前工作的目标

当前工作分成三层：

1. 把自定义重建数据转换成 MapAnything 可读的 WAI 格式。
2. 在 `Models/map-anything` 中注册自定义数据集、配置和训练脚本。
3. 用最小配置验证训练可以真正启动、前向、反向并保存 checkpoint。

这三层目前都已经完成了最小闭环验证。

## 2. 原始数据格式

原始数据按场景组织，每个 `location_x` 视为一个 scene，例如：

```text
vigor_chicago_processed/
  location_1/
    location_1_00.jpg
    location_1_00.exr
    location_1_00.npz
    ...
```

每帧当前使用的信息：

- `.jpg`：RGB 图像
- `.exr`：深度图
- `.npz`：
  - `intrinsics`：`3 x 3`
  - `cam2world`：`4 x 4`

当前接入默认假设：

- 深度是可直接用于监督的 metric depth
- 外参符合 MapAnything 预期的 OpenCV cam2world 约定
- 每个 `location_x` 都是一个独立场景

## 3. 已新增的代码

### 3.1 数据转换与 split 生成

新增脚本：

- [scripts/convert_vigor_chicago_to_wai.py](scripts/convert_vigor_chicago_to_wai.py)
- [scripts/prepare_vigor_chicago_splits.py](scripts/prepare_vigor_chicago_splits.py)

作用：

- 将 `jpg + exr + npz` 转为 WAI scene 结构
- 为每个 scene 生成 `scene_meta.json`
- 生成 train / val / test scene list

新增 GT depth covisibility 生成脚本：

- [scripts/generate_vigor_chicago_gt_covisibility.py](scripts/generate_vigor_chicago_gt_covisibility.py)

作用：

- 复用 MapAnything / WAI 原生 covisibility 计算逻辑
- 直接使用 VIGOR Chicago 已有的 GT depth `.exr`
- 为每个 scene 生成真实的 `pairwise_covisibility--NxN.npy`
- 更新每个 scene 的 `scene_meta.json`，让 dataset 直接读取真实 covisibility

### 3.2 自定义数据集注册

新增：

- [mapanything/datasets/wai/vigor_chicago.py](mapanything/datasets/wai/vigor_chicago.py)

更新：

- [mapanything/datasets/__init__.py](mapanything/datasets/__init__.py)

该数据集类目前已经能够：

- 从 split 文件读取 scene 列表
- 读取 `scene_meta.json`
- 从 covisibility 中采样多视角组合
- 返回训练需要的 `img / depthmap / camera_pose / camera_intrinsics`

### 3.3 Hydra 配置

新增配置：

- [configs/dataset/vigor_chicago_wai/default.yaml](configs/dataset/vigor_chicago_wai/default.yaml)
- [configs/dataset/vigor_chicago_wai/train/default.yaml](configs/dataset/vigor_chicago_wai/train/default.yaml)
- [configs/dataset/vigor_chicago_wai/val/default.yaml](configs/dataset/vigor_chicago_wai/val/default.yaml)
- [configs/dataset/vigor_chicago_wai/test/default.yaml](configs/dataset/vigor_chicago_wai/test/default.yaml)
- [configs/dataset/vigor_chicago_50_518.yaml](configs/dataset/vigor_chicago_50_518.yaml)
- [configs/machine/autodl_vigor.yaml](configs/machine/autodl_vigor.yaml)

这些配置提供了：

- 数据根目录和 metadata 根目录
- 50 个 location 的 pilot 配置
- 训练与验证的默认分辨率与 split 接入

### 3.4 训练脚本

新增训练脚本：

- [bash_scripts/train/examples/vigor_chicago_50_vggt_finetune.sh](bash_scripts/train/examples/vigor_chicago_50_vggt_finetune.sh)
- [bash_scripts/train/examples/vigor_chicago_50_pi3_finetune.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_finetune.sh)
- [bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh)
- [bash_scripts/train/examples/vigor_chicago_50_pi3_smoke_2gpu.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_smoke_2gpu.sh)

其中：

- 前两个脚本是单卡 / 通用入口模板
- `vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh` 是已经实际跑通过的 2-GPU 预训练微调脚本
- 最后一个脚本是已经验证过的 2-GPU 最小 smoke test 配置

## 4. 为了跑通本地环境做的修复

### 4.1 covisibility mmap 文件格式

MapAnything 的 WAI loader 读取 covisibility 时，要求文件名里编码矩阵 shape，例如：

```text
pairwise_covisibility--45x45.npy
```

因此 covisibility 生成脚本使用 MapAnything 自己的 `store_data(..., "mmap")` 来写矩阵。

### 4.2 covisibility 文件选择逻辑

`VigorChicagoWAI` dataset 已改成优先读取 `scene_meta.json` 里登记的 `scene_modalities.pairwise_covisibility`。

这意味着：

- 如果 scene 已生成真实 covisibility，就直接用真实矩阵
- 只有在缺少该字段时，才会回退到旧的目录扫描逻辑

这样可以避免旧文件名或目录结构继续干扰训练。

### 4.3 covisibility 当前定义

当前 VIGOR Chicago 使用的 covisibility 已不再是早期占位用的全 1 矩阵。

现在使用的是“MapAnything 原生风格”的 covisibility：

- 从 GT depth 出发
- 将一个视图的深度点重投影到其他视图
- 按深度一致性阈值计算 pairwise overlap score

对应实现复用了：

- [data_processing/wai_processing/scripts/covisibility.py](data_processing/wai_processing/scripts/covisibility.py)
- [data_processing/wai_processing/utils/covis_utils.py](data_processing/wai_processing/utils/covis_utils.py)

### 4.4 torchvision 图像读取兼容

当前环境下，仓库原始实现里的：

```python
decode_image(str(fname))
```

不能工作，已经修成：

```python
decode_image(read_file(str(fname)))
```

修复文件：

- [mapanything/utils/wai/io.py](mapanything/utils/wai/io.py)

## 5. 已经生成的数据

当前 pilot 数据规模：

- scene 数：`50`
- frame 总数：`1682`
- 真实 covisibility scene 数：`50`

转换总结文件：

- [../../outputs/dataset/vigor_chicago_wai/conversion_summary.json](../../outputs/dataset/vigor_chicago_wai/conversion_summary.json)

GT depth covisibility 批处理总结文件：

- [../../outputs/dataset/vigor_chicago_wai/gt_covisibility_summary_v0_gtdepth_native.json](../../outputs/dataset/vigor_chicago_wai/gt_covisibility_summary_v0_gtdepth_native.json)

当前 split：

- train：`location_1` 到 `location_40`
- val：`location_41` 到 `location_45`
- test：`location_46` 到 `location_50`

split 总结文件：

- [../../outputs/dataset/mapanything_metadata/vigor_chicago/split_summary.json](../../outputs/dataset/mapanything_metadata/vigor_chicago/split_summary.json)

## 6. 已补齐的环境依赖

为了跑通当前训练链路，安装了这些最小依赖：

- `hydra-core`
- `uniception==0.1.6`
- `plyfile`
- `safetensors`
- `jaxtyping`
- `timm`
- `termcolor`
- `minio`
- `torchmetrics`
- `torchaudio==2.3.0`

说明：

- 当前训练路径不依赖 `rerun-sdk`
- `rerun-sdk` 主要影响可视化脚本和 `--viz`

## 7. 已完成的验证

### 7.1 dataloader 验证

自定义 dataset 已经成功取样，返回结果包括：

- `num_views = 4`
- 图像 tensor shape：`(3, 518, 518)`
- 深度 shape：`(518, 518, 1)`
- 有效 `4 x 4` pose
- 有效 `3 x 3` intrinsics

说明数据接入已经通了。

### 7.1.1 covisibility 现状

当前 `vigor_chicago_wai` 的每个 scene 都已经生成了真实 covisibility，输出目录形如：

```text
location_x/
  covisibility/
    v0_gtdepth_native/
      pairwise_covisibility--NxN.npy
      generation_summary.json
```

例如：

- [../../outputs/dataset/vigor_chicago_wai/location_1/covisibility/v0_gtdepth_native/pairwise_covisibility--45x45.npy](../../outputs/dataset/vigor_chicago_wai/location_1/covisibility/v0_gtdepth_native/pairwise_covisibility--45x45.npy)

旧的全 1 `covisibility/v0/full_dense*` 已从数据目录中清理，避免与真实 covisibility 混用。

### 7.2 2-GPU smoke test 验证

当前机器可见：

- GPU 0：`NVIDIA GeForce RTX 4080 SUPER`
- GPU 1：`NVIDIA GeForce RTX 4080 SUPER`

已经成功完成一次 2-GPU 最小训练运行，输出目录：

- [../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu)

产物包括：

- [checkpoint-last.pth](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu/checkpoint-last.pth)
- [checkpoint-final.pth](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu/checkpoint-final.pth)
- [log.txt](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu/log.txt)

首个训练 step 的结果：

- `train_loss = 81.6795`
- 每卡显存峰值约 `18.7 GB`

这说明：

- 数据能读
- 模型能建
- 分布式能起
- 前向和反向能过
- checkpoint 能保存

### 7.3 真实预训练微调路径验证

在 smoke test 之后，又进一步验证了真正的预训练微调路径：

- `model=pi3`
- `model.model_config.load_pretrained_weights=true`
- 2 卡训练
- 小规模配置：
  - `num_views=2`
  - `max_num_of_imgs_per_gpu=2`
  - `train.overfit_num_sets=2`
  - `val.overfit_num_sets=1`
  - `test.overfit_num_sets=1`
  - `epochs=10`
  - `train transform=imgnorm`

验证结果：

- 程序能够正常进入 Hugging Face 官方 Pi3 权重加载流程
- 训练已经实际完成一次小规模 2-GPU 预训练微调
- `cuDNN` 日志里的 `Plan failed` 只是 warning，不是致命报错
- 输出目录：
  - [../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu)
- 产物包括：
  - [checkpoint-last.pth](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu/checkpoint-last.pth)
  - [checkpoint-final.pth](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu/checkpoint-final.pth)
  - [log.txt](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu/log.txt)
  - [train.log](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu/train.log)
- 训练时间约 `21s`
- 显存峰值约 `23.7 GB / GPU`
- 日志中可见 epoch 级 loss 已持续记录，不是只启动到下载权重
  - epoch 1: `train_loss = 12.1578`
  - epoch 5: `train_loss = 0.9059`
  - epoch 9: `train_loss = 2.8871`

这意味着：

- 当前 `pretrained_2gpu` 脚本使用的是官方预训练 Pi3 权重
- 它已经不只是“能启动”，而是完成了一次真实的小规模 fine-tuning
- 只是规模仍然被压得很小，用来先验证资源、显存和训练稳定性

## 8. 什么叫 smoke 配置

这里的 “smoke 配置” 指的是“冒烟测试配置”。

含义不是追求最终效果，而是先验证整条链路是否能跑通。它的目标是最小化变量，确认：

1. 数据加载正常
2. 模型初始化正常
3. loss 计算正常
4. 反向传播正常
5. checkpoint 保存正常

当前 smoke 配置的特点：

- 模型：`pi3`
- GPU：`2`
- `num_views = 2`
- 每卡 `max_num_of_imgs_per_gpu = 2`
- 训练 scene 只取很少一部分
- `epochs = 1`
- 训练增强改成 `imgnorm`
- 目标是“能跑通”，不是“效果最好”

## 9. 当前这次算不算微调

严格说，已经跑通的 smoke test **不算真正的微调**。

原因是：

- smoke test 用的是 `model.model_config.load_pretrained_weights=false`
- 也就是没有加载 Pi3 的预训练权重

所以它本质上是：

- “训练链路验证”
- “最小训练验证”
- 不是“从预训练模型继续训练”

真正的 fine-tuning 一般指：

1. 以预训练权重作为起点
2. 模型结构不变
3. 在你的自定义数据上继续优化

而当前项目状态已经进一步推进：`load_pretrained_weights=true` 的小规模 Pi3 fine-tune 不仅能够启动，而且已经完成了一次实际训练并保存 checkpoint。

因此当前更准确的表述是：

- smoke test 不算真正微调
- `pi3_finetune_pretrained_2gpu` 这条路径已经算真正的小规模预训练微调
- 当前问题已经不是“能不能做预训练 fine-tune”，而是“如何把它扩展成更正式的实验配置”

## 10. 当前是不是全量微调

是的，当前属于全量微调，不是只训 head。

原因：

- 当前配置没有冻结 encoder
- 也没有冻结 decoder / prediction head
- 优化器会更新整套模型参数
- 只是通过 `pi3_finetune` 把 encoder 的学习率设得更小

因此更准确的说法是：

- 当前是“全量微调”
- 但采用了“不同子模块不同学习率”的策略

它不是下面这些情况：

- 不是 linear probing
- 不是只训练 head
- 不是冻结 backbone 的 partial fine-tune

### 10.1 如果想冻结部分网络，当前怎么做

当前框架已经支持按子模块前缀设置不同学习率；当某个子模块的 `lr=0` 时，该子模块参数会在优化器分组时被直接冻结。

对应逻辑见：

- [configs/train_params/pi3_finetune.yaml](configs/train_params/pi3_finetune.yaml)
- [configs/train_params/finetune_heads_only.yaml](configs/train_params/finetune_heads_only.yaml)
- [mapanything/utils/train_tools.py](mapanything/utils/train_tools.py)

对当前 `Pi3Wrapper` 来说，实际参数前缀是 `model.xxx`，因为 wrapper 将 Pi3 主体挂在 `self.model` 下。已经核对到的前缀示例包括：

- `model.encoder`
- `model.decoder`
- `model.point_decoder`
- `model.point_head`
- `model.conf_decoder`
- `model.conf_head`
- `model.camera_decoder`
- `model.camera_head`
- `model.register_token`

这意味着如果想冻结 encoder，推荐直接在训练配置里写：

```yaml
train_params:
  submodule_configs:
    model.encoder:
      lr: 0
```

同理，如果想只训练 head，可以只给 `model.point_head` / `model.conf_head` / `model.camera_head` 保留非零学习率。

需要注意：

- 仓库里的通用 `freeze_encoder.yaml` 使用的是 `encoder` 前缀
- 对 `Pi3Wrapper` 不能直接假设这个前缀一定命中
- 最稳妥的做法是按 `named_parameters()` 的真实前缀写成 `model.encoder`

### 10.2 当前是否支持 LoRA / Adapter 风格微调

当前仓库没有原生的 LoRA / PEFT / adapter 支持，也没有现成脚本或配置示例。

也就是说：

- 没有 `lora` / `LoRA` / `peft` 的内置实现
- 没有现成的 LoRA fine-tune 入口脚本
- 当前可直接复用的现成例子主要是“部分冻结”而不是“参数高效微调”

但从训练框架机制上说，LoRA 不是完全不可行。原因是当前优化器只依赖：

- `named_parameters()`
- `requires_grad`
- `submodule_configs`

因此如果后续手动把 LoRA 模块插到 Pi3 的线性层或 attention 层中，并让：

- 原始权重冻结
- LoRA 参数可训练

那么 MapAnything 的训练循环本身可以带着这些参数训练。

真正需要额外处理的是：

- LoRA 注入代码
- 预训练权重与 LoRA 参数的加载逻辑
- checkpoint 保存 / 恢复策略
- 训练脚本和配置命名

结论是：

- “部分冻结”当前已经方便做
- “LoRA 微调”当前没有现成支持，但可以作为后续定制开发项

## 11. 当前损失函数、可替换损失与自定义难度

### 11.1 当前 Pi3 训练实际使用的损失

当前 Pi3 训练使用的 loss 配置是：

- [configs/loss/pi3_loss.yaml](configs/loss/pi3_loss.yaml)

当前 `train_criterion` / `test_criterion` 的总体形式为：

```text
ExcludeTopNPercentPixelLoss(
  FactoredGeometryRegr3DPlusNormalGMLoss(
    RobustRegressionLoss(...)
  )
)
```

其核心监督项包括：

- 世界坐标点 `pts3d`
- 相机坐标点 `pts3d_cam`
- `depth_z`
- `ray_directions`
- 成对相对位姿
- `normal` 与 `gm` 项

当前 Pi3 loss 的关键特点：

- `compute_pairwise_relative_pose_loss=true`
- `compute_world_frame_points_loss=true`
- `convert_predictions_to_view0_frame=true`
- `top_n_percent=5`
- 使用 robust regression 作为基础误差形式

### 11.2 当前仓库里已有的 loss 类型

loss 实现主要在：

- [mapanything/train/losses.py](mapanything/train/losses.py)

当前仓库中已实现的代表性 loss 类包括：

- `L1Loss`
- `L2Loss`
- `RobustRegressionLoss`
- `ConfLoss`
- `ExcludeTopNPercentPixelLoss`
- `ConfAndExcludeTopNPercentPixelLoss`
- `Regr3D`
- `FactoredGeometryRegr3D`
- `FactoredGeometryRegr3DPlusNormalGMLoss`
- `FactoredGeometryScaleRegr3D`
- `FactoredGeometryScaleRegr3DPlusNormalGMLoss`
- `DisentangledFactoredGeometryScaleRegr3D`
- `DisentangledFactoredGeometryScaleRegr3DPlusNormalGMLoss`

对应现成配置示例见：

- [configs/loss/pi3_loss.yaml](configs/loss/pi3_loss.yaml)
- [configs/loss/vggt_loss.yaml](configs/loss/vggt_loss.yaml)
- [configs/loss/overall_loss.yaml](configs/loss/overall_loss.yaml)
- [configs/loss/overall_disentangled_loss.yaml](configs/loss/overall_disentangled_loss.yaml)
- [configs/loss/moge2_loss.yaml](configs/loss/moge2_loss.yaml)
- 以及若干 ablation 配置，例如 `no_depth_loss` / `no_pose_loss` / `no_points_loss`

### 11.3 改 loss 是否方便

比较方便。

原因是当前训练循环不是把某个 loss 写死在代码里，而是在训练启动时从 Hydra 配置读取 loss 字符串，再动态实例化。

对应逻辑见：

- [mapanything/train/training.py](mapanything/train/training.py)

也就是说，改 loss 通常有两种层级：

1. 轻量改法：
   - 直接切换 `configs/loss/*.yaml`
   - 或者新建一个 loss yaml，重组已有 loss 类
2. 深度改法：
   - 在 `mapanything/train/losses.py` 里新增 loss 类
   - 再在 yaml 中把 `train_criterion` 写成新的构造表达式

### 11.4 自定义 loss 时最关键的约束

不是框架本身难改，而是：

- loss 读取的字段
- model wrapper 输出的字段
- dataset 提供的 GT 字段

这三者必须对齐。

例如如果新 loss 想额外监督一个新分支输出，那么：

- model 必须把这个新分支结果输出出来
- batch 中必须有对应 GT 或可构造 supervision
- loss 配置和实现必须读取到这个键

## 12. 如果改变模型结构，比如加入一个新的分支

如果只是改 Pi3 内部结构，例如增加一个新的 prediction branch，通常不只是改一个模型文件，而是要连带完成下面这些工作。

### 12.1 必改项

1. 改底层模型定义
   - 通常落在 [mapanything/models/external/pi3/models/pi3.py](mapanything/models/external/pi3/models/pi3.py)
2. 改 wrapper 输出接口
   - 对应 [mapanything/models/external/pi3/__init__.py](./mapanything/models/external/pi3/__init__.py)
3. 改 loss
   - 如果新分支要参与监督，就必须改 [mapanything/train/losses.py](mapanything/train/losses.py) 或新增 loss 配置
4. 改 train_params
   - 如果新分支需要单独学习率、冻结策略或 schedule，就要加到 `submodule_configs`

### 12.2 容易忽略但实际很关键的工作

- 预训练权重兼容
  - 当前 Pi3 通过 Hugging Face 官方权重加载
  - 一旦结构变了，新分支通常拿不到现成权重
  - 往往需要 `strict=False` 或单独初始化新分支
- checkpoint 兼容
  - 新结构会改变 state dict key
- benchmark / inference 兼容
  - 如果新分支改变输出语义，后续 benchmark 或推理代码也可能要同步改

### 12.3 什么时候需要注册成一个新模型

如果只是对现有 `Pi3Wrapper` 做少量扩展，可以继续沿用 `model=pi3` 的入口。

如果改动已经比较大，例如：

- 新增明显不同的结构分支
- 改了输出定义
- 不再希望和原始 Pi3 checkpoint 语义混用

那么更稳妥的做法是：

- 新建一个 wrapper
- 在 [mapanything/models/__init__.py](mapanything/models/__init__.py) 中注册新的 `model_str`
- 同时新增对应 `configs/model/*.yaml`

## 13. 哪些因素主要增加显存，哪些主要增加训练时间

### 13.1 对显存最敏感的因素

以下因素通常会显著增加显存占用：

1. `num_views`
2. `max_num_of_imgs_per_gpu`
3. 输入分辨率
4. 模型规模
5. 是否关闭 gradient checkpointing

更具体地说：

- `num_views` 是最关键因素之一，多视角 Transformer 在 `2 -> 4 -> 8` 时通常会明显变重
- `max_num_of_imgs_per_gpu` 对显存影响最直接
- 分辨率升高会同时推高 backbone、decoder 和 head 的激活显存
- 模型规模越大，参数和优化器状态显存越高
- 打开 gradient checkpointing 往往能省显存，但会牺牲速度

### 13.2 对训练时间最敏感的因素

以下因素更直接增加总训练时间：

1. `epochs`
2. 训练 scene 数 / 数据规模
3. `num_views`
4. 分辨率
5. 验证频率 `eval_freq`
6. 数据增强和 dataloader 复杂度

更具体地说：

- `epochs` 翻倍，训练总时间基本也会翻倍
- scene 数增多会拉长每个 epoch
- `num_views` 和分辨率不仅增显存，也会显著拖慢每 step
- `eval_freq` 越高，整体 wall-clock 时间越长
- 更强的数据增强一般更耗 CPU 和 data time，而不是主要吃 GPU 显存

### 13.3 主要是“时间增加大于显存增加”的因素

这类因素更偏时间成本：

- `epochs`
- `eval_freq`
- train/val/test scene 数
- dataloader `num_workers`
- 额外日志、可视化和评测

### 13.4 通常“显存和时间都会一起增加”的因素

这类最需要谨慎：

- `num_views`
- `max_num_of_imgs_per_gpu`
- 分辨率
- 更大的模型结构

## 14. 针对当前任务的典型配置范围

下面给的是针对你当前 2 张 `4080 SUPER`、Pi3、多视角重建、自定义 50-scene pilot 的实用范围。

### 14.1 最小验证配置

适合先确认训练稳定：

- `num_views`: `2`
- `max_num_of_imgs_per_gpu`: `2`
- `train_overfit_num_sets`: `2 ~ 8`
- `epochs`: `1 ~ 3`
- `eval_freq`: `0` 或 `1`
- train transform: `imgnorm`

特点：

- 最稳
- 最省显存
- 主要用于确认真实微调能持续跑

### 14.2 小规模真实 fine-tune 配置

适合开始看收敛：

- `num_views`: `2 ~ 4`
- `max_num_of_imgs_per_gpu`: `2 ~ 4`
- train scenes: `20 ~ 40`
- val scenes: `5`
- test scenes: `5`
- `epochs`: `5 ~ 20`
- `eval_freq`: `1` 或 `2`
- train transform: 先 `imgnorm`，稳定后可切回增强版

这是下一阶段最实用的区间。

### 14.3 50 个 location 的第一版正式训练建议

如果要开始做第一版正式实验，建议先从这里起步：

- `num_views=4`
- `max_num_of_imgs_per_gpu=2`
- `train scenes=40`
- `val scenes=5`
- `test scenes=5`
- `epochs=10 ~ 30`
- `eval_freq=1`
- `load_pretrained_weights=true`

如果显存紧张，优先保持：

- `num_views=4`
- 降低 `max_num_of_imgs_per_gpu`

也就是说，遇到显存不够时，优先减 batch，不要先减 scene 数和 epoch 数。

### 14.4 不建议一开始就拉太大的配置

不建议一开始直接上：

- `num_views >= 8`
- 很大的 batch
- 全 50 scene + 长 epoch + 强增强同时开启

原因是：

- 调试成本高
- 一旦出问题，不容易判断是数据、模型、显存还是训练策略的问题
- 对当前阶段不够稳妥

## 15. 当前仍然简化的地方

虽然链路已经跑通，但当前 pilot 仍然是简化版：

- smoke test 只用了极少 scene
- smoke test 没有加载预训练权重
- smoke test 把 `num_views` 压到了 `2`
- smoke test 为了稳定，把 train transform 改成了 `imgnorm`
- 已完成的 pretrained fine-tune 也仍然是小规模 overfit 风格配置：
  - `num_views=2`
  - 每卡 `max_num_of_imgs_per_gpu=2`
  - `train.overfit_num_sets=2`
  - `epochs=10`
- 目前还没有做更正式的 40/5/5 场景级训练和验证
- 当前仓库没有原生 LoRA / PEFT 支持
- 如果后续要做“冻结部分模块”或“参数高效微调”，还需要专门整理配置策略

这些都是刻意为“先跑通”做的简化，不是最终实验配置。

## 16. 下一步建议

下一阶段建议按这个顺序推进：

1. 保持 `pi3`，在已经跑通的小规模 pretrained fine-tune 基础上，扩大训练配置
   - 先从 `num_views=2 -> 4` 开始
   - 逐步增大 scene 数
   - 再增大 epoch 数
2. 在预训练微调能稳定跑通后，再系统比较不同微调策略：
   - 全量微调
   - 冻结 encoder
   - heads-only
   - 后续若需要，再评估是否值得接入 LoRA
3. 最后再做：
   - 更合理的 covisibility / overlap
   - 更完整的验证集评估
   - 与 `vggt` 的对比

## 17. 常用命令

重建 50-scene 的 WAI pilot 数据：

```bash
cd /root/autodl-tmp/Models/map-anything
PYTHONPATH=. python scripts/convert_vigor_chicago_to_wai.py --max_locations 50 --overwrite
python scripts/prepare_vigor_chicago_splits.py
```

运行已经验证过的 2-GPU smoke test：

```bash
cd /root/autodl-tmp/Models/map-anything
bash bash_scripts/train/examples/vigor_chicago_50_pi3_smoke_2gpu.sh
```

运行已经实际验证过的小规模 Pi3 pretrained fine-tune：

```bash
cd /root/autodl-tmp/Models/map-anything
bash bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh
```

## 18. 当前结论

截至现在，项目状态已经从“自定义数据还没接入 MapAnything”，推进到了：

- 自定义数据已转换成 WAI
- 自定义 dataset 已注册进 MapAnything
- Hydra 配置已接好
- 本地环境已补齐到可训练
- 2-GPU 端到端 smoke test 已成功完成
- 2-GPU 小规模 Pi3 预训练微调也已实际完成并保存 checkpoint

也就是说，当前已经不再是“能不能接进来”的问题，而是：

- 如何把当前的小规模预训练微调扩展成正式实验
- 如何选择合适的冻结 / 微调策略
- 是否需要为后续实验引入新的 loss 或新的结构分支


## 19. RS-Aerial benchmark 当前状态

在训练链路之外，当前还新增了一个面向“空中视图 + 遥感视图”任务的 benchmark 骨架，位置在：

- `benchmarking/rs_guided_dense_mv/`

### 19.1 当前 benchmark 分层

当前 benchmark 分为两层：

- Stage-0：空中视图 baseline
  - 入口：`benchmarking/rs_guided_dense_mv/benchmark.py`
  - 指标：`pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_ate_rmse`、`pose_auc_5`、`ray_dirs_err_deg`
- Stage-1：遥感视图单独几何评测
  - 入口：`benchmarking/rs_guided_dense_mv/benchmark_stage1.py`
  - 指标：`rs_pointmap_abs_rel`、`rs_height_mae`、`rs_height_rmse`

当前还没有实现 Stage-2 joint benchmark，因此：

- Stage-0 结果文件只包含空中视图指标
- Stage-1 结果文件只包含遥感视图指标
- 不会在同一个结果文件里同时出现两类指标

### 19.2 遥感数据接入

为接入 `exp_005` 新输出，新增了：

- `scripts/prepare_rs_aerial_benchmark_metadata.py`
- `mapanything/datasets/wai/vigor_chicago_rs_aerial.py`

当前遥感监督使用：

- `pixel_to_point_map.npz['xyz']`
- `valid_mask.npy`
- `height_map.npy`
- `info.json`

manifest 输出目录：

- `outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial`

### 19.3 运行脚本

当前 benchmark 相关脚本：

- Stage-0 Pi3：`bash_scripts/benchmark/rs_guided_dense_mv/pi3.sh`
- Stage-0 VGGT：`bash_scripts/benchmark/rs_guided_dense_mv/vggt.sh`
- Stage-1 Pi3：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage1.sh`
- Stage-2 Pi3：`bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage2.sh`

### 19.4 当前已验证结果

- Stage-0 Pi3：已跑通
- Stage-1 Pi3：已在 7 个可用遥感 scene 上跑通
- Stage-1 输出目录：`outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/pi3_stage1_7scenes_v2`
- Stage-1 当前平均结果：
  - `rs_pointmap_abs_rel: 1.4527127572468348`
  - `rs_height_mae: 1.033700440611158`
  - `rs_height_rmse: 1.1238195385251726`

更完整的 benchmark 定义与结构说明见：

- `RS_GUIDED_DENSE_MV_BENCHMARK_DESIGN_CN.md`


### 19.5 Stage-2 joint skeleton

当前已新增 Stage-2 joint benchmark 骨架：

- `benchmarking/rs_guided_dense_mv/benchmark_stage2.py`
- `configs/rs_aerial_stage2_benchmark.yaml`
- `configs/dataset/benchmark_vigor_chicago_rs_aerial_stage2.yaml`
- `bash_scripts/benchmark/rs_guided_dense_mv/pi3_stage2.sh`

其当前作用是：

- 仅在 paired scenes 上同时输出 aerial 指标与 remote 指标
- 保持结果组织为 per-scene / avg-across-scenes
- 当前已新增第一个真正的 cross-view 指标：`crossview_pointmap_gap_abs`

它当前还不是完整的跨视图一致性 benchmark，尚未实现：

- 联合点云对齐后的整体误差
- 空中 / 遥感预测点云重叠区域误差
- 其他真正的 cross-view consistency 指标
