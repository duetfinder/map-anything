# MapAnything 中 Pi3 微调与官方 Pi3 训练代码对比

本文档记录当前在 `Models/map-anything` 中使用 Pi3 进行微调时，与 `Models/Pi3` 官方训练代码之间的关键差异，并说明这些差异对横向 benchmark 的含义。

## 结论

- 当前 `MapAnything` 中的 Pi3 训练，不是“官方 Pi3 训练代码直接换数据集”。
- 它的本质是：把 Pi3 作为外部模型接入 `MapAnything` 的统一训练框架，并使用统一的数据接口、统一输出格式、统一损失体系和统一训练循环来进行微调。
- 因此，训练损失、数据增强、训练流程、默认微调策略都与官方 Pi3 不同。
- 这种做法适合做横向 benchmark 和多模型公平对比，但不适合拿来严格复现官方 Pi3 论文中的训练行为。

## 1. 损失函数差异

### 官方 Pi3

官方损失定义在 `Models/Pi3/pi3/models/loss.py`。

- 总损失为 `Pi3Loss = PointLoss + 0.1 * CameraLoss`
- `PointLoss`
  - 先对预测局部点图和 GT 做尺度对齐
  - 对局部点图做深度加权 L1
  - 可选 normal loss
  - 可选 global point loss
- `CameraLoss`
  - 基于所有视角对之间的相对位姿
  - 平移使用 Huber loss
  - 旋转使用角度误差
- 训练前会把 GT 和预测统一到 view0 坐标系，并做尺度归一化

### MapAnything 中的 Pi3

Pi3 在 `MapAnything` 中使用的是统一 loss 配置，定义在 `Models/map-anything/configs/loss/pi3_loss.yaml`。

- 总体形式为：
  - `ExcludeTopNPercentPixelLoss(FactoredGeometryRegr3DPlusNormalGMLoss(...))`
- 监督项包括：
  - 世界坐标点 `pts3d`
  - 相机坐标点 `pts3d_cam`
  - `depth_z`
  - `ray_directions`
  - 成对相对位姿
  - normal / gm 项
- 关键配置：
  - `compute_pairwise_relative_pose_loss=True`
  - `convert_predictions_to_view0_frame=True`
  - `compute_world_frame_points_loss=True`
  - `depth_type_for_loss='depth_z'`
  - `top_n_percent=5`

### 影响

- 两边 loss 不是同一个目标函数，loss 数值和收敛曲线不能直接对比。
- 官方 Pi3 更偏“Pi3 专用损失”。
- MapAnything 更偏“统一几何监督框架”。
- 对横向 benchmark 而言，后者更合理；对官方复现而言，前者更合理。

## 2. 数据增强差异

### 官方 Pi3

配置见 `Models/Pi3/configs/data/example.yaml`，实现见 `Models/Pi3/datasets/base/transforms.py` 和 `Models/Pi3/datasets/base/base_dataset.py`。

- 训练增强默认使用 `JitterJpegLossBlurring`
- 包含：
  - `ColorJitter`
  - `JpegLoss`
  - `Blurring`
- 同时有几何增强：
  - `aug_crop`
  - `aug_focal`
- 官方示例中常用 `frame_num=8`

### MapAnything

配置和实现见：

- `Models/map-anything/mapanything/datasets/base/base_dataset.py`
- `Models/map-anything/configs/dataset/vigor_chicago_wai/train/default.yaml`
- `Models/map-anything/bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh`

MapAnything 的通用训练增强较简单，主要是：

- `imgnorm`
- `colorjitter`
- `colorjitter+grayscale+gaublur`

当前我们实际跑通的 Pi3 微调脚本里，为了先稳定验证链路，显式覆盖成了：

- `dataset.vigor_chicago_wai.train.transform=imgnorm`

也就是说，当前这条 VIGOR Chicago 微调链路的增强强度明显弱于官方 Pi3 训练。

## 3. 训练流程差异

### 官方 Pi3

关键文件：

- `Models/Pi3/trainers/pi3_trainer.py`
- `Models/Pi3/pi3/models/pi3_training.py`
- `Models/Pi3/configs/model/pi3.yaml`

特点：

- 使用官方 Pi3 专用 trainer
- 使用官方 `Pi3Loss`
- 默认 `load_vggt: true`
- 默认 `freeze_encoder: true`
- 学习率调度默认是 `OneCycleLR`
- 默认是多数据集混训
- 默认多视角数通常更大，如 `frame_num=8`

### MapAnything

关键文件：

- `Models/map-anything/scripts/train.py`
- `Models/map-anything/mapanything/train/training.py`
- `Models/map-anything/mapanything/utils/inference.py`
- `Models/map-anything/mapanything/models/external/pi3/__init__.py`

特点：

- 使用 Hydra 统一配置系统
- 使用 MapAnything 通用训练循环
- 外部模型通过 wrapper 接入
- 所有模型输出统一格式
- loss 通过统一接口对 `batch + preds` 计算
- 优化器是统一的 AdamW 参数组逻辑
- 调度器来自 `train_params`，默认是 warmup + cosine 风格

## 4. 微调策略差异

### 官方 Pi3 默认策略

见 `Models/Pi3/configs/model/pi3.yaml`：

- `freeze_encoder: true`

这意味着官方默认更接近“冻结 encoder 的微调”，不是全量微调。

### 当前 MapAnything 中的 Pi3 微调

见：

- `Models/map-anything/configs/train_params/pi3_finetune.yaml`
- `Models/map-anything/configs/model/pi3.yaml`

特点：

- `load_pretrained_weights=true` 时，从 Hugging Face 加载官方公开 Pi3 权重
- 没有冻结 encoder
- 只是给 encoder 更小的学习率

因此，当前在 MapAnything 中的 Pi3 微调属于：

- 真实 fine-tune
- 全量微调
- 但使用 differential learning rates

## 5. 输入归一化差异

### 官方 Pi3

- 模型内部使用 ImageNet mean/std 归一化
- 数据增强中也有 `CustomNorm`

### MapAnything 中的 Pi3

见 `Models/map-anything/configs/model/pi3.yaml` 和 `Models/map-anything/mapanything/models/external/pi3/__init__.py`：

- `data_norm_type: "identity"`
- wrapper 要求输入图像先保持 identity 风格，再由 Pi3 模型内部处理

这不是主要矛盾，但在复现官方输入分布时需要注意。

## 6. 对横向 benchmark 的意义

对于当前任务，“带遥感图像引导的多视角重建”更像是：

- 一个新的任务设定
- 一个新的数据集
- 需要在统一框架下横向比较多个模型

因此，从工程目标上，使用 `MapAnything` 更合适，原因是：

- 它已经支持外部模型统一接入
- 它已经支持统一输出格式
- 它已经支持统一训练脚本、统一数据接口、统一 benchmark 入口
- 它天然适合做多模型公平比较

但代价也明确：

- 在 `MapAnything` 中训练出来的 Pi3，不等于官方 Pi3 训练范式下的 Pi3
- 其结果更应被解释为：
  - “Pi3 在 MapAnything 统一训练框架下的表现”
  - 而不是“官方 Pi3 原始训练范式下的表现”

## 7. 外部模型使用统一损失训练会有什么问题

### 可行性

是可行的。MapAnything 本身就是这样设计的，见：

- `Models/map-anything/train.md`
- `Models/map-anything/mapanything/models/external/README.md`

项目明确支持对外部模型进行：

- re-training
- fine-tuning
- benchmarking

### 潜在问题

统一损失训练外部模型时，主要风险有：

1. 输出语义错位
   - 如果 wrapper 输出的量和统一 loss 假设的语义不完全一致，训练会偏。

2. 原模型 inductive bias 被打破
   - 原模型可能是围绕某种特定损失、归一化、尺度约定设计的。
   - 换损失后，可能不再处于它最合适的优化目标下。

3. 和官方结果不可直接比较
   - 如果损失、增强、冻结策略都变了，训练结果不能直接当作官方复现结果。

4. 可能能训通，但未必最优
   - 统一框架通常能训通
   - 但未必达到该模型在官方训练范式下的上限

### 实际判断标准

对于你的任务，核心不是“是否完全保持 Pi3 原味”，而是：

- 在统一数据和统一评测下，是否能稳定训练
- 是否能公平比较不同模型
- 是否能在你的任务上取得有效结果

从这个目标出发，MapAnything 的统一训练框架是合理的。

## 8. MapAnything 是否有共同 benchmark 框架

有，而且是比较明确的共同 benchmark 框架。

现有 benchmark 目录：

- `Models/map-anything/benchmarking/dense_n_view/`
- `Models/map-anything/benchmarking/calibration/`
- `Models/map-anything/benchmarking/rmvd_mvs_benchmark/`

共同点：

- 都通过统一 model factory 初始化模型
- 都依赖统一输出格式
- 都围绕 WAI 数据格式组织数据
- 都使用 `mapanything.utils.metrics` 中的公共评测函数

其中与你最相关的是：

- `Models/map-anything/benchmarking/dense_n_view/benchmark.py`

它已经是一套“多视角稠密重建 benchmark”的共用基线。

## 9. 能不能新建一个 benchmark

可以，而且很适合这么做。

对你当前任务，更合理的做法是：

- 不去硬套现有公开 benchmark 的数据集
- 在 `MapAnything` 里新建一个面向你任务的新 benchmark

建议结构：

- `Models/map-anything/benchmarking/rs_guided_dense_mv/benchmark.py`
- `Models/map-anything/benchmarking/rs_guided_dense_mv/README.md`
- 配套 Hydra config
- 配套 bash script

建议复用的现有能力：

- 数据接口：沿用 WAI
- 模型接口：沿用统一 wrapper 输出
- 评测工具：尽量复用 `mapanything.utils.metrics`
- benchmark 主流程：参考 `benchmarking/dense_n_view/benchmark.py`

## 10. 当前建议

对你的任务，推荐方向是：

1. 继续以 `MapAnything` 作为主训练与 benchmark 框架
2. 把 Pi3、VGGT 等视为统一框架下的对比模型
3. 不把当前训练结果解释为官方原版结果
4. 后续单独为你的“遥感图像引导多视角重建”任务建立一个新的 benchmark

这条路线和你的实际目标是一致的：以统一标准比较模型在新任务上的表现，而不是深挖单个模型本身。
