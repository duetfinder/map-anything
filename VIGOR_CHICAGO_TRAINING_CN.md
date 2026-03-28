# VIGOR Chicago 训练说明

说明：本文档中的文件链接统一使用相对于当前 Markdown 文件的相对路径，不使用绝对路径。

本文档只记录 `VIGOR Chicago` 在 `Models/map-anything` 中的训练相关内容，与 benchmark 设计、RS-Aerial 指标定义等内容拆开维护。

## 0. 当前数据根目录

当前训练相关数据已经统一迁移到 `../../traindata`：

- aerial WAI：[`../../traindata/vigor_chicago_wai`](../../traindata/vigor_chicago_wai)
- RS 数据：[`../../traindata/vigor_chicago_rs`](../../traindata/vigor_chicago_rs)
- split / metadata：[`../../traindata/mapanything_metadata`](../../traindata/mapanything_metadata)

补充说明：

- `vigor_chicago_rs` 当前已经是实体文件，不再依赖软链接
- `vigor_chicago_wai` 当前仍使用相对链接组织 image / depth，但链接已按 `traindata` 修复
- joint training 的 dataset 数据层已新增并验证通过：`VigorChicagoJointRSAerial`

## 1. 当前训练状态

当前已经完成两条训练链路验证：

- aerial-only 2-GPU smoke test
  - 脚本：[bash_scripts/train/examples/vigor_chicago_50_pi3_smoke_2gpu.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_smoke_2gpu.sh)
  - 输出目录：[`../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu`](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_smoke_2gpu)
- aerial-only 2-GPU pretrained fine-tune
  - 脚本：[bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh)
  - 输出目录：[`../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu`](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago_50/pi3_finetune_pretrained_2gpu)

当前训练主数据集仍然是 aerial multi-view WAI 数据：

- dataset loader：[mapanything/datasets/wai/vigor_chicago.py](mapanything/datasets/wai/vigor_chicago.py)
- 数据配置：[configs/dataset/vigor_chicago_50_518.yaml](configs/dataset/vigor_chicago_50_518.yaml)
- 500-scene 正式配置：[configs/dataset/vigor_chicago_500_518.yaml](configs/dataset/vigor_chicago_500_518.yaml)

## 2. 当前 aerial-only 训练的数据接口

当前 `VigorChicagoWAI` 继承了 `BaseDataset`，其 `_get_views(...)` 返回的每个 view 至少提供：

- `img`
- `depthmap`
- `camera_pose`
- `camera_intrinsics`

对应实现见：

- [mapanything/datasets/wai/vigor_chicago.py](mapanything/datasets/wai/vigor_chicago.py)
- [mapanything/datasets/base/base_dataset.py](mapanything/datasets/base/base_dataset.py)

`BaseDataset` 后续会基于这些字段自动推导并补齐：

- `pts3d`
- `valid_mask`
- `pts3d_cam`
- `depth_along_ray`
- `camera_pose_quats`
- `camera_pose_trans`

这意味着当前训练 loss 默认假设：

- 每个训练 view 都有可解释的相机模型
- 每个训练 view 都能从 `depthmap + intrinsics + pose` 推出统一几何量

## 3. 当前遥感数据的组织方式

当前遥感图像已经有统一的数据组织，既可服务于 benchmark，也已经具备训练接入的基础。

相关代码：

- metadata 准备脚本：[scripts/prepare_rs_aerial_benchmark_metadata.py](scripts/prepare_rs_aerial_benchmark_metadata.py)
- dataset loader：[mapanything/datasets/wai/vigor_chicago_rs_aerial.py](mapanything/datasets/wai/vigor_chicago_rs_aerial.py)
- unified benchmark 配置：[configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml](configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml)
- unified benchmark 入口：[benchmarking/rs_guided_dense_mv/benchmark_unified.py](benchmarking/rs_guided_dense_mv/benchmark_unified.py)

当前 manifest 将同一 scene 下的 aerial scene 与 remote image 关联起来，包含：

- `aerial_scene_dir`
- `aerial_scene_meta`
- `remote_image_path`
- `remote_pointmap_path`
- `remote_valid_mask_path`
- `remote_height_map_path`
- `remote_info_path`

当前 remote 侧可直接读到的监督主要是：

- `remote_image`
- `remote_pointmap`
- `remote_valid_mask`
- `remote_height_map`

## 4. 当前最关键的约束

当前先明确一个前提：

- 卫星图像目前没有可靠、统一的相机模型
- 因此当前不考虑把它直接纳入现有的 pose / intrinsics / multi-view relative pose loss
- 这个问题先保留，不在当前阶段强行解决

这会直接带来一个工程结论：

- 遥感图像当前不能无缝复用 `BaseDataset` 的标准训练 view 接口
- 也不能直接作为“普通第 N 个 view”塞进现有 `FactoredGeometry...` 这类 loss 而完全不改代码

## 5. 如果把遥感图像加入训练，建议怎么做

在当前约束下，更稳妥的方案不是“把遥感图像当成普通 perspective view”，而是把它作为一个额外模态 / 辅助监督分支接入训练。

### 5.1 推荐的第一阶段目标

推荐先实现：

- aerial multi-view 训练主干保持不变
- 在 paired scenes 上额外读取 remote image
- 模型前向时允许输入 `aerial views + remote image`
- 但 remote 分支当前只吃几何监督，不吃相机位姿监督

也就是说，第一阶段更像：

- `aerial multi-view reconstruction loss`
- `+ remote geometry auxiliary loss`
- `- remote pose / view consistency loss`

### 5.2 推荐的数据组织方式

当前最值得复用的是已经重建到 `traindata` 下的 paired manifest，同时训练侧也已经具备直接读取统一 RS 根目录的 dataset。

推荐做法：

1. 保留 aerial WAI scene 目录不动。
2. 继续复用 `prepare_rs_aerial_benchmark_metadata.py` 生成的 paired manifest。
3. 新增一个训练专用 joint dataset，例如：
   - `VigorChicagoJointRSAerialTrain`
4. 这个 dataset 内部做两件事：
   - 先按当前 `VigorChicagoWAI` 逻辑采样 aerial multi-view
   - 再按 scene name 取对应的 remote sample

这样可以让 benchmark 和 training 共用同一份 scene 对齐关系。

## 6. 加入遥感图像后，训练侧需要改哪些地方

### 6.1 dataset 层

这是必须改的第一层。

原因：当前训练 dataset 的每个 view 必须有 `depthmap + camera_pose + camera_intrinsics`，而 remote image 当前不满足。

因此推荐新增一个 joint training dataset，而不是直接改坏 `VigorChicagoWAI`。

建议新增：

- [mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py](mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py)
- [mapanything/datasets/wai/vigor_chicago_rs.py](mapanything/datasets/wai/vigor_chicago_rs.py)

其返回结构建议分成两部分：

- `aerial_views`
  - 保持与当前 `BaseDataset` 兼容
- `remote_supervision`
  - `remote_image`
  - `remote_pointmap`
  - `remote_valid_mask`
  - `remote_height_map`
  - `remote_info`

如果后续希望最小化训练循环修改，也可以让 dataset 仍返回“views list”，但把 remote 样本额外挂在 batch 的 side channel 字段里，而不是伪装成普通 perspective view。

### 6.2 训练循环 / batch 组装

当前训练循环默认接收的是标准多视图 batch，并直接把它送给 model 和 loss。

因此如果加入 remote supervision，通常需要在训练循环中增加：

- 从 batch 里拿到 remote side-channel 数据
- 构造 remote 输入 view，例如仅包含：
  - `img`
  - `data_norm_type`
- 根据实验设计决定：
  - 是否做 `model(aerial_views)`
  - 是否做 `model(aerial_views + [remote_view])`
  - 是否还要单独做一次 `model([remote_view])`

如果沿用 benchmark 当前的联合推理方式，最接近的参考是：

- [benchmarking/rs_guided_dense_mv/benchmark_unified.py](benchmarking/rs_guided_dense_mv/benchmark_unified.py)

其中当前已经存在：

- `aerial_preds = model(batch)`
- `rs_preds = model([remote_view])`
- `joint_preds = model(batch + [remote_view])`

训练阶段可以借鉴这种输入拼接方式，但 loss 不能照 benchmark 的评测方式原样照搬。

### 6.3 loss 层

这是第二个必须改的地方。

当前主 loss 仍然应当保留 aerial 侧的现有训练目标：

- [configs/loss/pi3_loss.yaml](configs/loss/pi3_loss.yaml)
- [mapanything/train/losses.py](mapanything/train/losses.py)

但 remote 侧不能直接用当前依赖相机模型的监督项，至少当前阶段不建议使用：

- camera pose loss
- pairwise relative pose loss
- 基于 perspective 相机定义的 view-consistency loss

当前阶段更合适的 remote loss 候选是：

- remote pointmap regression loss
- remote height regression loss
- remote valid-mask 内的几何监督
- 可选的 remote normal / gradient 类监督

更具体地说，第一阶段可以考虑：

1. `remote_pointmap_loss`
   - 对 `remote_pointmap` 在 `remote_valid_mask` 上做 L1 / robust loss
2. `remote_height_loss`
   - 对预测点图的 `z` 或 height 派生量做监督
3. 不引入 remote pose loss

因此更合理的训练 loss 形态会变成：

```text
total_loss = aerial_multiview_loss + lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

### 6.4 配置层

需要新增至少两类配置：

- dataset config
  - 例如 `configs/dataset/vigor_chicago_50_rs_joint_518.yaml`
- loss config
  - 例如 `configs/loss/pi3_loss_rs_joint.yaml`

可选地还可以增加：

- train script
  - 例如 `bash_scripts/train/examples/vigor_chicago_50_pi3_rs_joint_finetune_2gpu.sh`

### 6.5 benchmark / logging 层

训练引入 remote 分支后，建议同步记录：

- aerial 主 loss
- remote pointmap loss
- remote height loss
- total loss

否则后续很难判断：

- 联合训练有没有帮助 aerial
- remote 分支是不是压过了 aerial 主任务
- 不同 loss 权重是否稳定

## 7. 加入遥感图像后，会有什么影响

### 7.1 对模型输入分布的影响

会显著增加跨域难度。

原因：

- aerial 图像是透视视角
- remote 图像是正射或近似正射的俯视图
- 两者纹理统计、尺度语义、几何先验都不同

因此即使模型接口允许把它们一起输入，也不代表当前 backbone 对这两种域天然等价。

### 7.2 对显存和速度的影响

如果把 remote image 当作额外输入 token / 额外 view：

- 显存会上升
- 每 step 时间会上升

直观上，`num_views=4` 的 aerial 训练如果再加 1 张 remote image，前向复杂度会更接近 `5` views，而不是 `4` views。

### 7.3 对 loss 稳定性的影响

联合训练最容易出现的问题不是代码报错，而是 loss 目标互相拉扯。

风险包括：

- aerial 目标与 remote 目标权重不平衡
- remote 几何标签分布与 aerial 深度标签分布差异很大
- 模型为了适应 remote supervision，反而破坏 aerial reconstruction

因此强烈建议：

- 先从很小的 `lambda_remote_*` 开始
- 先做 paired-scene 小规模实验
- 同时跟踪 aerial benchmark 是否退化

### 7.4 对 covisibility 采样逻辑的影响

当前 aerial 训练的 view sampling 依赖 scene 内 covisibility。

remote image 不是 scene 内普通帧，因此不适合直接纳入现有 covisibility 采样矩阵。

更合理的策略是：

- 先按当前逻辑采样 aerial multi-view
- 再 append 一个 scene-level remote image

也就是说：

- covisibility 继续只用于 aerial views
- remote image 由 scene pairing 决定，而不是由 covisibility 决定

## 8. 推荐的推进顺序

### 8.1 第一阶段：不改现有 aerial 主训练逻辑

目标：先验证 joint input + remote auxiliary loss 能不能稳定训练。

建议：

- 保持 aerial loss 完全不变
- 新增 remote pointmap / height auxiliary loss
- 不引入 remote pose loss
- 不要求 remote 进入 `BaseDataset` 标准 camera model 流程

### 8.2 第二阶段：比较三种训练方式

建议对比：

- aerial-only
- aerial + remote auxiliary loss
- aerial + remote auxiliary loss + joint forward input

这样可以先分离两个问题：

- 遥感监督本身有没有帮助
- 遥感图像作为额外输入有没有帮助

### 8.3 第三阶段：等相机模型问题更清楚后再升级几何约束

只有在 remote 的投影模型、全局坐标系关系、以及可解释相机参数更明确之后，才建议进一步尝试：

- remote pose-related losses
- aerial / remote cross-view consistency losses
- 更强的 shared-frame joint geometric losses

## 9. 当前建议改动点总表

如果要启动第一版 joint training，建议优先改这些文件：

- 新增训练 dataset
  - [mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py](mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py)
- [mapanything/datasets/wai/vigor_chicago_rs.py](mapanything/datasets/wai/vigor_chicago_rs.py)
- 更新 dataset registry
  - [mapanything/datasets/__init__.py](mapanything/datasets/__init__.py)
- 新增训练 dataset config
  - `configs/dataset/vigor_chicago_50_rs_joint_518.yaml`
- 新增 joint loss config
  - `configs/loss/pi3_loss_rs_joint.yaml`
- 视需要改训练循环
  - [mapanything/train/training.py](mapanything/train/training.py)
- 视需要新增 remote auxiliary loss 实现
  - [mapanything/train/losses.py](mapanything/train/losses.py)
- 新增训练脚本
  - `bash_scripts/train/examples/vigor_chicago_50_pi3_rs_joint_finetune_2gpu.sh`

## 10. 与状态总文档的关系

状态总文档见：

- [VIGOR_CHICAGO_STATUS.md](VIGOR_CHICAGO_STATUS.md)

建议后续分工：

- `VIGOR_CHICAGO_STATUS.md`
  - 记录全局进展、数据产物、benchmark 状态
- `VIGOR_CHICAGO_TRAINING_CN.md`
  - 记录训练设计、训练假设、loss、联合训练策略
