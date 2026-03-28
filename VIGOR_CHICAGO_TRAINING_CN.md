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
- 500-scene Pi3 debug 短跑脚本：[bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh)
- 500-scene Pi3 baseline 正式脚本：[bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh)
- 旧的 example 脚本：[bash_scripts/train/examples/vigor_chicago_500_pi3_finetune_pretrained_2gpu.sh](bash_scripts/train/examples/vigor_chicago_500_pi3_finetune_pretrained_2gpu.sh)

说明：

- `examples/vigor_chicago_500_pi3_finetune_pretrained_2gpu.sh` 与正式 baseline 在训练逻辑上已经非常接近。
- 新增的 [bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh) 是 `P0 Data Sanity` 调试入口，用于 1-epoch / 小样本短跑。
- 新增的 [bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh) 主要是把 baseline 重新命名、单独归档，并显式定义为 `P1 Baseline-Main` 实验入口。
- 新脚本同时暴露了 `NUM_VIEWS / BATCH_SIZE / OUTPUT_DIR`，而 P0 额外暴露了 `TRAIN_SETS / VAL_SETS / TEST_SETS`，便于在不改脚本的前提下做 debug 和 baseline 内部调参。

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

## 8. 当前对 baseline 的更具体判断

目前更合理的做法是：

- baseline 先固定成一条清晰的 `aerial-only` 正式训练线
- 不要在 baseline 阶段同时展开 `全量微调 / 冻结 encoder / 部分微调 / LoRA` 的系统对比
- `num_views` 的变化更适合视作 baseline 内部的训练配置选择，而不是单独的大实验方向

也就是说，当前优先回答的问题应当是：

1. 哪个模型先作为主 baseline。
2. baseline 先采用哪一种最稳妥的 fine-tune 方式。
3. baseline 在 `num_views=2` 还是 `num_views=4` 下更适合当前显存和训练稳定性。
4. 在 baseline 稳定之后，再决定是否需要做额外的 fine-tune strategy ablation。

对当前项目，推荐把 baseline 先收敛为：

- 500-scene aerial-only
- 直接加载官方预训练权重
- 先用当前已经跑通过的常规 fine-tune 路线
- encoder 先保持低学习率而不是完全冻结
- 不在第一阶段引入 LoRA

原因：

- 当前最主要的不确定性来自数据组织更新和 RS 联合训练设计，而不是 fine-tune strategy 本身。
- 如果 baseline 阶段同时比较太多训练方式，后续很难判断问题究竟来自数据、模型，还是训练策略。
- LoRA 当前仓库里没有内置支持，也没有现成例子；它更适合放到后续的策略对比，而不是第一条 baseline。

## 9. 训练实验安排表

下表的核心目的不是一次性把所有实验都做完，而是先把训练安排拆成“必须先做”和“后续可选对比”两层。

| 阶段 | 实验名 | 目标 | 模型建议 | 数据输入 | 训练方式 | 损失 | 对应脚本 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | Data Sanity | 确认 500-scene 数据与 joint dataset 可稳定取样 | `pi3` | aerial-only 或 joint dataset 读数检查 | 不正式训练，先做 dataloader / 1-epoch smoke | 当前 `pi3_loss` | [bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p0_pi3_data_sanity_500_pretrained_2gpu.sh) | 已有 joint dataset 基础，重点确认 `traindata` 迁移后读取稳定 |
| P1 | Baseline-Main | 建立正式对照组 | `pi3` | aerial-only, 500 scenes | 常规 pretrained fine-tune | `pi3_loss` | [bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh) | 当前最推荐的主 baseline |
| P1a | Baseline-Views | 在 baseline 内部选择合适的 `num_views` | `pi3` | aerial-only, 500 scenes | 与 P1 相同，只改 `num_views` | `pi3_loss` | 复用 [bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh](bash_scripts/train/vigor_chicago/p1_pi3_baseline_500_pretrained_2gpu.sh) | 本质上是调参，建议只比较 `2` 和 `4` |
| P1b | Baseline-FT-Strategy | 检查是否有必要做训练策略对比 | `pi3` | aerial-only, 500 scenes | 全量微调 / encoder 低 lr / encoder 冻结 | `pi3_loss` | 待后续按策略拆分脚本 | 不建议一开始就展开，只有在 baseline 不稳或效果不理想时再做 |
| P1c | Baseline-LoRA | 检查参数高效微调是否值得引入 | `pi3` | aerial-only, 500 scenes | LoRA / adapter | `pi3_loss` | 当前无脚本 | 当前仓库未原生支持，优先级低于 P1 / P1a / P2 |
| P2 | RS-Auxiliary | 先验证 RS supervision 本身是否有帮助 | `pi3` | aerial multi-view + remote supervision | 主干保持 aerial-only，多一个 remote 辅助监督 | `pi3_loss + lambda_pm * remote_pointmap_loss + lambda_h * remote_height_loss` | 待新增联合训练脚本 | 当前最值得优先实现的 joint training 方向 |
| P2a | RS-Loss-Ablation | 比较 remote loss 设计 | `pi3` | 同 P2 | 与 P2 相同 | 比较 `masked L1` / `robust` / `height-only` | 复用 P2 脚本并切 loss config | 先不要引入 remote pose loss |
| P3 | RS-Input | 检查把 remote image 作为额外输入是否有收益 | `pi3` 或 `vggt` | aerial views + remote image | joint forward | `aerial loss + remote auxiliary loss` | 待新增联合输入脚本 | 风险更高，放在 P2 后 |
| P4 | Model-Compare | 在稳定实验设置下比较不同模型 | `pi3` / `vggt` / `mapanything` / `da3` | 与选定任务一致 | 跟随对应 baseline / joint 设置 | 跟随模型对应主 loss | 视模型逐个补脚本 | 只有在 P1/P2 跑稳后再展开 |

补充解释：

- `P1` 才是当前真正意义上的 baseline。
- `P1a` 属于 baseline 内部调参，不需要把它抬升成一条完全独立的研究主线。
- `P1b` 和 `P1c` 是训练策略 ablation，不是 baseline 本身。
- 如果你的目标是尽快推进 RS 联合训练，`P1b / P1c` 完全可以后置。

## 10. 当前推荐的 loss 选择

结合当前代码和你的数据约束，建议这样分层：

### 10.1 aerial-only baseline

直接使用当前主 loss：

- [configs/loss/pi3_loss.yaml](configs/loss/pi3_loss.yaml)

也就是：

- `ExcludeTopNPercentPixelLoss(...)`
- 内部主项是 `FactoredGeometryRegr3DPlusNormalGMLoss(...)`
- 基础回归项是 `RobustRegressionLoss(alpha=0.5, scaling_c=0.05)`

这条线当前不要改 loss。

### 10.2 RS auxiliary 第一版

考虑到卫星图像当前没有可靠相机模型，第一版建议只加几何监督，不加视角相关项：

```text
total_loss = aerial_loss + lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

其中：

- `aerial_loss = pi3_loss`
- `remote_pointmap_loss`
  - 推荐先用 masked `L1`，其次再试 `RobustRegressionLoss`
- `remote_height_loss`
  - 推荐先用 masked `L1`
- 当前先不引入：
  - remote pose loss
  - remote relative pose loss
  - 依赖 perspective camera 的 remote view consistency loss

建议第一版权重从小值开始，例如：

- `lambda_remote_pm = 0.1`
- `lambda_remote_h = 0.05`

目的不是一次到位，而是先避免 remote 辅助项压过 aerial 主任务。

## 11. 模型选择建议

当前候选模型不只是 `MapAnything`，还包括 `da3 / pi3 / vggt`。从工程角度看，它们差别是比较大的。

### 11.1 MapAnything

特点：

- 仓库原生模型，训练脚本、ablation、loss 配置最完整。
- 适合做深度定制，例如几何输入控制、info sharing、结构改动。
- 但它也是当前改动面最大、训练链路最复杂的一条线。

工程判断：

- 如果你的目标是“尽快得到一个稳定 baseline”，它不是第一优先。
- 如果你的目标是“后续要深改结构、把 RS 联合训练做成框架级方案”，它反而最有潜力。

### 11.2 Pi3

特点：

- 当前 VIGOR Chicago 上已经有最完整、最直接的训练验证路径。
- 已经有现成脚本：[bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh](bash_scripts/train/examples/vigor_chicago_50_pi3_finetune_pretrained_2gpu.sh) 和 [bash_scripts/train/examples/vigor_chicago_500_pi3_finetune_pretrained_2gpu.sh](bash_scripts/train/examples/vigor_chicago_500_pi3_finetune_pretrained_2gpu.sh)
- wrapper 简单，前向接口清晰，当前最容易继续扩展。
- 冻结子模块也比较直接，因为 `named_parameters()` 前缀清楚。

工程判断：

- 当前最适合作为主 baseline。
- 也是最适合作为第一版 RS auxiliary training 的承载模型。

### 11.3 VGGT

特点：

- 同样是强多视角几何模型，训练上比 Pi3 更“完整相机模型导向”。
- 当前仓库里已经有 fine-tune 入口：[bash_scripts/train/examples/vigor_chicago_50_vggt_finetune.sh](bash_scripts/train/examples/vigor_chicago_50_vggt_finetune.sh)
- 但从工程复杂度和显存压力看，通常会比 Pi3 更重。

工程判断：

- 更适合作为第二候选 baseline 或对比模型。
- 如果只是想快速推进，不建议先拿它做第一主线。

### 11.4 DA3

特点：

- 当前仓库中是外部 wrapper 接入，不是本项目里最成熟的训练主线。
- 它本身更偏 foundation depth / camera wrapper 风格，当前没有你这套 VIGOR Chicago 的现成训练脚本。
- 如果后续要做系统训练，通常需要额外补更多实验脚本和验证。

工程判断：

- 当前不建议作为第一优先模型。
- 更适合后续在 baseline 稳定后做补充对比。

### 11.5 当前的实际推荐顺序

如果目标是“尽快把训练路线变得清晰且可执行”，推荐顺序是：

1. `pi3` 作为主 baseline。
2. `pi3` 上先做 RS auxiliary training。
3. `vggt` 作为第二对比模型。
4. `mapanything` 留给后续更深的结构化研究。
5. `da3` 放在更后面的补充实验。

## 12. 与状态总文档的关系

状态总文档见：

- [VIGOR_CHICAGO_STATUS.md](VIGOR_CHICAGO_STATUS.md)

建议后续分工：

- `VIGOR_CHICAGO_STATUS.md`
  - 记录全局进展、数据产物、benchmark 状态
- `VIGOR_CHICAGO_TRAINING_CN.md`
  - 记录训练设计、训练假设、loss、联合训练策略、实验表
