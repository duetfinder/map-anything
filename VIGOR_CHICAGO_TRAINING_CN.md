# VIGOR Chicago 训练说明

说明：本文档中的文件链接统一使用相对于当前 Markdown 文件的相对路径，不使用绝对路径。

本文档只记录 `VIGOR Chicago` 在 `Models/map-anything` 中的训练相关内容，与 benchmark 设计、RS-Aerial 指标定义等内容拆开维护。

loss 相关的独立汇总见：[MAPANYTHING_LOSSES_CN.md](MAPANYTHING_LOSSES_CN.md)。

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

补充：`P3 Joint-Input` 的第一版工程骨架已经补齐，并已完成一次 2-GPU debug smoke。

本次 `P3` smoke 运行信息：

- 脚本：[bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)
- 输出目录：[`../../outputs/mapanything_experiments/mapanything/training/vigor_chicago/p3_joint_input_debug`](../../outputs/mapanything_experiments/mapanything/training/vigor_chicago/p3_joint_input_debug)
- 运行形态：2 GPU, `num_views=2`, `batch_size=2`, `train/val/test overfit = 16/8/8`, `epochs=1`
- 结果：训练 1 epoch 完整跑通，验证阶段完整跑通，进程正常退出，已生成 `checkpoint-last / checkpoint-1 / checkpoint-best / checkpoint-final`
- 训练日志中已可同时记录：`loss`、`aerial_loss`、`remote_loss`、`rs_pointmap_loss`、`rs_height_loss`
- 本次 smoke 中实际修复了一个 joint batch 兼容问题：`loss_of_one_batch_multi_view(...)` 原先会对非 tensor side-channel 字段直接 `.to(device)`，现已改为仅对 tensor 做 device 搬运

本次 smoke 的日志量级可作为后续调参起点：

- 首个 train step 约为：`loss=57.18`, `aerial_loss=12.25`, `remote_loss=44.93`, `rs_pointmap_loss=419.73`, `rs_height_loss=295.38`
- 首轮 val 尾部约为：`loss=44.40`, `aerial_loss=2.60`, `remote_loss=41.80`, `rs_pointmap_loss=417.99`

从当前结果看：

- `P3` 的数据、joint forward、组合 loss、DDP、checkpoint 保存链路已经贯通
- remote 分支当前明显主导总 loss，下一阶段优先工作应是细化 `lambda_remote_*`、batch size、以及 `num_views` 的训练安排，而不是继续扩展新结构

已新增的 `P3` 相关入口：

- joint dataset config：[configs/dataset/vigor_chicago_rs_joint_518.yaml](configs/dataset/vigor_chicago_rs_joint_518.yaml)
- joint loss config：[configs/loss/pi3_loss_rs_joint.yaml](configs/loss/pi3_loss_rs_joint.yaml)
- joint loss 实现：[mapanything/train/losses.py](mapanything/train/losses.py)
- joint forward 组装：[mapanything/utils/inference.py](mapanything/utils/inference.py)
- debug 脚本：[bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)
- 500-scene 正式脚本：[bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh)

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

在当前约束下，更合理的路线不再是优先做“RS auxiliary supervision”，而是先把遥感图像当成一个独立输入域来验证，再进入联合输入训练。

### 5.1 推荐的第一阶段目标

推荐先实现：

- 先做 `RS-only` 训练
- 只输入 `remote image`
- 只监督 `remote_pointmap / remote_height_map / remote_valid_mask`
- 不要求 remote 进入 `BaseDataset` 标准 camera model 流程
- 不引入 remote pose / view consistency loss

也就是说，第一阶段更像：

- `remote-only geometry regression`
- `+ masked remote geometry supervision`
- `- remote pose / view consistency loss`

这样做的目的不是直接追求联合训练效果，而是先回答一个更基础的问题：

- 现有模型能不能适应 RS 这种输入域
- 在不依赖相机模型的前提下，RS 数据本身能不能形成稳定训练信号

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

训练阶段建议按两步走：

1. 先做 `model([remote_view])` 的 `RS-only` 训练。
2. 再做 `model(batch + [remote_view])` 的联合输入训练。

这样 loss 设计和问题解释都会更清楚。

### 6.3 loss 层

这是第二个必须改的地方。

当前如果做 `RS-only` 或 `Joint Input`，remote 侧都不能直接复用当前依赖相机模型的监督项，至少当前阶段不建议使用：

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
   - 对 `remote_height_map` 做 masked L1 / L2 loss
3. 不引入 remote pose loss

因此第一阶段更合理的 `RS-only` 训练 loss 形态是：

```text
total_loss = lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

在进入 `Joint Input` 后，再扩展为：

```text
total_loss = aerial_multiview_loss + lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

### 6.4 配置层

当前已经补齐一套可直接用于 `P3` 的第一版配置：

- dataset config
  - [configs/dataset/vigor_chicago_rs_joint_518.yaml](configs/dataset/vigor_chicago_rs_joint_518.yaml)
- loss config
  - [configs/loss/pi3_loss_rs_joint.yaml](configs/loss/pi3_loss_rs_joint.yaml)
- train script
  - [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)
  - [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh)

当前这套 `P3` 配置的设计点是：

- dataset 仍以 aerial multi-view 为主 batch
- remote supervision 通过 side-channel 附着在 aerial views 上
- 训练时在 `loss_of_one_batch_multi_view(...)` 内部构造第 `N+1` 个 `remote_view`
- `JointAerialRSLoss(...)` 内部把 `aerial` 与 `remote` 两段 supervision 分开计算再求和
- `loss.remote_pointmap_loss_weight / loss.remote_height_loss_weight` 已暴露为 Hydra 可覆写字段，便于脚本直接调节 `lambda_remote_*`

### 6.5 benchmark / logging 层

当前 `P3` smoke 已经能在训练日志里稳定记录：

- `loss`
- `aerial_loss`
- `remote_loss`
- `rs_pointmap_loss`
- `rs_height_loss`

训练引入 remote 分支后，建议继续同步记录：

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
| P2 | RS-Only | 先验证模型能否单独适应 RS 输入域 | `pi3` | remote image only | remote-only 几何回归 | `lambda_pm * remote_pointmap_loss + lambda_h * remote_height_loss` | [bash_scripts/train/vigor_chicago/p2_pi3_rs_only_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p2_pi3_rs_only_debug_2gpu.sh) | 已完成 2-GPU, 1-epoch smoke；当前是可运行的最小入口 |
| P2a | RS-Only-Loss-Ablation | 比较 RS-only 的 remote loss 设计 | `pi3` | remote image only | 与 P2 相同 | 比较 `pointmap-only L1` / `pointmap+height L1` / `pointmap robust + height L1` | [bash_scripts/train/vigor_chicago/p2a_pi3_rs_only_loss_ablation_2gpu.sh](bash_scripts/train/vigor_chicago/p2a_pi3_rs_only_loss_ablation_2gpu.sh) | 第一版只做 loss 与权重对比，不引入新结构 |
| P3 | Joint-Input | 检查 aerial + remote 同时输入是否有收益 | `pi3` 或 `vggt` | aerial views + remote image | joint forward | `aerial loss + lambda_pm * remote_pointmap_loss + lambda_h * remote_height_loss` | [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh) / [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh) | 已完成 2-GPU debug smoke；下一步进入权重与资源配置细化 |
| P4 | Model-Compare | 在稳定实验设置下比较不同模型 | `pi3` / `vggt` / `mapanything` / `da3` | 与选定任务一致 | 跟随对应 baseline / joint 设置 | 跟随模型对应主 loss | 视模型逐个补脚本 | 只有在 P1/P2/P3 跑稳后再展开 |

补充解释：

- `P1` 才是当前真正意义上的 baseline。
- `P1a` 属于 baseline 内部调参，不需要把它抬升成一条完全独立的研究主线。
- `P1b` 和 `P1c` 是训练策略 ablation，不是 baseline 本身。
- 当前新的研究主线不是“RS auxiliary supervision”，而是“先做 RS-only，再做 Joint Input”。
- 如果你的目标是尽快推进联合训练，`P1b / P1c` 仍然可以后置。

## 9.1 接下来如何细化 P3 训练

在当前 `P3 debug smoke` 已跑通之后，建议把后续训练继续拆成三小段，而不是直接上大规模正式训练。

### 9.1.1 P3a: Remote 权重细化

目标：先把 remote 分支对总 loss 的压制程度调到可控范围。

建议只扫两组权重：

- `loss.remote_pointmap_loss_weight`: `0.1 -> 0.05 -> 0.02`
- `loss.remote_height_loss_weight`: `0.01 -> 0.005 -> 0.002`

当前优先推荐两组起点：

- `lambda_remote_pm=0.05`, `lambda_remote_h=0.005`
- `lambda_remote_pm=0.02`, `lambda_remote_h=0.002`

判断标准：

- `aerial_loss` 不要因 remote 分支明显抬高
- `remote_loss` 仍能稳定下降
- 总 loss 中 remote 不应长期占绝对主导

### 9.1.2 P3b: 训练资源配置细化

目标：在不破坏稳定性的前提下确定联合训练的主配置。

建议顺序：

1. 先固定 `num_views=2`, `batch_size=2`
2. 在权重稳定后，再尝试 `batch_size=1` 与 `batch_size=2` 的速度/显存对比
3. 只有在 `num_views=2` 跑稳后，再考虑 `num_views=4`

当前不建议一开始就把 `num_views=4` 和更大 batch 叠加，因为 joint input 会把实际 forward 复杂度从 `2 views` 变成 `3 views`，再上 `4+1` 会明显抬高资源成本。

### 9.1.3 P3c: 500-scene 正式训练前的中间验证

目标：在正式 `P3` 训练前，先做一轮比 smoke 稍大的中间规模检查。

建议使用：

- `TRAIN_SETS=64`
- `VAL_SETS=16`
- `TEST_SETS=16`
- `epochs=3`
- 仍使用 [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)

只有当这一层稳定后，再切到 [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh)。

### 9.1.4 P3 正式训练前的记录口径

在进入 500-scene 正式训练前，建议固定比较口径：

- 与 `P1` 比较：看 aerial 主任务是否退化
- 与 `P2a` 比较：看 remote 监督是否仍然有效
- 在 `P3` 内部比较：统一用 `aerial_loss / remote_loss / rs_pointmap_loss / rs_pointmap_loss_weighted / total loss` 作为训练日志主字段
- 当前 `P3` 默认已关闭 joint `height` loss；只有在确认坐标系口径后才建议重新打开

## 9.2 P3 500 正式脚本 bug 记录与修复

2026-04-07 跑 `bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh` 时首次失败在 DataLoader 构建阶段，尚未进入模型训练。

错误信息：

```text
ValueError: batch_size should be a positive integer value, but got batch_size=0
```

根因不是 `inference.py` 或 `JointAerialRSLoss`，而是脚本默认参数不匹配：

```text
dataset.num_views=4
train_params.max_num_of_imgs_per_gpu=2
```

训练代码里验证 batch size 的计算方式是：

```python
test_batch_size = 2 * (max_num_of_imgs_per_gpu // dataset.num_views)
```

所以上述参数会得到 `2 * (2 // 4) = 0`，直接触发 PyTorch DataLoader 报错。

已修复：

- [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh)：默认 `NUM_VIEWS=2`，`NUM_GPUS` 改为支持环境变量或第 1 个参数，并新增 `BATCH_SIZE >= NUM_VIEWS` 启动前检查
- [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)：同步默认 `NUM_VIEWS=2`，并新增同样的启动前检查

重跑状态：

- 输出目录：[p3_joint_input_500_pretrained_2gpu](/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/vigor_chicago/p3_joint_input_500_pretrained_2gpu)
- 修复后已成功越过 train/test DataLoader 构建、模型初始化、`JointAerialRSLoss` 初始化，并进入训练迭代
- 第 0 epoch 训练完成：200 iter，总耗时约 `0:06:22`，显存峰值约 `27473 MB`
- 第 0 epoch 末训练均值：`loss≈46.55`, `aerial_loss≈2.91`, `remote_loss≈43.65`, `rs_pointmap_loss≈407.96`, `rs_height_loss≈285.09`
- 第 1 次验证完成：10 iter，总耗时约 `0:00:11`，验证均值 `loss≈42.34`, `aerial_loss≈1.59`, `remote_loss≈40.75`, `rs_pointmap_loss≈407.51`
- 已保存 `checkpoint-best.pth`，当前训练继续进入后续 epoch

后续注意：如果需要实验 `NUM_VIEWS=4`，必须至少设置 `BATCH_SIZE>=4`，例如：

```bash
NUM_VIEWS=4 BATCH_SIZE=4 bash bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh
```

但 P3 joint 实际前向是 `num_views` 个 aerial view 再加 1 个 remote view，`NUM_VIEWS=4` 会明显增加显存压力。当前正式训练仍建议先固定 `NUM_VIEWS=2, BATCH_SIZE=2`。

## 9.3 P3 Joint 坐标系对齐问题与修复

2026-04-08 进一步检查 `P3` 时发现，比 loss 尺度更根本的问题是 remote pointmap 与 aerial 主监督没有使用同一坐标参考系。

### 9.3.1 问题根因

当前 `Pi3` aerial 主 loss 会把 aerial GT 和 prediction 统一转换到 `view0` 参考系下比较；但 joint remote 分支原先直接把 `remote_pointmap` 当作全局坐标 GT，与 `pred['pts3d']` 直接做 L1。

由于当前 VIGOR Chicago 的 remote pointmap 来自航空多视角重建结果，本身处于场景全局坐标系；而 aerial 主损失内部使用的是 `view0` 参考系，因此二者之间至少差一个 `view0` 外参逆变换。

### 9.3.2 对齐验证脚本

已新增调试脚本：

- [scripts/debug_vigor_chicago_joint_alignment.py](scripts/debug_vigor_chicago_joint_alignment.py)

它会对单个 scene 导出并比较：

- `aerial_world`
- `aerial_view0`
- `remote_global`
- `remote_view0 = inv(T_view0_world) * remote_global`

首个验证 scene：`train / location_1`。输出：

- [location_1_alignment_summary.json](/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/vigor_chicago_joint_alignment/location_1_alignment_summary.json)
- [location_1_global_topdown.png](/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/vigor_chicago_joint_alignment/location_1_global_topdown.png)
- [location_1_view0_topdown.png](/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/vigor_chicago_joint_alignment/location_1_view0_topdown.png)

关键数值：

- `aerial_world vs remote_global`: `symmetric_mean_l2 ≈ 57.17`
- `aerial_view0 vs remote_global`: `symmetric_mean_l2 ≈ 224.47`
- `aerial_view0 vs remote_view0`: `symmetric_mean_l2 ≈ 63.04`

这说明：

- `remote_global` 明显更接近 `aerial_world`，而不是 `aerial_view0`
- 将 `remote_global` 通过 `view0` 外参逆变换到 `view0` 坐标后，会重新接近 aerial 主监督使用的坐标系

### 9.3.3 已实施修复

已修改：

- [mapanything/utils/inference.py](mapanything/utils/inference.py)
  - joint batch 构造时新增 `remote_pointmap_view0`
  - 做法是对 `remote_pointmap_global` 应用 `inv(camera_pose_view0)`
- [mapanything/train/losses.py](mapanything/train/losses.py)
  - `RSPointmapHeightLoss` 新增 `compare_in_view0_frame=True` 选项
  - joint remote loss 在该模式下会先把 remote prediction 用预测的 `view0` 位姿变换到预测 `view0` 参考系，再与 `remote_pointmap_view0` 比较
  - 新增 `rs_pointmap_loss_weighted` 日志字段，避免 raw / weighted 混淆
- [configs/loss/pi3_loss_rs_joint.yaml](configs/loss/pi3_loss_rs_joint.yaml)
  - 默认启用 `compare_in_view0_frame=True`
  - 默认将 `remote_height_loss_weight` 调整为 `0.0`
- [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)
- [bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_500_2gpu.sh)
  - 默认 `LAMBDA_REMOTE_H=0.0`

### 9.3.4 为什么先关闭 joint height loss

一旦把 remote pointmap 转到 `view0` 参考系，原始 `height_map` 的物理意义就不再是“全局竖直方向的高度”。因此当前修复版 `P3` 先只保留 pointmap 监督，暂时关闭 joint `height` loss，避免继续混入另一种坐标语义。

### 9.3.5 修复后的最小 smoke

已重新跑通最小 joint smoke：

- 脚本：[bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p3_pi3_joint_input_debug_2gpu.sh)
- 覆写参数：`TRAIN_SETS=4 VAL_SETS=4 TEST_SETS=4 NUM_VIEWS=2 BATCH_SIZE=2 NUM_GPUS=2`
- 输出目录：[p3_joint_input_debug_view0fix](/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/vigor_chicago/p3_joint_input_debug_view0fix)

运行结果：

- 完整跑通 `1 epoch train + val + checkpoint-final`，`exit code = 0`
- 训练日志已出现新的 `rs_pointmap_loss_weighted` 字段
- 第一个 train step 量级：`loss≈32.70`, `aerial_loss≈11.86`, `remote_loss≈20.85`, `rs_pointmap_loss≈208.49`, `rs_pointmap_loss_weighted≈20.85`
- 第一次验证起始量级：`loss≈25.83`, `aerial_loss≈1.69`, `remote_loss≈24.13`, `rs_pointmap_loss≈241.32`, `rs_pointmap_loss_weighted≈24.13`

当前结论：`P3` joint remote supervision 现在已经切换到与 aerial 主损失一致的 `view0` 口径，后续再做 `lambda_remote_pm` 扫描才有意义。

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

### 10.2 RS-only 第一版

考虑到卫星图像当前没有可靠相机模型，第一版建议把 RS 训练单独拆出来，只做几何监督，不加视角相关项：

```text
total_loss = lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

其中：

- `remote_pointmap_loss`
  - 推荐先用 masked `L1`，其次再试 `RobustRegressionLoss`
- `remote_height_loss`
  - 推荐先用 masked `L1`
- 当前先不引入：
  - remote pose loss
  - remote relative pose loss
  - 依赖 perspective camera 的 remote view consistency loss

建议第一版权重从小值开始，例如：

- `lambda_remote_pm = 1.0`
- `lambda_remote_h = 0.1`

这里和联合训练不同，`RS-only` 阶段不需要再担心 remote loss 压过 aerial 主任务，因为当前没有 aerial 主任务。

当前建议把 `P2a` 限定成一个很小的 loss ablation，而不是扩成复杂训练策略研究。第一版只比较三组：

- `pointmap-only L1`
  - 配置：[configs/loss/pi3_rs_only_pointmap_only_l1_loss.yaml](configs/loss/pi3_rs_only_pointmap_only_l1_loss.yaml)
- `pointmap + height L1`
  - 配置：[configs/loss/pi3_rs_only_pointmap_height_l1_loss.yaml](configs/loss/pi3_rs_only_pointmap_height_l1_loss.yaml)
- `pointmap robust + height L1`
  - 配置：[configs/loss/pi3_rs_only_pointmap_height_robust_loss.yaml](configs/loss/pi3_rs_only_pointmap_height_robust_loss.yaml)

从当前实现看，`P2a` 本质上就是两类调节：

- loss 项组合
  - 是否只用 `pointmap`，还是 `pointmap + height`
- pointmap 主回归的形式
  - `L1Loss` 还是 `RobustRegressionLoss`

也就是说，你的理解是对的：`P2a` 第一版不需要改模型结构，主要就是调节这两种损失的权重，以及比较 `L1` 和 `RobustRegressionLoss`。

为了避免不同训练 loss 的数值尺度不一致，`P2a` 当前统一建议把验证口径固定成 `masked pointmap L1`。也就是说：

- `train_criterion`
  - 各实验保持自己的训练 loss
- `test_criterion`
  - 统一使用 `pointmap-only masked L1`

这样记录到 `val/test` 日志里的 `loss_avg / loss_med / rs_pointmap_loss_*` 才能直接横向比较。

### 10.2.1 RS 数据增强建议：`random_scale_offset`

这条增强路线已经落地到训练 dataset 和 benchmark dataset，当前通过参数控制。

核心思想不是改变模型输入 tensor 的最终尺寸，而是改变从原始 `1024x1024` 遥感图生成 `518x518` 样本的方式。当前实现支持：

- `crop_mode='none'`
- `crop_mode='random_scale_center'`
- `crop_mode='random_scale_offset'`
- `crop_scale_range=[0.7, 1.0]` 这类比例范围
- `num_augmented_crops_per_sample`
  - 训练侧可显式把同一 base sample 扩成多个 crop variant
  - benchmark 侧建议保持 `1`，只通过 `crop_mode` 控制是否启用增强

`random_scale_offset` 的实现方式是：

- 原始数据仍然从 `1024x1024` 读取
- 在原图上随机选择一个方形 crop
- crop 尺寸按比例范围随机采样
- crop 位置允许随机偏移，不固定中心
- 最后再 resize 到固定 `518x518`

硬约束仍然不变，以下量必须同步做完全一致的几何变换：

- `remote_image`
- `remote_pointmap`
- `remote_valid_mask`
- `remote_height_map`

也就是说，不能只增强图像，必须同步变换标签。否则 pointmap / height 会和图像错位。

### 10.2.2 当前已实现的数据集控制项

当前训练 dataset `VigorChicagoRS` 已支持：

- 多 provider 展开
  - `provider='all'` 时会遍历每个 `location` 下的全部 provider
  - 当前实际 provider 包括：`Bing_Satellite`、`ESRI_Satellite`、`ESRI_Standard`、`Google_Satellite`、`OSM_Standard`、`Positron`、`Yandex_Satellite`
- `provider_sampling_mode='expand'`
  - 每个 `(scene, provider)` 视作一个 base sample
- `num_augmented_crops_per_sample`
  - 每个 base sample 可再扩成多个 crop variant
- `crop_mode`
- `crop_scale_range`
- `image_resize_mode`
- `label_resize_mode`

当前 benchmark dataset `VigorChicagoRSAerial` 也已支持：

- `provider` 控制评测时使用哪一种卫星 provider
- `crop_mode` 控制评测时是否使用遥感裁剪增强
- `crop_scale_range`
- `image_resize_mode`
- `label_resize_mode`

注意：benchmark 当前仍要求每个 scene 只对应一个 remote sample。因此如果 benchmark 使用更宽的 provider 选择，代码会为每个 scene 只保留一个 provider 样本，避免同一 scene 被重复覆盖。

### 10.2.3 当前缩放方式

在这次改动之前：

- `remote_image` 使用 `bilinear`
- `remote_pointmap / remote_valid_mask / remote_height_map` 使用 `nearest`

当前实现里，训练和 benchmark 都已经把缩放方式做成参数化：

- `image_resize_mode`
- `label_resize_mode`

当前 `RS` 相关 config 默认都改成了：

- `image_resize_mode='nearest'`
- `label_resize_mode='nearest'`

如果后续想回到旧行为，也可以显式覆盖成 `image_resize_mode='bilinear'`。

### 10.2.4 P2b RS-Augmentation

`P2b` 现在可以直接按下列对比做：

- `no_aug`
  - `crop_mode='none'`
- `random_scale_center`
  - `crop_mode='random_scale_center'`
- `random_scale_offset`
  - `crop_mode='random_scale_offset'`

其中最优先的是 `random_scale_offset`。

### 10.3 Joint Input 第一版

在 `RS-only` 跑通后，再进入联合输入训练：

```text
total_loss = aerial_loss + lambda_remote_pm * remote_pointmap_loss + lambda_remote_h * remote_height_loss
```

其中：

- `aerial_loss = pi3_loss`
- remote 侧 loss 仍然优先使用 `masked L1`
- 第一版不引入 remote pose-related losses

建议联合训练时再把 remote 权重重新压小，例如：

- `lambda_remote_pm = 0.1`
- `lambda_remote_h = 0.05`

### 10.4 P2 的具体起点定义

如果现在继续推进 `P2 RS-only`，更合理的起点不是“先把一整套联合训练补齐”，而是先定义一个最小、可控、对 `Pi3` 破坏最小的版本。

建议把 `P2` 起点定义成：

- 模型：`Pi3`
- 初始化：加载官方 pretrained weights
- 输入：`remote_image` 单视图输入
- 主监督：`remote_pointmap`
- 次监督：`remote_height_map`
- 不监督：pose / relative pose / ray directions / depth along ray / confidence

对应目标可以写成：

```text
total_loss = 1.0 * masked_L1(pred_pts3d, gt_remote_pointmap)
           + 0.1 * masked_L1(pred_height, gt_remote_height)
```

其中：

- `pred_pts3d`
  - 直接取 `Pi3` 输出中的 `pred['pts3d']`
- `gt_remote_pointmap`
  - 取 dataset 返回的 `remote_pointmap`
- `pred_height`
  - 第一版可从 `pred['pts3d'][..., 2]` 派生
- `gt_remote_height`
  - 取 `remote_height_map`
- mask
  - 统一使用 `remote_valid_mask`

保守建议：

- 如果一开始不能完全确认 `pred['pts3d'][..., 2]` 与 `remote_height_map` 的坐标语义严格一致，第一版甚至可以先只跑 `pointmap-only`。
- 也就是说，最小起点可以进一步缩成：

```text
total_loss = masked_L1(pred_pts3d, gt_remote_pointmap)
```

### 10.5 为了尽量不破坏 Pi3 原本多视角能力，P2 应该怎么控

仅靠 loss 还不够，`P2` 更关键的是“不要把原本用于多视角分解几何的部分训坏”。

因此建议第一版 `P2` 同时采用偏保守的训练策略：

- `camera_decoder` / `camera_head` 冻结
- `conf_decoder` / `conf_head` 冻结
- `encoder` 使用比 `P1` 更低的学习率
- 主要让 point-related 分支去适配 RS 域

这样做的原因是：

- `P2` 当前根本不监督 pose / camera / confidence
- 如果这些头仍然大幅更新，只会增加把原模型多视角能力训偏的风险
- `P2` 的目标是做一次受控的域适配，不是重新定义 `Pi3` 全部输出语义

### 10.6 当前实现层面的最小缺口

`P2` 当前不是只差一个 shell 脚本，它至少还缺这几层最小实现：

1. `RS-only` 训练 dataset 接入到 train dataloader
   - 已通过 [mapanything/datasets/wai/vigor_chicago_rs.py](mapanything/datasets/wai/vigor_chicago_rs.py) 的训练侧兼容改造补齐
   - 当前每个 sample 会按单视图 list 返回，并暴露 `make_sampler(...)` 所需属性
2. `RS-only` loss 接口
   - 已通过 [mapanything/train/losses.py](mapanything/train/losses.py) 中新增的 `RSPointmapHeightLoss` 补齐
   - 当前直接读取 `remote_pointmap / remote_height_map / remote_valid_mask`
3. `RS-only` dataset config
   - 已新增：[configs/dataset/vigor_chicago_rs_518.yaml](configs/dataset/vigor_chicago_rs_518.yaml)
4. `RS-only` loss config
   - 已新增：[configs/loss/pi3_rs_only_loss.yaml](configs/loss/pi3_rs_only_loss.yaml)
5. `RS-only` 脚本
   - 已新增：[bash_scripts/train/vigor_chicago/p2_pi3_rs_only_debug_2gpu.sh](bash_scripts/train/vigor_chicago/p2_pi3_rs_only_debug_2gpu.sh)

所以，`P2` 当前已经从“纯定义阶段”推进到了“最小 debug 入口已实际跑通”的阶段。当前已验证：2 GPU、1 epoch、`Pi3 + pretrained + RSPointmapHeightLoss` 可以完成 train loop、记录 loss，并正常保存 `checkpoint-last.pth` 与 `checkpoint-final.pth`。现阶段的状态不再是“只剩骨架”，而是“有可运行的 RS-only smoke 起点”，后续主要工作转为补更正式的 P2 配置与 loss ablation。

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
