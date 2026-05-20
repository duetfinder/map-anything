# P5: VGGT RS 联合训练实验规划

> 2026-05-20 更新：本文前半部分保留早期规划背景，最新可执行结论以文末“当前实验总结”章节为准。当前已经确认 p5 主线应使用 `aerial_avg_dis` shared normalization，而不是早期规划中的 remote-only `avg_dis`。

本文档整理 `Models/map-anything` 中将 `VGGT` 作为 `p5` 实验的整体规划，目标是回答两个问题：

1. `remote / satellite` 是否可以直接当作普通 view 输入 `VGGT`，只通过 `global pointmap` 监督完成联合训练。
2. 如果仅改 loss 不够，是否只需要轻量 domain signal，还是必须增加 remote 专属结构。

当前结论是：

- `p5` 不建议一开始就做重结构改造。
- 第一优先级应该是验证：`remote 作为普通 view 输入 + remote 只用 global pointmap 监督` 是否已经足够。
- 仅在这一最小改动路线效果不足时，再增加 `view-type embedding` 和轻量 remote 分支。

## 1. 现有仓库中的相关基础

### 1.1 VGGT 本体已有 point head

当前仓库里的 `VGGT` 本体同时包含：

- `camera_head`
- `depth_head`
- `point_head`

见：

- [mapanything/models/external/vggt/models/vggt.py](mapanything/models/external/vggt/models/vggt.py)

其中：

- `depth_head` 输出 `depth + depth_conf`
- `point_head` 输出 `world_points + world_points_conf`

这意味着 `VGGT` 本体天然支持 direct global pointmap 路线，不像默认 `MapAnything` 那样必须先经过 `raydirs + depth + pose` 再还原全局点图。

### 1.2 当前仓库里的 VGGT wrapper 没有直接使用 point head

当前 [mapanything/models/external/vggt/__init__.py](mapanything/models/external/vggt/__init__.py) 的 `VGGTWrapper` 只消费了：

- `camera_head`
- `depth_head`

然后通过几何函数恢复：

- `pts3d_cam`
- `pts3d`
- `ray_directions`
- `depth_along_ray`

也就是说：

- `VGGT` 本体有原生 `world_points`
- 但当前 wrapper 没把这条路线作为主输出接口暴露出来

这正是 `p5` 实验最值得先利用的地方。

### 1.3 当前仓库已有的 RS 联合训练思路

当前仓库里已经有 `MapAnythingRSJoint`：

- [mapanything/models/mapanything/rs_joint.py](mapanything/models/mapanything/rs_joint.py)
- [MAPANYTHING_RS_JOINT_MODEL_CN.md](MAPANYTHING_RS_JOINT_MODEL_CN.md)

它的核心经验是：

- 不强迫 remote 分支服从 perspective camera 几何
- remote 侧更适合 direct pointmap supervision
- shared trunk 可以保留 aerial / remote 的信息交互

这个经验对 `VGGT p5` 仍然成立，但 `VGGT` 比 `MapAnything` 更有利的一点是：它已有原生 point head，因此不必一开始就复制 `mapanything_rs_joint` 那种重结构改造。

## 2. 原始 mapanything 项目是怎么微调的

这里先明确用户关心的“原本项目怎么微调”。

### 2.1 Pi3 微调

脚本：

- [bash_scripts/train/finetuning/pi3_finetuning.sh](bash_scripts/train/finetuning/pi3_finetuning.sh)

配置特点：

- `dataset=megatrain_13d_518_many_ar_24ipg_8g`
- `dataset.num_views=4`
- `loss=pi3_loss`
- `model=pi3`
- `train_params=pi3_finetune`
- `epochs=10`
- `warmup_epochs=1`

对应 loss：

- [configs/loss/pi3_loss.yaml](configs/loss/pi3_loss.yaml)

其核心是：

- `FactoredGeometryRegr3DPlusNormalGMLoss`
- `depth_type_for_loss='depth_z'`
- `compute_pairwise_relative_pose_loss=True`
- `convert_predictions_to_view0_frame=True`
- `compute_world_frame_points_loss=True`

对应 train params：

- [configs/train_params/pi3_finetune.yaml](configs/train_params/pi3_finetune.yaml)

这里只对 `model.encoder` 设了更低学习率：

- `model.encoder.lr = 5e-07`
- 全局 `lr = 1e-05`

也就是：

- 原始 `pi3` 微调是“全模型微调，但 backbone 更低 lr”

### 2.2 VGGT 微调

脚本：

- [bash_scripts/train/finetuning/vggt_finetuning.sh](bash_scripts/train/finetuning/vggt_finetuning.sh)

配置特点：

- `dataset=megatrain_13d_518_many_ar_24ipg_8g`
- `dataset.num_views=4`
- `loss=vggt_loss`
- `model=vggt`
- `train_params=vggt_finetune`
- `epochs=10`
- `warmup_epochs=1`

对应 loss：

- [configs/loss/vggt_loss.yaml](configs/loss/vggt_loss.yaml)

其核心是：

- `ConfAndExcludeTopNPercentPixelLoss`
- 内层仍是 `FactoredGeometryRegr3DPlusNormalGMLoss`
- `depth_type_for_loss='depth_z'`
- `compute_pairwise_relative_pose_loss=False`
- `compute_world_frame_points_loss=True`

也就是：

- `VGGT` 原始微调仍然按“depth + pose 恢复出的统一几何输出”来训
- 并没有单独针对原生 `point_head` 做专门微调方案

对应 train params：

- [configs/train_params/vggt_finetune.yaml](configs/train_params/vggt_finetune.yaml)

目前仓库里可见的配置重点是：

- 全局 `lr = 1e-05`
- 全局 `min_lr = 1e-07`
- 对 `model.aggregator.patch_embed` 使用更低 lr：
  - `lr = 5e-07`
  - `min_lr = 5e-09`

因此可以把原始 `VGGT` 微调概括为：

- 仍是多视图几何统一输出监督
- backbone / patch embed 用更低学习率
- 不是专门为 RS / satellite pointmap 设计的

## 3. `remote_view_type_embedding` 是什么，和 modality embedding 有什么关系

这里单独回答这个概念问题。

### 3.1 它不等于 patch embed / special tokens / point head

下面几个东西是不同层面的模块：

- `patch_embed`
  - 图像转 patch tokens 的输入编码层
- `camera_token / register_token`
  - transformer 里的 special tokens
- `point_head`
  - 从共享 token 解码 dense geometry 的输出头
- `view_type_embedding`
  - 加在 token 上的“域/视图类型标识”

它们不能混为一谈。

### 3.2 `view_type_embedding` 和 modality embedding 的关系

`view_type_embedding` 可以理解为一种轻量版 `modality embedding / domain embedding`。

它的作用不是替代输入编码器，而是给 shared trunk 一个明确提示：

- 这些 tokens 来自 aerial view
- 那些 tokens 来自 remote / satellite view

如果 shared trunk 全共享，而输入域差异又比较大，这个信号通常是有帮助的，因为它：

- 参数量很小
- 几乎不增加工程复杂度
- 但能显式告诉模型“这两类视图不是完全同分布”

因此可以把它理解成：

- 是 modality/domain embedding 的近亲
- 但比“完整 remote 分支”轻很多

### 3.3 为什么 `p5` 应优先试它，而不是先拆 remote 分支

因为当前 `VGGT` 本体已经有：

- shared multi-view token 空间
- 原生 global point head

所以优先验证：

- 共享结构本身是否已经足够
- 只加一个 domain signal 是否已经够用

比一上来复制 remote encoder / remote head 更稳。

## 4. P5 实验范围

根据当前判断，`p5a` 和 `p5e` 先不做。

本轮 `p5` 只保留三组实验：

- `p5b`: 全共享结构，只改 remote supervision
- `p5c`: 全共享结构 + view-type embedding
- `p5d`: 轻量 remote 分支

实验目的如下。

### 4.1 p5b: Shared-all + loss-only

实验名建议：

- `p5b_vggt_joint_shared_all_loss_only`

核心假设：

- remote 可以直接作为普通 view 输入 `VGGT`
- 不需要拆分支
- 只需要把 remote 的监督改成 global pointmap loss

这是 `p5` 的主 baseline，优先级最高。

### 4.2 p5c: Shared-all + view-type embedding

实验名建议：

- `p5c_vggt_joint_shared_all_viewtype`

核心假设：

- 共享结构本身是对的
- 问题不在于必须拆 remote 分支
- 只是在 shared trunk 中需要一个显式 domain signal

这是在 `p5b` 基础上的最小增强版。

### 4.3 p5d: Remote patch/head branch

实验名建议：

- `p5d_vggt_joint_remote_patch_pointhead`

核心假设：

- remote 的域差异主要集中在输入侧和输出侧
- shared multi-view trunk 仍然应该保留
- 但 remote 的 patch embedding 和 point head 需要独立适配

这个实验只做轻量分支，不做整套 remote aggregator。

## 5. 三组实验的结构设计

## 5.1 p5b: remote 当普通 view 输入

结构：

- 不新增 remote encoder
- 不新增 remote point head
- 不新增 remote special tokens
- 所有 view 都进入同一个 `VGGT` aggregator
- 所有 view 都通过同一个 `point_head`

监督规则：

- aerial view:
  - 保持原始 `VGGT` 的统一几何监督路线
- remote view:
  - 只对 `point_head` 产生的 `world_points` 做监督
  - 不对 remote 施加 `depth_z / pose / ray_directions` 监督

优点：

- 改动最小
- 最能回答“是不是只改 loss 就够”

风险：

- shared token 空间可能仍然会把 remote 当成 perspective 图像来组织内部表征
- 如果域差异较大，remote loss 可能会和 aerial 几何监督相互干扰

## 5.2 p5c: 在 p5b 上增加 view-type embedding

结构：

- 与 `p5b` 相同
- 但在 token 进入 shared trunk 前，为不同 view 类型加不同 embedding

示意：

- aerial tokens += `aerial_view_type_embedding`
- remote tokens += `remote_view_type_embedding`

注意：

- 这不是拆 encoder
- 也不是拆 point head
- 只是给 shared trunk 一个域标签

优点：

- 几乎不增加结构复杂度
- 能显式缓解域混淆

风险：

- 如果 remote 与 aerial 的差异主要来自输入统计和纹理，而不仅是域标识，那么仅加 embedding 可能不够

## 5.3 p5d: 轻量 remote 分支

结构：

- 保留 shared `frame/global attention` trunk
- 单独增加：
  - `remote_patch_embed`
  - 可选 `remote_camera_token`
  - 可选 `remote_register_token`
  - `remote_point_head`
- 这些模块都用原始 `VGGT` 权重初始化

初始化：

- `remote_patch_embed <- aggregator.patch_embed`
- `remote_camera_token <- aggregator.camera_token`
- `remote_register_token <- aggregator.register_token`
- `remote_point_head <- point_head`

监督规则：

- aerial view:
  - 继续使用原始 `VGGT` 的统一几何监督
- remote view:
  - 使用 `remote_point_head` 的 `world_points`
  - 只做 pointmap supervision

优点：

- 允许 remote 输入域和输出头语义独立适配
- 同时保留 shared trunk 的跨域交互

风险：

- 工程复杂度明显高于 `p5b/p5c`
- 需要控制好训练阶段和学习率，否则更容易不稳定

## 6. Loss 设计

这里重点回答“损失怎么设计”。

## 6.1 不建议直接照搬原始 `vggt_loss`

原始 [configs/loss/vggt_loss.yaml](configs/loss/vggt_loss.yaml) 的设计是：

- 所有 view 都服从统一几何接口
- 都可以参与 `depth_z / pts3d_cam / pts3d / pose` 相关监督

但 `remote / satellite` 当前不应被强迫满足这套 perspective 语义。

因此 `p5` 不建议直接把 remote 当成普通训练目标去套 `vggt_loss`。

## 6.2 推荐的 joint loss 总体形式

推荐沿用现有 `pi3_loss_rs_joint.yaml` 的“aerial + remote 双部分”思想：

- aerial loss
- remote loss
- 两者相加

也就是建议新建一套 `vggt_loss_rs_joint.yaml`，结构上可以直接参考：

- [configs/loss/pi3_loss_rs_joint.yaml](configs/loss/pi3_loss_rs_joint.yaml)

推荐形式：

- `train_criterion = JointAerialRSLoss(aerial_criterion=..., remote_criterion=..., scale_remote_loss_by_num_aerial_views=...)`
- `test_criterion = JointAerialRSLoss(...)`

## 6.3 aerial loss 设计

建议 `p5` 的 aerial loss 直接沿用 `vggt_loss` 的主结构，不做激进改动。

推荐：

- `ConfAndExcludeTopNPercentPixelLoss`
- 内层 `FactoredGeometryRegr3DPlusNormalGMLoss`
- `depth_type_for_loss='depth_z'`
- `compute_pairwise_relative_pose_loss=False`
- `compute_world_frame_points_loss=True`

原因：

- 这样能最大程度保持原始 `VGGT` 已有能力
- 避免在同一次 `p5` 实验里同时改变结构和 aerial 监督，导致变量过多

## 6.4 remote loss 设计

remote loss 建议只做 direct pointmap 主监督，初版不要上复杂项。

推荐：

- 主损失：`RSPointmapHeightLoss`
- 其中：
  - `pointmap_loss_weight > 0`
  - `height_loss_weight = 0.0` 起步

也就是初版实际只启用：

- `pointmap` loss

不建议初版就加入：

- pose loss
- ray direction loss
- camera-frame depth loss
- relative pose loss

原因：

- remote 数据最稳定的监督就是 `world-frame pointmap`
- 其他监督都隐含了 perspective camera 结构假设

## 6.5 remote pointmap loss 的具体建议

推荐沿用当前 RS 路线里已有的归一化思路：

- `pointmap_norm_mode='avg_dis'`

推荐默认开关：

- `remote_compare_in_view0_frame=true`
- `remote_detach_pose_for_view0_align=false`
- `scale_remote_loss_by_num_aerial_views=true`

解释：

- `compare_in_view0_frame=true`
  - 便于保持和当前 VIGOR Chicago RS 路线一致
  - remote 监督与 aerial multi-view 对齐逻辑更统一
- `scale_remote_loss_by_num_aerial_views=true`
  - remote 只有一个 view 时，否则在多 aerial view 下占比容易过低

## 6.6 三组实验在 loss 上的区别

### p5b 的 loss

结构：

- `aerial_loss = 原始 vggt_loss 的 aerial 部分`
- `remote_loss = pointmap only`

建议默认值：

- `remote_pointmap_loss_weight = 1.0`
- `remote_height_loss_weight = 0.0`
- 扫描范围：`0.5 / 1.0 / 2.0`

### p5c 的 loss

与 `p5b` 完全相同。

区别只在模型结构里多了：

- `view_type_embedding`

这样才能把变量控制住，让 `p5c` 只回答“显式 domain signal 是否有帮助”。

### p5d 的 loss

主 loss 形式仍与 `p5b/p5c` 相同。

仍建议：

- aerial 保持原始统一几何 loss
- remote 只做 pointmap only

理由是：

- `p5d` 的变量已经是结构变化
- 不要再同时引入更复杂的 remote loss

## 7. 训练策略

## 7.1 p5b: 原则上可按原始 VGGT finetune 方式开始

`p5b` 因为结构不变，最适合沿用原始 `vggt_finetune` 风格：

- 全模型微调
- `patch_embed` 更低 lr

建议：

- 全局 `lr = 1e-05`
- `patch_embed lr = 5e-07`
- `epochs = 50`
- `warmup_epochs = 1`

和原始 `vggt_finetune` 的区别只在：

- loss 换成 RS joint 版本
- dataset 换成 `vigor_chicago_rs_joint_518`

## 7.2 p5c: 与 p5b 保持一致

`p5c` 不建议额外改训练策略。

建议与 `p5b` 保持相同：

- 相同 dataset
- 相同 loss
- 相同 train params

这样 `p5b/p5c` 的对比才是干净的。

## 7.3 p5d: 建议分阶段

因为 `p5d` 引入了 remote 专属参数，建议至少做两阶段训练：

### 阶段 1

- 冻结 shared trunk 的大部分参数
- 训练：
  - `remote_patch_embed`
  - `remote_point_head`
  - 可选 `remote special tokens`
  - `remote_view_type_embedding`

目的：

- 先让 remote 输入和输出头完成域适配

### 阶段 2

- 解冻 shared trunk 的后几层
- 继续联合训练

目的：

- 让 remote 与 aerial 在 shared token 空间中对齐

如果只做一阶段直接全开，训练不稳定的风险更大。

## 8. 推荐实验顺序

推荐的正式顺序：

1. `p5b_vggt_joint_shared_all_loss_only`
2. `p5c_vggt_joint_shared_all_viewtype`
3. `p5d_vggt_joint_remote_patch_pointhead`

推荐的判断逻辑：

- 如果 `p5b` 已经稳定且效果好，说明 remote 不一定需要单独分支。
- 如果 `p5b` 不稳定，而 `p5c` 改善明显，说明问题主要是 domain confusion，不必急着拆分支。
- 如果 `p5c` 仍明显不足，再做 `p5d`，说明 remote 需要更强的输入/输出侧专属适配。

## 9. 建议新增的配置与脚本

如果进入实现阶段，建议新增：

- `configs/loss/vggt_loss_rs_joint.yaml`
- `configs/model/vggt_rs_joint_viewtype.yaml`
- `configs/model/vggt_rs_joint_remote_head.yaml`
- `configs/train_params/vggt_rs_joint_finetune.yaml`

建议新增训练脚本：

- `bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_loss_only.sh`
- `bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh`
- `bash_scripts/train/Crossview/vggt/p5d_vggt_joint_remote_patch_pointhead.sh`

其中：

- `p5b` 可以直接复用 `model=vggt`
- `p5c` 需要新增带 `view_type_embedding` 的 `VGGT` 变体
- `p5d` 需要新增轻量 remote 分支版 `VGGT` 变体

## 9.1 当前脚本目录组织建议

当前 `Crossview` 训练脚本目录已经有：

- `bash_scripts/train/Crossview/mapanything`
- `bash_scripts/train/Crossview/pi3`
- `bash_scripts/train/Crossview/vggt`

因此本轮 `p5` 的 `VGGT` 脚本建议直接放在：

- `bash_scripts/train/Crossview/vggt/`

当前不建议再额外加一层子目录，例如：

- `p5/`
- `joint_rs/`
- `debug/`
- `formal/`

原因是：

- 当前 `VGGT` Crossview 实验脚本数量还少
- 只有 `p5b / p5c` 两个主实验脚本时，平铺最直接
- 目录层次过深会让调用和维护都更重

因此当前建议是：

- `p5b_vggt_joint_shared_all_loss_only.sh`
- `p5c_vggt_joint_shared_all_viewtype.sh`

都直接放在：

- `bash_scripts/train/Crossview/vggt/`

只有在后续出现以下情况时，再考虑加子目录：

- `VGGT` 下继续扩出 `debug / smoke / formal / ablation`
- `p5d / p5e / p5f` 等脚本显著增加
- 希望把 `RS-only`、`joint`、`benchmark` 分开管理

如果后续确实需要再拆层，优先建议的长期目录名是：

- `bash_scripts/train/Crossview/vggt/joint_rs/`

而不是按阶段号单独建：

- `bash_scripts/train/Crossview/vggt/p5/`

因为：

- `p5` 更适合表达实验序号
- `joint_rs` 更适合表达长期语义归类

## 10. 最终建议

当前最推荐的判断是：

- 不要先假设“remote 一定要单独分支”
- `p5` 应先把 `VGGT` 当作一个本身就具备 global point head 的模型来用
- 先验证“remote 作为普通输入 view + remote only pointmap supervision”是否已经够用

因此 `p5` 的中心不是：

- 一开始设计最复杂的 remote branch

而是：

- 用最少变量验证 `VGGT` 原生 `point_head` 在 RS joint 任务上的可迁移性

如果这个基础成立，再逐步增加：

- `view-type embedding`
- 轻量 remote patch/head branch

这会比直接复制一整套 remote encoder / remote transformer 更稳，也更容易解释实验结果。

---

## 11. 当前实验总结（2026-05-20）

本节是当前 p5 实验的最新状态，优先级高于前面的早期规划。

当前已经实现并验证的关键判断是：

- VGGT 的 global pointmap 语义是 canonical / first-view 坐标系，实际可按 view0 坐标系理解。
- Remote GT 不能直接用原始 world-frame pointmap 监督 VGGT point head，应使用 `remote_pointmap_view0`。
- 对 p5b，`REMOTE_COMPARE_GT_IN_VIEW0_ONLY=true` 是必要设置。
- `REMOTE_COMPARE_IN_VIEW0=false` 更符合 VGGT point head 语义，因为 remote pred 本身已经应在 view0/canonical 坐标系下，不需要再通过预测 pose 变换一次。
- 实验证明 remote-only `avg_dis` 归一化会造成普通视角和 remote 点云之间的尺度/偏移不一致风险。
- 当前 p5 主线应统一使用 `aerial_avg_dis`，即 shared / aerial-based normalization。

### 11.1 当前推荐默认参数

当前 p5b / p5c / p5d 应统一使用：

- `remote_compare_in_view0_frame=false`
- `remote_compare_gt_in_view0_frame_only=true`
- `remote_pointmap_norm_mode=aerial_avg_dis`
- `scale_remote_loss_by_num_aerial_views=true`
- `remote_height_loss_weight=0.0`

`aerial_avg_dis` 的含义：

- remote pred 不再用 remote 自己的点云尺度归一化。
- remote pred 除以普通视角预测点云组计算出的 `avg_dis` factor。
- remote GT 除以普通视角 GT 点云组计算出的 `avg_dis` factor。
- 这样 remote loss 和普通视角 loss 使用同一组尺度参考，保留归一化但避免 remote 独立尺度漂移。

### 11.2 当前已实现文件

模型入口：

- [mapanything/models/external/vggt/__init__.py](mapanything/models/external/vggt/__init__.py)

当前 `VGGTWrapper` 已支持：

- `use_point_head_for_remote`
- `use_view_type_bias`
- `ordinary_output_head`
- `remote_output_head`
- `use_remote_private_point_head`
- `output_point_head_for_consistency`

Loss：

- [configs/loss/vggt_loss_rs_joint.yaml](configs/loss/vggt_loss_rs_joint.yaml)
- [configs/loss/vggt_loss_rs_joint_shared_norm.yaml](configs/loss/vggt_loss_rs_joint_shared_norm.yaml)
- [configs/loss/vggt_loss_rs_joint_p5d.yaml](configs/loss/vggt_loss_rs_joint_p5d.yaml)
- [mapanything/train/losses.py](mapanything/train/losses.py)

训练脚本：

- [bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_loss_only.sh](bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_loss_only.sh)
- [bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh](bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh)
- [bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh](bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh)
- [bash_scripts/train/Crossview/vggt/p5d_vggt_remote_point_head_consistency.sh](bash_scripts/train/Crossview/vggt/p5d_vggt_remote_point_head_consistency.sh)

导出和检查脚本：

- [scripts/export_pointcloud_ply.py](scripts/export_pointcloud_ply.py)
- [scripts/export_vigor_joint_rs_gt_ply.py](scripts/export_vigor_joint_rs_gt_ply.py)

### 11.3 p5b 当前主线：shared-all + shared normalization

当前推荐 p5b 脚本：

- [bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh](bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh)

对应 loss：

- [configs/loss/vggt_loss_rs_joint_shared_norm.yaml](configs/loss/vggt_loss_rs_joint_shared_norm.yaml)

结构：

- 不新增 remote encoder。
- 不新增 remote private point head。
- 所有输入视角共享 VGGT encoder / aggregator。
- 普通视角走 `camera_head + depth_head`。
- Remote 视角走 VGGT 原生 `point_head`。

监督：

- 普通视角使用 VGGT 原始几何监督：`FactoredGeometryRegr3DPlusNormalGMLoss`。
- Remote 只启用 pointmap loss：`RSPointmapHeightLoss`。
- Remote GT 使用 `remote_pointmap_view0`。
- Remote 归一化使用 `aerial_avg_dis`。

推荐命令：

```bash
cd /root/autodl-tmp/Models/map-anything
bash bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh
```

低显存命令：

```bash
cd /root/autodl-tmp/Models/map-anything
CUDA_DEVICES=0 NUM_GPUS=1 NUM_VIEWS=2 BATCH_SIZE=2 \
bash bash_scripts/train/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm.sh 1
```

当前结论：

- `p5b_vggt_joint_shared_all_loss_only.sh` 是旧 baseline，默认 remote-only `avg_dis`，不再作为推荐主线。
- `p5b_vggt_joint_shared_all_shared_norm.sh` 是当前 p5b 正式主线。
- 后续 p5c / p5d 应继承该归一化方式，否则实验变量不干净。

### 11.4 p5c 当前定位：shared-all + view-type bias

脚本：

- [bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh](bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh)

实验目的：

- 在 p5b shared-all 基础上加入 view-type / domain signal。
- 验证 remote 与普通视角是否主要存在 shared trunk 中的 domain confusion。

注意：

- p5c 应使用与 p5b shared norm 完全一致的 remote loss。
- 如果脚本默认仍指向 `vggt_loss_rs_joint.yaml` 或默认 `avg_dis`，正式实验前应改成 `vggt_loss_rs_joint_shared_norm.yaml`，或者显式覆盖 `REMOTE_POINTMAP_NORM_MODE=aerial_avg_dis`。

建议命令：

```bash
cd /root/autodl-tmp/Models/map-anything
REMOTE_POINTMAP_NORM_MODE=aerial_avg_dis \
bash bash_scripts/train/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype.sh
```

### 11.5 p5d 当前定位：remote private point head + branch consistency

脚本：

- [bash_scripts/train/Crossview/vggt/p5d_vggt_remote_point_head_consistency.sh](bash_scripts/train/Crossview/vggt/p5d_vggt_remote_point_head_consistency.sh)

对应 loss：

- [configs/loss/vggt_loss_rs_joint_p5d.yaml](configs/loss/vggt_loss_rs_joint_p5d.yaml)

当前实际实现不是完整 remote encoder / remote patch branch，而是轻量版本：

- 普通视角：`ordinary_output_head=depth`
- Remote 视角：`remote_output_head=point`
- Remote 使用 `use_remote_private_point_head=true`
- `remote_private_point_head` 从原始 VGGT `point_head` 权重初始化
- 启用 `output_point_head_for_consistency=true`
- 对普通视角增加 `VGGTBranchConsistencyLoss`，约束 VGGT 原生 point head 输出靠近 `camera+depth` 路线输出

当前 p5d 默认已更新为：

- `REMOTE_POINTMAP_NORM_MODE=aerial_avg_dis`
- `LAMBDA_BRANCH_CONSISTENCY=0.2`
- `BRANCH_CONSISTENCY_NORM_MODE=null`
- `BRANCH_CONSISTENCY_DETACH_DEPTH=true`

推荐命令：

```bash
cd /root/autodl-tmp/Models/map-anything
bash bash_scripts/train/Crossview/vggt/p5d_vggt_remote_point_head_consistency.sh
```

已验证：

- p5d smoke 能跑通训练、反传和保存 checkpoint。
- p5d 当前已经和 p5b shared norm 使用同一种 `aerial_avg_dis` 归一化。

### 11.6 Loss 设计的当前解释

普通视角 loss 保持 VGGT finetune 主结构：

- `ConfAndExcludeTopNPercentPixelLoss`
- 内层 `FactoredGeometryRegr3DPlusNormalGMLoss`
- `depth_type_for_loss='depth_z'`
- `compute_pairwise_relative_pose_loss=False`
- `compute_world_frame_points_loss=True`

Remote loss 只启用：

- pointmap loss

Remote 暂不启用：

- pose loss
- ray direction loss
- camera-frame depth loss
- height loss

原因是 remote/satellite 不应被强迫满足普通 perspective camera 的局部几何语义。

### 11.7 为什么需要 shared normalization

旧 remote-only `avg_dis` 的问题是：

- remote pred 用 remote pred 自己的尺度 factor。
- remote GT 用 remote GT 自己的尺度 factor。
- 普通视角 loss 另用普通视角组的尺度 factor。

这使 remote 和普通视角虽然都做了归一化，但尺度约束来源不同。实际观察中，这会导致训练数据上也可能出现 remote 点云和普通视角点云的固定偏移。

当前 shared normalization 改为：

- remote pred 使用 aerial pred group 的尺度 factor。
- remote GT 使用 aerial GT group 的尺度 factor。

这样 remote loss 和 ordinary/aerial loss 共享尺度参考，更符合最终导出统一全局点云的目标。

### 11.8 导出建议

p5b / p5c 导出建议：

```bash
cd /root/autodl-tmp/Models/map-anything
python scripts/export_pointcloud_ply.py \
  --model vggt \
  --checkpoint_path /path/to/checkpoint-best.pth \
  --image_folder /path/to/scene \
  --output_path /path/to/output \
  --vggt_joint_remote_export \
  --remote_view_names zimage.png \
  --vggt_ordinary_output_head depth \
  --vggt_remote_output_head point
```

p5d 导出建议加上 private point head：

```bash
cd /root/autodl-tmp/Models/map-anything
python scripts/export_pointcloud_ply.py \
  --model vggt \
  --checkpoint_path /path/to/p5d/checkpoint-best.pth \
  --image_folder /path/to/scene \
  --output_path /path/to/output \
  --vggt_joint_remote_export \
  --remote_view_names zimage.png \
  --vggt_ordinary_output_head depth \
  --vggt_remote_output_head point \
  --vggt_use_remote_private_point_head
```

原始 VGGT baseline 导出：

```bash
cd /root/autodl-tmp/Models/map-anything
python scripts/export_pointcloud_ply.py \
  --model vggt \
  --checkpoint_path /root/autodl-tmp/outputs/checkpoints/vggt/model.pt \
  --image_folder /path/to/scene \
  --output_path /path/to/output
```

GT 检查结论：

- 已通过 [scripts/export_vigor_joint_rs_gt_ply.py](scripts/export_vigor_joint_rs_gt_ply.py) 验证 `vigor_joint_rs_gt_view0_raw_remote.ply` 中 ordinary 和 remote GT 是对齐的。
- 因此早期 p5b 的偏移更可能来自训练目标、归一化或输出分支，而不是原始 GT 未对齐。

### 11.9 当前推荐实验顺序

推荐正式顺序：

1. `p5b_vggt_joint_shared_all_shared_norm`
2. `p5c_vggt_joint_shared_all_viewtype`，但必须使用 `aerial_avg_dis`
3. `p5d_vggt_remote_point_head_consistency`

判断逻辑：

- 如果 p5b shared norm 已经明显缓解偏移，说明主要问题是归一化尺度来源。
- 如果 p5c 进一步改善，说明 domain signal 对 shared trunk 有帮助。
- 如果 p5d 进一步改善，说明 remote 需要独立 point head 适配，而不仅是共享 point head。

### 11.10 当前不推荐作为主线的设置

以下设置可以保留做 ablation，但不建议作为当前 p5 主实验默认值：

- `remote_pointmap_norm_mode=avg_dis`
- `remote_compare_in_view0_frame=true`
- 直接用原始 `vggt_loss.yaml` 训练 remote view
- 对 remote 强加 pose / ray direction / camera-frame depth loss
- 在未固定导出协议前同时扩大多个结构和 loss 变量
