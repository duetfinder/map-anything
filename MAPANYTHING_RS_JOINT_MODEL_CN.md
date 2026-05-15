# MapAnything RS-joint 模型设计说明

本文档单独总结 `MapAnything RS-joint` 变体的模型改动思路。训练过程、实验脚本、smoke 记录见：

- [VIGOR_CHICAGO_TRAINING_CN.md](VIGOR_CHICAGO_TRAINING_CN.md)
- [MAPANYTHING_LOSSES_CN.md](MAPANYTHING_LOSSES_CN.md)

当前实现代码与配置见：

- 模型实现：[mapanything/models/mapanything/rs_joint.py](mapanything/models/mapanything/rs_joint.py)
- 模型配置：[configs/model/mapanything_rs_joint.yaml](configs/model/mapanything_rs_joint.yaml)
- 微调参数：[configs/train_params/mapanything_rs_joint_finetune.yaml](configs/train_params/mapanything_rs_joint_finetune.yaml)

## 1. 问题背景

`MapAnything` 原始主模型默认针对 perspective multi-view 几何设计，aerial view 的输出语义通常是：

- `ray_directions`
- `depth_along_ray`
- `pose`
- 可选 `confidence / mask / scale`

然后再由这些量恢复出：

- `pts3d_cam`
- `pts3d`

这条路线默认依赖：

- 中心投影相机假设
- 可解释的相机位姿
- 可从 `intrinsics + pose + depth` 恢复几何

但 `remote / satellite` 输入并不天然满足这套假设。当前 VIGOR 的 remote 侧更稳定、直接的监督是：

- `remote_pointmap`
- `remote_valid_mask`
- `remote_height_map`

因此 `RS-joint` 的核心目标不是把 remote 伪装成普通 camera view，而是：

- 保留 aerial 分支的原始 `MapAnything` 几何能力
- 给 remote 单独增加一个不依赖中心投影假设的 direct pointmap 分支
- 仍让 aerial 与 remote 在中间层共享信息

## 2. 总体思路

当前 `RS-joint` 采用的是“共享中间 transformer，输入和输出分流”的结构：

- aerial 分支：
  - 保留原始 `MapAnything`
  - 继续输出 `raydirs + depth + pose (+ conf + mask + scale)`
- remote 分支：
  - 新增独立 `remote_encoder`
  - 新增独立 `remote DPT head`
  - 直接输出 `pointmap + confidence + mask`
- 中间层：
  - 继续共享原始 `info_sharing transformer`

一句话概括：

- `encoder` 分流
- `shared transformer` 共享
- `prediction head` 分流

## 3. 为什么不直接复用原始 dense head

原始 `MapAnything` 默认 dense head 的输出语义不是 pointmap，而是 factorized geometry。

如果直接让 remote 也输出：

- `ray_directions`
- `depth_along_ray`
- `pose`

那就等价于要求 remote 也服从 perspective camera 几何。这对当前 satellite 输入并不自然，训练上也会引入很多不稳定的伪约束。

因此 remote 侧更合理的做法是：

- 直接预测 `world-frame pointmap`
- 用 `pointmap` 误差做主监督

这也是 `RS-joint` 与默认 `MapAnything` 最大的结构差异。

## 4. 当前结构改动

### 4.1 新增 remote encoder

`RS-joint` 增加了：

- `remote_encoder`

实现上直接从 `aerial encoder` 深拷贝初始化，代码见 [rs_joint.py](mapanything/models/mapanything/rs_joint.py)。

设计动机：

- aerial 和 remote 都是 RGB 图像，底层纹理特征可以共享初始化
- 但两者域差异明显，不适合长期完全共享一个 encoder

所以这里采用：

- 初始化相同
- 训练时参数独立

### 4.2 新增 remote DPT head

`RS-joint` 新增：

- `remote_dpt_feature_head`
- `remote_dpt_regressor_head`
- `remote_dense_adaptor`

其中：

- `remote_dpt_feature_head` 复用 DPT 特征解码结构
- `remote_dpt_regressor_head` 输出维度改成 `5`

当前 `5` 个通道的语义是：

- `pointmap xyz`：3 通道
- `confidence`：1 通道
- `mask`：1 通道

最终 adaptor 类型是：

- `PointMapWithConfidenceAndMaskAdaptor`

这意味着 remote 侧输出不再走 `raydirs + depth` 还原，而是直接回归 pointmap。

### 4.3 共享 transformer 不改结构，只改接线

`info_sharing transformer` 本体没有改层数、block 类型和 hidden dim。

当前变化是：

- aerial token 由 `aerial encoder` 提供
- remote token 由 `remote encoder` 提供
- 两者一起送进同一个 shared transformer
- transformer 输出后再按 view 类型分别送到 aerial / remote head

所以严格说：

- 改的是 `forward` 路由
- 不是 transformer 结构定义本身

### 4.4 轻量 domain signal

为了让 shared transformer 能区分 aerial / remote 两种域，当前实现加了两个可学习向量：

- `aerial_view_type_embedding`
- `remote_view_type_embedding`

它们不是额外 token，而是对 encoder 输出 feature 做 broadcast 加法：

- aerial token 全部加上 `aerial_view_type_embedding`
- remote token 全部加上 `remote_view_type_embedding`

这样不会改变 token 长度，只会改变 token 内容分布。

这一步的作用是：

- 给 transformer 一个显式 domain signal
- 避免它把 remote token 完全当成普通 aerial token 处理

## 5. Forward 流程

当前 `forward()` 的逻辑可以概括为：

1. 判断每个 view 是否为 remote  
   当前通过 `view["instance"] == "remote"` 区分。

2. 分别编码  
   aerial view 走 `self.encoder`，remote view 走 `self.remote_encoder`。

3. 进入 shared transformer  
   所有 view 的 token 一起进入 `info_sharing`。

4. 分别解码  
   aerial view 走原始 `downstream_head`，remote view 走 `remote_dense_head`。

5. 按原始 view 顺序拼回输出  
   保持训练与评估接口不变。

如果一个 batch 内没有 remote view，当前实现会直接退回到原始 `MapAnything.forward()`。

## 6. 初始化策略

当前初始化遵循“尽量复用已有几何先验，但不强行共享错误语义”的原则。

### 6.1 可以直接镜像的部分

- `remote_encoder <- aerial_encoder`
- `remote_dpt_feature_head <- aerial_dpt_feature_head`

因为这些模块学习的是更通用的图像特征和 dense 解码特征。

### 6.2 只能部分镜像的部分

- `remote_dpt_regressor_head <- aerial_dpt_regressor_head`

这里只拷贝 shape 对得上的中间层，最终输出层不能完全照搬。原因是：

- aerial 默认输出语义是 `raydirs + depth + ...`
- remote 输出语义是 `pointmap + confidence + mask`

两者最后一层的通道语义并不一一对应。

因此当前策略是：

- 中间层尽量复用
- 最终几何输出层保留新初始化

### 6.3 载入 checkpoint 的策略

当前 `load_state_dict()` 做了一个额外处理：

- 如果 checkpoint 本身没有 `remote_*` 权重
- 则在载入 aerial 权重后，再自动镜像一次 remote 分支

这样可以直接从普通 `MapAnything` checkpoint 启动 `RS-joint` 微调。

## 7. 输出语义

### 7.1 aerial 输出

aerial 分支保持原始 `MapAnything` 语义，主要包括：

- `pts3d`
- `pts3d_cam`
- `ray_directions`
- `depth_along_ray`
- `cam_trans`
- `cam_quats`
- `metric_scaling_factor`
- 可选 `conf`
- 可选 `non_ambiguous_mask`

### 7.2 remote 输出

remote 分支当前只输出：

- `pts3d`
- `conf`
- `non_ambiguous_mask`
- `non_ambiguous_mask_logits`

注意：

- remote 当前没有 `pts3d_cam`
- remote 当前没有 `ray_directions`
- remote 当前没有 `depth_along_ray`
- remote 当前没有 `pose`

这是有意为之，因为 remote 分支当前不承担 perspective 相机几何语义。

## 8. Loss 设计

当前 `RS-joint` 的 loss 采取“aerial 保持原样，remote 最小化改动”的原则。

### 8.1 aerial loss

aerial 继续使用原始 `MapAnything` 的：

- `FactoredGeometryRegr3DPlusNormalGMLoss`

也就是继续监督：

- `pts3d`
- `pts3d_cam`
- `depth`
- `ray_directions`
- `pose`
- `normal`
- `gradient matching`

### 8.2 remote loss

remote 当前第一版只建议做：

- `pointmap` 主损失

可选再加：

- `height` 辅助损失

但不建议一开始就加：

- pose loss
- ray direction loss
- camera-frame depth loss

### 8.3 为什么 remote 只用 world-frame pointmap

当前 remote 更适合直接监督：

- `pred["pts3d"]`

而不是：

- `pred["pts3d_cam"]`

原因是 `pts3d_cam` 隐含了相机坐标系定义，而 remote 当前并没有稳定的 perspective camera model。

### 8.4 remote loss 的外层缩放

当前还增加了一个外层开关：

- `scale_remote_loss_by_num_aerial_views`

它的作用是：

- 当 aerial view 数增加时
- 把 remote loss 乘上 `num_aerial_views`

原因是：

- aerial 通常是多 view 累加监督
- remote 通常只有 1 个 view
- 如果权重固定，remote 在联合训练里的相对占比会越来越小

## 9. 训练策略

当前默认训练策略是微调，而不是从零训练。

### 9.1 冻结部分

默认冻结：

- `aerial encoder`
- 原几何输入 encoder
- 原 `dpt_feature_head`
- 原 `dpt_regressor_head`
- `pose_head`
- `scale_head`

### 9.2 训练部分

默认放开：

- `remote_encoder`
- `remote_dpt_feature_head`
- `remote_dpt_regressor_head`
- `info_sharing`
- `aerial_view_type_embedding`
- `remote_view_type_embedding`

这个策略的核心目的是：

- 尽量保持 aerial branch 已有能力
- 只让 remote branch 和 shared transformer 去适配新输入域

## 10. 当前实现的边界

当前版本是第一版可训练工程实现，不是最终结构。

它的边界主要有这些：

1. remote 目前只支持 direct pointmap 路线  
   还没有进一步加入更复杂的 normal / gm / consistency 设计。

2. transformer 只是共享，没有做更强的 modality-aware 结构  
   当前只有轻量 view-type bias，没有额外 cross-domain block。

3. remote 目前只通过 `instance == "remote"` 区分  
   属于最小侵入式工程方案。

4. remote head 目前只做 dense geometry  
   没有单独设计 remote 专属 pose / scale 分支。

## 11. 为什么当前方案是合理的

当前方案的核心优点是：

- 它最大程度复用了已经训练好的 `MapAnything` 权重
- 它没有强迫 remote 分支服从不自然的中心投影假设
- 它保留了 aerial / remote 在 shared transformer 中交互的能力
- 它的工程侵入面可控，便于继续做 loss 和训练策略迭代

因此从工程路径看，`RS-joint` 更像是：

- 以 `MapAnything` 为主体
- 给 remote 增加一个 direct pointmap 分支
- 再通过 joint fine-tuning 把这两个域接起来

而不是把 `pi3` 或默认 `MapAnything` 生硬改造成同一套输出语义。

## 12. 后续可继续迭代的方向

如果后续要继续深改，可以优先考虑：

1. 给 remote 增加更稳定的 surface regularization  
   例如 normal / gradient matching。

2. 在 shared transformer 前后加入更强的 domain-aware 设计  
   例如更显式的 modality token 或 cross-attention 约束。

3. 研究是否需要让 aerial / remote 在部分 encoder 层共享、部分层独立。

4. 重新标定 remote loss 权重与 aerial view 数的关系，形成更稳定的正式训练配方。
