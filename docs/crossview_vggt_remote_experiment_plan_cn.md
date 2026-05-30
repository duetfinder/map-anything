# Crossview VGGT Remote 实验规划

## 当前判断

p5f-lite 的结果说明，remote 分支已经被模型感知到，但没有形成对普通视角重建的稳定正向约束。mini benchmark 中 same remote 相比 aerial-only 的 pointmap 提升接近 0，且 same / blank / shuffled 的差距很小；可视化也显示 joint 输入会破坏普通视角重建。这更像是 remote token 过早进入 VGGT 全局混合后扰动了原始普通视角表征，而不是有效补充了场景信息。

因此下一轮实验的核心目标不是让 remote 自身预测更准，而是验证：remote 是否能在不损伤普通视角 baseline 的前提下，对普通视角输出提供 scene-specific 的增益。

## 实验原则

1. 先保护原始 VGGT 的普通视角能力。普通视角输入单独走原始 aggregator，避免 remote 在早期 self-attention 中污染普通 token。
2. remote 先独立编码，再通过很小的 late adapter 影响普通视角 token。adapter 初始 gate 为 0，使初始行为接近原始 VGGT。
3. 所有实验都必须跑 same / blank / shuffled remote 控制组。只有 same 明显优于 blank 和 shuffled，才说明模型用到了当前场景的 remote 信息。
4. checkpoint 选择不能只看训练 loss。优先看 ordinary path 是否受损、same gain 是否为正、same 相对 blank/shuffled 是否有特异性提升。
5. 当前仍在归一化空间训练 pointmap/height 约束；raw metric 不是主目标，但 ordinary damage 和 scene-specific gain 是关键诊断。

## 已实现实验矩阵

### p5g_no_fusion_split_remote

目标：诊断 split aggregator 本身是否保护了普通视角路径。

结构：ordinary views 和 remote views 分开跑 VGGT aggregator，之后合并 token，但不做 remote-to-aerial fusion。remote 使用 private point head，普通视角仍使用 depth 输出。

预期：普通视角指标应接近未训练 VGGT 或至少不弱于 p5f。如果这一项仍明显变差，说明问题不在 fusion，而在训练配置、loss 或 head/optimizer 影响了普通路径。

命令：

```bash
bash bash_scripts/train/Crossview/vggt/p5g_vggt_no_fusion_split_remote.sh
```

### p5g_film_split_remote

目标：低成本验证 remote 全局上下文是否能稳定增强普通视角。

结构：remote 独立编码后，对 ordinary patch token 做 gated FiLM residual。FiLM 只使用 remote patch tokens 的全局均值，参数量小、过拟合风险低。

优点：轻量，适合 2000 场景规模；如果有效，说明 remote 的全局场景先验足够提供增益。

风险：remote 正射图和地面透视图之间的局部对应关系不强，单个全局 context 可能过粗，增益有限。

命令：

```bash
bash bash_scripts/train/Crossview/vggt/p5g_vggt_film_split_remote.sh
```

### p5g_crossattn_split_remote

目标：主候选方案，验证 remote patch token 是否能通过 late cross-attention 给 ordinary patch token 提供更细粒度信息。

结构：ordinary patch tokens 作为 query，remote patch tokens 作为 key/value。remote token 数量默认下采样到 256，输出再经过 scalar gate 加回 ordinary patch tokens。

优点：比 FiLM 更有表达力，能学习局部/区域级 remote prior；同时 remote 不进入早期 VGGT self-attention，普通路径更可控。

风险：参数和计算更大，数据量不足时可能学到弱或非 scene-specific 的 shortcut。必须依赖 same / blank / shuffled 控制组判断。

命令：

```bash
bash bash_scripts/train/Crossview/vggt/p5g_vggt_crossattn_split_remote.sh
```

## 训练配置要点

p5g 使用 `configs/train_params/vggt_p5g_late_fusion.yaml`：

- 默认 `lr: 0`，冻结原始 VGGT trunk 和普通输出路径。
- 训练 late-fusion adapter：`remote_to_aerial_late_film` 或 `remote_to_aerial_late_cross_attention`。
- 训练 `remote_point_head`，用于 remote private point 输出。
- gate 初始为 0，使训练初始状态接近原始 VGGT，降低一开始就破坏普通视角的风险。

p5g 共享训练入口：

```bash
bash bash_scripts/train/Crossview/vggt/p5g_vggt_split_late_fusion.sh
```

常用覆盖示例：

```bash
FUSION_TYPE=cross_attention \
EPOCHS=50 \
LAMBDA_REMOTE_PM=4.0 \
LAMBDA_BRANCH_CONSISTENCY=0.2 \
bash bash_scripts/train/Crossview/vggt/p5g_vggt_split_late_fusion.sh
```

## Mini Benchmark

每个候选训练完成后，先跑 5-scene mini benchmark，并打开 same / blank / shuffled controls：

```bash
REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5g_no_fusion_unified.sh

REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5g_film_unified.sh

REMOTE_OVERFIT_NUM_SETS=5 bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5g_crossattn_unified.sh
```

默认输出：

```text
outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/<experiment>_mini_controls
```

## 指标汇总脚本

新增脚本：

```bash
python scripts/summarize_rs_guided_benchmark.py \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_raw_local_4v_overfit5 \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_p5b_shared_norm_4v_overfit5 \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_p5e_remote_head_attention_4v_overfit5 \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_crossview_p5f_lite_mini_controls \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p5g_vggt_film_split_remote_mini_controls \
  outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p5g_vggt_crossattn_split_remote_mini_controls \
  --reference outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_raw_local_4v_overfit5 \
  --output_csv outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p5g_summary.csv
```

脚本会读取每个目录下的 `rs_aerial_benchmark_results.json`，输出：

- `aerial_only__pointmaps_abs_rel`：没有 remote 输入时普通视角表现。
- `joint_same__pointmaps_abs_rel`：使用正确 remote 时普通视角表现。
- `same_gain__pointmaps_abs_rel`：`aerial_only - same`，正数代表 remote 带来提升。
- `specific_gain_blank__pointmaps_abs_rel`：`blank - same`，正数代表正确 remote 优于空 remote。
- `specific_gain_shuffled__pointmaps_abs_rel`：`shuffled - same`，正数代表正确 remote 优于错配 remote。
- `ordinary_damage_vs_reference__*`：相对 reference 的普通路径损伤，正数代表变差。
- `remote_damage__rs_height_mae_affine`：joint remote 相对 rs-only 的 remote 分支损伤，正数代表变差。
- `pass_rate_same_better_than_*`：逐场景 same 是否更好的比例。

## 判读标准

优先级最高的判断顺序：

1. `ordinary_damage_vs_reference__pointmaps_abs_rel` 接近 0 或为负。普通路径不能比原始 VGGT 明显差。
2. `same_gain__pointmaps_abs_rel > 0`。正确 remote 至少要提升 ordinary pointmap。
3. `specific_gain_blank__pointmaps_abs_rel > 0` 且 `specific_gain_shuffled__pointmaps_abs_rel > 0`。正确 remote 必须优于空 remote 和错配 remote。
4. `pass_rate_same_better_than_blank__pointmaps_abs_rel`、`pass_rate_same_better_than_shuffled__pointmaps_abs_rel` 应明显高于 0.5，理想值大于 0.6 或 0.7。
5. `same_gain__z_depth_abs_rel` 和 `same_gain__ray_dirs_err_deg` 可辅助判断 remote 是否只改善局部深度或是否破坏相机/射线估计。
6. 可视化只作为最后确认。若 PLY 中包含 remote 点，remote 正射投影误差会放大视觉混乱，因此先看 ordinary-only 和 controls 指标。

## 2026-05-27 追加实验：p5g-fixed 与 p5h

上一轮 p5g 结果暴露出两个问题：

1. `vggt_p5g_late_fusion` 继承了 `vggt_finetune`，导致 `model.aggregator.patch_embed` 仍有非零学习率。p5g 的原始设计是冻结 VGGT trunk，只训练 late adapter/remote head，因此这一点已修正为继承 `default`，并通过 `model: lr=0` 冻结原始 VGGT。
2. split aggregator 只能避免 remote 进入早期 aggregator，但原来的 camera/depth head 仍接收 combined tokens。这样 no-fusion 也可能被 remote pose token 影响。新增 `model.model_config.protect_ordinary_heads_from_remote=true` 后，普通视角 camera/depth head 只看 ordinary tokens；remote head 单独跑 remote tokens。late fusion 只把 remote 信息写入 ordinary patch tokens，不再让 remote token 直接进入 ordinary heads。

### p5g_fixedfreeze_protected

目的：验证修正 freeze 和 protected head 后，no-fusion 是否成为干净 sanity check。

训练：

```bash
NUM_GPUS=5 CUDA_DEVICES=0,1,2,3,4 BATCH_SIZE=8 EPOCHS=10 \
  bash bash_scripts/train/Crossview/vggt/p5g_vggt_no_fusion_fixedfreeze_protected.sh
```

评测：

```bash
REMOTE_OVERFIT_NUM_SETS=5 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5g_no_fusion_fixedfreeze_protected_unified.sh
```

预期：`same_gain__pointmaps_abs_rel` 应接近 0，same/blank/shuffled 也应接近；如果仍有明显差异，说明评测或 wrapper 仍有 remote 泄漏。

### p5h_p5e_base_protected

目的：不再从原始 VGGT 重新学 ordinary 能力，而是以当前最强的 p5e checkpoint 为 frozen base，只训练 late remote-to-aerial adapter。这样可以直接回答：remote adapter 能不能在 p5e 基础上带来额外 ordinary gain。

训练入口：

```bash
NUM_GPUS=5 CUDA_DEVICES=0,1,2,3,4 BATCH_SIZE=8 \
  bash bash_scripts/train/Crossview/vggt/p5h_vggt_p5e_base_film_protected.sh

NUM_GPUS=5 CUDA_DEVICES=0,1,2,3,4 BATCH_SIZE=8 \
  bash bash_scripts/train/Crossview/vggt/p5h_vggt_p5e_base_crossattn_protected.sh
```

默认设置：

- `BASE_CKPT` 指向 `p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth`。
- `model.pretrained=${BASE_CKPT}` 加载完整 wrapper checkpoint。
- `load_pretrained_weights=false`、`load_custom_ckpt=false`，避免再加载原始 VGGT。
- `remote_point_head`、`model.*`、view type embeddings 全部冻结。
- 只训练 `remote_to_aerial_late_*` adapter 和 gate。
- `SAVE_FREQ=0`、`KEEP_FREQ=0`，只保留 best/final，避免保存大量中间权重。

评测入口：

```bash
REMOTE_OVERFIT_NUM_SETS=5 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5h_no_fusion_unified.sh

REMOTE_OVERFIT_NUM_SETS=5 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5h_film_unified.sh

REMOTE_OVERFIT_NUM_SETS=5 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5h_crossattn_unified.sh
```

其中 `p5h_no_fusion` 是 benchmark-only sanity check，直接用 p5e checkpoint，不需要训练。

### 本轮成功标准

p5h 的判读要比 p5g 更严格：

1. `p5h_no_fusion` 应接近 p5e，确认 protected wrapper 没破坏 p5e base。
2. `p5h_crossattn` 或 `p5h_film` 的 `same_gain__pointmaps_abs_rel` 要为正。
3. `specific_gain_blank__pointmaps_abs_rel` 和 `specific_gain_shuffled__pointmaps_abs_rel` 都要为正，尤其 same 必须优于 shuffled。
4. `ordinary_damage_vs_reference__pointmaps_abs_rel` 以 p5e 为 reference 时应接近 0 或为负。
5. 若 p5h 对 p5e 没有提升，但 p5g cross-attn 仍有提升，说明 late adapter 主要在补 raw VGGT 的不足，而不是提供真正的 p5e 级 remote 增强。下一步应转向 ranking/margin loss 或 geometry-aware remote token。

## 后续可选方案

如果 p5g_crossattn 仍然 same≈blank≈shuffled：

1. 加 margin/ranking loss：直接优化 `same` 优于 `blank/shuffled`，让训练目标对 scene-specific remote 增益更敏感。
2. 加 teacher preservation：用原始 VGGT 对普通视角的输出作为 teacher，约束 ordinary path 不退化。
3. 限制 fusion 作用位置：只作用中后层 tokens 或只作用深度/point head 前的 patch tokens，避免影响 camera head。
4. 加 geometry-aware remote prior：把 remote 正射图先转换为粗 BEV/height/context token，减少正射影像与透视影像的投影假设冲突。

当前首轮建议顺序：先跑 `p5g_no_fusion_split_remote` 验证普通路径保护，再跑 `p5g_film_split_remote` 和 `p5g_crossattn_split_remote`，最后用 summary 脚本和可视化一起筛选下一步。

## 2026-05-27 运行记录：p5h protected 后续

当前已完成：

- `p5h_vggt_p5e_base_no_fusion_protected_mini_controls`：same/blank/shuffled 与 aerial-only 完全一致，说明 `protect_ordinary_heads_from_remote=true` 生效，remote 不再直接污染普通视角 head。
- `p5h_vggt_p5e_base_film_protected`：40 epoch 训练完成；best/final 均已跑 25-scene mini controls。

关键观察：

- p5h-film best：`same_gain(pointmaps_abs_rel)=0.0012188`，但 `specific_gain_blank=-0.0000707`、`specific_gain_shuffled=-0.0000135`。
- p5h-film final：same 也有表观提升，但 blank 比 same 更好。
- 因此 p5h-film 的提升目前不能解释为“正确 remote 内容带来的互补信息”，更像 late adapter 或 remote 输入存在非特异性偏置。

继续运行中的实验：

- `p5h_vggt_p5e_base_crossattn_protected`：4 GPU，`BATCH_SIZE=8`，仅训练 cross-attn late adapter，保留 protected heads。
- `p5h_vggt_p5e_base_film_unfreeze_viewtype_protected`：1 GPU，`BATCH_SIZE=8`，20 epoch，在 p5h-film 基础上额外解冻 `aerial_view_type_embedding` 和 `remote_view_type_embedding`，用于测试“类型偏置是否需要随 late adapter 共同调整”。

工程记录：

- `p5h_vggt_p5e_base_split_late_fusion.sh` 增加 `MASTER_PORT` 环境变量，避免多个 torchrun 并行时抢占默认 `29500` 端口。
- 当前并行资源分配：GPU0 跑 view-type 解冻对照，GPU1-4 跑 cross-attn；实测显存约 GPU0 38.6GB，GPU1-4 42.7GB。

下一步判据：

- 如果 cross-attn 或 view-type 解冻仍表现为 `same_gain > 0` 但 `same` 不优于 blank/shuffled，则说明普通 aerial loss 不能强迫模型利用正确 remote 内容。下一轮应转向显式控制组训练目标，例如 same-vs-blank/shuffled 的 ranking/contrastive loss。
- 如果任一实验出现 `specific_gain_blank > 0` 且 `specific_gain_shuffled > 0`，再扩大 benchmark 场景数并导出可视化点云。

## 2026-05-27 运行结果：p5h view-type 与 cross-attn

本轮完成：

- `p5h_vggt_p5e_base_film_unfreeze_viewtype_protected`：20 epoch，GPU0，`BATCH_SIZE=8`，在 p5h-film 基础上解冻 aerial/remote view type embedding。
- `p5h_vggt_p5e_base_crossattn_protected`：40 epoch，GPU1-4，`BATCH_SIZE=8`，只训练 cross-attn late adapter。
- 两个实验均使用 `checkpoint-best.pth` 跑 25-scene mini-control benchmark。

关键指标：

| 实验 | aerial-only | same | same gain | blank-specific | shuffled-specific | pass same>aerial | pass same>blank | pass same>shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p5h-film best | 0.059353 | 0.058135 | 0.001219 | -0.000071 | -0.000014 | 0.60 | 0.48 | 0.60 |
| p5h-film final | 0.059353 | 0.056938 | 0.002415 | -0.000603 | -0.000053 | 0.64 | 0.36 | 0.48 |
| p5h-film + view type unfreeze | 0.058719 | 0.056309 | 0.002410 | -0.000205 | 0.000083 | 0.64 | 0.40 | 0.48 |
| p5h-crossattn | 0.059353 | 0.056194 | 0.003159 | 0.000752 | 0.000287 | 0.76 | 0.56 | 0.56 |

结论：

- `p5h-crossattn` 是目前唯一同时满足 `same_gain > 0`、`same > blank`、`same > shuffled` 的方案，说明 cross-attn late fusion 比 film 更可能利用到 scene-specific remote 内容。
- 但 margin 仍然小，`same>blank` 和 `same>shuffled` 的 pass rate 只有 0.56，不能视为稳定成功。
- `film + view type unfreeze` 的 same gain 变大，但 blank 仍优于 same，说明“更早/可训练的 view type bias”本身不足以让模型学会正确 remote 匹配。
- no-fusion protected 对照已经确认 ordinary head 保护有效，因此当前瓶颈主要不是 remote 污染普通 head，而是训练目标没有强约束模型使用正确 remote 内容。

下一轮实验建议：

1. 以 `p5h-crossattn` 为主线，增加显式 control ranking loss：同一 batch 内构造 same/blank/shuffled remote，优化 `loss_same + margin(max(0, err_same - err_blank + m), max(0, err_same - err_shuffled + m))`。
2. 保持 `protect_ordinary_heads_from_remote=true`，继续只训练 late adapter/gate，避免再次破坏 p5e base。
3. 先做 10-20 epoch 小规模验证；若 `specific_gain_blank`、`specific_gain_shuffled` 和 pass rate 同时上升，再扩大 benchmark 场景数和可视化。
4. 暂停继续扩展 film/view-type 方向，除非 ranking loss 后需要更轻量的对照结构。

## 2026-05-27 运行结果：p5h cross-attn + remote-control ranking

本轮新增了 `remote_control_ranking_loss`，目标是显式约束真实 remote 输入优于 `blank/shuffled remote` 控制输入。实现上先尝试对 same/blank/shuffled 都保留梯度图，但 BS6/BS4 均 OOM；随后改为复用主 forward 的 same loss，blank/shuffled 控制分支使用 `no_grad` 只提供对照阈值，BS6 可稳定运行，显存约 33.7GB/GPU。

训练脚本：

- `bash_scripts/train/Crossview/vggt/p5h_vggt_p5e_base_crossattn_ranking_protected.sh`
- 实验目录：`outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5h_vggt_p5e_base_crossattn_ranking_w02_m001_nograd_bs6_protected`
- best 日志：`epoch=7`，验证 `40 @ VigorChicagoJointRSAerial_loss_avg=0.3258406222`
- 提前停止原因：epoch 8 验证回升到约 `0.32998`，继续到 20 epoch 的收益不明显。

mini controls benchmark：

| experiment | aerial_only | joint_same | same_gain | specific_gain_blank | specific_gain_shuffled | pass same>aerial | pass same>blank | pass same>shuffled | joint_global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p5h cross-attn protected | 0.059353 | 0.056194 | 0.003159 | 0.000752 | 0.000287 | 0.76 | 0.56 | 0.56 | 0.146032 |
| p5h cross-attn ranking w0.2 m0.01 no-grad | 0.059353 | 0.057052 | 0.002301 | 0.000384 | -0.000101 | 0.68 | 0.56 | 0.44 | 0.146054 |

结论：ranking 版没有改进，反而弱于原始 p5h cross-attn protected。训练日志里 blank 控制有时能被拉开，但 shuffled 控制长期贴近 same；benchmark 中 shuffled 甚至略优于 same，`specific_gain_shuffled < 0`。这说明当前 ranking 约束主要让模型对“是否有 remote 图像”有反应，但没有让模型学到稳定的“remote 与当前地面场景是否匹配”。因此，继续调这个轻量 ranking loss 的权重优先级不高。

下一步优先级调整：

1. 保留 p5h cross-attn protected 作为当前最好结构基线，不采用 ranking w0.2/m0.01 作为默认。
2. 如果继续做 ranking，应改成更强的错配负样本：同 batch hard negative、同城市近邻负样本、相似外观但不同位置负样本，而不是简单 shuffled。
3. 更值得做的是让 remote 进入方式带有几何选择性，例如基于 ground token query 的局部 remote token cross-attn、top-k remote token selection、或显式位置/方位编码；否则模型容易只学到 remote 的全局先验。
4. 评估上继续以 `same_gain`、`specific_gain_blank`、`specific_gain_shuffled`、`pass same>shuffled` 为主要判据；若 `specific_gain_shuffled <= 0`，即使 same 比 aerial-only 好，也不能认为 remote 真正提供了场景级互补信息。

## 2026-05-28 修正：P6A Conditional Remote Adapter

上一轮对 p5h cross-attn 和 ranking ablation 的结果说明，`same > blank/shuffled` 更适合作为 remote 是否有效的验证协议，而不应该直接作为主训练出发点。原因是如果把它强行写进主要 loss，模型可能通过让 blank/shuffled 变差、识别空图或错配分布、利用额外 token 的模式偏置来满足目标，而不一定学到真实的场景互补信息。

因此 P6A 的目标改为：训练一个受限的 conditional remote adapter，使模型默认接近 p5e/VGGT 普通视角能力；只有当 remote 信息能被 late adapter 转化为有用修正时，才对 ordinary tokens 产生小幅影响。`same/blank/shuffled` 仍然用于评估，而不是默认训练目标。

### P6A 要解决的问题

P5 系列暴露出的核心问题不是 remote loss 不下降，而是 remote 没有稳定成为 ordinary reconstruction 的有效条件信息：

1. p5f early mixing 会明显破坏普通视角重建，说明 remote 过早进入 VGGT 全局表征风险很大。
2. p5h protected late fusion 能减少普通 head 污染，但 same 相比 blank/shuffled 的优势很弱，说明模型可能只学到了 joint 模式偏置。
3. 直接 ranking ablation 没有改善 shuffled specificity，说明简单把评估关系写入 loss 不足以解决 remote 匹配问题。

P6A 因此不再追求 remote 分支自身重建得更准，也不默认追求 same 在训练中压过 blank/shuffled，而是先建立一个更稳的条件注入机制。

### P6A 结构

P6A 继续沿用 p5h 的 protected late cross-attn 结构：

```text
ordinary images -> frozen p5e/VGGT ordinary path -> ordinary tokens
remote image    -> split remote aggregator/head       -> remote tokens

ordinary patch tokens += gate * cross_attn(ordinary patch tokens, remote patch tokens)
```

关键约束：

- ordinary 和 remote 使用 split aggregator，remote 不进入早期 ordinary self-attention。
- `protect_ordinary_heads_from_remote=true`，ordinary camera/depth/point head 不直接看到 remote tokens。
- 只训练 late adapter/gate，原始 VGGT trunk 和普通输出路径冻结。
- `LATE_GATE_INIT=1e-3`，初始行为仍接近 base，但避免精确 0 gate 让 cross-attn 分支初始没有梯度。
- remote pointmap/height loss 默认仍为 0，避免再次把目标变成“让正射图像按透视 view 重建”。

### P6A 训练目标

P6A 默认目标是：

```text
L_total = L_ordinary_same
        + lambda_preserve * L_blank_preserve
        + lambda_gate * L_gate
        + lambda_delta * L_weighted_delta
```

各项含义：

- `L_ordinary_same`：使用正确 remote 输入时，对普通视角输出计算原有 aerial reconstruction loss。这是主任务，仍然要求 joint forward 的普通视角输出变好。
- `L_blank_preserve`：将 remote 替换成 blank 后，再计算普通视角 aerial loss。它不是为了让 blank 变强，而是约束模型在 remote 无信息时不能明显偏离可用 ordinary path。
- `L_gate`：对 late fusion gate 做 L1/L2 正则，防止 remote adapter 无条件变成强扰动。
- `L_weighted_delta`：对 `gate * remote_delta` 的幅度做正则，约束 remote 修正量小而可控，避免用 remote delta 覆盖 ordinary token。

默认参数：

```yaml
remote_blank_preserve_loss_weight: 0.05
remote_late_gate_l1_weight: 1e-03
remote_late_gate_l2_weight: 0.0
remote_late_weighted_delta_l2_weight: 1e-04
remote_control_ranking_loss_weight: 0.0
```

这里最重要的是最后一项：P6A 默认关闭 ranking。`same > blank/shuffled` 只作为 benchmark 结果来判断 remote 是否自然有效。

### 为什么 blank-preserve 合理

`blank-preserve` 和 `same > blank/shuffled` 的性质不同。它不是要求 same 赢 blank，而是给模型一个退化路径：当 remote 没有可用信息时，joint 架构应该尽量保持 ordinary reconstruction，而不是因为多了一个 remote slot 就改变普通视角输出。

因此它约束的是鲁棒性和普通路径保护：

```text
blank remote -> 不应破坏 ordinary path
same remote  -> 如果有用，可以在受限 gate 下提供增益
shuffled remote -> 不参与默认训练，只在评估中检查是否被错误使用
```

这比直接 ranking 更不容易泄漏评估目标，也更符合最终需求：remote 是可忽略的辅助条件，而不是必须被使用的输入。

### Ranking 的位置

P6A 仍保留 weak ranking 脚本，但只作为诊断 ablation：

```bash
bash bash_scripts/train/Crossview/vggt/p6a_vggt_conditional_remote_adapter_weak_ranking.sh
```

解释规则：

- 如果 P6A 无 ranking 已经 `same > blank/shuffled`，说明结构和普通 reconstruction 目标足以自然利用 remote。
- 如果只有 weak ranking 有效，要谨慎，因为可能是训练目标贴合了评估协议，而不是 remote 真正变成稳定几何条件。
- 如果 weak ranking 仍无效，说明问题不在 loss 权重，而在 remote token 表达、错配负样本质量或正射几何建模。

### 已实现代码

新增配置：

```text
configs/train_params/vggt_p6_conditional_remote_adapter.yaml
```

新增训练脚本：

```text
bash_scripts/train/Crossview/vggt/p6a_vggt_conditional_remote_adapter.sh
bash_scripts/train/Crossview/vggt/p6a_vggt_raw_base_conditional_remote_adapter.sh
bash_scripts/train/Crossview/vggt/p6a_vggt_p5e_base_conditional_remote_adapter.sh
bash_scripts/train/Crossview/vggt/p6a_vggt_conditional_remote_adapter_weak_ranking.sh
```

其中 `p6a_vggt_conditional_remote_adapter.sh` 和 `p6a_vggt_raw_base_conditional_remote_adapter.sh` 默认使用本地 VGGT 官方权重：`/root/autodl-tmp/outputs/checkpoints/vggt/model.pt`。加载方式是 `BASE_CKPT=null`、`LOAD_PRETRAINED_WEIGHTS=false`、`LOAD_CUSTOM_CKPT=true`、`CUSTOM_CKPT_PATH=/root/autodl-tmp/outputs/checkpoints/vggt/model.pt`，即通过 `VGGTWrapper.model.load_state_dict(...)` 加载内层 VGGT 权重。注意不要使用 `/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth` 作为 VGGT raw base；它是 MapAnything checkpoint，key 为 `encoder.model.*`，不能正确加载到 `VGGTWrapper.model.aggregator.*`。

`p6a_vggt_p5e_base_conditional_remote_adapter.sh` 只作为历史对照，使用：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth
```

训练代码新增日志项：

```text
remote_blank_preserve_aerial_loss
remote_blank_preserve_loss_weighted
remote_to_aerial_late_gate_l1
remote_to_aerial_late_gate_l1_weighted
late_gate_abs
late_delta_l2
late_weighted_delta_l2
late_weighted_delta_l2_weighted
```

这些日志的判读方式：

- `remote_blank_preserve_aerial_loss` 应稳定，不应随训练明显恶化。
- `remote_to_aerial_late_gate_l1` / `late_gate_abs` 不应快速变成很大，否则说明 remote adapter 变成强扰动。
- `late_weighted_delta_l2` 应保持小量级；如果很大且可视化变差，说明 remote delta 覆盖了 ordinary token。
- 这些正则项本身下降不代表成功，成功仍要看 benchmark controls 和可视化。

### 推荐运行

主实验：VGGT 官方 raw base。这个是 P6A 默认主线，用来验证 remote adapter 能否在较干净的普通视角能力上产生增益。

```bash
NUM_GPUS=4 CUDA_DEVICES=1,2,3,4 BATCH_SIZE=8 EPOCHS=40 \
  bash bash_scripts/train/Crossview/vggt/p6a_vggt_raw_base_conditional_remote_adapter.sh
```

更保守的 raw-base gate 版本：

```bash
NUM_GPUS=4 CUDA_DEVICES=1,2,3,4 BATCH_SIZE=8 EPOCHS=40 \
PRESERVE_WEIGHT=0.1 GATE_L1_WEIGHT=3e-03 WEIGHTED_DELTA_L2_WEIGHT=3e-04 \
  bash bash_scripts/train/Crossview/vggt/p6a_vggt_raw_base_conditional_remote_adapter.sh
```

p5e-base 历史对照：只用于判断 p5e 预训练痕迹是否影响结论，不作为默认主实验。

```bash
NUM_GPUS=4 CUDA_DEVICES=1,2,3,4 BATCH_SIZE=8 EPOCHS=40 \
  bash bash_scripts/train/Crossview/vggt/p6a_vggt_p5e_base_conditional_remote_adapter.sh
```

弱 ranking 诊断：

```bash
NUM_GPUS=4 CUDA_DEVICES=1,2,3,4 BATCH_SIZE=6 EPOCHS=20 \
  bash bash_scripts/train/Crossview/vggt/p6a_vggt_conditional_remote_adapter_weak_ranking.sh
```

### Base 选择原则

P6A 不应默认依赖 p5e。p5e 的价值只是历史对照，因为它本身已经经过 remote 相关训练，普通视角能力和 remote bias 都可能被改变。若以 p5e 为唯一 base，后续结果很难区分是 P6A adapter 有效，还是 p5e 预训练痕迹在起作用。

因此 base 优先级为：

1. `VGGT official raw base`：主实验。结论最干净，能直接判断 conditional remote adapter 是否在强 ordinary baseline 上有效。
2. `p5e base`：对照实验。只回答“在已有 remote finetune 痕迹上，P6A 是否还能补一点”。

判读时也要分开：

```text
P6A raw-base 成功：最有价值，说明结构本身有希望。
P6A p5e-base 成功但 raw-base 失败：结论不干净，可能依赖 p5e bias。
两者都 same≈blank≈shuffled：问题更可能在 remote 表达/几何建模。
raw-base ordinary-only 明显差于 raw VGGT：保护机制仍不足，不能继续加 remote。
```

### P6A 成功标准

训练日志只用于确认实验没有失控，不能作为最终结论。最终仍看 mini controls：

1. `ordinary_damage_vs_reference__pointmaps_abs_rel` 接近 0 或为负。
2. `same_gain__pointmaps_abs_rel > 0`。
3. `specific_gain_blank__pointmaps_abs_rel > 0`。
4. `specific_gain_shuffled__pointmaps_abs_rel > 0`，且优先级高于 blank。
5. `pass_rate_same_better_than_shuffled__pointmaps_abs_rel > 0.6` 才能认为有初步 scene-specific remote 使用。
6. 可视化中 ordinary-only 和 joint-same 都不能比 p5e/raw VGGT 明显崩坏。

如果 P6A 仍然表现为 `same≈blank≈shuffled`，下一步不应继续调 remote loss 权重，而应转向 remote 表达方式：例如 geometry-aware remote prior、位置/方位编码、hard negative 采样，或把正射图像编码为 BEV/footprint/height prior，而不是普通 VGGT view。



## P6B：Joint Frame Remote Alignment（2026-05-30 更新）

P6A 废弃。P6A 的 split/protected 结构适合验证 remote 是否能作为条件输入影响 ordinary branch，但它不保证 remote branch 点云和普通视角点云处在同一坐标系，因此不满足当前目标。

P6B 的目标改为：remote view 的 `pts3d` 必须直接表示 ordinary view0 坐标系下的场景点，并且 remote 点云应能和普通视角点云一起导出到同一个 PLY 中可视化对齐。

硬约束：

- 不使用 `use_split_remote_aggregator`。
- 不使用 `protect_ordinary_heads_from_remote`。
- remote view 使用 point output head。
- remote point loss 显式对齐 `remote_pointmap_view0`。
- `remote_pointmap_loss_weight` 必须大于 0。

首轮实验：

- `p6b_vggt_joint_remote_alignment_shared_head_w03`：remote 使用共享 point head，验证 joint frame 最小可行性。
- `p6b_vggt_joint_remote_alignment_private_head_w03`：remote 使用 private point head，避免 remote 成像差异直接污染共享 point head。
- `p6b_vggt_joint_remote_alignment_private_head_viewtype_w03`：private head 加 post-aggregator view type bias，给模型一个轻量模态提示。

默认主实验为 private head，`remote_pointmap_loss_weight=0.3`。后续根据 remote PLY 对齐情况扫 `0.1 / 0.3 / 1.0`。

成功判据：

- `--include_remote_points` 导出的 remote 点云和 ordinary 点云大体对齐。
- benchmark 中 `rs_pointmap_loss` / `rs_pointmap_loss_raw_metric` 相对 P5F/P6A 明显改善。
- same remote 优于 blank/shuffled remote，但这是评估结果，不直接作为训练损失。
- ordinary branch 不明显劣于 raw VGGT/P5B。

运行入口：

```bash
NUM_GPUS=2 CUDA_DEVICES=0,1 BATCH_SIZE=8 EPOCHS=50 \
LAMBDA_REMOTE_PM=0.3 \
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head.sh
```

mini benchmark：

```bash
CUDA_DEVICE=0 REMOTE_OVERFIT_NUM_SETS=10 \
EXPERIMENT_NAME=p6b_vggt_joint_remote_alignment_private_head_w03 \
bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p6b_mini_controls.sh
```


### P6B 训练实现备注

2026-05-30 首次启动时 `BATCH_SIZE=10` 在 2x3090 上 OOM。P6B 同时运行 ordinary depth、shared point consistency 和 remote private point head，显存高于 P6A，因此默认使用 `BATCH_SIZE=8`。`model.track_head` 已显式冻结，避免无关参数进入优化器。

## 2026-05-30 规划更新：丢弃 P6A，转向 P6B remote branch alignment

P6A 已丢弃。原因不是简单实现问题，而是目标错位：P6A 把 remote 分支拆成 private frame 后，导出的 remote 点云与普通视角点云存在整体角度/坐标错乱；而当前真正关心的是 `remote branch point cloud alignment`，即 remote 视图预测出的点云要直接落在 ordinary/view0 对齐空间中。继续优化 P6A 的普通视角 conditioning 或 remote 点云可视化没有意义。

P6B 的目标更窄：保留 VGGT joint forward 的共享坐标系，让 remote view 输出 `pts3d`，并用数据里的 `remote_pointmap_view0` 监督它。这个实验不追求 remote 自身绝对 metric 准，而是先验证 remote branch 是否能学到和 ordinary/view0 一致的点云空间。如果这一点失败，后续任何 remote-to-ordinary 增强都缺少几何基础。

### P6B 当前实现

新增训练入口：

```bash
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head.sh
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_shared_head.sh
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head_viewtype.sh
```

当前主实验使用 private remote point head：

```bash
NUM_GPUS=2 CUDA_DEVICES=0,1 BATCH_SIZE=5 NUM_VIEWS=4 EPOCHS=50 \
LAMBDA_REMOTE_PM=0.3 SAVE_FREQ=0 KEEP_FREQ=0 PRINT_FREQ=20 \
EXPERIMENT_NAME=p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly \
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head.sh
```

关键配置：

- official VGGT 权重：`/root/autodl-tmp/outputs/checkpoints/vggt/model.pt`。
- `use_split_remote_aggregator=false`，避免再进入 P6A 式 private frame。
- `ordinary_output_head=depth`，普通视角仍走 depth/camera 路径。
- `remote_output_head=point`，remote 只训练 point 输出。
- `use_remote_private_point_head=true`，普通 `model.point_head` 冻结，remote 用独立 point head。
- `output_point_head_for_consistency=false`，没有 branch consistency 时不再额外跑 shared point head。
- `remote_compare_in_view0_frame=false` 且 `remote_compare_gt_in_view0_frame_only=true`，直接使用 view0-frame 的 remote GT。
- same/blank/shuffled 仍只用于评估，不写入训练 loss。

### 工程修正

P6B private head 的显存瓶颈主要来自 DPT head。已做三处修正：

1. remote private point head 只在 remote views 上运行，然后 scatter 回输出列表；不再对所有 ordinary+remote views 运行 remote head。
2. DPT head 训练时使用 activation checkpoint，降低 head backward 显存。
3. DDP 增加 `train_params.ddp_static_graph` 和 `train_params.ddp_find_unused_parameters`。P6B private wrapper 默认 `static_graph=true`、`find_unused_parameters=false`，解决 reentrant checkpoint 下的 `mark variable ready twice` 问题。

显存边界：

- `BATCH_SIZE=8`：backward OOM，约 47.4GB 被打满。
- `BATCH_SIZE=6`：backward OOM，仍约 47.3GB。
- `BATCH_SIZE=5`：已稳定进入训练，`max mem` 约 46.8GB，两张 RTX 3090 48GB 基本吃满。
- `BATCH_SIZE=4`：可稳定，但显存利用率略低；仅作为 fallback。

### 当前运行状态

当前正在运行：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly
```

第 0 个 epoch 已完成并保存 `checkpoint-best.pth`。早期训练表现正常：

- train loss 从首步约 `24.57` 下降到 epoch0 末约 `0.92`。
- train `remote_loss` 从约 `0.0657` 下降到约 `0.0099`。
- val loss epoch1 约 `1.5244`，val `remote_loss` 约 `0.0146`。
- `rs_pointmap_loss_raw_metric` 暂时没有明显下降趋势，这符合归一化训练设定下的预期；它只作为诊断，不作为 checkpoint 主目标。

### P6B 评估计划

训练完成后先跑 mini controls：

```bash
CUDA_DEVICE=0 REMOTE_OVERFIT_NUM_SETS=10 \
EXPERIMENT_NAME=p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly \
OUTPUT_DIR=/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly_mini_controls \
bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p6b_mini_controls.sh
```

再汇总：

```bash
python scripts/summarize_rs_guided_benchmark.py \
  /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly_mini_controls \
  --output_dir /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly_mini_controls
```

主要看两类结果：

1. remote branch alignment：remote point/height 指标是否比 raw VGGT 和 P5 系列更合理，PLY 中 remote 点云是否能和 ordinary 点云落到同一 view0 空间。
2. ordinary path damage：`aerial_only` 和 `joint_same` 的普通视角指标是否明显差于 raw VGGT/P5B。如果 P6B 只改善 remote branch 但伤普通路径，下一步需要 teacher preservation 或冻结更多 ordinary trunk。

如果 P6B private head 有效，下一步做两个小对照：

- `private_head_viewtype`：打开 view type bias，判断类型偏置是否能让 remote head 更稳定。
- `shared_head`：不用 private remote head，测试共享 point head 是否足以对齐 remote；若效果差，说明 remote 的正射投影差异确实需要专门 head。

### P6B 首轮训练与 mini benchmark 结果

实际执行：

```bash
NUM_GPUS=2 CUDA_DEVICES=0,1 BATCH_SIZE=5 NUM_VIEWS=4 EPOCHS=50 \
LAMBDA_REMOTE_PM=0.3 SAVE_FREQ=0 KEEP_FREQ=0 PRINT_FREQ=20 \
EXPERIMENT_NAME=p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly \
bash bash_scripts/train/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head.sh
```

训练跑到 epoch10 后手动停止，原因是第 10 个验证点没有超过当前 best，且已有足够信息进入 mini benchmark。`checkpoint-best.pth` 保留在：

```text
outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly/checkpoint-best.pth
```

best checkpoint 的保存时间为 2026-05-30 02:44，对应早期验证最优 median。训练过程中 `BATCH_SIZE=5` 稳定运行，`max mem` 约 46835 MiB，基本吃满 48GB GPU；`BATCH_SIZE=6/8` 都在 backward OOM。

10-scene mini controls 输出目录：

```text
outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly_mini_controls
```

summary 关键结果：

| metric | value |
| --- | ---: |
| aerial_only pointmaps_abs_rel | 0.0674151 |
| joint_same pointmaps_abs_rel | 0.0705461 |
| same_gain pointmaps_abs_rel | -0.00313103 |
| specific_gain_blank pointmaps_abs_rel | +0.00371139 |
| specific_gain_shuffled pointmaps_abs_rel | +0.00116381 |
| pass_rate same better than aerial | 0.20 |
| pass_rate same better than blank | 0.80 |
| pass_rate same better than shuffled | 0.60 |
| same_gain z_depth_abs_rel | -0.00137855 |
| same_gain ray_dirs_err_deg | +0.0761856 |
| remote_damage rs_height_mae_affine | -1.42882 |
| joint_global_pointmaps_abs_rel | 0.0669546 |

判读：

1. P6B private head 确实让 remote branch 更有用：`remote_damage__rs_height_mae_affine=-1.42882`，joint remote height 比 rs-only 更好。
2. same remote 对 ordinary path 仍不是正向增强：`same_gain__pointmaps_abs_rel=-0.00313`，`pass_rate_same_better_than_aerial=0.20`。
3. 但 same 相比 blank/shuffled 有特异性优势：`specific_gain_blank>0`、`specific_gain_shuffled>0`，pass rate 分别为 0.80/0.60。这说明模型不是完全忽略 remote 内容，而是 remote 内容带来的信息还不足以抵消 joint forward 对 ordinary path 的扰动。
4. 当前 P6B 比 P6A 更合理，因为它没有追求 split private frame 下的无意义 remote PLY 对齐，而是在 joint frame 中学习 remote 点云；但它还不是最终方案。

下一步优先级：

1. 跑 `private_head_viewtype` 小对照。目标是让模型更早区分 remote/ordinary token，降低正射图像对 ordinary token 的表示污染，同时保留 P6B 的 remote branch alignment。
2. 跑 `shared_head` 小对照。若 shared head 的 remote alignment 明显差于 private head，说明正射 remote 需要专门 head；若接近，则 private head 不是必要复杂度。
3. 给 P6B 加 ordinary teacher preservation：用 raw VGGT 或冻结 ordinary-only 输出约束 ordinary path，不让 joint same 为了 remote alignment 损伤 aerial-only。这个比直接把 same/blank/shuffled 写进训练 loss 更合理，因为 same 优于 controls 应该是评估结果，而不是硬编码出发点。
4. 若 viewtype + preservation 仍不能让 `same_gain>0`，再考虑更强的 late/gated fusion，而不是回到 P6A split frame。

