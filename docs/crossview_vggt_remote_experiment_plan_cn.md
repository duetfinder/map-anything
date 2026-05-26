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

## 后续可选方案

如果 p5g_crossattn 仍然 same≈blank≈shuffled：

1. 加 margin/ranking loss：直接优化 `same` 优于 `blank/shuffled`，让训练目标对 scene-specific remote 增益更敏感。
2. 加 teacher preservation：用原始 VGGT 对普通视角的输出作为 teacher，约束 ordinary path 不退化。
3. 限制 fusion 作用位置：只作用中后层 tokens 或只作用深度/point head 前的 patch tokens，避免影响 camera head。
4. 加 geometry-aware remote prior：把 remote 正射图先转换为粗 BEV/height/context token，减少正射影像与透视影像的投影假设冲突。

当前首轮建议顺序：先跑 `p5g_no_fusion_split_remote` 验证普通路径保护，再跑 `p5g_film_split_remote` 和 `p5g_crossattn_split_remote`，最后用 summary 脚本和可视化一起筛选下一步。
