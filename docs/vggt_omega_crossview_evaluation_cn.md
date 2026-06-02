# VGGT-Omega CrossView 训练与评测记录

本文档记录 `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega` 下三个 VGGT-Omega CrossView 训练实验，以及对应的 `rs_guided_dense_mv` New York mini controls benchmark 结果。

## 评测环境

- 项目目录：`/root/autodl-tmp/Models/map-anything`
- 原始 VGGT-Omega 权重：`/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt`
- Benchmark 脚本：`bash_scripts/benchmark/rs_guided_dense_mv/run_crossview_finetuned_unified.sh`
- Benchmark 数据：`newyork` mini controls
- Benchmark 设置：
  - `REMOTE_OVERFIT_NUM_SETS=10`
  - `REMOTE_CONTROL_MODES=[same,blank,shuffled]`
  - `BATCH_SIZE=1`
  - `resolution=512x512`
  - 默认 `NUM_VIEWS=4`，除 view sweep 外

`benchmarking/rs_guided_dense_mv/benchmark_unified.py` 已补充 `vggt_omega` 支持，并支持 raw VGGT-Omega checkpoint 作为 baseline 加载。

## 三个训练实验

| 实验目录 | num_views | remote_views | 训练城市 | max_num_of_imgs_per_gpu | 训练时间 | best val loss |
|---|---:|---:|---|---:|---:|---:|
| `p1_vggt_omega_joint_depth_512` | 4 | 1 | chicago | 12 | 0:48:36 | `0.3376 @ epoch 43` |
| `p1_vggt_omega_joint_depth_512_1gpu_2v` | 2 | 1 | chicago | 2 | 6:07:58 | `0.3564 @ epoch 50` |
| `p1_vggt_omega_joint_depth_512_all` | 4 | 2 | all cities | 8 | 1:46:06 | `0.6159 @ epoch 47` |

说明：`all` 实验使用全部城市，并且 remote views 从 1 变成 2，训练任务更难，因此训练日志里的 validation loss 不能和 chicago-only 实验直接等价比较。最终以同配置 benchmark 的泛化结果为主。

## New York Mini Benchmark 对比

评测命令模板：

```bash
cd /root/autodl-tmp/Models/map-anything

NUM_VIEWS=4 \
BATCH_SIZE=1 \
REMOTE_OVERFIT_NUM_SETS=10 \
REMOTE_CONTROL_MODES='[same,blank,shuffled]' \
MODEL_NAME=vggt_omega \
CKPT_PATH=<checkpoint-path> \
OUTPUT_DIR=<output-dir> \
CUDA_DEVICE=0 \
bash bash_scripts/benchmark/rs_guided_dense_mv/run_crossview_finetuned_unified.sh \
  'dataset.resolution_train=${dataset.resolution_options.512_1_00_ar}' \
  'dataset.resolution_val=${dataset.resolution_options.512_1_00_ar}'
```

输出目录：

| 名称 | 输出目录 |
|---|---|
| raw | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_raw_1b_512_mini_controls` |
| p1 4v chicago | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_mini_controls` |
| p1 2v chicago | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_1gpu_2v_mini_controls` |
| p1 4v all | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_all_mini_controls` |

核心指标：

| 模型 | aerial point ↓ | aerial AUC5 ↑ | joint point ↓ | joint global ↓ | joint AUC5 ↑ | joint ray ↓ | RS-only MAE ↓ | joint RS MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.1008 | 52.33 | 0.0881 | 0.0842 | 69.33 | 1.1254 | 17.94 | 20.02 |
| `p1_4v_chicago` | 0.0580 | 91.00 | 0.0574 | 0.0545 | 92.33 | 0.2833 | 16.35 | 16.32 |
| `p1_2v_chicago` | 0.0575 | 86.67 | 0.0565 | 0.0546 | 90.00 | 0.3150 | 14.35 | 16.22 |
| `p1_4v_all` | 0.0579 | 89.67 | 0.0564 | 0.0533 | 92.67 | 0.2464 | 12.58 | 16.05 |

结论：

- 三个 finetuned checkpoint 都显著优于 raw VGGT-Omega。
- `p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth` 综合最好：`joint_global_pointmaps_abs_rel`、`joint pose_auc_5`、`joint ray_dirs_err_deg`、`RS-only height MAE`、`joint RS MAE` 均为最优或接近最优。
- `p1_vggt_omega_joint_depth_512/checkpoint-best.pth` 在 chicago validation loss 和部分 aerial pose 指标上更稳，但 New York 泛化略低于 all。
- `p1_vggt_omega_joint_depth_512_1gpu_2v/checkpoint-best.pth` 的 point 指标接近最好，但 pose/ray 指标弱于 4-view 训练。

推荐默认使用：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth
```

如果只关注 chicago validation loss 或单 remote-view 设置，则使用：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512/checkpoint-best.pth
```

## 卫星输入是否有效

Benchmark 中的 remote controls 含义：

- `same`：正常匹配的卫星图输入。
- `blank`：卫星图替换成常数图，近似没有有效卫星信息。
- `shuffled`：卫星图替换成其他 scene 的图，近似错误卫星信息。

因此看卫星输入收益时，主要比较：

- `same vs blank`：有卫星输入 vs 无有效卫星输入。
- `same vs shuffled`：有正确卫星输入 vs 错误卫星输入。

以综合最好的 `p1_vggt_omega_joint_depth_512_all` 为例，`NUM_VIEWS=4` 时：

| mode | pointmaps_abs_rel ↓ | pose_auc_5 ↑ | ray_dirs_err_deg ↓ | z_depth_abs_rel ↓ |
|---|---:|---:|---:|---:|
| same | 0.05638 | 92.67 | 0.24637 | 0.06521 |
| blank | 0.06047 | 89.33 | 0.28125 | 0.06806 |
| shuffled | 0.06065 | 86.33 | 0.26890 | 0.06825 |

`same` 相比 `blank`：

- point error 降低约 6.75%。
- pose AUC5 提升 +3.33。
- ray error 降低约 12.40%。
- z-depth error 降低约 4.19%。

结论：卫星输入确实被模型利用，并带来可测提升，但在 4-view 条件下提升幅度中等。

## 卫星输入收益与普通图像数量的关系

为了分析普通图像数量是否影响卫星输入收益，固定同一个 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth
```

分别跑 `NUM_VIEWS=2/3/4` 的 same/blank/shuffled mini controls。

输出目录：

| views | 输出目录 |
|---:|---|
| 2 | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_all_2v_mini_controls` |
| 3 | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_all_3v_mini_controls` |
| 4 | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/vggt_omega_p1_joint_depth_512_all_mini_controls` |

原始指标：

| views | mode | point ↓ | AUC5 ↑ | ray err ↓ | z depth ↓ |
|---:|---|---:|---:|---:|---:|
| 2 | same | 0.05761 | 92.00 | 0.29202 | 0.04962 |
| 2 | blank | 0.06589 | 84.00 | 0.42505 | 0.05617 |
| 2 | shuffled | 0.06707 | 82.00 | 0.43565 | 0.05776 |
| 3 | same | 0.05605 | 93.33 | 0.30863 | 0.04809 |
| 3 | blank | 0.06131 | 88.67 | 0.31363 | 0.05078 |
| 3 | shuffled | 0.06105 | 90.67 | 0.35091 | 0.05232 |
| 4 | same | 0.05638 | 92.67 | 0.24637 | 0.06521 |
| 4 | blank | 0.06047 | 89.33 | 0.28125 | 0.06806 |
| 4 | shuffled | 0.06065 | 86.33 | 0.26890 | 0.06825 |

`same vs blank` 提升幅度：

| views | point 下降 | point 相对提升 | AUC5 提升 | ray err 下降 | z depth 下降 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.00828 | 12.57% | +8.00 | 31.30% | 11.66% |
| 3 | 0.00526 | 8.58% | +4.67 | 1.59% | 5.30% |
| 4 | 0.00408 | 6.75% | +3.33 | 12.40% | 4.19% |

结论：当前实验支持“普通图像 view 数少时，卫星输入提升更明显”的判断。

- `2 views` 时卫星输入收益最大，point 提升 12.57%，AUC5 提升 +8，ray error 降低 31.30%。
- `3 views` 时收益下降到中等水平。
- `4 views` 时普通多视角图像本身约束更强，卫星输入仍有效，但边际收益变小。

解释：低 view 数下普通图像提供的多视角几何约束不足，卫星图更像额外的全局结构/高度先验；当 view 数增加后，普通图像已经能提供较充分的几何约束，因此卫星图的相对收益下降。

## 注意事项

- `metric_point_l1` 和 `metric_scale_abs_rel` 为 `nan` 是预期现象，因为当前 `vggt_omega` wrapper 没有输出 MapAnything 的 `metric_scaling_factor`。
- 当前 benchmark 是 New York mini controls，样本数为 10 paired scenes。结论适合作为趋势判断，正式论文/报告建议扩展到更多城市和完整 validation/test split。
- `same vs blank/shuffled` 是判断模型是否使用卫星输入的关键对照，而不是只看 joint average。
