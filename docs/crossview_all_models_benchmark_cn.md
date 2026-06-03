# Crossview 全部已训练模型统一评测记录

评测时间：2026-06-01；补充 `vggt_omega_p2_joint_depth_512_all_2`：2026-06-02。

## 评测口径

- 数据：`benchmark_vigor_chicago_rs_aerial`，城市 `newyork`。
- 样本：`REMOTE_OVERFIT_NUM_SETS=10`，即 10 个 paired scene 的 mini benchmark。
- 输入：`NUM_VIEWS=4`，`BATCH_SIZE=1`。
- 卫星图对照：`REMOTE_CONTROL_MODES=[same,blank,shuffled]`。
- 分辨率：Pi3/VGGT 使用 518，VGGT-Omega 使用 512。
- 输出目录：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls`。
- 批量脚本：`scripts/evaluate_crossview_all_models.py`。

指标方向：`pointmaps_abs_rel`、`joint_global_pointmaps_abs_rel`、`ray_dirs_err_deg`、`rs_height_mae_affine` 越低越好；`pose_auc_5` 越高越好。

## 运行状态

- 共发现并评测 26 个模型结果 JSON，最终均成功生成 `rs_aerial_benchmark_results.json`。
- 2026-06-02 追加评测 `vggt_omega_p2_joint_depth_512_all_2`，使用同一 New York 10-scene mini benchmark、4 views、512 分辨率。
- 第一次 VGGT 评测会下载/缓存 5GB 级 base 权重，因此 `vggt_p5b_shared_norm` 耗时明显更长；后续 VGGT 从缓存加载。

## 绝对精度 Top 10

按 `joint_global_pointmaps_abs_rel` 从低到高排序：

| rank | model | joint_global | joint_point | joint_auc5 | joint_ray | RS-only MAE | joint RS MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `vggt_p5e_remote_head_attention_viewtype` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 2 | `vggt_p5h_crossattn_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 3 | `vggt_p5h_film_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 4 | `vggt_p5h_film_unfreeze_viewtype_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 5 | `vggt_p6b_private_head_2` | 0.0464 | 0.0504 | 93.33 | 0.2843 | 12.29 | 16.10 |
| 6 | `vggt_omega_p2_joint_depth_512_all_2` | 0.0492 | 0.0508 | 97.00 | 0.2006 | 12.92 | 16.00 |
| 7 | `vggt_p5d_remote_point_head_consistency` | 0.0522 | 0.0561 | 86.00 | 0.4080 | 17.57 | 16.20 |
| 8 | `vggt_omega_p1_joint_depth_512_all` | 0.0533 | 0.0564 | 92.67 | 0.2464 | 12.58 | 16.05 |
| 9 | `vggt_p6b_private_head_1` | 0.0543 | 0.0581 | 86.67 | 0.4158 | 17.45 | 16.39 |
| 10 | `vggt_omega_p1_joint_depth_512` | 0.0545 | 0.0574 | 92.33 | 0.2833 | 16.35 | 16.32 |

## 卫星输入收益 Top 10

按 `blank_point - same_point` 从高到低排序；正数表示真实卫星图比空白卫星图更好。

| rank | model | same point | blank point | delta | rel | AUC delta | ray rel | z rel | shuffled point |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `vggt_p5g_no_fusion_fixedfreeze_protected` | 0.1565 | 0.1634 | 0.0069 | +4.20% | 4.00 | +10.11% | -10.27% | 0.1550 |
| 2 | `vggt_p6a_raw_base_conditional_remote_adapter` | 0.1565 | 0.1634 | 0.0069 | +4.20% | 4.00 | +10.11% | -10.27% | 0.1550 |
| 3 | `vggt_p5g_no_fusion_split_remote` | 0.0836 | 0.0884 | 0.0048 | +5.43% | 0.00 | -16.59% | +2.12% | 0.0841 |
| 4 | `vggt_omega_p1_joint_depth_512_all` | 0.0564 | 0.0605 | 0.0041 | +6.75% | 3.33 | +12.40% | +4.19% | 0.0606 |
| 5 | `vggt_p6b_private_head_w03_bs5_static_remoteonly` | 0.0705 | 0.0743 | 0.0037 | +5.00% | -4.00 | +13.31% | +2.91% | 0.0717 |
| 6 | `vggt_p5e_remote_head_attention_viewtype` | 0.0501 | 0.0533 | 0.0032 | +6.03% | 3.00 | +11.70% | +7.82% | 0.0517 |
| 7 | `vggt_p5h_crossattn_protected` | 0.0501 | 0.0533 | 0.0032 | +6.03% | 3.00 | +11.70% | +7.82% | 0.0517 |
| 8 | `vggt_p5h_film_protected` | 0.0501 | 0.0533 | 0.0032 | +6.03% | 3.00 | +11.70% | +7.82% | 0.0517 |
| 9 | `vggt_p5h_film_unfreeze_viewtype_protected` | 0.0501 | 0.0533 | 0.0032 | +6.03% | 3.00 | +11.70% | +7.82% | 0.0517 |
| 10 | `vggt_p7_remote_head_projection_aux_trunk` | 0.0610 | 0.0637 | 0.0027 | +4.24% | 0.67 | +15.26% | +1.19% | 0.0632 |

## 全量结果

| family | model | res | aerial point | aerial AUC5 | joint global | joint point | joint AUC5 | joint ray | same-vs-blank point delta | same-vs-blank rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi3 | `pi3_p3_modality_embedding` | 518 | 0.0567 | 89.67 | 0.0548 | 0.0576 | 89.00 | 0.3711 | -0.0003 | -0.61% |
| pi3 | `pi3_p3_modality_embedding_remote_head` | 518 | 0.0569 | 88.33 | 0.0568 | 0.0576 | 86.33 | 0.3778 | -0.0006 | -1.01% |
| pi3 | `pi3_p3_zero_covis` | 518 | 0.0574 | 88.33 | 0.0569 | 0.0580 | 84.67 | 0.4562 | -0.0003 | -0.47% |
| pi3 | `pi3_p3_base` | 518 | 0.0602 | 84.00 | 0.0606 | 0.0617 | 84.00 | 0.4624 | -0.0012 | -1.96% |
| pi3 | `pi3_p3_freeze_shared` | 518 | 0.0731 | 87.33 | 0.0698 | 0.0738 | 80.67 | 0.4679 | -0.0003 | -0.40% |
| vggt | `vggt_p5e_remote_head_attention_viewtype` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_crossattn_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_film_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_film_unfreeze_viewtype_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p6b_private_head_2` | 518 | 0.0510 | 92.00 | 0.0464 | 0.0504 | 93.33 | 0.2843 | 0.0016 | +2.99% |
| vggt_omega | `vggt_omega_p2_joint_depth_512_all_2` | 512 | 0.0527 | 93.67 | 0.0492 | 0.0508 | 97.00 | 0.2006 | 0.0026 | +4.91% |
| vggt | `vggt_p5d_remote_point_head_consistency` | 518 | 0.0564 | 87.67 | 0.0522 | 0.0561 | 86.00 | 0.4080 | 0.0004 | +0.63% |
| vggt_omega | `vggt_omega_p1_joint_depth_512_all` | 512 | 0.0579 | 89.67 | 0.0533 | 0.0564 | 92.67 | 0.2464 | 0.0041 | +6.75% |
| vggt | `vggt_p6b_private_head_1` | 518 | 0.0576 | 89.67 | 0.0543 | 0.0581 | 86.67 | 0.4158 | 0.0010 | +1.73% |
| vggt_omega | `vggt_omega_p1_joint_depth_512` | 512 | 0.0580 | 91.00 | 0.0545 | 0.0574 | 92.33 | 0.2833 | 0.0026 | +4.38% |
| vggt_omega | `vggt_omega_p1_joint_depth_512_1gpu_2v` | 512 | 0.0575 | 86.67 | 0.0546 | 0.0565 | 90.00 | 0.3150 | 0.0020 | +3.42% |
| vggt | `vggt_p7_remote_head_projection_aux_trunk` | 518 | 0.0606 | 87.33 | 0.0564 | 0.0610 | 87.67 | 0.4018 | 0.0027 | +4.24% |
| vggt | `vggt_p5c_viewtype` | 518 | 0.0565 | 87.33 | 0.0567 | 0.0558 | 87.67 | 0.3885 | 0.0008 | +1.49% |
| vggt | `vggt_p5b_shared_norm_2` | 518 | 0.0567 | 89.33 | 0.0592 | 0.0560 | 87.00 | 0.4301 | 0.0008 | +1.34% |
| vggt | `vggt_p5f_lite_early_bias_gated_residual` | 518 | 0.0674 | 86.00 | 0.0626 | 0.0673 | 83.00 | 0.4175 | 0.0000 | +0.01% |
| vggt | `vggt_p6b_private_head_w03_bs5_static_remoteonly` | 518 | 0.0674 | 76.00 | 0.0670 | 0.0705 | 73.00 | 0.6179 | 0.0037 | +5.00% |
| vggt | `vggt_p5b_shared_norm` | 518 | 0.0590 | 83.00 | 0.0671 | 0.0566 | 86.00 | 0.3941 | 0.0024 | +4.13% |
| vggt | `vggt_p5g_no_fusion_split_remote` | 518 | 0.0805 | 75.00 | 0.1014 | 0.0836 | 74.00 | 0.6579 | 0.0048 | +5.43% |
| vggt | `vggt_p5g_crossattn_split_remote` | 518 | 0.0913 | 82.33 | 0.1091 | 0.0932 | 78.33 | 0.6354 | 0.0006 | +0.60% |
| vggt | `vggt_p5g_film_split_remote` | 518 | 0.0908 | 80.67 | 0.1132 | 0.0924 | 79.33 | 0.7067 | 0.0001 | +0.15% |
| vggt | `vggt_p6a_raw_base_conditional_remote_adapter` | 518 | 0.1663 | 33.00 | 0.1469 | 0.1565 | 38.33 | 1.4736 | 0.0069 | +4.20% |
| vggt | `vggt_p5g_no_fusion_fixedfreeze_protected` | 518 | 0.1663 | 33.00 | 0.1727 | 0.1565 | 38.33 | 1.4736 | 0.0069 | +4.20% |

## p5e 与 p5h 指标完全一致的说明

`vggt_p5e_remote_head_attention_viewtype`、`vggt_p5h_crossattn_protected`、`vggt_p5h_film_protected`、`vggt_p5h_film_unfreeze_viewtype_protected` 在本表中的主要指标完全一致。已检查：

- 四个评测目录使用的 checkpoint 路径不同。
- checkpoint 文件大小/hash 不同，不是同一个文件复制。
- p5h 系列训练时从 p5e checkpoint 初始化，并且 remote/branch consistency loss 权重为 0，主要训练 late remote-to-aerial fusion。
- p5h checkpoint 中 late fusion gate 很小：cross-attn 约 -0.0104，film 约 0.0139，film-unfreeze 约 0.0233。
- 修正 p5h 评测 override 字段名后单独复测，结果仍与 p5e 只有浮点末位差异。

因此，这几行不能解读成四种结构独立取得了同样的最优结果。更合理的解读是：在当前 10-scene、4-view、普通视角 joint 指标上，p5h 的 late-fusion 改动没有带来可测差异，结果基本等价于其初始化来源 p5e。选择模型时应把它们视作同一组候选，并优先用更大样本或更直接的 remote-to-aerial 消融复核。

## VGGT-Omega p2 更新

`vggt_omega_p2_joint_depth_512_all_2` 对应训练目录：

`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all_2`

这组训练相对旧 `vggt_omega_p1_joint_depth_512_all` 的关键差异：

- 移除训练阶段 confidence / aleatoric uncertainty loss，使用 `vggt_loss_rs_joint_no_conf`。
- `warmup_epochs` 从 1 提到 3。
- `max_num_of_imgs_per_gpu` 从 8 提到 12。
- 仍使用全部城市、4 views、VGGT-Omega 512、`lr=1e-5`、cosine schedule。

训练验证集上，p2 明显优于旧 all：

| metric | p1 all | p2 all_2 | change |
|---|---:|---:|---:|
| val loss_avg | 0.6159 | 0.4960 | -19.47% |
| val aerial_loss_avg | 0.7601 | 0.6010 | -20.92% |
| val remote_loss_avg | 0.1179 | 0.0977 | -17.12% |
| val rs_pointmap_loss_avg | 0.0295 | 0.0244 | -17.12% |

New York mini benchmark 上也有提升：

| metric | p1 all | p2 all_2 | change |
|---|---:|---:|---:|
| aerial pointmaps_abs_rel | 0.0579 | 0.0527 | -8.93% |
| aerial pose_auc_5 | 89.67 | 93.67 | +4.00 |
| joint pointmaps_abs_rel | 0.0564 | 0.0508 | -9.97% |
| joint_global_pointmaps_abs_rel | 0.0533 | 0.0492 | -7.61% |
| joint pose_auc_5 | 92.67 | 97.00 | +4.33 |
| joint ray_dirs_err_deg | 0.2464 | 0.2006 | -18.60% |
| joint z_depth_abs_rel | 0.0652 | 0.0583 | -10.65% |
| RS-only height MAE | 12.58 | 12.92 | +2.73% |
| joint RS height MAE | 16.05 | 16.00 | -0.31% |

解读：p2 在普通视角几何和 joint 几何上提升明确，尤其 pose AUC、ray 和 pointmap 指标；RS-only 高度 MAE 略差，joint RS MAE 基本持平。由于 p2 同时改变了 no-conf loss、warmup 和 batch size，这个结果不能作为单因素消融，但可以作为当前 VGGT-Omega 系列更强的候选配置。

## 真实卫星内容收益筛选

只看 `same < blank` 还不够，因为 shuffled 也提供了 remote token 和遥感图像分布。如果 `same` 没有优于 `shuffled`，说明收益可能不是来自正确匹配的卫星内容。下面列出 `same` 同时优于 `blank` 和 `shuffled` 的主要模型：

| model | same point | blank point | shuffled point | blank delta | shuffled delta | joint global | joint AUC5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vggt_p5g_no_fusion_split_remote` | 0.0836 | 0.0884 | 0.0841 | 0.0048 | 0.0005 | 0.1014 | 74.00 |
| `vggt_omega_p1_joint_depth_512_all` | 0.0564 | 0.0605 | 0.0606 | 0.0041 | 0.0043 | 0.0533 | 92.67 |
| `vggt_p6b_private_head_w03_bs5_static_remoteonly` | 0.0705 | 0.0743 | 0.0717 | 0.0037 | 0.0012 | 0.0670 | 73.00 |
| `vggt_p5e_remote_head_attention_viewtype` | 0.0501 | 0.0533 | 0.0517 | 0.0032 | 0.0017 | 0.0457 | 92.00 |
| `vggt_p5h_crossattn_protected` | 0.0501 | 0.0533 | 0.0517 | 0.0032 | 0.0017 | 0.0457 | 92.00 |
| `vggt_p5h_film_protected` | 0.0501 | 0.0533 | 0.0517 | 0.0032 | 0.0017 | 0.0457 | 92.00 |
| `vggt_p5h_film_unfreeze_viewtype_protected` | 0.0501 | 0.0533 | 0.0517 | 0.0032 | 0.0017 | 0.0457 | 92.00 |
| `vggt_p7_remote_head_projection_aux_trunk` | 0.0610 | 0.0637 | 0.0632 | 0.0027 | 0.0022 | 0.0564 | 87.67 |
| `vggt_omega_p2_joint_depth_512_all_2` | 0.0508 | 0.0534 | 0.0533 | 0.0026 | 0.0025 | 0.0492 | 97.00 |
| `vggt_omega_p1_joint_depth_512` | 0.0574 | 0.0600 | 0.0614 | 0.0026 | 0.0040 | 0.0545 | 92.33 |
| `vggt_p5b_shared_norm` | 0.0566 | 0.0590 | 0.0591 | 0.0024 | 0.0025 | 0.0671 | 86.00 |

这个筛选下，`vggt_p5e_remote_head_attention_viewtype` / `vggt_p5h_*` 的 `joint_global` 仍最好；`vggt_omega_p2_joint_depth_512_all_2` 的 `joint point`、`pose_auc_5`、`ray` 更强，是 VGGT-Omega 系列当前最好的 checkpoint。旧 `vggt_omega_p1_joint_depth_512_all` 的卫星内容 delta 更大，但 p2 的绝对精度更高。

## 初步结论

- 绝对 `joint_global_pointmaps_abs_rel` 当前最好的是 `vggt_p5e_remote_head_attention_viewtype`，`joint_global=0.0457`，`joint_point=0.0501`，`pose_auc_5=92.00`。
- 追加的 `vggt_omega_p2_joint_depth_512_all_2` 是 VGGT-Omega 系列当前最强结果：`joint_global=0.0492`，`joint_point=0.0508`，`pose_auc_5=97.00`，`joint_ray=0.2006`；相比旧 `p1 all`，joint point 提升 9.97%，joint global 提升 7.61%。
- 对卫星图最敏感、same 相比 blank 的 point 提升最大的是 `vggt_p5g_no_fusion_fixedfreeze_protected`，`delta=0.0069`，相对提升 +4.20%。
- 解读卫星输入是否有效时，优先看 `same` 是否同时优于 `blank` 和 `shuffled`。如果只优于 blank 但接近 shuffled，可能更多是额外 remote token/分布扰动带来的收益，而不是正确利用真实卫星内容。
- 这是 10 scene mini benchmark，适合快速筛选；最终模型选择仍建议用更大 paired scene 数量或完整 benchmark 复核。
