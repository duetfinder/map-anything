# Crossview 全部已训练模型统一评测记录

评测时间：2026-06-01；补充 `vggt_omega_p2_joint_depth_512_all_2`：2026-06-02；补充 P7 projection-aux 两城/全城市/curriculum/nocrop/lowtrunklr/P5E-lowtrunkfull 结果：2026-06-04/05；补充 P7 P5B-head freeze/aux-only、oldP7 aggtail2 remote-head、MoGe2 remote prior、projection-MoGe aux 与 PI3 projection-aux MoGe 迁移验证：2026-06-06。

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

- 文档累计记录 55 个模型结果。2026-06-04/05 新增 `vggt_p7_p5b_shared_norm_projection_aux_full_2city`、`vggt_p7_p5b_shared_norm_projection_aux_allcities_best/final`、`vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final`、`vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best`、`vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best/final`、`vggt_p7_p5e_private_viewtype_projection_aux_allcities_best/final`、`vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_best/final`、`vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final`、`vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final`，使用同一 New York 10-scene mini benchmark、4 views、518 分辨率。其中 P7-P5E lowtrunkfull best/final benchmark 完全一致，摘要表只保留 final。2026-06-06 新增 `vggt_p7_diagnostic_oldp7_trunk_p5b_remote_head_final`、`vggt_p7_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_final`、`vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best`、`vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_final`、`vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_final`、`vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final/e6_final/zdist2_final/zhigh2q80_final`、`vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final`、`vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_final/e4_final`、`vggt_p7_moge2_balanced20x4_private_tokens_mogegrad001_edge0002_fixed_best/final`、`vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_best/final`、`pi3_p7_remote_head_projection_aux_moge_final`。
- 2026-06-02 追加评测 `vggt_omega_p2_joint_depth_512_all_2`，使用同一 New York 10-scene mini benchmark、4 views、512 分辨率。
- 第一次 VGGT 评测会下载/缓存 5GB 级 base 权重，因此 `vggt_p5b_shared_norm` 耗时明显更长；后续 VGGT 从缓存加载。

## 绝对精度 Top 10

按 `joint_global_pointmaps_abs_rel` 从低到高排序：

| rank | model | joint_global | joint_point | joint_auc5 | joint_ray | RS-only MAE | joint RS MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final` | 0.0448 | 0.0485 | 95.67 | 0.2958 | 10.10 | 16.73 |
| 2 | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_best` | 0.0448 | 0.0486 | 95.67 | 0.2956 | 10.01 | 16.70 |
| 3 | `vggt_p5e_remote_head_attention_viewtype` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 4 | `vggt_p5h_crossattn_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 5 | `vggt_p5h_film_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 6 | `vggt_p5h_film_unfreeze_viewtype_protected` | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 15.64 |
| 7 | `vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final` | 0.0460 | 0.0491 | 93.67 | 0.2990 | 9.60 | 16.90 |
| 8 | `vggt_p6b_private_head_2` | 0.0464 | 0.0504 | 93.33 | 0.2843 | 12.29 | 16.10 |
| 9 | `vggt_p7_p5e_private_viewtype_projection_aux_allcities_final` | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 16.93 |
| 10 | `vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final` | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 16.93 |

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
| pi3 | `pi3_p7_remote_head_projection_aux_moge_final` | 518 | 0.0587 | 86.33 | 0.0561 | 0.0576 | 87.67 | 0.3916 | 0.0005 | +0.88% |
| pi3 | `pi3_p3_modality_embedding_remote_head` | 518 | 0.0569 | 88.33 | 0.0568 | 0.0576 | 86.33 | 0.3778 | -0.0006 | -1.01% |
| pi3 | `pi3_p3_zero_covis` | 518 | 0.0574 | 88.33 | 0.0569 | 0.0580 | 84.67 | 0.4562 | -0.0003 | -0.47% |
| pi3 | `pi3_p3_base` | 518 | 0.0602 | 84.00 | 0.0606 | 0.0617 | 84.00 | 0.4624 | -0.0012 | -1.96% |
| pi3 | `pi3_p3_freeze_shared` | 518 | 0.0731 | 87.33 | 0.0698 | 0.0738 | 80.67 | 0.4679 | -0.0003 | -0.40% |
| vggt | `vggt_p5e_remote_head_attention_viewtype` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_crossattn_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_film_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p5h_film_unfreeze_viewtype_protected` | 518 | 0.0530 | 89.33 | 0.0457 | 0.0501 | 92.00 | 0.2926 | 0.0032 | +6.03% |
| vggt | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final` | 518 | 0.0494 | 93.33 | 0.0448 | 0.0485 | 95.67 | 0.2958 | 0.0009 | +1.81% |
| vggt | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_best` | 518 | 0.0494 | 93.33 | 0.0448 | 0.0486 | 95.67 | 0.2956 | 0.0009 | +1.75% |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final` | 518 | 0.0500 | 92.33 | 0.0460 | 0.0491 | 93.67 | 0.2990 | 0.0008 | +1.67% |
| vggt | `vggt_p6b_private_head_2` | 518 | 0.0510 | 92.00 | 0.0464 | 0.0504 | 93.33 | 0.2843 | 0.0016 | +2.99% |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_allcities_final` | 518 | 0.0501 | 93.33 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0004 | +0.83% |
| vggt | `vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final` | 518 | 0.0501 | 93.33 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0004 | +0.81% |
| vggt | `vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final` | 518 | 0.0501 | 93.33 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 0.0004 | +0.81% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final` | 518 | 0.0493 | 94.00 | 0.0471 | 0.0484 | 95.33 | 0.3009 | 0.0011 | +2.16% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best` | 518 | 0.0498 | 94.67 | 0.0472 | 0.0487 | 96.00 | 0.2997 | 0.0012 | +2.37% |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final` | 518 | 0.0496 | 92.67 | 0.0472 | 0.0494 | 94.33 | 0.2903 | 0.0009 | +1.76% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_best` | 518 | 0.0490 | 92.33 | 0.0474 | 0.0485 | 95.33 | 0.2893 | 0.0007 | +1.41% |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final` | 518 | 0.0497 | 92.67 | 0.0474 | 0.0489 | 95.33 | 0.2906 | 0.0010 | +2.07% |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_final` | 518 | 0.0498 | 92.33 | 0.0474 | 0.0495 | 94.67 | 0.2912 | 0.0009 | +1.82% |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final` | 518 | 0.0499 | 92.33 | 0.0475 | 0.0492 | 95.33 | 0.2913 | 0.0010 | +2.02% |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final` | 518 | 0.0499 | 92.67 | 0.0475 | 0.0492 | 94.33 | 0.2895 | 0.0010 | +2.10% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_final` | 518 | 0.0493 | 92.67 | 0.0474 | 0.0486 | 95.67 | 0.2948 | 0.0008 | +1.54% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best` | 518 | 0.0508 | 88.67 | 0.0482 | 0.0495 | 92.33 | 0.3247 | 0.0015 | +3.03% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final` | 518 | 0.0524 | 93.00 | 0.0483 | 0.0506 | 93.00 | 0.2941 | 0.0021 | +4.04% |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_full_2city` | 518 | 0.0506 | 90.67 | 0.0485 | 0.0491 | 92.33 | 0.2961 | 0.0016 | +3.17% |
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
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final` | 0.0506 | 0.0527 | 0.0533 | 0.0021 | 0.0027 | 0.0483 | 93.00 |
| `vggt_p7_p5b_shared_norm_projection_aux_full_2city` | 0.0491 | 0.0507 | 0.0524 | 0.0016 | 0.0033 | 0.0485 | 92.33 |
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best` | 0.0495 | 0.0510 | 0.0516 | 0.0015 | 0.0021 | 0.0482 | 92.33 |
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best` | 0.0487 | 0.0499 | 0.0502 | 0.0012 | 0.0015 | 0.0472 | 96.00 |
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final` | 0.0484 | 0.0495 | 0.0499 | 0.0011 | 0.0014 | 0.0471 | 95.33 |
| `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final` | 0.0492 | 0.0502 | 0.0505 | 0.0010 | 0.0013 | 0.0475 | 95.33 |
| `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final` | 0.0492 | 0.0503 | 0.0505 | 0.0010 | 0.0013 | 0.0475 | 94.33 |
| `vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final` | 0.0491 | 0.0500 | 0.0507 | 0.0008 | 0.0016 | 0.0460 | 93.67 |
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_final` | 0.0486 | 0.0494 | 0.0503 | 0.0008 | 0.0016 | 0.0474 | 95.67 |
| `vggt_p7_p5b_shared_norm_projection_aux_allcities_best` | 0.0485 | 0.0492 | 0.0510 | 0.0007 | 0.0025 | 0.0474 | 95.33 |
| `vggt_p7_p5e_private_viewtype_projection_aux_allcities_final` | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 | 0.0466 | 93.00 |
| `vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final` | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 | 0.0466 | 93.00 |
| `vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final` | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 | 0.0466 | 93.00 |

这个筛选下，`vggt_p5e_remote_head_attention_viewtype` / `vggt_p5h_*` 的 `joint_global` 仍最好；P7-P5E lowtrunkfull 把 P7 路线的 `joint_global` 进一步推到 0.0460，并且 RS-only MAE 降到 9.60，但 `joint point/AUC` 仍不如 P7-P5B lowtrunklr final。P7-P5H film protected weak/strong ranking 基本复现旧 P7-P5E，未拉开 satellite delta。P7-P5B lowtrunklr final 是当前 P7-P5B 综合最好权重：`joint_global=0.0471`、`joint_point=0.0484`，同时 blank delta 从旧 best 的 `0.0007` 提到 `0.0011`；代价是 ray 从 `0.2893` 退到 `0.3009`。nocrop warmstart 把 same-vs-blank delta 提到 `0.0015`，但 absolute reconstruction 和 AUC 退化，不是新 best。`vggt_omega_p2_joint_depth_512_all_2` 的 `pose_auc_5`、`ray` 更强，是 VGGT-Omega 系列当前最好的 checkpoint。旧 `vggt_omega_p1_joint_depth_512_all` 的卫星内容 delta 更大，但 p2 的绝对精度更高。

## P7 P5B shared-norm projection-aux full 两城更新

新增模型：

```text
vggt_p7_p5b_shared_norm_projection_aux_full_2city
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu/checkpoint-final.pth
```

benchmark 配置：

```text
NUM_VIEWS=4
REMOTE_OVERFIT_NUM_SETS=10
REMOTE_CONTROL_MODES=[same,blank,shuffled]
CITY=newyork
REMOTE_PROVIDER=Google_Satellite
resolution=518
vggt_export_mode=mixed
use_remote_projection_aux_head=true
remote_projection_aux_hidden_dim=96
remote_projection_aux_use_rgb=true
remote_projection_aux_use_coord=true
remote_projection_aux_image_stem_dim=32
remote_projection_aux_positive_slope=true
remote_projection_aux_num_blocks=6
```

关键结果：

| metric | value |
|---|---:|
| aerial pointmaps_abs_rel | 0.0506 |
| aerial pose_auc_5 | 90.67 |
| joint_global_pointmaps_abs_rel | 0.0485 |
| joint pointmaps_abs_rel | 0.0491 |
| joint pose_auc_5 | 92.33 |
| joint ray_dirs_err_deg | 0.2961 |
| RS-only height MAE | 12.60 |
| joint RS height MAE | 16.47 |
| same point | 0.0491 |
| blank point | 0.0507 |
| shuffled point | 0.0524 |
| same-vs-blank delta | 0.0016 |
| same-vs-shuffled delta | 0.0033 |

解读：

- 这组 full 两城 P7-P5B projection-aux 的绝对重建精度明显强于早期 `vggt_p7_remote_head_projection_aux_trunk`：`joint_global 0.0564 -> 0.0485`，`joint_point 0.0610 -> 0.0491`。
- 它的 `same` 同时优于 `blank` 和 `shuffled`，说明正确卫星图内容有实际贡献，不只是 remote token 分布扰动。
- 但卫星内容收益幅度仍低于 p5e/p5h 和 VGGT-Omega p2；projection aux 已经证明机制可学，下一步瓶颈更可能在 remote 信息如何进入 ordinary reconstruction，而不是 aux head 本身。

## P7 P5B shared-norm projection-aux 全城市更新

新增模型：

```text
vggt_p7_p5b_shared_norm_projection_aux_allcities_best
vggt_p7_p5b_shared_norm_projection_aux_allcities_final
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun/checkpoint-best.pth
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_warm2city_e30_b8_2gpu_rerun/checkpoint-final.pth
```

训练使用 Chicago/NewYork/SanFrancisco/Seattle 全部 clean metadata city split，从两城 full checkpoint warm-start，`LAMBDA_PROJ_REL_HEIGHT=0.75`，其他 projection aux 结构与两城版本一致。

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 |
| allcities final | 0.0474 | 0.0486 | 95.67 | 0.2948 | 9.68 | 0.0486 | 0.0494 | 0.0503 | 0.0008 |
| curric 2v->4v final | 0.0483 | 0.0506 | 93.00 | 0.2941 | 10.10 | 0.0506 | 0.0527 | 0.0533 | 0.0021 |
| full 2city final | 0.0485 | 0.0491 | 92.33 | 0.2961 | 12.60 | 0.0491 | 0.0507 | 0.0524 | 0.0016 |

解读：

- 全城市训练明确提升绝对精度：相对两城 final，`joint_global 0.0485 -> 0.0474`，`joint_point 0.0491 -> 0.0485/0.0486`，`pose_auc_5 92.33 -> 95.33/95.67`。
- `best` 和 `final` 很接近；`best` 的 point/ray 略好，`final` 的 AUC 略好。后续默认可用 `final`，若按 pointmap 排名则用 `best`。
- `same < blank < shuffled` 仍成立，说明真实卫星内容没有被忽略；但 same-vs-blank delta 降到 `0.0007-0.0008`。这说明全城市主要提升来自更强的主重建和泛化，卫星内容敏感性没有同步增强。
- 追加的 2-view curriculum 再切回 4-view 后，same-vs-blank delta 增到 `0.0021`，但 `joint_point` 退到 `0.0506`、`joint_global` 退到 `0.0483`。因此它只能说明低 view 预热会提高 remote 内容敏感性，不能作为当前最佳 checkpoint。

## P7 P5E private-head projection-aux 全城市短跑

新增模型：

```text
vggt_p7_p5e_private_viewtype_projection_aux_allcities_best
vggt_p7_p5e_private_viewtype_projection_aux_allcities_final
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static/checkpoint-best.pth
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_h075_warm_p5bfinal_e12_b8_2gpu_static/checkpoint-final.pth
```

配置差异：

```text
use_view_type_bias=true
use_remote_private_point_head=true
remote_output_head=point
output_point_head_for_consistency=false
train_params=vggt_p7_p5e_projection_aux
EPOCHS=12
warmstart=P7-P5B allcities final
```

best/final benchmark 完全一致，关键结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 |
| P7-P5B aux allcities best | 0.0474 | 0.0485 | 95.33 | 0.2893 | 9.53 | 0.0485 | 0.0492 | 0.0510 | 0.0007 |
| original p5e baseline | 0.0457 | 0.0501 | 92.00 | 0.2926 | 9.76 | 0.0501 | 0.0533 | 0.0517 | 0.0032 |

解读：

- P7-P5E aux 提升了 P7 路线的 `joint_global`，从 P7-P5B allcities 的 0.0474 到 0.0466，接近原始 p5e/p5h 基线的 0.0457。
- 但 `joint_point` 不如 P7-P5B allcities：0.0495 vs 0.0485；AUC 也从 95+ 降到 93.00。
- `same < blank < shuffled` 仍成立，但 blank delta 只有 0.0004，说明 private remote head 提高了全局对齐/尺度类指标，却没有增强真实卫星内容依赖。

## P7 P5E private-viewtype projection-aux lowtrunkfull 更新

新增模型：

```text
vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_best
vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-best.pth
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-final.pth
```

配置差异：

```text
warmstart=P7-P5E private-viewtype projection-aux final
train_params=vggt_p7_p5e_projection_aux_lowtrunk_full
use_view_type_bias=true
use_remote_private_point_head=true
remote_output_head=point
output_point_head_for_consistency=false
ordinary camera/depth/point heads lr=2e-6
remote_point_head lr=1e-5
projection aux heads lr=5e-5
EPOCHS=6
BATCH_SIZE=9
```

best/final benchmark 完全一致，关键结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5E lowtrunkfull | 0.0460 | 0.0491 | 93.67 | 0.2990 | 9.60 | 0.0491 | 0.0500 | 0.0507 | 0.0008 | 0.0016 |
| P7-P5E aux allcities final | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5B lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 0.0484 | 0.0495 | 0.0499 | 0.0011 | 0.0014 |

解读：

- 解冻普通 camera/depth/point heads 并用低 LR 微调后，P7-P5E 的 `joint_global` 从 0.0466 提升到 0.0460，RS-only MAE 从 10.11 降到 9.60，说明 private remote head + projection aux 的 remote 重建/全局对齐继续有效。
- `joint_point` 从 0.0495 到 0.0491 有改善，但仍不如 P7-P5B lowtrunklr final 的 0.0484；AUC 93.67 也低于 P7-P5B 的 95.33。
- satellite delta 从 0.0004 提到 0.0008，说明真实卫星内容依赖略增强，但还没超过 P7-P5B lowtrunklr final 的 0.0011。
- 因此这组是当前 P7 机制/remote 重建最好的候选，不是综合 ordinary reconstruction 最好的默认导出候选。

## P7 P5H film protected weak-ranking 短跑

新增模型：

```text
vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank005_e8_b8_2gpu/checkpoint-final.pth
```

配置差异：

```text
BASE_CKPT=P7-P5E aux allcities final
FUSION_TYPE=film
PROTECT_ORDINARY_HEADS=true
remote_control_ranking_loss_weight=0.05
remote_control_ranking_modes=[blank,shuffled]
EPOCHS=8
```

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5H film rank005 | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 |
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.2971 | 10.11 | 0.0495 | 0.0499 | 0.0510 | 0.0004 |

解读：

- P7-P5H film protected weak-ranking 没有带来可测提升：`joint_global 0.046629 -> 0.046639`，`joint_point 0.049535 -> 0.049543`，same-vs-blank delta `0.000412 -> 0.000401`。
- 训练中 late gate 约 0.028，ranking weighted loss 约 0.0009，信号太弱；adapter-only 的 P5H 后接训练基本没有改变普通视角重建行为。
- 这条结果说明，单独给 late fusion 加很弱的 blank/shuffled ranking 不是当前最有效方向。要增强卫星内容依赖，需要更直接的 paired remote-to-aerial 监督或更强的融合路径，而不是只训练小 gate adapter。

### strong-ranking 复核

新增模型：

```text
vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final
```

训练 checkpoint：

```text
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5h_film_protected_from_p5e_aux_rank05_gate005_e6_b8_2gpu/checkpoint-final.pth
```

相对 weak-ranking，改动是 `remote_control_ranking_loss_weight=0.5`、`LATE_GATE_INIT=0.05`、`EPOCHS=6`。训练后 late gate 确实更大：`0.028 -> 0.044`，ranking weighted loss 约 `0.0115`，但 benchmark 仍没有可测变化：

| model | joint global | joint point | joint AUC5 | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5H rank05 gate005 | 0.0466 | 0.0495 | 93.00 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5H rank005 | 0.0466 | 0.0495 | 93.00 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |
| P7-P5E aux allcities | 0.0466 | 0.0495 | 93.00 | 0.0495 | 0.0499 | 0.0510 | 0.0004 | 0.0015 |

结论：把 ranking 权重提高 10 倍、gate 初值提高到 0.05 后，adapter 参数确实变化，但对统一 benchmark 的 ordinary reconstruction 和 satellite delta 仍不可测。adapter-only P5H ranking 路线可以暂时停止。

## P7 P5B private oldP7 + pmgrad / P5B-anchor 复核

新增模型：

```text
vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_final
vggt_p7_p5b_parallel_token_aux_p5b_anchor_h035_final
```

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | same | blank | shuffled | blank delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7-P5B oldP7 private pmgrad05 | 0.0446 | 0.0485 | 94.67 | 0.2995 | 10.47 | 0.0485 | 0.0495 | 0.0498 | 0.0011 |
| P7-P5B oldP7 private h035 best | 0.0446 | 0.0485 | 95.33 | 0.2957 | 10.53 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| P7-P5B lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 0.0484 | 0.0495 | 0.0499 | 0.0011 |
| P7-P5B p5b-anchor h035 final | 0.0523 | 0.0560 | 86.33 | 0.4257 | 17.62 | 0.0560 | 0.0567 | 0.0576 | 0.0007 |
| P7 diagnostic oldP7 trunk + P5B remote head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.27 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| P7 freeze P5B remote head h035 final | 0.0466 | 0.0501 | 95.67 | 0.2856 | 10.16 | 0.0501 | 0.0509 | 0.0514 | 0.0008 |
| P7 diagnostic P5B-head aux-only h035 best | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.27 | 0.0485 | 0.0494 | 0.0497 | 0.0009 |
| P7 diagnostic P5B-head height001 final | 0.0467 | 0.0497 | 95.33 | 0.2862 | 10.16 | 0.0497 | 0.0506 | 0.0507 | 0.0009 |
| P7 diagnostic P5B-head height003 final | 0.0467 | 0.0497 | 95.67 | 0.2891 | 10.16 | 0.0497 | 0.0507 | 0.0507 | 0.0010 |
| P7 diagnostic P5B-head height001 trunklr5e8 final | 0.0474 | 0.0489 | 95.33 | 0.2906 | 10.24 | 0.0489 | 0.0499 | 0.0501 | 0.0010 |
| P7 diagnostic P5B-head height001 trunklr5e8 e6 final | 0.0474 | 0.0495 | 94.67 | 0.2912 | 10.20 | 0.0495 | 0.0504 | 0.0505 | 0.0009 |
| P7 diagnostic P5B-head height001 zdist2 trunklr5e8 final | 0.0475 | 0.0492 | 95.33 | 0.2913 | 10.22 | 0.0492 | 0.0502 | 0.0505 | 0.0010 |
| P7 diagnostic P5B-head height001 zhigh2q80 trunklr5e8 final | 0.0475 | 0.0492 | 94.33 | 0.2895 | 10.21 | 0.0492 | 0.0503 | 0.0505 | 0.0010 |
| P7 diagnostic P5B-head height001 trunklr1e7 final | 0.0472 | 0.0494 | 94.33 | 0.2903 | 10.21 | 0.0494 | 0.0502 | 0.0504 | 0.0009 |

448 remote-only PLY 统计显示：

| model | same z mean/std | blank z mean/std | 备注 |
|---|---:|---:|---|
| P5B original | 0.9058 / 0.0456 | 0.5571 / 0.0972 | remote 点云高度分布较合理 |
| P7 oldP7 private pmgrad05 | 0.8540 / 0.0428 | 0.4383 / 0.1088 | 与旧 oldP7 private 基本相同，仍整体压低 |
| P7 p5b-anchor h035 | 0.9053 / 0.0457 | 0.5565 / 0.0977 | 几乎完整保住 P5B remote 点云 |
| P7 diagnostic P5B-head aux-only h035 | 0.8980 / 0.0479 | 0.4461 / 0.1137 | 当前 P7 视觉 remote-only 最稳候选，benchmark 等同 diagnostic |
| P7 diagnostic P5B-head height001 | 0.8928 / 0.0471 | 0.4376 / 0.1139 | 小幅高于 freeze remote head，但仍低于 diagnostic/aux-only |
| P7 diagnostic P5B-head height003 | 0.8927 / 0.0472 | 0.4371 / 0.1137 | 与 height001 基本相同，未进一步恢复高度 |
| P7 diagnostic P5B-head height001 trunklr5e8 | 0.8967 / 0.0477 | 0.4439 / 0.1140 | 降低 trunk LR 后高度更接近 diagnostic/aux-only，但仍未超过 |
| P7 diagnostic P5B-head height001 trunklr5e8 e6 | 0.8953 / 0.0476 | 0.4423 / 0.1140 | 加长训练改善 val pointmap，但 remote-only 高度略低于 e3，benchmark 未超过 e3 |
| P7 diagnostic P5B-head height001 zdist2 trunklr5e8 | 0.8965 / 0.0476 | 0.4436 / 0.1139 | z mean/std 约束改善 aux 日志，不改善实际 remote-only PLY 或 benchmark |
| P7 diagnostic P5B-head height001 zhigh2q80 trunklr5e8 | 0.8965 / 0.0476 | 0.4440 / 0.1140 | 高 z 区域额外 pointmap z L1 仍未改善实际 remote-only PLY |
| P7 diagnostic P5B-head height001 trunklr1e7 | 0.8953 / 0.0476 | 0.4420 / 0.1139 | 高度介于 height001 和 trunklr5e8 之间，benchmark 未超过 trunklr5e8 |

解读：

- `pmgrad05` 没有解决 remote 点云高度压低问题；benchmark 与 oldP7 private h035 基本持平，且 AUC 略差。
- `p5b-anchor` 用 P5B warmstart、极低 trunk LR、冻结 remote point head/patch embed，只训练 parallel-token aux head。它成功保住 P5B remote-only 可视化高度分布，并且训练/验证中 rel-height aux 可学。
- 但 `p5b-anchor` 的 New York mini benchmark 明显退化，说明它更适合作为“projection aux 可学且不必破坏 P5B remote head”的诊断，不适合作为当前默认重建模型。
- `diagnostic P5B-head aux-only` 保住了 diagnostic remote-only 高度，并把 fixed-token projection aux 训练到 `val aux loss=0.0800`，但 benchmark 不变；它是当前视觉 remote 点云候选，不是新的综合 benchmark best。
- `height001` 相比 freeze remote head 小幅改善：remote-only z mean `0.8913 -> 0.8928`，`joint_point 0.0501 -> 0.0497`，RS-only MAE 基本持平略好；但还没回到 diagnostic/aux-only 的 `0.8980/0.0485`。
- `height003` 与 `height001` 基本持平：remote-only z mean `0.8927`，`joint_point=0.0497`，说明单纯继续加大 absolute height 权重已经接近平台。
- `height001 trunklr5e8` 比 `height001/003` 更好地保住 remote-only 高度：z mean 回升到 `0.8967`，benchmark `joint_point=0.0489` 也优于 `0.0497`；但它仍略低于 diagnostic/aux-only 的 `0.0485/0.8980`，说明降低 trunk LR 是有效方向但不是最终解。
- `height001 trunklr1e7` 的验证 pointmap 收敛快于 `trunklr5e8`，但 PLY 高度回落到 `0.8953`，benchmark `joint_point=0.0494`、AUC5 `94.33` 也不如 `trunklr5e8`；因此不是当前 best。
- `height001 trunklr5e8 e6` 把训练验证 pointmap 从 e3 的 `0.0562` 继续降到 `0.0495`，aux 高区也对齐到 `0.1516/0.1510`；但 New York benchmark 退到 `joint_point=0.0495`、AUC5 `94.67`，remote-only z mean 也从 `0.8967` 降到 `0.8953`。因此不应继续单纯加长低 LR 训练。
- `height001 zdist2 trunklr5e8` 让 val high20 rel-height 从 epoch1 的 `0.1895/0.1510` 校到 epoch3 的 `0.1511/0.1510`，但 448 remote-only PLY 几乎等同 e3，benchmark `joint_point=0.0492` 也未超过 `trunklr5e8 e3` 的 `0.0489`。mean/std 型 z 分布约束过粗，不足以修复 remote 高区飞点。
- `height001 zhigh2q80 trunklr5e8` 在 GT z top 区域上额外监督 pointmap z，val high-z loss 从 `0.0506` 降到 `0.0447`，但 PLY 与 zdist2/e3 基本相同，benchmark `joint_point=0.0492`、AUC5 `94.33` 也未超过 e3。说明当前问题不是简单的 pointmap 子集重加权能解决。
- 当前默认候选仍不应换成 anchor/aux-only；综合 benchmark 仍优先 oldP7 private h035 / P7-P5B lowtrunklr / P7-P5E lowtrunkfull，视觉 remote 点云则优先 diagnostic P5B-head aux-only。

## P7 oldP7 aggtail2 remote-head + z-gradient 复核

新增模型：

```text
vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_final
vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_e4_final
```

训练设置：

- warmstart：`p7_diagnostic_oldp7_trunk_p5b_remote_head/checkpoint-final.pth`
- 可训练参数：`remote_point_head`，parallel token projection aux head，`model.aggregator.frame_blocks/global_blocks.{22,23}`，其余普通 head 和大部分 trunk 冻结。
- remote pointmap：`LAMBDA_REMOTE_PM=4.0`，`LAMBDA_REMOTE_RAW_PM=0.001`，`LAMBDA_REMOTE_PM_GRAD=0.05`，gradient channels=`z`。
- aux：tokens source，hidden 96，6 blocks，rel height 0.35，offset 0.75，global slope 0.05。
- anchor：`remote_point_head_param_anchor_loss_weight=500000`，relative L2。

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | joint RS MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7 oldP7 aggtail2 e4 final | 0.0448 | 0.0486 | 95.67 | 0.2954 | 10.07 | 16.67 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 oldP7 aggtail2 e2 final | 0.0450 | 0.0486 | 95.67 | 0.2958 | 9.72 | 16.65 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7-P5B oldP7 private h035 final | 0.0446 | 0.0485 | 95.33 | 0.2957 | 10.53 | 16.31 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7-P5B lowtrunklr final | 0.0471 | 0.0484 | 95.33 | 0.3009 | 9.94 | 16.50 | 0.0484 | 0.0495 | 0.0499 | 0.0011 | 0.0014 |
| P7-P5E lowtrunkfull final | 0.0460 | 0.0491 | 93.67 | 0.2990 | 9.60 | 16.90 | 0.0491 | 0.0500 | 0.0507 | 0.0008 | 0.0016 |
| P7 diagnostic oldP7 trunk + P5B remote head | 0.0475 | 0.0485 | 95.33 | 0.2957 | 10.27 | 16.86 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |

448 remote-only PLY 统计：

| model | same z mean/std | blank z mean/std | 输出目录 |
|---|---:|---:|---|
| P7 oldP7 aggtail2 e4 final | 0.8473 / 0.0454 | 0.4080 / 0.1066 | `debug/plyview/448/vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_e4_final` |
| P7 oldP7 aggtail2 e2 final | 0.8545 / 0.0465 | 0.4110 / 0.1066 | `debug/plyview/448/vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_final` |
| P7 diagnostic P5B-head aux-only h035 | 0.8980 / 0.0479 | 0.4461 / 0.1137 | 旧复核输出 |

解读：

- e4 相比 e2 的训练内 test remote_loss 从 `0.1805` 小幅到 `0.1784`，但 benchmark `same` remote pointmap 基本不变：`0.0485505 -> 0.0485519`。
- e4 的 `joint_global` 略好于 e2：`0.0450 -> 0.0448`，但 `RS-only MAE` 退化：`9.72 -> 10.07`，`joint RS MAE` 也没有改善。
- e2/e4 都保持 `same < blank < shuffled`，说明 remote 图像内容仍有可测贡献；但 aggtail2 没有扩大 satellite delta。
- e4 remote-only z mean 从 e2 的 `0.8545` 进一步降到 `0.8473`，低于 diagnostic P5B-head aux-only 的 `0.8980`。这与用户观察的高区 remote 点云飞掉/高度不稳一致：benchmark 小幅波动不能代表局部形状已经修复。
- 因此 aggtail2 低 LR 微调可以改善 global 对齐，但不是更好的 remote 点云模型；继续单纯拉长训练没有价值。

## P7 MoGe2 remote prior 快速验证

新增模型：

```text
vggt_p7_moge2_balanced20x4_private_tokens_mogegrad001_edge0002_fixed_final
vggt_p7_moge2_balanced20x4_private_tokens_mogegrad001_edge0002_fixed_best
```

训练设置：

- warmstart：`p7_oldp7_train_remotehead_nonreentrant_aggtail2lr1e7_raw001_gradz005_paramanchor500k_lowlr3e6_h003_e2_b24_4gpu/checkpoint-final.pth`
- 训练数据：4 城 balanced 80 scene list，每城 20 scene，Google/Bing remote provider；MoGe2 prior 使用 `moge_residual_p95 <= 30` 质量门控。
- 可训练参数：`remote_point_head`，parallel token projection aux head，`model.aggregator.frame_blocks/global_blocks.{22,23}`，其余普通 head 和大部分 trunk 冻结。
- remote pointmap：`LAMBDA_REMOTE_PM=4.0`，`LAMBDA_REMOTE_RAW_PM=0.001`，`LAMBDA_REMOTE_PM_GRAD=0.05`，gradient channels=`z`。
- MoGe2 prior：`LAMBDA_REMOTE_MOGE_GRAD=0.01`，`LAMBDA_REMOTE_MOGE_EDGE=0.002`，`REMOTE_MOGE_PRIOR_MIN_WEIGHT=0.03`。
- aux：tokens source，hidden 96，6 blocks，rel height 0.35，offset 0.75，global slope 0.05。
- anchor：`remote_point_head_param_anchor_loss_weight=500000`，relative L2。

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | joint RS MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7 MoGe2 final | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.97 | 16.72 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 MoGe2 best | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.96 | 16.72 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 oldP7 aggtail2 e4 final | 0.0448 | 0.0486 | 95.67 | 0.2954 | 10.07 | 16.67 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 oldP7 aggtail2 e2 final | 0.0450 | 0.0486 | 95.67 | 0.2958 | 9.72 | 16.65 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |

输出：

```text
benchmark summary:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls/p7_moge2_fixed_summary

448 PLY:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/p7_moge2_balanced20x4_private_tokens_mogegrad001_edge0002_fixed_final
```

解读：

- 训练稳定，MoGe prior 确认进入 loss；训练中 `rs_moge_required_present=1`，有效 prior ratio 约 `0.23-0.36`，weighted MoGe loss 约 `0.015-0.018`，没有压过主 pointmap 监督。
- `final` 和 `best` benchmark 几乎一致，说明这个 30 epoch/80 scene 快速验证已经到小样本平台。
- MoGe2 prior 没有证明能显著降低 remote 点云高度误差：`RS-only MAE=9.96/9.97`，弱于 oldP7 aggtail2 e2 的 `9.72`，强于 e4 的 `10.07`。
- `joint_global=0.0448` 略好于 oldP7 aggtail2 e2/e4，但幅度很小；`same < blank < shuffled` 仍成立，卫星内容收益没有扩大。
- 因此 MoGe2 作为弱局部梯度/边缘先验是可训练的，但当前小样本/低权重设置没有给出“remote 局部形状明显修复”的可靠证据。是否视觉更好需要查看同目录 `mapanything_pointcloud_same_remote.ply`。

## P7 projection-MoGe aux 快速验证

新增模型：

```text
vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final
vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_best
```

训练设置：

- warmstart：`p7_oldp7_train_remotehead_nonreentrant_aggtail2lr1e7_raw001_gradz005_paramanchor500k_lowlr3e6_h003_e2_b24_4gpu/checkpoint-final.pth`
- 训练数据：4 城 balanced 80 scene list，每城 20 scene，Google/Bing remote provider；MoGe2 prior 使用 `moge_residual_p95 <= 30` 质量门控。
- 可训练参数：`remote_point_head`，parallel token projection aux head，`model.aggregator.frame_blocks/global_blocks.{22,23}`，其余普通 head 和大部分 trunk 冻结。
- remote pointmap MoGe prior 关闭：`LAMBDA_REMOTE_MOGE_GRAD=0`，`LAMBDA_REMOTE_MOGE_EDGE=0`。
- projection-aux MoGe prior 打开：`LAMBDA_PROJ_MOGE_GRAD=0.02`，`LAMBDA_PROJ_MOGE_EDGE=0.005`，`PROJ_MOGE_PRIOR_MIN_WEIGHT=0.03`。
- aux：tokens source，hidden 96，6 blocks，rel height 0.35，offset 0.75，global slope 0.05。

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | joint RS MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P7 projection-MoGe aux final | 0.0448 | 0.0485 | 95.67 | 0.2958 | 10.10 | 16.73 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 projection-MoGe aux best | 0.0448 | 0.0486 | 95.67 | 0.2956 | 10.01 | 16.70 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 MoGe2 remote prior final | 0.0448 | 0.0485 | 95.33 | 0.2955 | 9.97 | 16.72 | 0.0485 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |
| P7 oldP7 aggtail2 e2 final | 0.0450 | 0.0486 | 95.67 | 0.2958 | 9.72 | 16.65 | 0.0486 | 0.0494 | 0.0497 | 0.0009 | 0.0012 |

输出：

```text
checkpoint:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu/checkpoint-final.pth

benchmark:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final

PLY:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final
```

解读：

- projection-MoGe aux 训练稳定，训练中 `rs_projection_moge_required_present=1`，有效 prior ratio 约 `0.25-0.38`，说明 MoGe prior 确实进入 projection-aux height 分支。
- `joint_global=0.0448` 进入当前表内 Top 1/2，但它与前一个 MoGe2 remote prior/oldP7 aggtail2 系列几乎同平台，主要是继续保持了 oldP7 private remote-head 的综合指标。
- 它没有改善 remote-only 高度：`RS-only MAE=10.01/10.10`，弱于 oldP7 aggtail2 e2 的 `9.72`，也弱于 MoGe2 remote prior final 的 `9.97`。
- 因此“MoGe 辅助 projection-aux height”这个方向工程上可行，但当前弱 edge/gradient prior 没有证明能修复 remote 局部形状；如果可视化仍无明显变化，下一步不应只加长这一路线，而应尝试更强的可控结构监督或把多任务头迁移到更强 base 模型。

## PI3 projection-aux MoGe 迁移验证

新增模型：

```text
pi3_p7_remote_head_projection_aux_moge_final
```

训练设置：

- warmstart：`pi3/p3_pi3_modality_embedding_remote_head/checkpoint-final.pth`
- 结构：`Pi3ModalityEmbeddingRemoteHeadWrapper`，保留 PI3 remote private point decoder/head；新增并行 `remote_projection_aux` token head，输入 PI3 shared decoder tokens、remote RGB 和坐标，输出 `rel_height`、`offset_xy`、`global_dir_xy`、`global_slope`。
- 训练数据：4 城 balanced 80 scene list，每城 20 scene；Google/Bing remote provider。
- loss：remote pointmap weight `6.0`，projection rel height `0.35`，offset `0.75`，global slope `0.05`；projection-aux MoGe prior 使用 `grad=0.10`、`edge=0.02`、`min_weight=0.05`。
- 训练资源：4 GPU，`BATCH_SIZE=8` 已达到约 79GB/卡峰值；保存策略保留 `best/final`，删除 `last` 以节省存储。

关键 benchmark 结果：

| model | joint global | joint point | joint AUC5 | joint ray | RS-only MAE | joint RS MAE | same | blank | shuffled | blank delta | shuffled delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PI3 projection-aux MoGe final | 0.0561 | 0.0576 | 87.67 | 0.3916 | 23.96 | 23.77 | 0.0576 | 0.0581 | 0.0598 | 0.0005 | 0.0021 |
| PI3 P3 modality embedding | 0.0548 | 0.0576 | 89.00 | 0.3711 | - | - | 0.0576 | 0.0573 | - | -0.0003 | - |
| PI3 P3 remote head | 0.0568 | 0.0576 | 86.33 | 0.3778 | - | - | 0.0576 | 0.0570 | - | -0.0006 | - |

输出：

```text
checkpoint:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p7_pi3_remote_head_projection_aux_moge_balanced20x4_grad010_edge002_e30_b8_4gpu/checkpoint-final.pth

benchmark:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/crossview_all_models_4v_mini_controls/pi3_p7_remote_head_projection_aux_moge_final_10scene

PLY:
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/pi3_p7_remote_head_projection_aux_moge
/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/pi3_p7_remote_head_projection_aux_moge
```

解读：

- 这轮不是候选模型。`joint_global=0.0561` 介于 PI3 P3 modality embedding 和 PI3 P3 remote head 附近，但显著弱于当前 VGGT P7 系列；`RS-only MAE=23.96`、`joint RS MAE=23.77` 很差。
- 训练日志显示 projection aux `rel_height_pred_abs_mean` 从约 `0.097` 缓慢降到约 `0.082`，高区 GT 常在几十米量级，说明 aux height 基本停在小常数解；MoGe gradient/edge loss 虽下降，但没有把显式高度拉起来。
- `same < blank < shuffled` 成立，说明 remote 图像内容仍有一点贡献；但这个贡献没有转化为可靠 remote 点云高度。
- 结论：把多任务思路直接迁移到 PI3 是可运行的，但当前 aux head 输出尺度/初始化和 height 目标不匹配。下一步不应继续加长这条训练，而应改 `rel_height` 参数化，例如引入目标归一化、可学习输出尺度/非零初始化，或先只训练 normalized height head 让 aux 分支脱离常数解。

## 初步结论

- 绝对 `joint_global_pointmaps_abs_rel` 当前最好的一组来自 P7-P5B/private/oldP7/MoGe2 系列；projection-MoGe aux final/best 达到 `joint_global=0.0448`、`joint_point=0.0485/0.0486`，但仍未证明改善 remote 高度误差。
- 新增 oldP7 private `pmgrad05` 没有带来有效视觉改善；`p5b-anchor` 保住了 P5B remote 点云高度，但 benchmark 明显退化，不能作为默认权重。`diagnostic P5B-head aux-only` 保住了 `z_mean=0.8980` 的 remote-only 高度并有可用 aux 输出，但 benchmark 等同 diagnostic，仍不是综合 best。`height001` 是小幅正向的折中实验，`height003` 未进一步改善；`height001 trunklr5e8` 进一步保住高度并恢复部分 benchmark，`trunklr1e7`、`trunklr5e8 e6`、`zdist2 trunklr5e8`、`zhigh2q80 trunklr5e8` 都未超过它。`oldP7 aggtail2 e4` 只小幅改善 `joint_global`，没有改善 `same` remote pointmap 或 remote-only 高度，继续加长训练价值不高。MoGe2 prior 可训练、未破坏 joint 指标，但在当前快速验证中也没有显著降低 RS-only MAE；下一步应控制 trunk/feature 漂移、加入 teacher-style output anchor，或扩大质量门控后的 MoGe prior 数据量并复核视觉 PLY，而不是继续提高 height 权重、加长训练、只约束全局 z 分布或对 pointmap 子集重加权。
- 新增的全城市 P7-P5B projection-aux 是当前 P7 主联训路线里 point/AUC 最强的一组：`joint_global=0.0474`，`joint_point=0.0485/0.0486`，`pose_auc_5=95.33/95.67`。它超过两城 P7，并进入绝对精度 Top 10。
- P7-P5E lowtrunkfull 把 P7 路线 `joint_global` 进一步推到 0.0460，且 RS-only MAE 最好；但 `joint_point=0.0491`、`AUC=93.67` 仍不如 P7-P5B lowtrunklr final 的 `0.0484/95.33`，更像是改善 remote 重建和全局对齐，而不是全面提升 ordinary reconstruction。
- P7-P5E projection-aux 原始冻结普通头版本 `joint_global=0.0466`，`joint_point/AUC/satellite delta` 不如 P7-P5B allcities；P7-P5H film protected weak/strong ranking 基本复现旧 P7-P5E，没有改善 satellite delta。
- P7 全城市和两城都满足 `same < blank < shuffled`，说明正确卫星图内容有可测收益；但全城市 same-vs-blank delta 只有 `0.0004-0.0008`，低于两城 P7 的 `0.0016`，更低于 p5e/p5h 和 VGGT-Omega p2。
- 追加的 `vggt_omega_p2_joint_depth_512_all_2` 是 VGGT-Omega 系列当前最强结果：`joint_global=0.0492`，`joint_point=0.0508`，`pose_auc_5=97.00`，`joint_ray=0.2006`；相比旧 `p1 all`，joint point 提升 9.97%，joint global 提升 7.61%。
- 对卫星图最敏感、same 相比 blank 的 point 提升最大的是 `vggt_p5g_no_fusion_fixedfreeze_protected`，`delta=0.0069`，相对提升 +4.20%。
- 解读卫星输入是否有效时，优先看 `same` 是否同时优于 `blank` 和 `shuffled`。如果只优于 blank 但接近 shuffled，可能更多是额外 remote token/分布扰动带来的收益，而不是正确利用真实卫星内容。
- 这是 10 scene mini benchmark，适合快速筛选；最终模型选择仍建议用更大 paired scene 数量或完整 benchmark 复核。
