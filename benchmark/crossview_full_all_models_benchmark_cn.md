# Crossview 全量模型统一评测

本文档只保留当前正式评测的关键结论。完整逐模型、逐 scene 原始结果保存在评测输出目录；本文档不展开原始 JSON 的所有字段。

## 评测口径

- 结果根目录：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8`。
- 汇总文件：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8/summary_key_metrics.csv`、`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8/summary_key_metrics.json`、`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8/failed_jobs.json`。
- 数据与流程：沿用 `Models/map-anything/docs/crossview_all_models_benchmark_cn.md` 的 Crossview 评测流程，使用全部测试样本选择口径中的 `n_scenes=2` 与 `n_scenes=8` 两个典型值。
- 输入：`NUM_VIEWS=4`；只评测真实匹配卫星图的 Joint 输入，不做 `same`、`blank`、`shuffled` 对照。
- 批量与并发：`BATCH_SIZE=8`，`CUDA_VISIBLE_DEVICES=0,1`，`workers=4`。
- 模型范围：`outputs/mapanything_experiments/mapanything/training/Crossview` 下可发现的 checkpoint，加上未微调基线 `pi3_raw_pretrained_image_input` 与 `vggt_raw_pretrained_image_input`。
- 成功结果：245 个模型在 `n_scenes=2` 与 `n_scenes=8` 均有结果，共 490 条汇总记录。
- 失败结果：`vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best` 在两个 n_scenes 下均失败，原因是 checkpoint zip archive 损坏。

指标口径：

- Aerial-only：`aerial_pointmaps_abs_rel`、`aerial_pose_auc_5` 等。
- Joint aerial：`joint_pointmaps_abs_rel`、`joint_pose_auc_5` 等。
- Remote-only：`remote_pointmaps_abs_rel` 映射自 `rs_only.average.rs_point_abs_rel`。
- Joint remote：`joint_remote_pointmaps_abs_rel` 映射自 `joint.average.rs_point_abs_rel`。
- 当前 `rs_point_abs_rel` 已按 remote 自身作为 view0，并使用和 dense aerial `pointmaps_abs_rel` 一致的 `avg_dis` 尺度归一化；旧全局坐标辅助值保存在 `*_abs_rel_global_aux`，不进入主榜。
- 卫星输入收益：误差类用 `aerial - joint`，越大表示加入卫星图后变好；AUC 用 `joint - aerial`。

## 记录的模型

| family | 模型数 | 主要覆盖特点 |
|---|---:|---|
| pi3 | 17 | P3 baseline/embedding; projection aux; MoGe prior; 未微调基线 |
| vggt | 229 | P5B/shared-norm; view-type; remote point-head; private remote head; overlap loss; teacher; stage training; no-crop; parallel-token aux; projection aux; film/protected fusion; Crossview checkpoint |

补充：两个未微调基线均已纳入：`pi3_raw_pretrained_image_input`、`vggt_raw_pretrained_image_input`。

## 绝对精度 Top 10

排序：按 `joint_pointmaps_abs_rel` 从低到高；并列时参考 `joint_remote_pointmaps_abs_rel` 与 `joint_pose_auc_5`。

### n_scenes=2

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | vggt | `vggt_p7_frombest_gatedres_preagg_e3_b8_2gpu_final` | Crossview checkpoint | 0.0500 | 95.0000 | 0.2915 | 1.8672 | 1.9732 |
| 2 | vggt | `vggt_p7_p5b_shared_norm_projection_aux_full_2city` | P5B/shared-norm; projection aux | 0.0505 | 90.0000 | 0.4204 | 1.2893 | 1.2903 |
| 3 | vggt | `vggt_p7_proj_views8_from_robustoverlap_e3_b4_2gpu_final` | projection aux; overlap loss | 0.0508 | 93.3333 | 0.3246 | 1.8667 | 1.9743 |
| 4 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0509 | 91.6667 | 0.3659 | 1.8696 | 1.9750 |
| 5 | vggt | `vggt_p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu_best` | P5B/shared-norm; NewYork subset; Chicago subset | 0.0510 | 90.0000 | 0.4224 | 1.2897 | 1.2903 |
| 6 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head | 0.0514 | 91.6667 | 0.4160 | 1.9067 | 1.9749 |
| 7 | vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final` | P5B/shared-norm; projection aux | 0.0516 | 93.3333 | 0.3896 | 1.2961 | 1.2928 |
| 8 | vggt | `vggt_p7_proj_denseh015_highq50_from_robustoverlap_e4_b8_2gpu_final` | projection aux; overlap loss | 0.0516 | 91.6667 | 0.4021 | 1.8667 | 1.9736 |
| 9 | vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best` | P5B/shared-norm; projection aux | 0.0516 | 93.3333 | 0.3920 | 1.2963 | 1.2929 |
| 10 | vggt | `vggt_p7_frombest_gridglobal_selfcontained_e2_b8_2gpu_best` | Crossview checkpoint | 0.0518 | 93.3333 | 0.3025 | 1.8670 | 1.9727 |

### n_scenes=8

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | vggt | `vggt_p7_frombest_gatedres_preagg_e3_b8_2gpu_final` | Crossview checkpoint | 0.0397 | 93.7500 | 0.3301 | 1.8673 | 1.9757 |
| 2 | vggt | `vggt_p7_proj_views8_from_robustoverlap_e3_b4_2gpu_final` | projection aux; overlap loss | 0.0399 | 94.5833 | 0.2851 | 1.8672 | 1.9771 |
| 3 | vggt | `vggt_p7_proj_moge_agglr1e7_private_tokens_warmprivbest_e10_b8_4gpu_final` | projection aux; MoGe prior | 0.0399 | 95.4167 | 0.2848 | 1.8684 | 1.9788 |
| 4 | vggt | `vggt_p7_proj_moge_agglr1e7_private_tokens_warmprivbest_e10_b8_4gpu_best` | projection aux; MoGe prior | 0.0399 | 95.0000 | 0.2862 | 1.8688 | 1.9788 |
| 5 | vggt | `vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_e4_final` | Crossview checkpoint | 0.0401 | 95.8333 | 0.2887 | 1.8696 | 1.9784 |
| 6 | vggt | `vggt_p7_diagnostic_oldp7_frozen_trunk_train_remotehead_aux_h035_height001_final` | diagnostic | 0.0401 | 95.4167 | 0.2893 | 1.8771 | 1.9786 |
| 7 | vggt | `vggt_p7_diagnostic_oldp7_trunk_p5b_remote_head_final` | P5B/shared-norm; diagnostic | 0.0401 | 95.4167 | 0.2893 | 1.8771 | 1.9786 |
| 8 | vggt | `vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best` | P5B/shared-norm; diagnostic | 0.0401 | 95.4167 | 0.2893 | 1.8771 | 1.9786 |
| 9 | vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_lowlr3e6_h003_e2_b32_4gpu_final` | Crossview checkpoint | 0.0401 | 95.4167 | 0.2893 | 1.8656 | 1.9781 |
| 10 | vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_paramanchor5k_lowlr3e6_h003_e2_b32_4gpu_final` | Crossview checkpoint | 0.0401 | 95.4167 | 0.2893 | 1.8661 | 1.9782 |

## 卫星输入收益 Top 10

排序：按 `aerial_pointmaps_abs_rel - joint_pointmaps_abs_rel` 从高到低。`remote_point_rel` 是 Remote-only 到 Joint remote 的相对改善，正数表示 Joint remote 更好。

### n_scenes=2

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_best` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 2 | vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_final` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 3 | vggt | `vggt_p7_dptinit_globalrecon_normgt_nooffset_2city_e12_b16_2gpu_best` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 4 | vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_best` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 5 | vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_final` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 6 | vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_best` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 7 | vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_final` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 8 | vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_best_vggtbase_only` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 9 | vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_best` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |
| 10 | vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_final` | 0.1970 | 0.1875 | 0.0096 | +4.85% | 48.3333 | 61.6667 | 13.3333 | 1.9710 | 1.9008 | 0.0702 | +3.56% |

### n_scenes=8

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_best` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 2 | vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_final` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 3 | vggt | `vggt_p7_dptinit_globalrecon_normgt_nooffset_2city_e12_b16_2gpu_best` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 4 | vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_best` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 5 | vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_final` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 6 | vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_best` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 7 | vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_final` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 8 | vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_best_vggtbase_only` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 9 | vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_best` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |
| 10 | vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_final` | 0.1759 | 0.1639 | 0.0120 | +6.83% | 37.5000 | 41.6667 | 4.1667 | 1.9769 | 1.8679 | 0.1090 | +5.51% |

## 全量结果

每个模型一行，只展示最重要的可视化字段；完整字段见输出目录下的 `summary_key_metrics.csv/json` 和各模型 `rs_aerial_benchmark_results.json`。

| family | record_label | ckpt_type | 主要特点 | n2_status | n2_joint_point | n2_joint_auc5 | n2_remote | n2_joint_remote | n2_point_gain | n8_status | n8_joint_point | n8_joint_auc5 | n8_remote | n8_joint_remote | n8_point_gain |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| pi3 | `pi3_p3_base` | - | P3 baseline/embedding | ok | 0.0665 | 85.0000 | 1.4725 | 1.4797 | 0.0022 | ok | 0.0528 | 82.9167 | 1.4846 | 1.5248 | -0.0014 |
| pi3 | `pi3_p3_freeze_shared` | - | P3 baseline/embedding | ok | 0.1190 | 90.0000 | 1.4696 | 1.4575 | 0.0046 | ok | 0.0710 | 78.7500 | 1.4825 | 1.5084 | -0.0008 |
| pi3 | `pi3_p3_modality_embedding` | - | P3 baseline/embedding | ok | 0.0592 | 91.6667 | 1.4694 | 1.4631 | 0.0013 | ok | 0.0479 | 89.5833 | 1.4787 | 1.4995 | -0.0004 |
| pi3 | `pi3_p3_modality_embedding_remote_head` | - | P3 baseline/embedding | ok | 0.0593 | 88.3333 | 1.6493 | 1.6092 | 0.0031 | ok | 0.0495 | 85.4167 | 1.6635 | 1.6081 | -0.0006 |
| pi3 | `pi3_p3_pi3_base_final` | final | P3 baseline/embedding | ok | 0.0643 | 86.6667 | 1.4737 | 1.4812 | 0.0027 | ok | 0.0494 | 89.1667 | 1.4836 | 1.5237 | -0.0002 |
| pi3 | `pi3_p3_pi3_freeze_shared_final` | final | P3 baseline/embedding | ok | 0.1181 | 90.0000 | 1.4705 | 1.4607 | 0.0042 | ok | 0.0708 | 79.5833 | 1.4831 | 1.5084 | -0.0015 |
| pi3 | `pi3_p3_pi3_modality_embedding_final` | final | P3 baseline/embedding | ok | 0.0591 | 91.6667 | 1.4700 | 1.4640 | 0.0012 | ok | 0.0477 | 88.7500 | 1.4791 | 1.5004 | -0.0003 |
| pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | final | P3 baseline/embedding | ok | 0.0593 | 88.3333 | 1.6493 | 1.6092 | 0.0031 | ok | 0.0495 | 85.4167 | 1.6635 | 1.6081 | -0.0006 |
| pi3 | `pi3_p3_pi3_zero_covis_final` | final | P3 baseline/embedding | ok | 0.0582 | 85.0000 | 1.4754 | 1.4611 | 0.0010 | ok | 0.0491 | 83.7500 | 1.4861 | 1.5044 | -0.0007 |
| pi3 | `pi3_p3_zero_covis` | - | P3 baseline/embedding | ok | 0.0593 | 86.6667 | 1.4717 | 1.4545 | 0.0011 | ok | 0.0496 | 83.7500 | 1.4826 | 1.5013 | -0.0005 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_denseheight_h080_lr3e6_balanced20x4_e8_b8_4gpu_best` | best | projection aux | ok | 0.0627 | 95.0000 | 1.6319 | 1.6116 | 0.0022 | ok | 0.0538 | 89.1667 | 1.6446 | 1.6094 | -0.0027 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_denseheight_h080_lr3e6_balanced20x4_e8_b8_4gpu_final` | final | projection aux | ok | 0.0618 | 91.6667 | 1.6230 | 1.5923 | 0.0039 | ok | 0.0519 | 86.2500 | 1.6347 | 1.5956 | -0.0005 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_moge_balanced20x4_grad010_edge002_e30_b8_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0606 | 91.6667 | 1.6545 | 1.6313 | 0.0040 | ok | 0.0495 | 87.0833 | 1.6675 | 1.6317 | 0.0015 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_moge_balanced20x4_grad010_edge002_e30_b8_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0605 | 91.6667 | 1.6469 | 1.6233 | 0.0046 | ok | 0.0490 | 87.0833 | 1.6598 | 1.6249 | 0.0015 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_moge_relscale32_offsetscale8_warmbadfinal_e12_b8_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0618 | 91.6667 | 1.6486 | 1.6303 | 0.0020 | ok | 0.0547 | 80.8333 | 1.6589 | 1.6289 | -0.0015 |
| pi3 | `pi3_p7_pi3_remote_head_projection_aux_moge_relscale32_offsetscale8_warmbadfinal_e12_b8_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0618 | 91.6667 | 1.6486 | 1.6303 | 0.0020 | ok | 0.0547 | 80.8333 | 1.6589 | 1.6289 | -0.0015 |
| pi3 | `pi3_raw_pretrained_image_input` | raw | 未微调基线 | ok | 0.1536 | 76.6667 | 1.4496 | 1.4083 | 0.0067 | ok | 0.1606 | 47.0833 | 1.4718 | 1.4339 | -0.0388 |
| vggt | `vggt_p5b_shared_norm` | - | P5B/shared-norm | ok | 0.0561 | 88.3333 | 1.9764 | 1.8924 | 0.0004 | ok | 0.0480 | 86.6667 | 1.9788 | 1.8706 | 0.0009 |
| vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | final | P5B/shared-norm | ok | 0.0567 | 88.3333 | 1.9766 | 1.8917 | 0.0003 | ok | 0.0473 | 87.5000 | 1.9791 | 1.8699 | 0.0008 |
| vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | final | view-type | ok | 0.0568 | 83.3333 | 1.9765 | 1.8811 | -0.0009 | ok | 0.0461 | 89.5833 | 1.9788 | 1.8634 | 0.0005 |
| vggt | `vggt_p5c_viewtype` | - | view-type | ok | 0.0583 | 83.3333 | 1.9763 | 1.8802 | -0.0007 | ok | 0.0474 | 89.1667 | 1.9787 | 1.8622 | 0.0005 |
| vggt | `vggt_p5d_remote_point_head_consistency` | - | remote point-head | ok | 0.0574 | 85.0000 | 1.9756 | 1.8668 | 0.0002 | ok | 0.0481 | 86.6667 | 1.9784 | 1.8653 | 0.0004 |
| vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | final | remote point-head | ok | 0.0577 | 86.6667 | 1.9773 | 1.8942 | -0.0002 | ok | 0.0473 | 87.9167 | 1.9795 | 1.8732 | 0.0004 |
| vggt | `vggt_p5e_remote_head_attention_viewtype` | - | view-type; private remote head | ok | 0.0509 | 91.6667 | 1.9750 | 1.8696 | 0.0035 | ok | 0.0411 | 91.2500 | 1.9777 | 1.8716 | 0.0032 |
| vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | final | view-type; private remote head | ok | 0.0514 | 91.6667 | 1.9749 | 1.9067 | 0.0035 | ok | 0.0414 | 91.6667 | 1.9777 | 1.8796 | 0.0034 |
| vggt | `vggt_p7_allcities_fromrobust_stageA2teacher_pm4_overlap6_fullfinetune_e60_b8_2gpu_best` | best | overlap loss; teacher; stage training | ok | 0.0523 | 93.3333 | 1.2937 | 1.2945 | 0.0014 | ok | 0.0410 | 92.9167 | 1.3030 | 1.3051 | 0.0010 |
| vggt | `vggt_p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_to4v_e4_b8_2gpu_best` | best | P5B/shared-norm | ok | 0.0576 | 93.3333 | 1.2903 | 1.2931 | 0.0016 | ok | 0.0463 | 91.2500 | 1.3011 | 1.3043 | 0.0011 |
| vggt | `vggt_p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_warmbest_e4_b8_2gpu_best` | best | P5B/shared-norm | ok | 0.0552 | 85.0000 | 1.2927 | 1.2955 | 0.0001 | ok | 0.0427 | 86.2500 | 1.3029 | 1.3070 | 0.0008 |
| vggt | `vggt_p7_allcities_p5b_joint_pm4_aux_h075_lowover15_curric2v_warmbest_e4_b8_2gpu_final` | final | P5B/shared-norm | ok | 0.0547 | 93.3333 | 1.2920 | 1.2965 | 0.0002 | ok | 0.0415 | 94.5833 | 1.3023 | 1.3080 | 0.0013 |
| vggt | `vggt_p7_allcities_p5b_joint_pm4_aux_h075_lowover15_nocrop_warmbest_e8_b8_2gpu_final` | final | P5B/shared-norm; no-crop | ok | 0.0520 | 88.3333 | 1.2907 | 1.2975 | 0.0007 | ok | 0.0407 | 91.6667 | 1.3013 | 1.3097 | 0.0017 |
| vggt | `vggt_p7_allcities_p5b_parallel_token_aux_h075_lowtrunklr_warmbest_e4_b9_4gpu_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0523 | 91.6667 | 1.2927 | 1.2984 | 0.0011 | ok | 0.0406 | 94.5833 | 1.3023 | 1.3104 | 0.0015 |
| vggt | `vggt_p7_allcities_p5b_parallel_token_aux_h075_lowtrunklr_warmbest_e4_b9_4gpu_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0521 | 91.6667 | 1.2927 | 1.2976 | 0.0008 | ok | 0.0401 | 95.0000 | 1.3022 | 1.3095 | 0.0015 |
| vggt | `vggt_p7_allcities_p5b_parallel_token_aux_pm6_auxhalf_lowtrunklr_warmparfinal_e6_b9_4gpu_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0526 | 95.0000 | 1.2925 | 1.2960 | -0.0000 | ok | 0.0402 | 95.8333 | 1.3017 | 1.3077 | 0.0012 |
| vggt | `vggt_p7_allcities_p5b_parallel_token_aux_pm6_auxhalf_lowtrunklr_warmparfinal_e6_b9_4gpu_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0526 | 95.0000 | 1.2925 | 1.2960 | -0.0000 | ok | 0.0402 | 95.8333 | 1.3017 | 1.3077 | 0.0012 |
| vggt | `vggt_p7_allcities_p5e_private_viewtype_projection_aux_midtrunkfull_warmp5elowfull_e4_b9_2gpu_best` | best | view-type; private remote head; projection aux | ok | 0.0523 | 88.3333 | 1.9752 | 1.8698 | 0.0010 | ok | 0.0406 | 93.3333 | 1.9780 | 1.8687 | 0.0012 |
| vggt | `vggt_p7_allcities_p5e_private_viewtype_projection_aux_midtrunkfull_warmp5elowfull_e4_b9_2gpu_final` | final | view-type; private remote head; projection aux | ok | 0.0523 | 88.3333 | 1.9752 | 1.8698 | 0.0010 | ok | 0.0406 | 93.3333 | 1.9780 | 1.8687 | 0.0012 |
| vggt | `vggt_p7_allcities_p5h_film_diffblank_rank02_gate005_e4_b8_2gpu_best` | best | film/protected fusion | ok | 0.0526 | 86.6667 | 1.9760 | 1.8844 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9786 | 1.8707 | 0.0015 |
| vggt | `vggt_p7_allcities_p5h_film_protected_from_p5e_aux_rank005_e8_b8_2gpu_best` | best | private remote head; film/protected fusion | ok | 0.0526 | 86.6667 | 1.9760 | 1.8844 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9786 | 1.8707 | 0.0015 |
| vggt | `vggt_p7_allcities_p5h_film_protected_from_p5e_aux_rank05_gate005_e6_b8_2gpu_best` | best | private remote head; film/protected fusion | ok | 0.0526 | 86.6667 | 1.9760 | 1.8844 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9786 | 1.8707 | 0.0015 |
| vggt | `vggt_p7_aux_pointresidual_offset_gt_w10_pm8_e4_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0546 | 86.6667 | 1.9730 | 1.8684 | -0.0005 | ok | 0.0417 | 92.5000 | 1.9758 | 1.8693 | 0.0005 |
| vggt | `vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0542 | 86.6667 | 1.9707 | 1.8661 | -0.0006 | ok | 0.0410 | 92.5000 | 1.9734 | 1.8683 | 0.0006 |
| vggt | `vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_b9_2gpu_best_slim` | best | Crossview checkpoint | ok | 0.0542 | 85.0000 | 1.9687 | 1.8675 | -0.0009 | ok | 0.0406 | 91.6667 | 1.9706 | 1.8666 | 0.0006 |
| vggt | `vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0540 | 86.6667 | 1.9688 | 1.8675 | -0.0005 | ok | 0.0404 | 92.9167 | 1.9706 | 1.8685 | 0.0007 |
| vggt | `vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_cont_e12_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0544 | 91.6667 | 1.9694 | 1.8672 | -0.0005 | ok | 0.0408 | 92.9167 | 1.9706 | 1.8689 | 0.0008 |
| vggt | `vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_b9_2gpu_best_slim` | best | Crossview checkpoint | ok | 0.0551 | 86.6667 | 1.9690 | 1.8646 | -0.0012 | ok | 0.0413 | 92.5000 | 1.9714 | 1.8683 | 0.0005 |
| vggt | `vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0549 | 85.0000 | 1.9692 | 1.8661 | -0.0008 | ok | 0.0412 | 92.0833 | 1.9716 | 1.8702 | 0.0006 |
| vggt | `vggt_p7_auxonly_global_recon_frombest_e6_b32_2gpu_best` | best | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_auxonly_global_recon_frombest_e6_b32_2gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_auxonly_recon_gt_frombest_e8_b32_2gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_chicago_newyork_full_p5b_joint_pm4_aux_lowover15_e50_b8_2gpu_best` | best | P5B/shared-norm; NewYork subset; Chicago subset | ok | 0.0510 | 90.0000 | 1.2903 | 1.2897 | 0.0005 | ok | 0.0409 | 92.0833 | 1.3021 | 1.3019 | 0.0019 |
| vggt | `vggt_p7_diagnostic_oldp7_frozen_trunk_train_remotehead_aux_h035_height001_final` | final | diagnostic | ok | 0.0523 | 93.3333 | 1.9758 | 1.8743 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9786 | 1.8771 | 0.0014 |
| vggt | `vggt_p7_diagnostic_oldp7_trunk_p5b_remote_head_final` | final | P5B/shared-norm; diagnostic | ok | 0.0523 | 93.3333 | 1.9758 | 1.8743 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9786 | 1.8771 | 0.0014 |
| vggt | `vggt_p7_diagnostic_p5b_warm_privatehead_freeze_remotehead_aux_h035_height001_trunklr5e8_final` | final | P5B/shared-norm; diagnostic | ok | 0.0563 | 88.3333 | 1.9750 | 1.8661 | -0.0001 | ok | 0.0481 | 87.5000 | 1.9780 | 1.8602 | 0.0007 |
| vggt | `vggt_p7_diagnostic_p5b_warm_privatehead_frozen_trunk_remotehead_auxonly_h035_height001_final` | final | P5B/shared-norm; diagnostic | ok | 0.0561 | 88.3333 | 1.9749 | 1.8665 | 0.0004 | ok | 0.0480 | 86.6667 | 1.9779 | 1.8605 | 0.0009 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_final` | final | P5B/shared-norm; diagnostic | ok | 0.0543 | 93.3333 | 1.9761 | 1.8683 | -0.0002 | ok | 0.0415 | 95.4167 | 1.9789 | 1.8713 | 0.0008 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherxyz5_trunklr2e7_final` | final | P5B/shared-norm; diagnostic; teacher | ok | 0.0542 | 91.6667 | 1.9760 | 1.8680 | 0.0001 | ok | 0.0415 | 95.0000 | 1.9789 | 1.8707 | 0.0009 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_teacherz5_trunklr2e7_final` | final | P5B/shared-norm; diagnostic; teacher | ok | 0.0543 | 91.6667 | 1.9759 | 1.8701 | -0.0001 | ok | 0.0413 | 94.1667 | 1.9788 | 1.8729 | 0.0010 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr1e7_final` | final | P5B/shared-norm; diagnostic | ok | 0.0537 | 91.6667 | 1.9759 | 1.8714 | 0.0001 | ok | 0.0412 | 94.1667 | 1.9787 | 1.8743 | 0.0009 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_e6_final` | final | P5B/shared-norm; diagnostic | ok | 0.0536 | 91.6667 | 1.9759 | 1.8715 | 0.0002 | ok | 0.0412 | 94.5833 | 1.9788 | 1.8744 | 0.0008 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_trunklr5e8_final` | final | P5B/shared-norm; diagnostic | ok | 0.0530 | 91.6667 | 1.9758 | 1.8729 | 0.0004 | ok | 0.0406 | 95.4167 | 1.9787 | 1.8758 | 0.0012 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zdist2_trunklr5e8_final` | final | P5B/shared-norm; diagnostic | ok | 0.0535 | 91.6667 | 1.9758 | 1.8724 | 0.0001 | ok | 0.0410 | 95.4167 | 1.9787 | 1.8753 | 0.0011 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height001_zhigh2q80_trunklr5e8_final` | final | P5B/shared-norm; diagnostic | ok | 0.0535 | 91.6667 | 1.9759 | 1.8725 | 0.0002 | ok | 0.0410 | 94.1667 | 1.9787 | 1.8754 | 0.0012 |
| vggt | `vggt_p7_diagnostic_p5bhead_freeze_remotehead_aux_h035_height003_final` | final | P5B/shared-norm; diagnostic | ok | 0.0542 | 93.3333 | 1.9761 | 1.8684 | 0.0001 | ok | 0.0415 | 95.8333 | 1.9789 | 1.8713 | 0.0009 |
| vggt | `vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_auxonly_h035_best` | best | P5B/shared-norm; diagnostic | ok | 0.0523 | 93.3333 | 1.9758 | 1.8743 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9786 | 1.8771 | 0.0014 |
| vggt | `vggt_p7_diagnostic_p5bhead_frozen_trunk_remotehead_heads_aux_h035_height001_final` | final | P5B/shared-norm; diagnostic | ok | 0.0528 | 93.3333 | 1.9758 | 1.8743 | 0.0003 | ok | 0.0403 | 95.4167 | 1.9786 | 1.8771 | 0.0011 |
| vggt | `vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best` | best | DPT init | failed | nan | nan | nan | nan | nan | failed | nan | nan | nan | nan | nan |
| vggt | `vggt_p7_dptinit_fullfinetune_fromrobust_nocrop_validqheight_gtglobal_e60_b8_2gpu_best` | best | DPT init; no-crop | ok | 0.0523 | 93.3333 | 1.9757 | 1.8824 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9784 | 1.8723 | 0.0010 |
| vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_best` | best | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_final` | final | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_globalrecon_normgt_nooffset_2city_e12_b16_2gpu_best` | best | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_best` | best | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu_final` | final | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_remotehead_aggtail2_2city_e12_b16_2gpu_best` | best | DPT init | ok | 0.1864 | 65.0000 | 1.9710 | 1.9012 | 0.0094 | ok | 0.1625 | 43.7500 | 1.9769 | 1.8678 | 0.0119 |
| vggt | `vggt_p7_dptinit_remotehead_aggtail2_2city_e12_b16_2gpu_final` | final | DPT init | ok | 0.1864 | 65.0000 | 1.9710 | 1.9011 | 0.0094 | ok | 0.1625 | 43.7500 | 1.9769 | 1.8678 | 0.0119 |
| vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_best` | best | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_dptinit_remotehead_joint_2city_e8_b16_2gpu_final` | final | DPT init | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p7_frombase_robustoverlap_lowaux_e4_b8_2gpu_final` | final | overlap loss | ok | 0.0603 | 90.0000 | 1.9783 | 1.8634 | 0.0005 | ok | 0.0518 | 85.8333 | 1.9809 | 1.8558 | -0.0004 |
| vggt | `vggt_p7_frombest_auxrecon_gt_safe_e4_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0527 | 90.0000 | 1.9728 | 1.8661 | 0.0009 | ok | 0.0416 | 95.0000 | 1.9750 | 1.8687 | 0.0017 |
| vggt | `vggt_p7_frombest_denseheight_densegoffset_e2_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0531 | 90.0000 | 1.9737 | 1.8666 | 0.0003 | ok | 0.0409 | 92.5000 | 1.9767 | 1.8682 | 0.0014 |
| vggt | `vggt_p7_frombest_denseheight_densegoffset_e8_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0525 | 91.6667 | 1.9744 | 1.8656 | 0.0015 | ok | 0.0414 | 94.5833 | 1.9771 | 1.8660 | 0.0015 |
| vggt | `vggt_p7_frombest_denseheight_main_nooffset_e2_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0536 | 90.0000 | 1.9735 | 1.8669 | 0.0002 | ok | 0.0410 | 92.5000 | 1.9765 | 1.8682 | 0.0013 |
| vggt | `vggt_p7_frombest_gatedres_preagg_e3_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0500 | 95.0000 | 1.9732 | 1.8672 | 0.0017 | ok | 0.0397 | 93.7500 | 1.9757 | 1.8673 | 0.0021 |
| vggt | `vggt_p7_frombest_gridglobal_selfcontained_e2_b8_2gpu_best` | best | Crossview checkpoint | ok | 0.0518 | 93.3333 | 1.9727 | 1.8670 | 0.0006 | ok | 0.0415 | 92.0833 | 1.9750 | 1.8694 | 0.0009 |
| vggt | `vggt_p7_frombest_gridglobal_selfcontained_e2_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0518 | 93.3333 | 1.9727 | 1.8670 | 0.0006 | ok | 0.0415 | 92.0833 | 1.9750 | 1.8694 | 0.0009 |
| vggt | `vggt_p7_frombest_highz_normrecon_globalgt_e3_b8_2gpu_best` | best | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_frombest_highz_normrecon_globalgt_e3_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_frombest_normrecon_globalgt_e3_b8_2gpu_best` | best | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_frombest_normrecon_globalgt_e3_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_frombest_overlap12_pm4_e3_b8_2gpu_final` | final | overlap loss | ok | 0.0530 | 91.6667 | 1.9720 | 1.8669 | 0.0005 | ok | 0.0419 | 90.8333 | 1.9745 | 1.8672 | 0.0006 |
| vggt | `vggt_p7_frombest_splitfilm_denseh_highq_e3_b6_2gpu_final` | final | film/protected fusion | ok | 0.0548 | 88.3333 | 1.9762 | 1.9762 | 0.0000 | ok | 0.0458 | 86.2500 | 1.9790 | 1.9790 | 0.0001 |
| vggt | `vggt_p7_frombest_top50_highz008_overlap6_e3_b8_2gpu_final` | final | overlap loss | ok | 0.0546 | 93.3333 | 1.9725 | 1.8674 | 0.0004 | ok | 0.0428 | 90.4167 | 1.9747 | 1.8672 | 0.0005 |
| vggt | `vggt_p7_frombest_zdist02_highz006_e3_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0533 | 91.6667 | 1.9727 | 1.8667 | 0.0006 | ok | 0.0417 | 90.0000 | 1.9751 | 1.8685 | 0.0007 |
| vggt | `vggt_p7_fromrobust_resetaux_stageA2_heightrepair_simple_e30_b32_2gpu_best` | best | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_fromrobust_stageA2_heightrepair_auxonly_gtglobal_e40_b32_2gpu_ddpfix_best` | best | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_fullfinetune_fromrobust_resetaux_pm4_overlap6_stageA2simple_e60_b8_2gpu_ddpfix_best` | best | overlap loss; stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_googleonly_robustoverlap_from_best_e4_b8_2gpu_final` | best | overlap loss | ok | 0.0541 | 98.3333 | 1.9734 | 1.8660 | 0.0010 | ok | 0.0419 | 94.1667 | 1.9757 | 1.8686 | 0.0011 |
| vggt | `vggt_p7_moge2_balanced20x4_private_tokens_raw001_gradz005_mogegrad001_edge0002_h003_warme2_e30_b24_4gpu_fixed_best` | best | MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8667 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9784 | 1.8684 | 0.0014 |
| vggt | `vggt_p7_moge2_balanced20x4_private_tokens_raw001_gradz005_mogegrad001_edge0002_h003_warme2_e30_b24_4gpu_fixed_final` | final | MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8667 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9784 | 1.8685 | 0.0014 |
| vggt | `vggt_p7_newyork_metadata_top256_p5b_joint_pm4_aux_capacity_e6_b8_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0671 | 88.3333 | 1.2810 | 1.2941 | -0.0007 | ok | 0.0519 | 85.4167 | 1.2945 | 1.3009 | 0.0020 |
| vggt | `vggt_p7_newyork_metadata_top256_p5b_joint_pm4_aux_lowover15_e6_b8_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0690 | 85.0000 | 1.2819 | 1.2931 | -0.0021 | ok | 0.0524 | 86.2500 | 1.2954 | 1.2995 | 0.0011 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm1_aux_capacity_e6_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0631 | 78.3333 | 1.2923 | 1.2959 | -0.0004 | ok | 0.0494 | 84.5833 | 1.3016 | 1.3083 | 0.0005 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm1_aux_capacity_e6_final_155` | final | P5B/shared-norm; NewYork subset | ok | 0.0631 | 78.3333 | 1.2923 | 1.2959 | -0.0004 | ok | 0.0494 | 84.5833 | 1.3016 | 1.3083 | 0.0005 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm2_aux_capacity_e6_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0657 | 83.3333 | 1.2824 | 1.2918 | 0.0005 | ok | 0.0537 | 80.4167 | 1.2954 | 1.2989 | -0.0000 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm2_aux_capacity_e6_final_157` | final | P5B/shared-norm; NewYork subset | ok | 0.0657 | 83.3333 | 1.2824 | 1.2918 | 0.0005 | ok | 0.0537 | 80.4167 | 1.2954 | 1.2989 | -0.0000 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm4_aux_capacity_e6_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0610 | 83.3333 | 1.2842 | 1.2895 | -0.0007 | ok | 0.0493 | 87.5000 | 1.2976 | 1.3018 | -0.0003 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm4_aux_capacity_e6_final_159` | final | P5B/shared-norm; NewYork subset | ok | 0.0610 | 83.3333 | 1.2842 | 1.2895 | -0.0007 | ok | 0.0493 | 87.5000 | 1.2976 | 1.3018 | -0.0003 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm4_aux_offset2_e6_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0632 | 83.3333 | 1.2825 | 1.2946 | 0.0004 | ok | 0.0537 | 79.5833 | 1.2957 | 1.3007 | -0.0009 |
| vggt | `vggt_p7_newyork_top128_p5b_joint_pm4_aux_offset2_e6_final_161` | final | P5B/shared-norm; NewYork subset | ok | 0.0632 | 83.3333 | 1.2825 | 1.2946 | 0.0004 | ok | 0.0537 | 79.5833 | 1.2957 | 1.3007 | -0.0009 |
| vggt | `vggt_p7_newyork_top128_p5b_stage_pm1_to_pm4_aux_capacity_e4_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0605 | 83.3333 | 1.2893 | 1.2908 | 0.0015 | ok | 0.0524 | 81.2500 | 1.3004 | 1.3019 | 0.0035 |
| vggt | `vggt_p7_newyork_top128_p5b_stage_pm1_to_pm4_aux_capacity_e4_final_163` | final | P5B/shared-norm; NewYork subset | ok | 0.0605 | 83.3333 | 1.2893 | 1.2908 | 0.0015 | ok | 0.0524 | 81.2500 | 1.3004 | 1.3019 | 0.0035 |
| vggt | `vggt_p7_newyork_top256_p5b_joint_pm4_aux_capacity_e6_final` | final | P5B/shared-norm; NewYork subset | ok | 0.0657 | 88.3333 | 1.2819 | 1.2955 | -0.0002 | ok | 0.0510 | 86.2500 | 1.2952 | 1.3029 | 0.0020 |
| vggt | `vggt_p7_newyork_top256_p5b_joint_pm4_aux_capacity_e6_final_165` | final | P5B/shared-norm; NewYork subset | ok | 0.0657 | 88.3333 | 1.2819 | 1.2955 | -0.0002 | ok | 0.0510 | 86.2500 | 1.2952 | 1.3029 | 0.0020 |
| vggt | `vggt_p7_offset_only_tokenaux_from_stagea_e6_b8_2gpu_final` | final | stage training | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_offset_only_tokenaux_nopointdetach_fixedlow_e4_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_offset_residual_pointbase_normbase_token_e4_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_e4_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9755 | 1.8674 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8696 | 0.0014 |
| vggt | `vggt_p7_oldp7_train_remotehead_aggtail2_raw001_gradz005_anchor500k_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9755 | 1.8678 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8700 | 0.0014 |
| vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_lowlr3e6_h003_e2_b32_4gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9752 | 1.8648 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9781 | 1.8656 | 0.0014 |
| vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_paramanchor5k_lowlr3e6_h003_e2_b32_4gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9752 | 1.8653 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9782 | 1.8661 | 0.0014 |
| vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_raw0005_paramanchor50k_lowlr3e6_h003_e2_b32_4gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9753 | 1.8666 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9783 | 1.8678 | 0.0014 |
| vggt | `vggt_p7_oldp7_train_remotehead_nonreentrant_raw001_paramanchor500k_lowlr3e6_h003_e2_b32_4gpu_final` | final | Crossview checkpoint | ok | 0.0523 | 93.3333 | 1.9756 | 1.8690 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9785 | 1.8710 | 0.0014 |
| vggt | `vggt_p7_p5b_noauxhead_chicago_crop08_e20_best` | best | P5B/shared-norm; Chicago subset | ok | 0.0589 | 86.6667 | 1.9774 | 1.9009 | 0.0008 | ok | 0.0528 | 81.2500 | 1.9794 | 1.8682 | 0.0007 |
| vggt | `vggt_p7_p5b_noauxhead_chicago_crop08_e20_final` | final | P5B/shared-norm; Chicago subset | ok | 0.0607 | 86.6667 | 1.9782 | 1.8998 | -0.0009 | ok | 0.0517 | 84.1667 | 1.9804 | 1.8644 | 0.0001 |
| vggt | `vggt_p7_p5b_parallel_token_aux_p5b_anchor_h035_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0563 | 86.6667 | 1.2801 | 1.2803 | -0.0001 | ok | 0.0480 | 85.8333 | 1.2956 | 1.2939 | 0.0007 |
| vggt | `vggt_p7_p5b_parallel_token_aux_preservep5b_h035_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0565 | 90.0000 | 1.2846 | 1.2803 | 0.0000 | ok | 0.0456 | 90.4167 | 1.2990 | 1.2933 | 0.0005 |
| vggt | `vggt_p7_p5b_parallel_token_aux_preservep5b_h035_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0565 | 90.0000 | 1.2846 | 1.2803 | 0.0000 | ok | 0.0456 | 90.4167 | 1.2990 | 1.2933 | 0.0005 |
| vggt | `vggt_p7_p5b_parallel_token_aux_recoverp5b_h035_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0583 | 88.3333 | 1.2847 | 1.2815 | -0.0015 | ok | 0.0450 | 90.4167 | 1.2984 | 1.2935 | 0.0002 |
| vggt | `vggt_p7_p5b_parallel_token_aux_recoverp5b_h035_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0573 | 88.3333 | 1.2850 | 1.2814 | -0.0013 | ok | 0.0442 | 91.2500 | 1.2985 | 1.2933 | 0.0006 |
| vggt | `vggt_p7_p5b_private_oldp7_p5bhead_freeze_remotehead_aux_h035_final` | final | P5B/shared-norm | ok | 0.0548 | 93.3333 | 1.9762 | 1.8672 | -0.0001 | ok | 0.0417 | 95.8333 | 1.9790 | 1.8701 | 0.0009 |
| vggt | `vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly_h035_final` | final | P5B/shared-norm | ok | 0.0548 | 93.3333 | 1.9762 | 1.8672 | -0.0001 | ok | 0.0417 | 95.8333 | 1.9790 | 1.8701 | 0.0009 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0523 | 93.3333 | 1.9760 | 1.8666 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9787 | 1.8676 | 0.0014 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0520 | 93.3333 | 1.9762 | 1.8670 | 0.0009 | ok | 0.0403 | 95.8333 | 1.9790 | 1.8671 | 0.0013 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_e2_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0520 | 93.3333 | 1.9762 | 1.8670 | 0.0009 | ok | 0.0403 | 95.8333 | 1.9790 | 1.8671 | 0.0013 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0523 | 93.3333 | 1.9760 | 1.8666 | 0.0007 | ok | 0.0401 | 95.4167 | 1.9787 | 1.8676 | 0.0014 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_pmgrad05_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0522 | 91.6667 | 1.9761 | 1.8670 | 0.0009 | ok | 0.0402 | 94.5833 | 1.9788 | 1.8669 | 0.0014 |
| vggt | `vggt_p7_p5b_private_p5bhead_oldp7_parallel_token_aux_h035_zheight001_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0529 | 93.3333 | 1.9758 | 1.8670 | 0.0003 | ok | 0.0404 | 94.5833 | 1.9784 | 1.8672 | 0.0011 |
| vggt | `vggt_p7_p5b_private_remote_parallel_token_aux_recoverp5b_h035_best` | best | P5B/shared-norm; parallel-token aux | ok | 0.0578 | 90.0000 | 1.9782 | 1.8660 | -0.0014 | ok | 0.0446 | 90.8333 | 1.9810 | 1.8648 | 0.0003 |
| vggt | `vggt_p7_p5b_private_remote_parallel_token_aux_recoverp5b_h035_final` | final | P5B/shared-norm; parallel-token aux | ok | 0.0578 | 90.0000 | 1.9782 | 1.8660 | -0.0014 | ok | 0.0446 | 90.8333 | 1.9810 | 1.8648 | 0.0003 |
| vggt | `vggt_p7_p5b_projaux_light_detach_chicago_crop08_e20_best` | best | P5B/shared-norm; projection aux; Chicago subset | ok | 0.0647 | 81.6667 | 1.2817 | 1.2894 | -0.0004 | ok | 0.0525 | 85.8333 | 1.2974 | 1.2989 | -0.0006 |
| vggt | `vggt_p7_p5b_projaux_light_detach_chicago_crop08_e20_final` | final | P5B/shared-norm; projection aux; Chicago subset | ok | 0.0657 | 83.3333 | 1.2816 | 1.2892 | -0.0004 | ok | 0.0528 | 85.0000 | 1.2972 | 1.2989 | -0.0006 |
| vggt | `vggt_p7_p5b_projaux_light_grad_chicago_crop08_e20_best` | best | P5B/shared-norm; projection aux; Chicago subset | ok | 0.0602 | 81.6667 | 1.2802 | 1.2868 | -0.0003 | ok | 0.0508 | 85.8333 | 1.2962 | 1.2965 | 0.0001 |
| vggt | `vggt_p7_p5b_projaux_light_grad_chicago_crop08_e20_final` | final | P5B/shared-norm; projection aux; Chicago subset | ok | 0.0600 | 83.3333 | 1.2801 | 1.2865 | -0.0008 | ok | 0.0506 | 86.2500 | 1.2960 | 1.2963 | -0.0001 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_best` | best | P5B/shared-norm; projection aux | ok | 0.0518 | 91.6667 | 1.2922 | 1.2953 | 0.0010 | ok | 0.0406 | 94.5833 | 1.3021 | 1.3075 | 0.0012 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_curric2v_to4v_final` | final | P5B/shared-norm; projection aux | ok | 0.0569 | 91.6667 | 1.2912 | 1.2937 | 0.0012 | ok | 0.0429 | 92.9167 | 1.3016 | 1.3048 | 0.0010 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_final` | final | P5B/shared-norm; projection aux | ok | 0.0520 | 91.6667 | 1.2925 | 1.2953 | 0.0011 | ok | 0.0404 | 95.8333 | 1.3024 | 1.3076 | 0.0014 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_h0005_final` | final | P5B/shared-norm; projection aux | ok | 0.0535 | 93.3333 | 1.2909 | 1.2949 | 0.0016 | ok | 0.0427 | 91.6667 | 1.3018 | 1.3067 | 0.0009 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_best` | best | P5B/shared-norm; projection aux | ok | 0.0516 | 93.3333 | 1.2929 | 1.2963 | 0.0014 | ok | 0.0404 | 96.2500 | 1.3026 | 1.3082 | 0.0016 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_lowtrunklr2e6_warmbest_final` | final | P5B/shared-norm; projection aux | ok | 0.0516 | 93.3333 | 1.2928 | 1.2961 | 0.0009 | ok | 0.0401 | 95.4167 | 1.3026 | 1.3080 | 0.0013 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_allcities_nocrop_warmbest_best` | best | P5B/shared-norm; projection aux; no-crop | ok | 0.0520 | 88.3333 | 1.2907 | 1.2975 | 0.0007 | ok | 0.0407 | 91.6667 | 1.3013 | 1.3097 | 0.0017 |
| vggt | `vggt_p7_p5b_shared_norm_projection_aux_full_2city` | - | P5B/shared-norm; projection aux | ok | 0.0505 | 90.0000 | 1.2903 | 1.2893 | 0.0008 | ok | 0.0404 | 92.0833 | 1.3021 | 1.3016 | 0.0021 |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_allcities_best` | best | view-type; private remote head; projection aux | ok | 0.0526 | 86.6667 | 1.9746 | 1.8715 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9778 | 1.8714 | 0.0015 |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_allcities_final` | final | view-type; private remote head; projection aux | ok | 0.0526 | 86.6667 | 1.9746 | 1.8715 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9778 | 1.8714 | 0.0015 |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_best` | best | view-type; private remote head; projection aux | ok | 0.0521 | 88.3333 | 1.9749 | 1.8704 | 0.0016 | ok | 0.0403 | 94.1667 | 1.9778 | 1.8685 | 0.0014 |
| vggt | `vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_final` | final | view-type; private remote head; projection aux | ok | 0.0521 | 88.3333 | 1.9749 | 1.8704 | 0.0016 | ok | 0.0403 | 94.1667 | 1.9778 | 1.8685 | 0.0014 |
| vggt | `vggt_p7_p5h_film_diffblank_rank02_gate005_allcities_final` | final | film/protected fusion | ok | 0.0526 | 86.6667 | 1.9746 | 1.8715 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9778 | 1.8714 | 0.0015 |
| vggt | `vggt_p7_p5h_film_protected_from_p5e_aux_rank005_allcities_final` | final | private remote head; film/protected fusion | ok | 0.0526 | 86.6667 | 1.9746 | 1.8715 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9778 | 1.8714 | 0.0015 |
| vggt | `vggt_p7_p5h_film_protected_from_p5e_aux_rank05_gate005_allcities_final` | final | private remote head; film/protected fusion | ok | 0.0526 | 86.6667 | 1.9746 | 1.8715 | 0.0018 | ok | 0.0406 | 93.3333 | 1.9778 | 1.8714 | 0.0015 |
| vggt | `vggt_p7_pointhead_pointoffset_gt_e4_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0536 | 86.6667 | 1.9753 | 1.8686 | -0.0006 | ok | 0.0408 | 92.0833 | 1.9781 | 1.8709 | 0.0009 |
| vggt | `vggt_p7_pointhead_pointoffset_gt_w10_headlr1e4_e4_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0542 | 88.3333 | 1.9734 | 1.8686 | -0.0007 | ok | 0.0411 | 93.3333 | 1.9760 | 1.8585 | 0.0006 |
| vggt | `vggt_p7_pointhead_pointoffset_gt_w10_pm8_pure_start2_ramp2_e6_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0546 | 86.6667 | 1.9741 | 1.8671 | -0.0004 | ok | 0.0413 | 92.0833 | 1.9770 | 1.8666 | 0.0006 |
| vggt | `vggt_p7_pointhead_pointoffset_gt_w3_pm8_headlr1e4_e4_b9_2gpu_final` | final | Crossview checkpoint | ok | 0.0543 | 86.6667 | 1.9749 | 1.8686 | -0.0006 | ok | 0.0415 | 92.9167 | 1.9778 | 1.8674 | 0.0005 |
| vggt | `vggt_p7_proj_denseh015_highq50_from_robustoverlap_e4_b8_2gpu_final` | final | projection aux; overlap loss | ok | 0.0516 | 91.6667 | 1.9736 | 1.8667 | 0.0004 | ok | 0.0421 | 91.6667 | 1.9761 | 1.8684 | 0.0005 |
| vggt | `vggt_p7_proj_denseh015_highq50_midtrunk_from_best_e4_b8_2gpu_final` | best | projection aux | ok | 0.0532 | 95.0000 | 1.9748 | 1.8677 | 0.0012 | ok | 0.0415 | 94.1667 | 1.9776 | 1.8679 | 0.0012 |
| vggt | `vggt_p7_proj_denseh02_from_robustoverlap_e3_b8_2gpu_final` | final | projection aux; overlap loss | ok | 0.0540 | 95.0000 | 1.9739 | 1.8672 | 0.0002 | ok | 0.0425 | 90.8333 | 1.9765 | 1.8682 | 0.0008 |
| vggt | `vggt_p7_proj_denseh_mogeshape_soft_from_best_e4_b8_2gpu_final` | best | projection aux; MoGe prior | ok | 0.0523 | 91.6667 | 1.9734 | 1.8664 | -0.0001 | ok | 0.0420 | 92.5000 | 1.9760 | 1.8678 | 0.0005 |
| vggt | `vggt_p7_proj_denseh_tokenres_g005_from_best_e4_b8_2gpu_final` | best | projection aux | ok | 0.0525 | 95.0000 | 1.9747 | 1.8672 | 0.0008 | ok | 0.0409 | 93.7500 | 1.9776 | 1.8668 | 0.0009 |
| vggt | `vggt_p7_proj_headonly_denseh_highq_from_best_e4_b32_2gpu_final` | best | projection aux | ok | 0.0523 | 93.3333 | 1.9741 | 1.8678 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9769 | 1.8679 | 0.0010 |
| vggt | `vggt_p7_proj_moge_agglr1e7_private_tokens_warmprivbest_e10_b8_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0522 | 93.3333 | 1.9761 | 1.8669 | 0.0009 | ok | 0.0399 | 95.0000 | 1.9788 | 1.8688 | 0.0014 |
| vggt | `vggt_p7_proj_moge_agglr1e7_private_tokens_warmprivbest_e10_b8_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0522 | 93.3333 | 1.9761 | 1.8663 | 0.0009 | ok | 0.0399 | 95.4167 | 1.9788 | 1.8684 | 0.0014 |
| vggt | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8675 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8692 | 0.0014 |
| vggt | `vggt_p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8675 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8691 | 0.0014 |
| vggt | `vggt_p7_proj_moge_aux_validq90_private_tokens_projmg05_edge01_h035_warmprojbest_e20_b32_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9756 | 1.8679 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8694 | 0.0014 |
| vggt | `vggt_p7_proj_moge_aux_validq90_private_tokens_projmg05_edge01_h035_warmprojbest_e20_b32_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9756 | 1.8679 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8694 | 0.0014 |
| vggt | `vggt_p7_proj_moge_denseheight_aux_validq90_private_tokens_h100_tail2lr5e7_warmprojbest_e12_b32_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0524 | 93.3333 | 1.9755 | 1.8674 | 0.0006 | ok | 0.0401 | 95.4167 | 1.9784 | 1.8688 | 0.0013 |
| vggt | `vggt_p7_proj_moge_denseheight_aux_validq90_private_tokens_h100_tail2lr5e7_warmprojbest_e12_b32_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0524 | 93.3333 | 1.9755 | 1.8674 | 0.0006 | ok | 0.0401 | 95.4167 | 1.9784 | 1.8688 | 0.0013 |
| vggt | `vggt_p7_proj_moge_denseheight_aux_validq90_private_tokens_h150_warmprojbest_e10_b32_4gpu_best` | best | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8675 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8689 | 0.0014 |
| vggt | `vggt_p7_proj_moge_denseheight_aux_validq90_private_tokens_h150_warmprojbest_e10_b32_4gpu_final` | final | projection aux; MoGe prior | ok | 0.0523 | 93.3333 | 1.9755 | 1.8675 | 0.0007 | ok | 0.0401 | 95.8333 | 1.9784 | 1.8689 | 0.0014 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_frombase_e8_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight | ok | 0.0578 | 91.6667 | 1.9780 | 1.8674 | 0.0000 | ok | 0.0455 | 92.5000 | 1.9818 | 1.8563 | 0.0015 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_lowcovis_warmh5best_e4_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight | ok | 0.0521 | 88.3333 | 1.9751 | 1.8673 | 0.0015 | ok | 0.0429 | 87.0833 | 1.9776 | 1.8648 | 0.0004 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_overlappm2_d3_rel005_warmh5best_e8_b8_4gpu_best` | best | projection aux; MoGe prior; pmheight; overlap loss | ok | 0.0534 | 96.6667 | 1.9739 | 1.8675 | 0.0016 | ok | 0.0410 | 93.3333 | 1.9768 | 1.8692 | 0.0014 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_overlappm2_d3_rel005_warmh5best_e8_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight; overlap loss | ok | 0.0534 | 96.6667 | 1.9739 | 1.8675 | 0.0016 | ok | 0.0410 | 93.3333 | 1.9768 | 1.8692 | 0.0014 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_prior0_balanced20x4_private_tokens_warmbest_e6_b8_4gpu_best` | best | projection aux; MoGe prior; pmheight | ok | 0.0522 | 93.3333 | 1.9748 | 1.8671 | 0.0015 | ok | 0.0415 | 92.5000 | 1.9774 | 1.8664 | 0.0010 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_prior0_balanced20x4_private_tokens_warmbest_e6_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight | ok | 0.0522 | 93.3333 | 1.9748 | 1.8671 | 0.0015 | ok | 0.0415 | 92.5000 | 1.9774 | 1.8664 | 0.0010 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_cont_e4_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight; robust pointmap | ok | 0.0519 | 91.6667 | 1.9748 | 1.8662 | 0.0005 | ok | 0.0412 | 92.5000 | 1.9774 | 1.8676 | 0.0011 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight; robust pointmap | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_views6_warmh5best_e4_b6_4gpu_final` | final | projection aux; MoGe prior; pmheight | ok | 0.0519 | 91.6667 | 1.9738 | 1.8663 | 0.0020 | ok | 0.0421 | 90.8333 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_proj_moge_pmheight_h5_zerocovis_train_warmh5best_e3_b8_4gpu_final` | final | projection aux; MoGe prior; pmheight | ok | 0.0540 | 93.3333 | 1.9757 | 1.8654 | 0.0010 | ok | 0.0427 | 90.0000 | 1.9784 | 1.8692 | 0.0012 |
| vggt | `vggt_p7_proj_moge_robustpm5_balanced20x4_private_tokens_warmh5best_e8_b8_4gpu_best` | best | projection aux; MoGe prior; robust pointmap | ok | 0.0590 | 90.0000 | 1.9713 | 1.8691 | 0.0037 | ok | 0.0470 | 91.6667 | 1.9736 | 1.8714 | 0.0037 |
| vggt | `vggt_p7_proj_moge_robustpm5_balanced20x4_private_tokens_warmh5best_e8_b8_4gpu_final` | final | projection aux; MoGe prior; robust pointmap | ok | 0.0530 | 93.3333 | 1.9726 | 1.8667 | 0.0018 | ok | 0.0403 | 95.0000 | 1.9750 | 1.8682 | 0.0019 |
| vggt | `vggt_p7_proj_moge_robustpm5_overlappm4_d3_rel005_warmrobustbest_e6_b8_4gpu_best` | best | projection aux; MoGe prior; robust pointmap; overlap loss | ok | 0.0520 | 86.6667 | 1.9745 | 1.8675 | 0.0020 | ok | 0.0409 | 92.0833 | 1.9770 | 1.8661 | 0.0012 |
| vggt | `vggt_p7_proj_moge_robustpm5_overlappm4_d3_rel005_warmrobustbest_e6_b8_4gpu_final` | final | projection aux; MoGe prior; robust pointmap; overlap loss | ok | 0.0520 | 86.6667 | 1.9745 | 1.8675 | 0.0020 | ok | 0.0409 | 92.0833 | 1.9770 | 1.8661 | 0.0012 |
| vggt | `vggt_p7_proj_moge_robustpm5_overlappm4_frombase_e6_b8_4gpu_final` | final | projection aux; MoGe prior; robust pointmap; overlap loss | ok | 0.0582 | 88.3333 | 1.9777 | 1.8680 | 0.0004 | ok | 0.0471 | 88.7500 | 1.9804 | 1.8570 | 0.0020 |
| vggt | `vggt_p7_proj_robusttop10_overlap8_from_best_e3_b8_2gpu_final` | best | projection aux; robust pointmap; overlap loss; P8 point-only/joint | ok | 0.0526 | 93.3333 | 1.9748 | 1.8672 | 0.0008 | ok | 0.0412 | 93.7500 | 1.9776 | 1.8677 | 0.0008 |
| vggt | `vggt_p7_proj_tokenres_g001_from_robustoverlap_e4_b8_2gpu_v2_final` | final | projection aux; overlap loss | ok | 0.0526 | 95.0000 | 1.9748 | 1.8675 | 0.0009 | ok | 0.0410 | 94.1667 | 1.9777 | 1.8671 | 0.0008 |
| vggt | `vggt_p7_proj_views8_from_robustoverlap_e3_b4_2gpu_final` | final | projection aux; overlap loss | ok | 0.0508 | 93.3333 | 1.9743 | 1.8667 | 0.0015 | ok | 0.0399 | 94.5833 | 1.9771 | 1.8672 | 0.0015 |
| vggt | `vggt_p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_2gpu_best` | best | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_2gpu_final` | final | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_2gpu_best` | best | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_2gpu_final` | final | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_2gpu_best` | best | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_2gpu_final` | final | scale-free | ok | 0.0523 | 93.3333 | 1.9740 | 1.8679 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8678 | 0.0010 |
| vggt | `vggt_p7_scratch_denseheight_densegoffset_e12_b8_2gpu_final` | final | Crossview checkpoint | ok | 0.0558 | 81.6667 | 1.9719 | 1.8646 | 0.0012 | ok | 0.0446 | 90.4167 | 1.9746 | 1.8651 | 0.0007 |
| vggt | `vggt_p7_stageA2_height_gtglobalrecon_frozentrunk_fromA_e6_b36_2gpu_best` | best | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8680 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8679 | 0.0010 |
| vggt | `vggt_p7_stageA2_height_gtglobalrecon_frozentrunk_fromA_e6_b36_2gpu_final` | final | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8680 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8679 | 0.0010 |
| vggt | `vggt_p7_stageA2_height_gtglobalrecon_fullfinetune_fromA2_e80_b8_2gpu_best` | best | stage training | ok | 0.0523 | 93.3333 | 1.2937 | 1.2945 | 0.0014 | ok | 0.0410 | 92.9167 | 1.3030 | 1.3051 | 0.0010 |
| vggt | `vggt_p7_stageA_height_gtdir_frozentrunk_e6_b32_2gpu_best` | best | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8680 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8679 | 0.0010 |
| vggt | `vggt_p7_stageA_height_gtdir_frozentrunk_e6_b32_2gpu_final` | final | stage training | ok | 0.0523 | 93.3333 | 1.9740 | 1.8680 | 0.0014 | ok | 0.0410 | 92.9167 | 1.9768 | 1.8679 | 0.0010 |
| vggt | `vggt_p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b8_2gpu_best` | best | no-crop | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_token_fullfinetune_heightonly_highweight_nocrop_validqheight_e30_b8_2gpu_final` | final | no-crop | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_token_stageb_gtglobal_recon100_fromheight_e20_b8_2gpu_best` | best | stage training | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_token_stageb_gtglobal_recon100_fromheight_e20_b8_2gpu_final` | final | stage training | ok | 0.0540 | 86.6667 | 1.9756 | 1.9098 | -0.0006 | ok | 0.0413 | 92.9167 | 1.9789 | 1.9217 | 0.0008 |
| vggt | `vggt_p7_vggt_p5b_shared_norm_projection_aux_best` | best | P5B/shared-norm; projection aux | ok | 0.0540 | 90.0000 | 1.2856 | 1.2879 | -0.0003 | ok | 0.0454 | 88.3333 | 1.2997 | 1.2994 | -0.0001 |
| vggt | `vggt_p7_vggt_p5b_shared_norm_projection_aux_final` | final | P5B/shared-norm; projection aux | ok | 0.0531 | 90.0000 | 1.2851 | 1.2853 | -0.0014 | ok | 0.0409 | 93.7500 | 1.2997 | 1.2965 | 0.0005 |
| vggt | `vggt_p8_joint_allcities_4sat_private_remoteagg_from_rsonlybest_worldgt_e12_b8_2gpu_final` | final | P8 point-only/joint; private remote aggregator | ok | 0.2568 | 28.3333 | 0.1098 | 0.0234 | 0.0000 | ok | 0.1825 | 27.0833 | 0.1335 | 0.0281 | 0.0000 |
| vggt | `vggt_p8_joint_allcities_4sat_private_remoteagg_worldgt_headonly_lowlr_fromlowlr2_e8_b8_2gpu_best` | best | P8 point-only/joint; private remote aggregator | ok | 0.2568 | 28.3333 | 0.1102 | 0.0210 | 0.0000 | ok | 0.1825 | 27.0833 | 0.1341 | 0.0261 | 0.0000 |
| vggt | `vggt_p8_joint_allcities_4sat_private_remoteagg_worldgt_lowlr2_fromlowlrfinal_e12_b8_2gpu_final` | final | P8 point-only/joint; private remote aggregator | ok | 0.2568 | 28.3333 | 0.1101 | 0.0211 | 0.0000 | ok | 0.1825 | 27.0833 | 0.1343 | 0.0259 | 0.0000 |
| vggt | `vggt_p8_joint_allcities_4sat_private_remoteagg_worldgt_lowlr_frome12_e8_b8_2gpu_final` | final | P8 point-only/joint; private remote aggregator | ok | 0.2568 | 28.3333 | 0.1092 | 0.0221 | 0.0000 | ok | 0.1825 | 27.0833 | 0.1335 | 0.0268 | 0.0000 |
| vggt | `vggt_p8_joint_nychicago_4sat_private_remoteagg_fromrsonly_worldgt_e80_b8_2gpu_final` | final | P8 point-only/joint; private remote aggregator; Chicago subset | ok | 0.1662 | 86.6667 | 0.0656 | 0.0178 | 0.0000 | ok | 0.1081 | 66.6667 | 0.1086 | 0.0184 | 0.0000 |
| vggt | `vggt_p8_joint_nychicago_4sat_teacheranchor_remoteonly_e60_b8_2gpu_final` | final | teacher; P8 point-only/joint; Chicago subset | ok | 0.0707 | 81.6667 | 1.9798 | 1.9496 | -0.0027 | ok | 0.0657 | 79.1667 | 1.9847 | 1.8864 | -0.0023 |
| vggt | `vggt_p8_pointonly_4view_2city_nocrop_lowtrunklr_e6_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only; no-crop | ok | 0.0672 | 83.3333 | 1.9788 | 1.9253 | -0.0024 | ok | 0.0526 | 84.5833 | 1.9824 | 1.8583 | -0.0001 |
| vggt | `vggt_p8_pointonly_4view_2city_nocrop_lowtrunklr_e6_b9_2gpu_final` | final | P8 point-only/joint; point-only; no-crop | ok | 0.0672 | 83.3333 | 1.9791 | 1.9247 | -0.0028 | ok | 0.0527 | 85.4167 | 1.9827 | 1.8575 | -0.0004 |
| vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_best_modelonly` | best | P8 point-only/joint; point-only; no-crop | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only; no-crop | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_best_vggtbase_only` | best | P8 point-only/joint; point-only; no-crop | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p8_pointonly_4view_allcities_nocrop_midtrunklr_e10_b9_2gpu_final` | final | P8 point-only/joint; point-only; no-crop | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_best` | best | P8 point-only/joint; point-only | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p8_pointonly_8view_allcities_lowcovis_remoteheadonly_fromvggt_e12_b48_2gpu_final` | final | P8 point-only/joint; point-only | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p8_pointonly_8view_allcities_nocrop_lowtrunklr_e12_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only; no-crop | ok | 0.0564 | 90.0000 | 1.9800 | 1.8954 | -0.0023 | ok | 0.0478 | 89.5833 | 1.9831 | 1.8497 | -0.0021 |
| vggt | `vggt_p8_pointonly_8view_allcities_nocrop_lowtrunklr_e12_b9_2gpu_final` | final | P8 point-only/joint; point-only; no-crop | ok | 0.0555 | 90.0000 | 1.9800 | 1.8966 | -0.0014 | ok | 0.0471 | 90.0000 | 1.9831 | 1.8520 | -0.0014 |
| vggt | `vggt_p8_pointonly_8view_allcities_remotedominant_fromvggt_e16_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_p8_pointonly_8view_allcities_remotedominant_fromvggt_e16_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_p8_pointonly_8view_allcities_remoteheadonly_fromvggt_e12_b48_2gpu_final_best_slim` | best | P8 point-only/joint; point-only | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p8_pointonly_8view_allcities_remoteheadonly_fromvggt_e12_b48_2gpu_final_final_slim` | final | P8 point-only/joint; point-only | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |
| vggt | `vggt_p8_pointonly_8view_allcities_shared_pointhead_fromvggt_e12_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only | ok | 0.1353 | 73.3333 | 1.9734 | 1.8735 | 0.0078 | ok | 0.0912 | 64.5833 | 1.9742 | 1.8620 | 0.0026 |
| vggt | `vggt_p8_pointonly_remotedominant_aerial0_frommidtrunk_e8_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only | ok | 0.0835 | 76.6667 | 1.9760 | 1.8999 | 0.0037 | ok | 0.0747 | 67.9167 | 1.9765 | 1.8894 | 0.0041 |
| vggt | `vggt_p8_pointonly_remotedominant_aerial0_frommidtrunk_e8_b9_2gpu_final` | final | P8 point-only/joint; point-only | ok | 0.0791 | 78.3333 | 1.9765 | 1.8993 | 0.0033 | ok | 0.0705 | 68.7500 | 1.9770 | 1.8893 | 0.0039 |
| vggt | `vggt_p8_pointonly_remotedominant_aerial0_frommidtrunk_e8_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0791 | 78.3333 | 1.9765 | 1.8993 | 0.0033 | ok | 0.0705 | 68.7500 | 1.9770 | 1.8893 | 0.0039 |
| vggt | `vggt_p8_pointonly_scalealigned_max1000_w02_top5_frommidtrunk_e4_b9_2gpu_final` | final | P8 point-only/joint; point-only | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_scalealigned_max1000_w02_top5_frommidtrunk_e4_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_scalealigned_top5_frommidtrunk_e6_b9_2gpu_final` | final | P8 point-only/joint; point-only | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_scalealigned_top5_frommidtrunk_e6_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0551 | 88.3333 | 1.9800 | 1.8945 | 0.0016 | ok | 0.0449 | 91.2500 | 1.9824 | 1.8806 | 0.0007 |
| vggt | `vggt_p8_pointonly_scalealigned_w05_top5_fromvggtbest_e6_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_p8_pointonly_scalealigned_w05_top5_fromvggtbest_e6_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_p8_pointonly_shapenorm_w2_top5_fromvggtbest_e8_b9_2gpu_best_slim` | best | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_p8_pointonly_shapenorm_w2_top5_fromvggtbest_e8_b9_2gpu_final_slim` | final | P8 point-only/joint; point-only | ok | 0.0891 | 70.0000 | 1.9748 | 1.8731 | 0.0069 | ok | 0.0776 | 57.9167 | 1.9747 | 1.8731 | 0.0049 |
| vggt | `vggt_raw_pretrained_image_input` | raw | 未微调基线 | ok | 0.1875 | 61.6667 | 1.9710 | 1.9008 | 0.0096 | ok | 0.1639 | 41.6667 | 1.9769 | 1.8679 | 0.0120 |

## 失败记录

| n_scenes | record_label | 原因 | log |
|---:|---|---|---|
| 2 | `vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best` | checkpoint zip archive 损坏，PytorchStreamReader 无法读取 central directory | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8/n2/vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best/codex_run.log` |
| 8 | `vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best` | checkpoint zip archive 损坏，PytorchStreamReader 无法读取 central directory | `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_all_models_4v_n2_n8_remote_norm_b8/n8/vggt_p7_dptinit_auxonly_2city_highbucket_e6_b8_2gpu_best/codex_run.log` |
