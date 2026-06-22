# Crossview NUM_VIEWS Hard-20 评测

## 评测目的

本轮评测不再使用 `n_scenes=2/8` 作为主变量；`n_scenes` 只是评测样本数量，不反映多视角输入规模。新的主变量改为 aerial 输入视角数 `NUM_VIEWS`，用于观察模型指标随 aerial 视角数变化的趋势。

## 评测口径

- 视角数 sweep：`NUM_VIEWS = 2, 4, 8, 16, 32`。
- 固定场景：同一批 20 个 hard scenes，避免不同视角数下样本集合变化污染结论。
- hard scenes 选择方式：先用未微调 `vggt_raw_pretrained_image_input` 在 New York 当前 paired scenes 上跑 `NUM_VIEWS=4`，再按 Aerial-only `pointmaps_abs_rel` 从高到低选择最差 20 个场景。
- 模型范围：保留 raw baseline、Pi3、VGGT P5/P6、VGGT Omega 等非 P7/P8 模型；排除所有 label/path 中包含 `p7` 或 `p8` 的模型。
- remote 主指标：`rs_point_abs_rel` 使用 Sim(3) 最优对齐后的 remote 点云误差；remote-only 与 joint-remote 口径一致，可以直接计算相对收益。
- remote 辅助指标：`rs_point_abs_rel_view0` 保留为 joint 坐标一致性参考；`rs_point_abs_rel_flattened` 保留为旧口径/debug 对照。
- 不使用 remote 对照输入：不评测 `same/blank/shuffled`。

## 固定场景

scene list:
`/root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial/val/Crossview_rs_aerial_hard20_vggt_raw_4v_newyork.npy`

选择明细:
`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/hard_scene_source_vggt_raw_4v_newyork25/hard20_scenes.json`

| rank | scene | source aerial_pointmaps_abs_rel |
|---:|---|---:|
| 1 | `newyork__location_468` | 1.0140 |
| 2 | `newyork__location_461` | 0.7415 |
| 3 | `newyork__location_475` | 0.4104 |
| 4 | `newyork__location_453` | 0.3147 |
| 5 | `newyork__location_467` | 0.2702 |
| 6 | `newyork__location_451` | 0.2373 |
| 7 | `newyork__location_464` | 0.2295 |
| 8 | `newyork__location_466` | 0.2190 |
| 9 | `newyork__location_459` | 0.2128 |
| 10 | `newyork__location_455` | 0.2013 |
| 11 | `newyork__location_458` | 0.1988 |
| 12 | `newyork__location_462` | 0.1876 |
| 13 | `newyork__location_471` | 0.1673 |
| 14 | `newyork__location_452` | 0.1567 |
| 15 | `newyork__location_463` | 0.1426 |
| 16 | `newyork__location_457` | 0.1383 |
| 17 | `newyork__location_469` | 0.1296 |
| 18 | `newyork__location_473` | 0.1190 |
| 19 | `newyork__location_465` | 0.1175 |
| 20 | `newyork__location_454` | 0.1059 |

## 输出路径

正式输出根目录建议使用：
`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8`

正式评测状态：已完成。

结果规模：
- `NUM_VIEWS=2/4/8/16/32` 各 20 个模型结果。
- 共 100 个 `rs_aerial_benchmark_results.json`。

聚合结果：
- CSV: `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8/aggregate_results.csv`
- JSON: `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8/aggregate_results.json`
- completion summary: `/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8/completion_summary.json`

smoke test 已验证：
`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8_smoke`

smoke test 配置：
- 模型：`pi3_raw_pretrained_image_input`
- `NUM_VIEWS=2`
- hard-20 scene list
- `paired_scene_count=20`
- 结果状态：ok

## 正式运行命令

```bash
cd /root/autodl-tmp/Models/map-anything
python scripts/evaluate_crossview_all_models.py \
  --out-root /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8 \
  --num-views 2 4 8 16 32 \
  --scene-list-path /root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial/val/Crossview_rs_aerial_hard20_vggt_raw_4v_newyork.npy \
  --remote-control-modes none \
  --exclude-label-patterns p7,p8 \
  --exclude-patterns debug,smoke,overfit,probe,p7,p8 \
  --skip-missing \
  --discover-checkpoints \
  --cuda-devices 0,1 \
  --workers 2 \
  --batch-size 4
```

说明：`NUM_VIEWS=16/32` 显存压力更高；如果 OOM，优先把 `--batch-size` 降到 `1` 或降低 `--workers`。

## 运行补充

首次正式运行使用 `--batch-size 4 --workers 2`。其中 `NUM_VIEWS=32` 的 9 个 VGGT 任务发生 CUDA OOM，随后使用 `--batch-size 1` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 只重跑失败任务，全部成功。

重跑命令：

```bash
cd /root/autodl-tmp/Models/map-anything
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/evaluate_crossview_all_models.py \
  --out-root /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/crossview_num_views_hard20_no_p7p8 \
  --num-views 32 \
  --scene-list-path /root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial/val/Crossview_rs_aerial_hard20_vggt_raw_4v_newyork.npy \
  --remote-control-modes none \
  --skip-missing \
  --discover-checkpoints \
  --cuda-devices 0,1 \
  --workers 2 \
  --batch-size 1 \
  --force \
  --only \
    vggt_raw_pretrained_image_input \
    vggt_p5b_shared_norm \
    vggt_p5c_viewtype \
    vggt_p5d_remote_point_head_consistency \
    vggt_p5e_remote_head_attention_viewtype \
    vggt_p5b_vggt_joint_shared_all_shared_norm_final \
    vggt_p5c_vggt_joint_shared_all_viewtype_final \
    vggt_p5d_vggt_remote_point_head_consistency_final \
    vggt_p5e_vggt_remote_head_attention_viewtype_final
```

## 记录的模型

| family | 模型数 | 主要覆盖特点 |
| --- | ---: | --- |
| pi3 | 11 | P3; final; modality embedding; freeze shared; zero-covis; 未微调基线 |
| vggt | 9 | final; view-type; P5B/shared-norm; remote point-head; private remote head; 未微调基线 |

补充：未微调基线已纳入：`pi3_raw_pretrained_image_input`、`vggt_raw_pretrained_image_input`。P7/P8 已全部排除。

## 绝对精度 Top 10

排序：按 `joint_pointmaps_abs_rel` 从低到高；并列时参考 `joint_remote_pointmaps_abs_rel` 与 `joint_pose_auc_5`。

### NUM_VIEWS=2

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head; final | 0.0572 | 89.0000 | 0.2791 | 0.0517 | 0.0501 |
| 2 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0583 | 89.0000 | 0.2704 | 0.0212 | 0.0322 |
| 3 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | view-type; final | 0.0730 | 80.0000 | 0.4837 | 0.0504 | 0.0570 |
| 4 | vggt | `vggt_p5c_viewtype` | view-type | 0.0743 | 81.0000 | 0.5184 | 0.0517 | 0.0569 |
| 5 | pi3 | `pi3_p3_pi3_modality_embedding_final` | P3; modality embedding; final | 0.0744 | 82.0000 | 0.4894 | 0.0340 | 0.0599 |
| 6 | pi3 | `pi3_p3_modality_embedding` | P3; modality embedding | 0.0744 | 80.0000 | 0.4933 | 0.0340 | 0.0600 |
| 7 | pi3 | `pi3_p3_modality_embedding_remote_head` | P3; modality embedding | 0.0748 | 81.0000 | 0.4415 | 0.0443 | 0.0692 |
| 8 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | P3; modality embedding; final | 0.0748 | 81.0000 | 0.4415 | 0.0443 | 0.0692 |
| 9 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | P5B/shared-norm; final | 0.0754 | 83.0000 | 0.4700 | 0.0531 | 0.0569 |
| 10 | vggt | `vggt_p5b_shared_norm` | P5B/shared-norm | 0.0758 | 81.0000 | 0.4851 | 0.0537 | 0.0576 |

### NUM_VIEWS=4

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head; final | 0.0522 | 94.0000 | 0.2926 | 0.0515 | 0.0501 |
| 2 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0532 | 93.6667 | 0.2897 | 0.0196 | 0.0322 |
| 3 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | remote point-head; final | 0.0631 | 86.0000 | 0.4324 | 0.0502 | 0.0564 |
| 4 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | view-type; final | 0.0633 | 84.6667 | 0.4275 | 0.0492 | 0.0570 |
| 5 | pi3 | `pi3_p3_pi3_modality_embedding_final` | P3; modality embedding; final | 0.0637 | 89.8333 | 0.3832 | 0.0314 | 0.0599 |
| 6 | pi3 | `pi3_p3_modality_embedding` | P3; modality embedding | 0.0637 | 89.8333 | 0.3808 | 0.0314 | 0.0600 |
| 7 | vggt | `vggt_p5d_remote_point_head_consistency` | remote point-head | 0.0643 | 85.6667 | 0.4607 | 0.0260 | 0.0511 |
| 8 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | P5B/shared-norm; final | 0.0643 | 85.6667 | 0.4454 | 0.0539 | 0.0569 |
| 9 | pi3 | `pi3_p3_pi3_zero_covis_final` | P3; zero-covis; final | 0.0644 | 85.1667 | 0.4146 | 0.0337 | 0.0594 |
| 10 | vggt | `vggt_p5c_viewtype` | view-type | 0.0648 | 82.8333 | 0.4388 | 0.0508 | 0.0569 |

### NUM_VIEWS=8

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head; final | 0.0477 | 91.0000 | 0.3034 | 0.0508 | 0.0501 |
| 2 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0480 | 90.2500 | 0.2917 | 0.0182 | 0.0322 |
| 3 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | view-type; final | 0.0533 | 88.8214 | 0.3179 | 0.0521 | 0.0570 |
| 4 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | remote point-head; final | 0.0537 | 89.2143 | 0.3168 | 0.0508 | 0.0564 |
| 5 | vggt | `vggt_p5d_remote_point_head_consistency` | remote point-head | 0.0544 | 89.2500 | 0.3284 | 0.0239 | 0.0511 |
| 6 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | P5B/shared-norm; final | 0.0546 | 88.2857 | 0.3289 | 0.0555 | 0.0569 |
| 7 | vggt | `vggt_p5c_viewtype` | view-type | 0.0547 | 87.0000 | 0.3301 | 0.0539 | 0.0569 |
| 8 | vggt | `vggt_p5b_shared_norm` | P5B/shared-norm | 0.0550 | 87.6786 | 0.3422 | 0.0559 | 0.0576 |
| 9 | pi3 | `pi3_p3_pi3_modality_embedding_final` | P3; modality embedding; final | 0.0559 | 92.3571 | 0.2997 | 0.0286 | 0.0599 |
| 10 | pi3 | `pi3_p3_modality_embedding_remote_head` | P3; modality embedding | 0.0560 | 92.0000 | 0.3025 | 0.0400 | 0.0692 |

### NUM_VIEWS=16

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0528 | 89.3250 | 0.3097 | 0.0181 | 0.0322 |
| 2 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head; final | 0.0530 | 90.2833 | 0.3065 | 0.0505 | 0.0501 |
| 3 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | view-type; final | 0.0550 | 90.9250 | 0.2942 | 0.0525 | 0.0570 |
| 4 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | remote point-head; final | 0.0552 | 90.4833 | 0.2990 | 0.0508 | 0.0564 |
| 5 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | P5B/shared-norm; final | 0.0554 | 89.8250 | 0.2999 | 0.0581 | 0.0569 |
| 6 | vggt | `vggt_p5d_remote_point_head_consistency` | remote point-head | 0.0558 | 90.3167 | 0.3036 | 0.0238 | 0.0511 |
| 7 | vggt | `vggt_p5b_shared_norm` | P5B/shared-norm | 0.0559 | 89.5583 | 0.3080 | 0.0584 | 0.0576 |
| 8 | vggt | `vggt_p5c_viewtype` | view-type | 0.0565 | 88.6750 | 0.3022 | 0.0554 | 0.0569 |
| 9 | pi3 | `pi3_p3_pi3_zero_covis_final` | P3; zero-covis; final | 0.0577 | 89.3667 | 0.3198 | 0.0311 | 0.0594 |
| 10 | pi3 | `pi3_p3_pi3_modality_embedding_final` | P3; modality embedding; final | 0.0591 | 91.9583 | 0.2733 | 0.0291 | 0.0599 |

### NUM_VIEWS=32

| rank | family | record_label | 主要特点 | joint_pointmaps_abs_rel | joint_pose_auc_5 | joint_ray_dirs_err_deg | joint_remote_pointmaps_abs_rel | remote_pointmaps_abs_rel |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 1 | vggt | `vggt_p5e_remote_head_attention_viewtype` | view-type; private remote head | 0.0523 | 87.1573 | 0.3079 | 0.0181 | 0.0322 |
| 2 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | view-type; private remote head; final | 0.0524 | 88.0484 | 0.3040 | 0.0503 | 0.0501 |
| 3 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | view-type; final | 0.0548 | 88.4415 | 0.3099 | 0.0520 | 0.0570 |
| 4 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | remote point-head; final | 0.0552 | 88.4496 | 0.3044 | 0.0504 | 0.0564 |
| 5 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | P5B/shared-norm; final | 0.0554 | 87.7419 | 0.3151 | 0.0568 | 0.0569 |
| 6 | vggt | `vggt_p5d_remote_point_head_consistency` | remote point-head | 0.0560 | 88.4294 | 0.3163 | 0.0240 | 0.0511 |
| 7 | vggt | `vggt_p5b_shared_norm` | P5B/shared-norm | 0.0562 | 87.7500 | 0.3272 | 0.0569 | 0.0576 |
| 8 | vggt | `vggt_p5c_viewtype` | view-type | 0.0563 | 86.8468 | 0.3179 | 0.0545 | 0.0569 |
| 9 | pi3 | `pi3_p3_pi3_zero_covis_final` | P3; zero-covis; final | 0.0569 | 88.8669 | 0.3232 | 0.0320 | 0.0594 |
| 10 | pi3 | `pi3_p3_pi3_modality_embedding_final` | P3; modality embedding; final | 0.0575 | 90.9919 | 0.2650 | 0.0301 | 0.0599 |

## 卫星输入收益 Top 10

排序：按 `aerial_pointmaps_abs_rel - joint_pointmaps_abs_rel` 从高到低。`point_rel` 与 `remote_point_rel` 均为误差相对下降，正数表示 Joint 更好。

### NUM_VIEWS=2

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | vggt | `vggt_raw_pretrained_image_input` | 0.2762 | 0.2396 | 0.0365 | +13.23% | 42.0000 | 49.0000 | 7.0000 | 0.0647 | 0.0581 | 0.0066 | +10.28% |
| 2 | pi3 | `pi3_p3_freeze_shared` | 0.1085 | 0.0842 | 0.0243 | +22.40% | 73.0000 | 72.0000 | -1.0000 | 0.0630 | 0.0413 | 0.0218 | +34.52% |
| 3 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.1051 | 0.0838 | 0.0213 | +20.29% | 75.0000 | 72.0000 | -3.0000 | 0.0631 | 0.0412 | 0.0219 | +34.75% |
| 4 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | 0.0713 | 0.0572 | 0.0141 | +19.72% | 85.0000 | 89.0000 | 4.0000 | 0.0501 | 0.0517 | -0.0016 | -3.09% |
| 5 | vggt | `vggt_p5e_remote_head_attention_viewtype` | 0.0723 | 0.0583 | 0.0139 | +19.30% | 83.0000 | 89.0000 | 6.0000 | 0.0322 | 0.0212 | 0.0110 | +34.25% |
| 6 | vggt | `vggt_p5c_viewtype` | 0.0877 | 0.0743 | 0.0134 | +15.28% | 74.0000 | 81.0000 | 7.0000 | 0.0569 | 0.0517 | 0.0052 | +9.13% |
| 7 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | 0.0851 | 0.0730 | 0.0122 | +14.29% | 76.0000 | 80.0000 | 4.0000 | 0.0570 | 0.0504 | 0.0066 | +11.65% |
| 8 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | 0.0814 | 0.0754 | 0.0060 | +7.35% | 77.0000 | 83.0000 | 6.0000 | 0.0569 | 0.0531 | 0.0038 | +6.74% |
| 9 | vggt | `vggt_p5b_shared_norm` | 0.0817 | 0.0758 | 0.0059 | +7.24% | 80.0000 | 81.0000 | 1.0000 | 0.0576 | 0.0537 | 0.0039 | +6.70% |
| 10 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | 0.0801 | 0.0762 | 0.0039 | +4.83% | 78.0000 | 79.0000 | 1.0000 | 0.0564 | 0.0520 | 0.0043 | +7.70% |

### NUM_VIEWS=4

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | vggt | `vggt_raw_pretrained_image_input` | 0.2385 | 0.2257 | 0.0128 | +5.36% | 41.8333 | 46.0000 | 4.1667 | 0.0647 | 0.0569 | 0.0079 | +12.15% |
| 2 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | 0.0559 | 0.0522 | 0.0037 | +6.68% | 91.5000 | 94.0000 | 2.5000 | 0.0501 | 0.0515 | -0.0013 | -2.65% |
| 3 | vggt | `vggt_p5e_remote_head_attention_viewtype` | 0.0566 | 0.0532 | 0.0033 | +5.86% | 91.8333 | 93.6667 | 1.8333 | 0.0322 | 0.0196 | 0.0125 | +38.93% |
| 4 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | 0.0666 | 0.0633 | 0.0033 | +4.90% | 85.0000 | 84.6667 | -0.3333 | 0.0570 | 0.0492 | 0.0078 | +13.74% |
| 5 | vggt | `vggt_p5c_viewtype` | 0.0679 | 0.0648 | 0.0031 | +4.51% | 82.3333 | 82.8333 | 0.5000 | 0.0569 | 0.0508 | 0.0061 | +10.75% |
| 6 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | 0.0657 | 0.0643 | 0.0014 | +2.19% | 84.6667 | 85.6667 | 1.0000 | 0.0569 | 0.0539 | 0.0030 | +5.27% |
| 7 | vggt | `vggt_p5b_shared_norm` | 0.0665 | 0.0651 | 0.0014 | +2.16% | 84.1667 | 85.0000 | 0.8333 | 0.0576 | 0.0542 | 0.0034 | +5.90% |
| 8 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0658 | 0.0649 | 0.0009 | +1.37% | 92.1667 | 89.1667 | -3.0000 | 0.0692 | 0.0421 | 0.0271 | +39.13% |
| 9 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0658 | 0.0649 | 0.0009 | +1.37% | 92.1667 | 89.1667 | -3.0000 | 0.0692 | 0.0421 | 0.0271 | +39.13% |
| 10 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0639 | 0.0637 | 0.0002 | +0.30% | 92.1667 | 89.8333 | -2.3333 | 0.0599 | 0.0314 | 0.0285 | +47.58% |

### NUM_VIEWS=8

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | vggt | `vggt_raw_pretrained_image_input` | 0.2271 | 0.2168 | 0.0103 | +4.55% | 45.7500 | 48.1786 | 2.4286 | 0.0647 | 0.0588 | 0.0059 | +9.10% |
| 2 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | 0.0493 | 0.0477 | 0.0016 | +3.20% | 89.6786 | 91.0000 | 1.3214 | 0.0501 | 0.0508 | -0.0007 | -1.39% |
| 3 | vggt | `vggt_p5e_remote_head_attention_viewtype` | 0.0494 | 0.0480 | 0.0014 | +2.84% | 88.7857 | 90.2500 | 1.4643 | 0.0322 | 0.0182 | 0.0139 | +43.35% |
| 4 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0689 | 0.0676 | 0.0014 | +1.96% | 87.1429 | 86.0000 | -1.1429 | 0.0631 | 0.0353 | 0.0278 | +44.06% |
| 5 | pi3 | `pi3_p3_freeze_shared` | 0.0696 | 0.0684 | 0.0012 | +1.70% | 87.0000 | 85.2143 | -1.7857 | 0.0630 | 0.0354 | 0.0276 | +43.82% |
| 6 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | 0.0536 | 0.0533 | 0.0002 | +0.46% | 89.5714 | 88.8214 | -0.7500 | 0.0570 | 0.0521 | 0.0049 | +8.59% |
| 7 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0561 | 0.0560 | 0.0002 | +0.33% | 92.4643 | 92.0000 | -0.4643 | 0.0692 | 0.0400 | 0.0292 | +42.18% |
| 8 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0561 | 0.0560 | 0.0002 | +0.33% | 92.4643 | 92.0000 | -0.4643 | 0.0692 | 0.0400 | 0.0292 | +42.18% |
| 9 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | 0.0546 | 0.0546 | 0.0001 | +0.11% | 88.5714 | 88.2857 | -0.2857 | 0.0569 | 0.0555 | 0.0015 | +2.55% |
| 10 | vggt | `vggt_p5b_shared_norm` | 0.0550 | 0.0550 | -0.0000 | -0.00% | 88.3214 | 87.6786 | -0.6429 | 0.0576 | 0.0559 | 0.0017 | +2.96% |

### NUM_VIEWS=16

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | 0.0537 | 0.0530 | 0.0007 | +1.38% | 89.5750 | 90.2833 | 0.7083 | 0.0501 | 0.0505 | -0.0003 | -0.65% |
| 2 | vggt | `vggt_p5e_remote_head_attention_viewtype` | 0.0534 | 0.0528 | 0.0006 | +1.06% | 88.5583 | 89.3250 | 0.7667 | 0.0322 | 0.0181 | 0.0141 | +43.75% |
| 3 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | 0.0551 | 0.0550 | 0.0001 | +0.18% | 90.5833 | 90.9250 | 0.3417 | 0.0570 | 0.0525 | 0.0045 | +7.89% |
| 4 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0599 | 0.0598 | 0.0001 | +0.09% | 92.4833 | 92.0833 | -0.4000 | 0.0692 | 0.0400 | 0.0291 | +42.11% |
| 5 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0599 | 0.0598 | 0.0001 | +0.09% | 92.4833 | 92.0833 | -0.4000 | 0.0692 | 0.0400 | 0.0291 | +42.11% |
| 6 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | 0.0554 | 0.0554 | 0.0000 | +0.04% | 89.5167 | 89.8250 | 0.3083 | 0.0569 | 0.0581 | -0.0011 | -1.99% |
| 7 | vggt | `vggt_p5c_viewtype` | 0.0566 | 0.0565 | 0.0000 | +0.02% | 88.6500 | 88.6750 | 0.0250 | 0.0569 | 0.0554 | 0.0014 | +2.49% |
| 8 | vggt | `vggt_p5b_shared_norm` | 0.0559 | 0.0559 | 0.0000 | +0.00% | 89.2167 | 89.5583 | 0.3417 | 0.0576 | 0.0584 | -0.0008 | -1.34% |
| 9 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | 0.0551 | 0.0552 | -0.0000 | -0.05% | 90.1750 | 90.4833 | 0.3083 | 0.0564 | 0.0508 | 0.0056 | +9.95% |
| 10 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0590 | 0.0591 | -0.0001 | -0.10% | 92.2750 | 91.9583 | -0.3167 | 0.0599 | 0.0291 | 0.0309 | +51.48% |

### NUM_VIEWS=32

| rank | family | record_label | aerial_point | joint_point | point_delta | point_rel | aerial_auc5 | joint_auc5 | auc5_delta | remote_point | joint_remote_point | remote_point_delta | remote_point_rel |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | vggt | `vggt_raw_pretrained_image_input` | 0.1578 | 0.1556 | 0.0022 | +1.41% | 51.3427 | 51.7702 | 0.4274 | 0.0647 | 0.0547 | 0.0100 | +15.51% |
| 2 | vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | 0.0528 | 0.0524 | 0.0004 | +0.75% | 87.7298 | 88.0484 | 0.3185 | 0.0501 | 0.0503 | -0.0002 | -0.33% |
| 3 | vggt | `vggt_p5e_remote_head_attention_viewtype` | 0.0526 | 0.0523 | 0.0003 | +0.57% | 86.9315 | 87.1573 | 0.2258 | 0.0322 | 0.0181 | 0.0141 | +43.73% |
| 4 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0584 | 0.0583 | 0.0001 | +0.17% | 90.3004 | 90.1915 | -0.1089 | 0.0692 | 0.0409 | 0.0283 | +40.93% |
| 5 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0584 | 0.0583 | 0.0001 | +0.17% | 90.3004 | 90.1915 | -0.1089 | 0.0692 | 0.0409 | 0.0283 | +40.93% |
| 6 | vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | 0.0547 | 0.0548 | -0.0001 | -0.13% | 88.0867 | 88.4415 | 0.3548 | 0.0570 | 0.0520 | 0.0050 | +8.75% |
| 7 | vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | 0.0553 | 0.0554 | -0.0001 | -0.19% | 87.6552 | 87.7419 | 0.0867 | 0.0569 | 0.0568 | 0.0002 | +0.26% |
| 8 | vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | 0.0551 | 0.0552 | -0.0001 | -0.19% | 88.3488 | 88.4496 | 0.1008 | 0.0564 | 0.0504 | 0.0060 | +10.57% |
| 9 | vggt | `vggt_p5b_shared_norm` | 0.0560 | 0.0562 | -0.0001 | -0.23% | 87.7823 | 87.7500 | -0.0323 | 0.0576 | 0.0569 | 0.0007 | +1.25% |
| 10 | vggt | `vggt_p5c_viewtype` | 0.0562 | 0.0563 | -0.0001 | -0.26% | 86.6048 | 86.8468 | 0.2419 | 0.0569 | 0.0545 | 0.0024 | +4.15% |

## Remote 点云收益 Top 10

排序：按 `remote_pointmaps_abs_rel - joint_remote_pointmaps_abs_rel` 从高到低。主 remote 指标为 Sim(3) 对齐后的 `rs_point_abs_rel`。

### NUM_VIEWS=2

| rank | family | record_label | remote_point | joint_remote_point | remote_point_delta | remote_point_rel | aerial_point | joint_point | point_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi3 | `pi3_p3_modality_embedding` | 0.0600 | 0.0340 | 0.0259 | +43.24% | 0.0742 | 0.0744 | -0.0002 |
| 2 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0599 | 0.0340 | 0.0259 | +43.23% | 0.0744 | 0.0744 | 0.0001 |
| 3 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0692 | 0.0443 | 0.0249 | +36.02% | 0.0766 | 0.0748 | 0.0018 |
| 4 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0692 | 0.0443 | 0.0249 | +36.02% | 0.0766 | 0.0748 | 0.0018 |
| 5 | pi3 | `pi3_p3_pi3_zero_covis_final` | 0.0594 | 0.0354 | 0.0240 | +40.35% | 0.0770 | 0.0770 | -0.0001 |
| 6 | pi3 | `pi3_p3_zero_covis` | 0.0597 | 0.0362 | 0.0235 | +39.32% | 0.0776 | 0.0765 | 0.0011 |
| 7 | pi3 | `pi3_p3_pi3_base_final` | 0.0612 | 0.0391 | 0.0221 | +36.07% | 0.0788 | 0.0768 | 0.0020 |
| 8 | vggt | `vggt_p5d_remote_point_head_consistency` | 0.0511 | 0.0291 | 0.0220 | +43.00% | 0.0799 | 0.0772 | 0.0028 |
| 9 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0631 | 0.0412 | 0.0219 | +34.75% | 0.1051 | 0.0838 | 0.0213 |
| 10 | pi3 | `pi3_p3_freeze_shared` | 0.0630 | 0.0413 | 0.0218 | +34.52% | 0.1085 | 0.0842 | 0.0243 |

### NUM_VIEWS=4

| rank | family | record_label | remote_point | joint_remote_point | remote_point_delta | remote_point_rel | aerial_point | joint_point | point_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi3 | `pi3_p3_modality_embedding` | 0.0600 | 0.0314 | 0.0285 | +47.58% | 0.0639 | 0.0637 | 0.0002 |
| 2 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0599 | 0.0314 | 0.0285 | +47.58% | 0.0639 | 0.0637 | 0.0002 |
| 3 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0692 | 0.0421 | 0.0271 | +39.13% | 0.0658 | 0.0649 | 0.0009 |
| 4 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0692 | 0.0421 | 0.0271 | +39.13% | 0.0658 | 0.0649 | 0.0009 |
| 5 | pi3 | `pi3_p3_pi3_zero_covis_final` | 0.0594 | 0.0337 | 0.0257 | +43.21% | 0.0643 | 0.0644 | -0.0001 |
| 6 | vggt | `vggt_p5d_remote_point_head_consistency` | 0.0511 | 0.0260 | 0.0252 | +49.25% | 0.0643 | 0.0643 | 0.0000 |
| 7 | pi3 | `pi3_p3_zero_covis` | 0.0597 | 0.0346 | 0.0251 | +42.09% | 0.0651 | 0.0656 | -0.0005 |
| 8 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0631 | 0.0385 | 0.0245 | +38.88% | 0.0752 | 0.0753 | -0.0001 |
| 9 | pi3 | `pi3_p3_freeze_shared` | 0.0630 | 0.0385 | 0.0245 | +38.84% | 0.0760 | 0.0760 | 0.0000 |
| 10 | pi3 | `pi3_p3_base` | 0.0622 | 0.0380 | 0.0242 | +38.90% | 0.0666 | 0.0673 | -0.0007 |

### NUM_VIEWS=8

| rank | family | record_label | remote_point | joint_remote_point | remote_point_delta | remote_point_rel | aerial_point | joint_point | point_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi3 | `pi3_p3_modality_embedding` | 0.0600 | 0.0286 | 0.0314 | +52.37% | 0.0558 | 0.0561 | -0.0002 |
| 2 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0599 | 0.0286 | 0.0313 | +52.26% | 0.0557 | 0.0559 | -0.0003 |
| 3 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0692 | 0.0400 | 0.0292 | +42.18% | 0.0561 | 0.0560 | 0.0002 |
| 4 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0692 | 0.0400 | 0.0292 | +42.18% | 0.0561 | 0.0560 | 0.0002 |
| 5 | pi3 | `pi3_p3_base` | 0.0622 | 0.0331 | 0.0290 | +46.68% | 0.0598 | 0.0603 | -0.0005 |
| 6 | pi3 | `pi3_p3_pi3_base_final` | 0.0612 | 0.0329 | 0.0283 | +46.26% | 0.0580 | 0.0584 | -0.0004 |
| 7 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0631 | 0.0353 | 0.0278 | +44.06% | 0.0689 | 0.0676 | 0.0014 |
| 8 | pi3 | `pi3_p3_freeze_shared` | 0.0630 | 0.0354 | 0.0276 | +43.82% | 0.0696 | 0.0684 | 0.0012 |
| 9 | pi3 | `pi3_p3_pi3_zero_covis_final` | 0.0594 | 0.0318 | 0.0276 | +46.47% | 0.0553 | 0.0561 | -0.0008 |
| 10 | pi3 | `pi3_p3_zero_covis` | 0.0597 | 0.0325 | 0.0272 | +45.55% | 0.0565 | 0.0575 | -0.0010 |

### NUM_VIEWS=16

| rank | family | record_label | remote_point | joint_remote_point | remote_point_delta | remote_point_rel | aerial_point | joint_point | point_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0599 | 0.0291 | 0.0309 | +51.48% | 0.0590 | 0.0591 | -0.0001 |
| 2 | pi3 | `pi3_p3_modality_embedding` | 0.0600 | 0.0291 | 0.0308 | +51.43% | 0.0591 | 0.0592 | -0.0001 |
| 3 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0692 | 0.0400 | 0.0291 | +42.11% | 0.0599 | 0.0598 | 0.0001 |
| 4 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0692 | 0.0400 | 0.0291 | +42.11% | 0.0599 | 0.0598 | 0.0001 |
| 5 | pi3 | `pi3_p3_base` | 0.0622 | 0.0335 | 0.0287 | +46.16% | 0.0615 | 0.0616 | -0.0001 |
| 6 | pi3 | `pi3_p3_pi3_base_final` | 0.0612 | 0.0328 | 0.0284 | +46.37% | 0.0597 | 0.0600 | -0.0002 |
| 7 | pi3 | `pi3_p3_pi3_zero_covis_final` | 0.0594 | 0.0311 | 0.0283 | +47.64% | 0.0576 | 0.0577 | -0.0001 |
| 8 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0631 | 0.0349 | 0.0282 | +44.67% | 0.0662 | 0.0671 | -0.0009 |
| 9 | pi3 | `pi3_p3_freeze_shared` | 0.0630 | 0.0350 | 0.0281 | +44.53% | 0.0669 | 0.0676 | -0.0007 |
| 10 | pi3 | `pi3_p3_zero_covis` | 0.0597 | 0.0320 | 0.0277 | +46.45% | 0.0597 | 0.0601 | -0.0004 |

### NUM_VIEWS=32

| rank | family | record_label | remote_point | joint_remote_point | remote_point_delta | remote_point_rel | aerial_point | joint_point | point_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi3 | `pi3_p3_modality_embedding` | 0.0600 | 0.0301 | 0.0299 | +49.77% | 0.0575 | 0.0577 | -0.0002 |
| 2 | pi3 | `pi3_p3_pi3_modality_embedding_final` | 0.0599 | 0.0301 | 0.0298 | +49.69% | 0.0574 | 0.0575 | -0.0002 |
| 3 | pi3 | `pi3_p3_modality_embedding_remote_head` | 0.0692 | 0.0409 | 0.0283 | +40.93% | 0.0584 | 0.0583 | 0.0001 |
| 4 | pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | 0.0692 | 0.0409 | 0.0283 | +40.93% | 0.0584 | 0.0583 | 0.0001 |
| 5 | pi3 | `pi3_p3_pi3_zero_covis_final` | 0.0594 | 0.0320 | 0.0274 | +46.19% | 0.0566 | 0.0569 | -0.0003 |
| 6 | vggt | `vggt_p5d_remote_point_head_consistency` | 0.0511 | 0.0240 | 0.0271 | +53.10% | 0.0559 | 0.0560 | -0.0002 |
| 7 | pi3 | `pi3_p3_zero_covis` | 0.0597 | 0.0326 | 0.0271 | +45.37% | 0.0584 | 0.0588 | -0.0005 |
| 8 | pi3 | `pi3_p3_pi3_freeze_shared_final` | 0.0631 | 0.0364 | 0.0266 | +42.22% | 0.0642 | 0.0649 | -0.0007 |
| 9 | pi3 | `pi3_p3_freeze_shared` | 0.0630 | 0.0364 | 0.0266 | +42.24% | 0.0651 | 0.0659 | -0.0008 |
| 10 | pi3 | `pi3_p3_base` | 0.0622 | 0.0362 | 0.0259 | +41.72% | 0.0619 | 0.0636 | -0.0016 |

## 全量结果

每个模型一行，只展示最重要的可视化字段；完整字段见输出目录下的 `aggregate_results.csv/json` 和各模型 `rs_aerial_benchmark_results.json`。`gain` 为 `aerial_pointmaps_abs_rel - joint_pointmaps_abs_rel`。

| family | record_label | ckpt_type | 主要特点 | v2_joint_point | v2_joint_auc5 | v2_remote | v2_joint_remote | v2_gain | v4_joint_point | v4_joint_auc5 | v4_remote | v4_joint_remote | v4_gain | v8_joint_point | v8_joint_auc5 | v8_remote | v8_joint_remote | v8_gain | v16_joint_point | v16_joint_auc5 | v16_remote | v16_joint_remote | v16_gain | v32_joint_point | v32_joint_auc5 | v32_remote | v32_joint_remote | v32_gain |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pi3 | `pi3_p3_base` | - | P3 | 0.0782 | 76.0000 | 0.0622 | 0.0405 | 0.0004 | 0.0673 | 87.0000 | 0.0622 | 0.0380 | -0.0007 | 0.0603 | 88.6786 | 0.0622 | 0.0331 | -0.0005 | 0.0616 | 90.4917 | 0.0622 | 0.0335 | -0.0001 | 0.0636 | 86.0282 | 0.0622 | 0.0362 | -0.0016 |
| pi3 | `pi3_p3_freeze_shared` | - | P3; freeze shared | 0.0842 | 72.0000 | 0.0630 | 0.0413 | 0.0243 | 0.0760 | 84.0000 | 0.0630 | 0.0385 | 0.0000 | 0.0684 | 85.2143 | 0.0630 | 0.0354 | 0.0012 | 0.0676 | 88.0833 | 0.0630 | 0.0350 | -0.0007 | 0.0659 | 86.9073 | 0.0630 | 0.0364 | -0.0008 |
| pi3 | `pi3_p3_modality_embedding` | - | P3; modality embedding | 0.0744 | 80.0000 | 0.0600 | 0.0340 | -0.0002 | 0.0637 | 89.8333 | 0.0600 | 0.0314 | 0.0002 | 0.0561 | 92.2500 | 0.0600 | 0.0286 | -0.0002 | 0.0592 | 91.8917 | 0.0600 | 0.0291 | -0.0001 | 0.0577 | 90.7016 | 0.0600 | 0.0301 | -0.0002 |
| pi3 | `pi3_p3_modality_embedding_remote_head` | - | P3; modality embedding | 0.0748 | 81.0000 | 0.0692 | 0.0443 | 0.0018 | 0.0649 | 89.1667 | 0.0692 | 0.0421 | 0.0009 | 0.0560 | 92.0000 | 0.0692 | 0.0400 | 0.0002 | 0.0598 | 92.0833 | 0.0692 | 0.0400 | 0.0001 | 0.0583 | 90.1915 | 0.0692 | 0.0409 | 0.0001 |
| pi3 | `pi3_p3_pi3_base_final` | final | P3; final | 0.0768 | 81.0000 | 0.0612 | 0.0391 | 0.0020 | 0.0658 | 89.3333 | 0.0612 | 0.0374 | -0.0001 | 0.0584 | 90.7857 | 0.0612 | 0.0329 | -0.0004 | 0.0600 | 92.5333 | 0.0612 | 0.0328 | -0.0002 | 0.0643 | 88.4698 | 0.0612 | 0.0354 | -0.0027 |
| pi3 | `pi3_p3_pi3_freeze_shared_final` | final | P3; freeze shared; final | 0.0838 | 72.0000 | 0.0631 | 0.0412 | 0.0213 | 0.0753 | 85.3333 | 0.0631 | 0.0385 | -0.0001 | 0.0676 | 86.0000 | 0.0631 | 0.0353 | 0.0014 | 0.0671 | 88.3417 | 0.0631 | 0.0349 | -0.0009 | 0.0649 | 87.3266 | 0.0631 | 0.0364 | -0.0007 |
| pi3 | `pi3_p3_pi3_modality_embedding_final` | final | P3; modality embedding; final | 0.0744 | 82.0000 | 0.0599 | 0.0340 | 0.0001 | 0.0637 | 89.8333 | 0.0599 | 0.0314 | 0.0002 | 0.0559 | 92.3571 | 0.0599 | 0.0286 | -0.0003 | 0.0591 | 91.9583 | 0.0599 | 0.0291 | -0.0001 | 0.0575 | 90.9919 | 0.0599 | 0.0301 | -0.0002 |
| pi3 | `pi3_p3_pi3_modality_embedding_remote_head_final` | final | P3; modality embedding; final | 0.0748 | 81.0000 | 0.0692 | 0.0443 | 0.0018 | 0.0649 | 89.1667 | 0.0692 | 0.0421 | 0.0009 | 0.0560 | 92.0000 | 0.0692 | 0.0400 | 0.0002 | 0.0598 | 92.0833 | 0.0692 | 0.0400 | 0.0001 | 0.0583 | 90.1915 | 0.0692 | 0.0409 | 0.0001 |
| pi3 | `pi3_p3_pi3_zero_covis_final` | final | P3; zero-covis; final | 0.0770 | 78.0000 | 0.0594 | 0.0354 | -0.0001 | 0.0644 | 85.1667 | 0.0594 | 0.0337 | -0.0001 | 0.0561 | 88.8929 | 0.0594 | 0.0318 | -0.0008 | 0.0577 | 89.3667 | 0.0594 | 0.0311 | -0.0001 | 0.0569 | 88.8669 | 0.0594 | 0.0320 | -0.0003 |
| pi3 | `pi3_p3_zero_covis` | - | P3; zero-covis | 0.0765 | 75.0000 | 0.0597 | 0.0362 | 0.0011 | 0.0656 | 83.5000 | 0.0597 | 0.0346 | -0.0005 | 0.0575 | 86.9286 | 0.0597 | 0.0325 | -0.0010 | 0.0601 | 87.5667 | 0.0597 | 0.0320 | -0.0004 | 0.0588 | 87.1008 | 0.0597 | 0.0326 | -0.0005 |
| pi3 | `pi3_raw_pretrained_image_input` | raw | 未微调基线 | 0.2046 | 58.0000 | 0.0678 | 0.0569 | -0.0100 | 0.1561 | 62.0000 | 0.0678 | 0.0516 | -0.0176 | 0.1741 | 65.8929 | 0.0678 | 0.0465 | -0.0272 | 0.1188 | 69.7333 | 0.0678 | 0.0451 | -0.0012 | 0.1282 | 66.4698 | 0.0678 | 0.0470 | -0.0031 |
| vggt | `vggt_p5b_shared_norm` | - | P5B/shared-norm | 0.0758 | 81.0000 | 0.0576 | 0.0537 | 0.0059 | 0.0651 | 85.0000 | 0.0576 | 0.0542 | 0.0014 | 0.0550 | 87.6786 | 0.0576 | 0.0559 | -0.0000 | 0.0559 | 89.5583 | 0.0576 | 0.0584 | 0.0000 | 0.0562 | 87.7500 | 0.0576 | 0.0569 | -0.0001 |
| vggt | `vggt_p5b_vggt_joint_shared_all_shared_norm_final` | final | P5B/shared-norm; final | 0.0754 | 83.0000 | 0.0569 | 0.0531 | 0.0060 | 0.0643 | 85.6667 | 0.0569 | 0.0539 | 0.0014 | 0.0546 | 88.2857 | 0.0569 | 0.0555 | 0.0001 | 0.0554 | 89.8250 | 0.0569 | 0.0581 | 0.0000 | 0.0554 | 87.7419 | 0.0569 | 0.0568 | -0.0001 |
| vggt | `vggt_p5c_vggt_joint_shared_all_viewtype_final` | final | view-type; final | 0.0730 | 80.0000 | 0.0570 | 0.0504 | 0.0122 | 0.0633 | 84.6667 | 0.0570 | 0.0492 | 0.0033 | 0.0533 | 88.8214 | 0.0570 | 0.0521 | 0.0002 | 0.0550 | 90.9250 | 0.0570 | 0.0525 | 0.0001 | 0.0548 | 88.4415 | 0.0570 | 0.0520 | -0.0001 |
| vggt | `vggt_p5c_viewtype` | - | view-type | 0.0743 | 81.0000 | 0.0569 | 0.0517 | 0.0134 | 0.0648 | 82.8333 | 0.0569 | 0.0508 | 0.0031 | 0.0547 | 87.0000 | 0.0569 | 0.0539 | -0.0000 | 0.0565 | 88.6750 | 0.0569 | 0.0554 | 0.0000 | 0.0563 | 86.8468 | 0.0569 | 0.0545 | -0.0001 |
| vggt | `vggt_p5d_remote_point_head_consistency` | - | remote point-head | 0.0772 | 78.0000 | 0.0511 | 0.0291 | 0.0028 | 0.0643 | 85.6667 | 0.0511 | 0.0260 | 0.0000 | 0.0544 | 89.2500 | 0.0511 | 0.0239 | -0.0001 | 0.0558 | 90.3167 | 0.0511 | 0.0238 | -0.0001 | 0.0560 | 88.4294 | 0.0511 | 0.0240 | -0.0002 |
| vggt | `vggt_p5d_vggt_remote_point_head_consistency_final` | final | remote point-head; final | 0.0762 | 79.0000 | 0.0564 | 0.0520 | 0.0039 | 0.0631 | 86.0000 | 0.0564 | 0.0502 | 0.0001 | 0.0537 | 89.2143 | 0.0564 | 0.0508 | -0.0000 | 0.0552 | 90.4833 | 0.0564 | 0.0508 | -0.0000 | 0.0552 | 88.4496 | 0.0564 | 0.0504 | -0.0001 |
| vggt | `vggt_p5e_remote_head_attention_viewtype` | - | view-type; private remote head | 0.0583 | 89.0000 | 0.0322 | 0.0212 | 0.0139 | 0.0532 | 93.6667 | 0.0322 | 0.0196 | 0.0033 | 0.0480 | 90.2500 | 0.0322 | 0.0182 | 0.0014 | 0.0528 | 89.3250 | 0.0322 | 0.0181 | 0.0006 | 0.0523 | 87.1573 | 0.0322 | 0.0181 | 0.0003 |
| vggt | `vggt_p5e_vggt_remote_head_attention_viewtype_final` | final | view-type; private remote head; final | 0.0572 | 89.0000 | 0.0501 | 0.0517 | 0.0141 | 0.0522 | 94.0000 | 0.0501 | 0.0515 | 0.0037 | 0.0477 | 91.0000 | 0.0501 | 0.0508 | 0.0016 | 0.0530 | 90.2833 | 0.0501 | 0.0505 | 0.0007 | 0.0524 | 88.0484 | 0.0501 | 0.0503 | 0.0004 |
| vggt | `vggt_raw_pretrained_image_input` | raw | 未微调基线 | 0.2396 | 49.0000 | 0.0647 | 0.0581 | 0.0365 | 0.2257 | 46.0000 | 0.0647 | 0.0569 | 0.0128 | 0.2168 | 48.1786 | 0.0647 | 0.0588 | 0.0103 | 0.1676 | 52.4583 | 0.0647 | 0.0546 | -0.0023 | 0.1556 | 51.7702 | 0.0647 | 0.0547 | 0.0022 |
