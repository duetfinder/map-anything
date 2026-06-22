# P7 Remote 重建改进实验跟踪

更新时间：2026-06-13

本文档用于替代过长的阶段性口述记录，只维护“做过什么、输出在哪里、结论是什么”。正式 benchmark 统一使用 `rs_guided_dense_mv/newyork/remote_pointmetric20`；但 remote 点云筛选不能只看 raw metric `rs_point_l1`，还必须同时看 `rs_point_l1_scale_aligned`、`rs_height_mae_affine` 和 PLY 可视化。注意：现有 `rs_point_l1_centered` 只去除了预测/GT 的平移中心，没有做尺度归一化，因此它仍是绝对尺度下的点云误差；`rs_point_l1_scale_aligned` 是当前新增的尺度无关 remote 点云形状指标，只允许平移和正尺度对齐，不允许负尺度反射。

## 当前结论

当前需要分两类保留和比较：

`p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu`

它的 raw benchmark 结果目前最好：`RS-only rs_point_l1=292.558`，`Joint rs_point_l1=292.122`，仍作为常规 remote point head 的参考基线。

`p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_b9_2gpu`

它代表当前 scale-free 多任务方向的最强证据：aux xyz residual benchmark 的 `RS-only rs_point_l1_scale_aligned=30.108`，与 2-city xyz residual 的 `30.039` 基本持平，明显好于 xy-only 的 `43.629` 和旧 projection/global 系列的约 `95`。但 raw/centered 仍约 `291/95`，说明它学到的是 normalized 形状修正，不是绝对尺度/中心/真实高度恢复。

总体判断：

- P7 aux 显式投影 height/offset 是可学习的，但它和 remote point head 的最终点云质量耦合仍弱。
- MoGe 伪标签作为弱先验有帮助但幅度有限，主要改善局部高度/边缘约束，不足以单独解决 remote 点云形状问题。
- 增加普通视角数量、low-covis、zero-covis 都没有直接改善 remote 点云误差。
- 从 aux 输出反投影回点云后再和 GT pointmap 算损失时，必须先按 pointmap loss 的尺度归一化对齐；已验证这能避免错误的 metric 尺度约束，但低权重加入后仍没有改善局部形状。
- 目前最可靠的改进来自：P5B private remote point head + token-source projection aux + MoGe height/gradient/edge 弱先验 + overlap 区域强化 + 较鲁棒的 top-percent pointmap 损失。
- 已明确训练和评估都应按尺度无关处理：point cloud 侧使用中心化 + 正尺度对齐，height 侧使用 normalized/affine 评价；`rs_point_l1_centered` 不能当作尺度无关指标。
- 从 GT `pixel_to_point_map` 拟合每个 remote 样本的低维 global direction/slope target 后，监督覆盖稳定、训练链路正常，但 aux global vector 仍接近常数输出，正式 `rs_point_l1_scale_aligned` 仍约 `95.49`；因此当前瓶颈不是旧 projection_aux 稀疏 direction 标签本身。
- 最近一轮 dense height、dense global-offset、aux reconstructed point loss、DPT/head 初始化、从 base 重新训练、更多 view、strong overlap 等方向都没有让 remote 点云质量发生稳定突破。当前更像是损失调度和任务耦合方式的问题，而不是训练数据完全错误或模型容量明显不够。
- 新增 self-contained grid global 反投影实验后，point head 验证 `rs_pointmap_loss` 有所改善，但 `grid_global_to_gt` 点云形状损失基本停在 `0.765` 左右；说明只把 global 重建改成 `grid_xy + height/slope/dir` 还不能让 aux head 独立学出稳定 remote 点云。
- “aux offset/xyz 作为 point head normalized residual”是目前最明确可训练的 aux 目标。xy-only 版本把 `rs_point_l1_scale_aligned` 从约 `95` 降到 `43.6`；xyz 版本进一步到 `30.0`，all-cities 训练复现到 `30.1`，说明 aux head 能学到尺度无关的 remote 局部形状修正。但 raw/centered 指标仍约 `291/95`，height affine 也不稳，因此还没有解决绝对尺度/中心/真实高度恢复。

## P7 Aux 标签为什么更稀疏

`rs pointmap loss` 只要求 remote 像素有合法 3D 点。只要 `pts3d/valid_mask` 有效，就可以直接对预测点云做 L1 或 robust loss。因此它的监督范围基本等于 remote 点云标签的有效区域。

P7 aux 的 height、offset、global-slope 等标签不是直接监督 xyz，而是把 remote 点云解释成显式投影机制的中间量。一个像素除了有合法 3D 点，还需要满足更多条件：

- 能稳定转换成相对高度、投影 offset 或全局坡度；
- 有可用的参考平面、参考高度或归一化尺度；
- 投影方向、中心化和尺度估计不退化；
- 不属于明显异常值，否则会污染 bucket、contrast、affine 统计；
- 如果使用 MoGe prior，还要求伪标签 confidence、edge 或 height mask 有效；
- 如果使用 overlap 强化，还要求 remote 点投影到普通视角后与普通视角深度一致。

因此 pointmap 是“有点就能监督”，aux 是“这个点还要能被稳定解释为投影机制的一部分”。后一种条件更严格，所以有效 mask 会明显更小。

这会带来三个训练影响：

- aux 梯度更稀疏，batch 间波动更大；
- 模型更容易先学到平坦背景、低矮区域或少量高置信区域的局部统计；
- aux 与 pointmap 的监督域不完全一致，可能出现 aux height/offset 可学习，但 remote point head 点云质量没有同步改善。

目前的 MoGe height、overlap pointmap、top-percent robust loss 都是在缓解这个稀疏监督问题，但现有结果说明它们还没有完全解决 remote 局部点云形状不稳的问题。

## 实验表

| 方向 | 关键结果 | 结论 | 训练脚本/入口 | 训练输出 | Benchmark / 可视化输出 |
|---|---|---|---|---|---|
| P7-P5B 基础 shared-norm projection aux | 可视化比 P5E 私有 viewtype 稳，但高区 remote 点云仍会飞 | 基础 P7-P5B 有一定提升空间，但不是最终解 | `bash_scripts/train/Crossview/vggt/p7_vggt_p5b_shared_norm_projection_aux.sh` | `outputs/.../training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu` | 已支持 `scripts/export_pointcloud_ply.py` 自动导出 mixed + remote-only |
| P7-P5E private-viewtype | 可视化出现局部条纹；用户观察弱于 P5B | 暂不作为主线 | 同上，P5E/private-viewtype 配置 | `outputs/.../training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu` | `debug/plyview/448/...p5e_private...` |
| 标签一致性检查：aux 标签反投影点云 | `aux_original_xyz_world_common` 与 pointmap 标签整体对齐；`offset` 重建一致；`projected_centered` 有整体倾斜 | 多任务标签本身没有明显互相矛盾；投影坐标/中心化可视化需谨慎解释 | `scripts/export_projection_aux_reconstruction.py` / 辅助检查脚本 | 非训练实验 | `outputs/.../debug/projection_aux_reconstruct/newyork_448_Google_Satellite` |
| 稠密化后 aux 标签反投影检查 | dense height 与原 aux height 在 common 区域一致：461/493 的 abs diff P50 约 `0.65/0.22m`，P95 约 `2.84/2.70m`；dense global 反投影回 pointmap 的 L2 误差约 `1e-6`；但 pointmap/aux common mask 只有 `4.48%/2.26%` | 标签值本身可信，主要问题是 aux 监督域和 pointmap 主监督域严重错位；继续训练时不应只加 aux 权重，应优先把 aux 改成 pointmap-derived dense/soft 监督或显式对齐监督域 | `scripts/reconstruct_remote_pointcloud_from_projection_aux.py --export_dense_pointmap_height` | 非训练实验 | `outputs/.../debug/projection_aux_reconstruct_dense/newyork_461_Google_Satellite`、`newyork_493_Google_Satellite` |
| Pointmap-derived dense height 主监督，关闭 offset | smoke test 证明 dense mask 等于 pointmap valid，exact dense prediction loss 为 0；训练中 `dense_rel_height_mask_ratio` 约 `0.18`，明显大于 common；但 benchmark `RS-only rs_point_l1=295.067`，`Joint rs_point_l1=294.667`，未超过 best；aux global PLY 稳定，aux offset 仍有长尾飞点 | 直接改 height 主监督是正确且可训练的，但单独这样做仍不能改善 remote pointmap；问题不只是 height mask，offset head/point head 之间仍缺少可靠耦合，后续不要依赖旧 offset 输出 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 robust-overlap best warm-start，`PROJ_DENSE_REL_HEIGHT_WEIGHT=0.45`、`PROJ_DENSE_REL_HEIGHT_EXCLUDE_HARD_MASK=false`，旧 rel-height 低权重 `0.05`，per-pixel offset/consistency/reconstruction 全关闭 | `outputs/.../training/Crossview/vggt/p7_frombest_denseheight_main_nooffset_e2_b8_2gpu` | benchmark `.../vggt_p7_frombest_denseheight_main_nooffset_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_denseheight_main_nooffset_final`、`493/...` |
| Parallel aux head / token-source aux | aux 不再依赖预测 pointmap 作为输入，结构更合理 | 成为后续主线结构 | `p7_vggt_p5b_shared_norm_projection_aux.sh` | 多个 `p7_proj_moge_*private_tokens*` 输出 | `debug/plyview/461_1`、`debug/plyview/493` |
| MoGe height/edge/gradient prior 初版 | 可视化能看，但整体变化不大 | MoGe 伪标签可作为弱辅助，不能只靠它 | `p7_vggt_p5b_shared_norm_projection_aux.sh` | `outputs/.../training/Crossview/vggt/p7_proj_moge_aux_private_tokens_best` 附近实验 | `debug/plyview/461_1/vggt_p7_proj_moge_aux_private_tokens_best`、`debug/plyview/493/...` |
| MoGe PM-height h5 prior0 | 用户观察这版可视化更好；remote benchmark 约 `292-293` 区间 | 当前重要基线之一 | 同上 | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_prior0_balanced20x4_private_tokens_warmbest_e6_b8_4gpu` | `debug/plyview/461_1/vggt_p7_proj_moge_pmheight_h5_prior0_best`、`debug/plyview/493/...`；benchmark `.../vggt_p7_proj_moge_pmheight_h5_prior0_best` |
| 从 base checkpoint 重新微调 | `RS-only rs_point_l1=295.146`，`Joint rs_point_l1=294.714` | aux height 可学，但 remote pointmap 没提升；从零/从 base 不解决主问题 | 同上 | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_frombase_e8_b8_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_frombase_final`；PLY：`debug/plyview/461_1/...frombase_final`、`493/...` |
| overlap 区域严格监督 + robust/top20 | `RS-only rs_point_l1=292.558`，`Joint rs_point_l1=292.122`，joint aerial 也较稳 | 目前最好的 P7 remote 数字，作为主参考 | `p7_vggt_p5b_shared_norm_projection_aux.sh`，新增 overlap loss | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_final`；PLY：`debug/plyview/461_1/...robusttop20_overlappm6_lowaux_final`、`493/...` |
| 增加普通视角到 6-view | `RS-only rs_point_l1=295.070`，`Joint rs_point_l1=294.648` | 对 remote 点云无帮助；可能只改善普通视角/pose | 同上，`NUM_VIEWS=6` | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_views6_warmh5best_e4_b6_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_views6_warmh5best_final` |
| low-covis 普通视角采样 | `RS-only rs_point_l1=295.073`，`Joint rs_point_l1=294.656`，affine height 略好 | 对 remote pointmap 没提升，暂不继续 | 同上，训练采样 low-covis | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_lowcovis_warmh5best_e4_b8_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_lowcovis_warmh5best_final` |
| zero-covis 普通视角采样 | `RS-only rs_point_l1=295.075`，`Joint rs_point_l1=294.654` | 没提升；zero-covis 不是主解 | 同上，训练采样 `zero_covis`，val/test 保持标准 | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_zerocovis_train_warmh5best_e3_b8_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_zerocovis_train_warmh5best_final`；PLY：`debug/plyview/461_1/...zerocovis_train_warmh5best_final`、`493/...` |
| PI3 remote 多任务迁移 | 用户观察结果有大问题 | 暂停 PI3；先不作为主线 | `bash_scripts/train/Crossview/pi3/*` | 若干临时 PI3 输出 | 部分 PLY 可视化 |
| vggt_omega remote 多任务迁移 | 本地缺少 `/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt` | 需要先补齐 omega checkpoint，否则不做随机初始化实验 | `bash_scripts/train/Crossview/vggt_omega/p7_vggt_omega_projection_aux.sh` | 未启动有效训练 | 无 |
| robust-overlap 低学习率续训 | `RS-only rs_point_l1=295.075`，`Joint rs_point_l1=294.659` | 没有超过当前 best；低学习率续训没有带来边际提升 | `p7_vggt_p5b_shared_norm_projection_aux.sh` | `outputs/.../training/Crossview/vggt/p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_cont_e4_b8_4gpu` | benchmark `.../vggt_p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_cont_final`；PLY：`debug/plyview/461_1/...lowaux_cont_final`、`493/...` |
| Pointmap 直接派生 dense height 标签可视化 | common 区域与原 aux height 基本一致：448/461/493 的 abs diff P50 约 `0.52/0.65/0.22m`，P95 约 `4.11/2.84/2.70m`；448 覆盖从 `14.4%` 提到 `20.5%` | 直接派生 height 几何上可用；训练时应作为 soft/low-weight dense aux，避免新增低矮背景主导 | `scripts/visualize_dense_pointmap_height_aux.py` | 非训练实验 | `outputs/.../debug/dense_pointmap_height_aux/newyork_448_Google_Satellite`、`newyork_461_Google_Satellite`、`newyork_493_Google_Satellite` |
| Pointmap 派生 dense height 训练，soft 全新增区域 | `RS-only rs_point_l1=295.072`，`Joint rs_point_l1=294.666`；但 `RS-only rs_height_mae_affine=8.437`，比之前约 `9m` 有改善 | 稠密 height 对高度校准有信号，但全新增区域没有改善 remote pointmap；下一步改成高幅值/高结构优先，降低低背景干扰 | `p7_vggt_p5b_shared_norm_projection_aux.sh`，新增 `PROJ_DENSE_REL_HEIGHT_*` | `outputs/.../training/Crossview/vggt/p7_proj_denseh02_from_robustoverlap_e3_b8_2gpu` | benchmark `.../vggt_p7_proj_denseh02_from_robustoverlap_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_denseh02_from_robustoverlap_final`、`493/...` |
| Pointmap 派生 dense height 训练，高幅值区域优先 | `RS-only rs_point_l1=295.070`，`Joint rs_point_l1=294.667`；`RS-only rs_height_mae_affine=7.988` | 高幅值 dense height 进一步改善 height affine，但仍没有转化为 remote pointmap 几何提升；说明单纯稠密化 aux height 不够，需要结构上让 aux/token 表征影响 point head | 同上，`PROJ_DENSE_REL_HEIGHT_MIN_ABS_QUANTILE=0.5` | `outputs/.../training/Crossview/vggt/p7_proj_denseh015_highq50_from_robustoverlap_e4_b8_2gpu` | benchmark `.../vggt_p7_proj_denseh015_highq50_from_robustoverlap_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_denseh015_highq50_from_robustoverlap_final`、`493/...` |
| Dense high-q height + midtrunk 低 LR | `RS-only rs_point_l1=295.075`，`Joint rs_point_l1=294.663`；`RS-only rs_height_mae_affine=8.921` | 没有超过当前 best；开放少量 shared aggregator 参数没有把 dense aux height 转化为 remote pointmap 改善，继续加大 trunk LR 的收益证据不足 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `configs/train_params/vggt_p7_p5b_parallel_token_aux_midtrunklr.yaml`；从当前 best warm-start，aggregator frame/global blocks 以 `5e-7` 小幅微调 | `outputs/.../training/Crossview/vggt/p7_proj_denseh015_highq50_midtrunk_from_best_e4_b8_2gpu` | benchmark `.../vggt_p7_proj_denseh015_highq50_midtrunk_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_denseh015_highq50_midtrunk_from_best_final`、`493/...` |
| Dense high-q height + token residual g0.05 | `RS-only rs_point_l1=295.076`，`Joint rs_point_l1=294.665`；`RS-only rs_height_mae_affine=9.079` | 没有超过当前 best；把 dense aux/token residual 直接接到 point head 仍未改善 remote 点云，说明瓶颈不只是 aux 表征是否能传到 head | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_parallel_token_aux_residual_lowtrunklr.yaml`；从当前 best warm-start，token-source aux + dense high-q height，remote tokens 经过 gated residual adapter 后进入 remote point head | `outputs/.../training/Crossview/vggt/p7_proj_denseh_tokenres_g005_from_best_e4_b8_2gpu` | benchmark `.../vggt_p7_proj_denseh_tokenres_g005_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_denseh_tokenres_g005_from_best_final`、`493/...` |
| Self-contained grid global 反投影损失 | 2 epoch 短训中验证 `rs_pointmap_loss` 从约 `0.0282` 降到 `0.0211`，但新增 `rs_projection_grid_global_to_gt_loss` 仍约 `0.7647`；benchmark：`RS-only rs_point_l1=295.063`，`Joint rs_point_l1=294.663`，`RS-only scale_aligned=95.505`，`Joint scale_aligned=95.493` | point head 没被拖坏，但没有超过 robust-overlap best；aux 的 `height + slope + dir + pixel-grid` 自包含反投影还没有学出可靠点云。问题更集中在 global 参数化/损失耦合，而不是只缺 GT base | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 `p7_proj_moge_pmheight_h5_robusttop20_overlappm6_lowaux_warmh5best_e6_b8_4gpu` warm-start；`PROJ_GRID_GLOBAL_TO_GT=0.15`、`PROJ_GRID_GLOBAL_TO_GT_HIGH_Z_QUANTILE=0.5`、`PROJ_GLOBAL_TARGET_FROM_POINTMAP=true`、offset/recon-to-gt 关闭 | `outputs/.../training/Crossview/vggt/p7_frombest_gridglobal_selfcontained_e2_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_gridglobal_selfcontained_e2_final`；PLY：`debug/plyview/493/vggt_p7_gridglobal_selfcontained_e2_final`、`debug/plyview/461_1/vggt_p7_gridglobal_selfcontained_e2_final`；重点看 `mapanything_pointcloud_same_remote.ply` 与 `mapanything_pointcloud_same_aux_grid_global_remote.ply` |
| Aux offset 作为 point head normalized xy residual | 2-city / 4 epoch 快速验证：`corrected_mae_norm_mean` 从初始约 `0.279` 降到最终约 `0.080`，base MAE 约 `0.288-0.314`；`rs_pointmap_loss` 最终约 `0.017-0.025`，未被拖坏 | 这是 offset-only 系列第一个明确可训练的 aux offset 目标；它直接修正最终 point head xy，而不要求 point head 内部 offset gauge 等于 GT projection offset。短训只验证可学性，后续用 e12 和 xyz 版本判断实际收益 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；`PROJ_POINT_RESIDUAL_OFFSET_TO_GT=10`、`LAMBDA_REMOTE_PM=8`、其他 projection aux 子损失关闭，从 height-only best warm-start | `outputs/.../training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_e4_b9_2gpu` | PLY：`debug/plyview/493/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_e4_final`、`debug/plyview/461_1/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_e4_final`；新增 `*_aux_point_residual_remote.ply` 和 `*_aux_point_residual_norm_remote.ply` |
| Aux xy residual 保留权重 e12 continuation | 从 e4 final 继续训练并显式设置 `train_params.warmstart_exclude_prefixes=[]`，避免 aux head 被重新初始化；最终 `corrected_mae_norm_mean=0.0573`，base MAE `0.3100`，`rs_pointmap_loss=0.0219`；aux xy metric：`RS-only scale_aligned=43.629`，`Joint scale_aligned=68.629`，raw/centered 仍约 `292/95` | 训练度量继续改善，且尺度无关 xy 形状有明确提升；但它只修 xy、不修 z，raw/centered 几乎不变，说明绝对尺度/中心/高度仍由 point head 决定 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；`PROJ_POINT_RESIDUAL_OFFSET_TO_GT=10`、`LAMBDA_REMOTE_PM=8`、`train_params.warmstart_exclude_prefixes=[]`，从 e4 residual final warm-start | `outputs/.../training/Crossview/vggt/p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_b9_2gpu` | Benchmark：`benchmarking/.../remote_pointmetric20/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_final_auxresmetric`；PLY：`debug/plyview/493/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_final`、`debug/plyview/461_1/vggt_p7_aux_pointresidual_offset_gt_w10_pm8_warme4_noexclude_e12_final`；summary residual mean/p95：493 `0.259/0.510`，461_1 `0.256/0.524` |
| Aux xyz residual 修正 point head normalized xyz | best val：`corrected_mae_norm_mean=0.1245`，`corrected_z_mae_norm_mean=0.1088`；final val：`0.1571/0.1460`；aux xyz benchmark：`RS-only rs_point_l1=291.204`，`centered=95.193`，`scale_aligned=30.039`，`height_affine=21.697`；`Joint scale_aligned=47.349` | 目前 residual 系列最强的尺度无关 remote shape 改善，证明 aux head 能学到 point head 到 GT 的 normalized xyz 修正；但 raw/centered 仍未突破，z residual 的真实高度尺度不稳。下一步应把它作为 teacher/repair 目标，研究尺度恢复或让 point head 吸收该尺度无关修正，而不是直接当最终 metric 输出 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；`PROJ_POINT_RESIDUAL_XYZ_TO_GT=10`、`LAMBDA_REMOTE_PM=8`、所有旧 projection aux 子损失显式为 `0`，从 e12 xy residual final warm-start 并保留 aux 权重 | `outputs/.../training/Crossview/vggt/p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_b9_2gpu`；保留 `checkpoint-best-slim.pth` 与 `checkpoint-final.pth`，已删除 `checkpoint-last.pth` 和 14G 原 best | Benchmark：`benchmarking/.../remote_pointmetric20/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best_auxxyzmetric`；PLY：`debug/plyview/493/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best`、`debug/plyview/461_1/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_pure_warme12_e6_best`；summary：493 residual `0.272/0.549`、z `1.273/1.391`，461_1 residual `0.282/0.586`、z `1.383/1.524` |
| Aux xyz residual all-cities continuation | best val 出现在 epoch 4：`rs_projection_point_residual_xyz_to_gt_loss=0.0537`，`corrected_mae_norm_mean=0.1145`，`corrected_z_mae_norm_mean=0.0876`；final val `0.0539/0.1221/0.1095`；aux xyz benchmark：`RS-only rs_point_l1=291.241`，`centered=95.206`，`scale_aligned=30.108`，`height_affine=16.588`；`Joint scale_aligned=50.417` | 全城市训练没有突破 2-city scale-aligned 上界，但复现了约 `30` 的尺度无关形状误差，并且 best val 更稳；说明 residual 目标不是小数据偶然过拟合。raw/centered 仍不变，下一步应研究让 point head 吸收 residual 或增加尺度/中心恢复，而不是继续单独拉长 aux residual 训练 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；`TRAIN_CITIES=[chicago,newyork,sanfrancisco,seattle]`，从 2-city xyz best warm-start，`PROJ_POINT_RESIDUAL_XYZ_TO_GT=10`、`LAMBDA_REMOTE_PM=8`，旧 projection aux 子损失全关，保留 aux 权重 | `outputs/.../training/Crossview/vggt/p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_b9_2gpu`；保留 `checkpoint-best-slim.pth` 与 `checkpoint-final.pth`，已删除 14G best/last | Benchmark：`benchmarking/.../remote_pointmetric20/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best_auxxyzmetric`；PLY：`debug/plyview/493/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best`、`debug/plyview/461_1/vggt_p7_aux_pointresidual_xyz_gt_w10_pm8_allcities_warm2citybest_e8_best`；summary：493 residual `0.270/0.523`、z `1.153/1.282`，461_1 residual `0.264/0.532`、z `1.311/1.454` |
| Head-only repair + dense high-q height | `RS-only rs_point_l1=295.073`，`Joint rs_point_l1=294.661`；`RS-only rs_height_mae_affine=8.899` | 没有超过当前 best；只修 remote head/aux head 不能恢复 remote 几何，说明问题不只是 head 容量或 head-only 校准 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_train_remotehead_aux`；从当前 best warm-start，冻结 trunk，只训练 remote point head 和 projection aux | `outputs/.../training/Crossview/vggt/p7_proj_headonly_denseh_highq_from_best_e4_b32_2gpu` | benchmark `.../vggt_p7_proj_headonly_denseh_highq_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_headonly_denseh_highq_from_best_final`、`493/...` |
| Dense soft height + MoGe aux shape fallback | `RS-only rs_point_l1=295.070`，`Joint rs_point_l1=294.665`；`RS-only rs_height_mae_affine=9.224`，`Aerial pointmaps_abs_rel=0.05236` | 没有超过当前 best；更稠密的 aux/MoGe shape 能提供高度侧信号，但仍不能转化为 remote pointmap 几何收益，下一步应优先验证 provider/标签质量，而不是继续增加 aux 权重 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux`；从当前 best warm-start，保留 robust/top20/overlap，加入 pointmap-derived dense height soft mask、MoGe aux height/gradient/edge 弱监督 | `outputs/.../training/Crossview/vggt/p7_proj_denseh_mogeshape_soft_from_best_e4_b8_2gpu` | benchmark `.../vggt_p7_proj_denseh_mogeshape_soft_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_denseh_mogeshape_soft_from_best_final`、`493/...` |
| Google-only provider 反事实 | `RS-only rs_point_l1=295.074`，`Joint rs_point_l1=294.664`；`RS-only rs_height_mae_affine=8.517` | 没有超过当前 best；单纯去掉 Bing/provider mixing 不能修复 remote pointmap，标签问题更可能是 scene/region 级噪声或训练目标传递问题 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux`；从当前 best warm-start，只用 `RS_PROVIDER=Google_Satellite`，all cities/no crop，保留 robust/top20/overlap | `outputs/.../training/Crossview/vggt/p7_googleonly_robustoverlap_from_best_e4_b8_2gpu` | benchmark `.../vggt_p7_googleonly_robustoverlap_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_googleonly_robustoverlap_from_best_final`、`493/...` |
| 从 base VGGT 重新微调 P7 robust-overlap | `RS-only rs_point_l1=295.153`，`Joint rs_point_l1=294.696`；`RS-only rs_height_mae_affine=14.241` | 没有超过当前 best；从 base 重新微调没有自然突破，问题不只是 warm-start 局部解。aux 标签/height gap 可学习，但没有转化为更好的 remote pointmap | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux`；不使用当前 P7 warm-start，直接从本地 VGGT checkpoint 初始化，all cities/no crop，保留 robust/top20/overlap/lowaux | `outputs/.../training/Crossview/vggt/p7_frombase_robustoverlap_lowaux_e4_b8_2gpu` | benchmark `.../vggt_p7_frombase_robustoverlap_lowaux_final`；PLY：`debug/plyview/461_1/vggt_p7_frombase_robustoverlap_lowaux_final`、`493/...` |
| Top50 pointmap + stronger high-z | `RS-only rs_point_l1=295.064`，`Joint rs_point_l1=294.666`；`RS-only rs_height_mae_affine=8.616` | 没有超过当前 best；高处飞点问题不是因为 top20 裁剪简单漏训高结构，放宽裁剪和加高处权重只改善/保持 height affine，未改善 remote pointmap | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux`；从当前 best warm-start，把 robust pointmap 从 top20 放宽到 top50，并提高高处区域权重 `LAMBDA_REMOTE_HIGH_Z=0.08` | `outputs/.../training/Crossview/vggt/p7_frombest_top50_highz008_overlap6_e3_b8_2gpu` | benchmark `.../vggt_p7_frombest_top50_highz008_overlap6_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_top50_highz008_overlap6_final`、`493/...` |
| Overlap-consistent 区域强监督 | `RS-only rs_point_l1=295.068`，`Joint rs_point_l1=294.671`；`RS-only rs_height_mae_affine=8.640` | 没有超过当前 best；remote/普通视角 GT 重合区域能被筛出并参与更强 loss，但简单加权不能把 remote pointmap 拉回，说明问题不只是可信 overlap 区域监督不够 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux`；从当前 best warm-start，保持 top20/no-crop，把普通 remote pointmap 设 `LAMBDA_REMOTE_PM=4`，把 overlap pointmap 提到 `LAMBDA_REMOTE_OVERLAP_PM=12` | `outputs/.../training/Crossview/vggt/p7_frombest_overlap12_pm4_e3_b8_2gpu` | benchmark `.../vggt_p7_frombest_overlap12_pm4_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_overlap12_pm4_final`、`493/...` |
| Scale-normalized aux reconstructed point loss | `RS-only rs_point_l1=291.286`，`Joint rs_point_l1=290.837`，但 `Joint rs_point_l1_centered=97.460`，弱于 robust-overlap best 的 `95.394`；`RS-only rs_height_mae_affine=10.211`，`Joint=16.347`；训练日志中 pred/target norm factor 接近，说明尺度对齐路径生效 | 修正了损失定义中的尺度问题，但低权重 global recon-to-GT 只带来 raw L1 小幅波动，没有改善 centered/local shape，也没有稳定提升 height；不应简单放大该损失，否则可能由不稳定 aux 反向拖坏 point head。若继续，应做高结构/高置信 mask 的重建损失，而不是全 mask 均匀约束 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 robust-overlap best warm-start，新增 `PROJ_RECON_GLOBAL_TO_GT=0.05`，并打开 `PROJ_RECON_TO_GT_USE_POINTMAP_NORM=true`，即 aux global 反投影点云和 GT pointmap 比较前先按 pointmap norm 尺度对齐；direct offset/recon consistency 不作为主改动 | `outputs/.../training/Crossview/vggt/p7_frombest_normrecon_globalgt_e3_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_frombest_normrecon_globalgt_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_normrecon_globalgt_final`、`493/...`，包含 mixed、remote-only、`aux_offset_remote`、`aux_global_remote` |
| High-z masked scale-normalized aux reconstructed point loss | benchmark 与上一行全区域 normrecon 完全一致：`RS-only rs_point_l1=291.286`，`Joint rs_point_l1=290.837`，`Joint rs_point_l1_centered=97.460`，`Joint rs_height_mae_affine=16.347`；训练中 `rs_projection_reconstruct_high_z_mask_ratio` 约 `0.06-0.08`，pred/target norm factor 接近，说明高 z mask 和尺度归一化均生效 | 高结构 mask 没有把 aux reconstructed point loss 转化为更好的 remote point head；这进一步说明瓶颈不在 recon loss 的尺度或背景稀释，而在 aux 分支与 point head 的耦合/参数化。短期不再继续同类 recon-to-GT 权重微调 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；新增 `PROJ_RECON_TO_GT_HIGH_Z_QUANTILE=0.6`、`PROJ_RECON_TO_GT_HIGH_Z_MIN_PIXELS=64`，并把 `PROJ_RECON_GLOBAL_TO_GT` 提到 `0.2`；其余 robust/top20/overlap/MoGe 设置沿用当前 best | `outputs/.../training/Crossview/vggt/p7_frombest_highz_normrecon_globalgt_e3_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_frombest_highz_normrecon_globalgt_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_highz_normrecon_globalgt_final`、`493/...` |
| 正尺度无关 dense height + global recon 正式训练 | 正结构/正尺度 benchmark：`RS-only rs_point_l1=295.072`，`Joint=294.660`；`RS-only rs_point_l1_scale_aligned=95.505`，`Joint=95.493`，几乎持平；`RS-only rs_height_mae_affine=8.933`，`Joint=15.106`，height 明显变差。训练中 pointmap loss 稳定，但 `global_dir_cosine` 长期偏负，dense height 高/低区域仍过预测 | 这次明确按尺度无关训练/评估跑通，但没有带来 remote 点云突破；问题不是绝对尺度约束本身，而是 aux 的 direction/height/reconstruction 没有被组织成稳定机制。后续不应继续延长同配置，应优先改 global direction 参数化/监督和阶段式 teacher-student 一致性 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 robust-overlap best warm-start，`PROJ_REL_HEIGHT_SCALE_MODE=gt_pointmap_norm`，`PROJ_DENSE_REL_HEIGHT_WEIGHT=0.45`，`PROJ_RECON_GLOBAL_TO_GT=0.12`，`PROJ_RECON_TO_GT_USE_POINTMAP_NORM=true`，关闭 direct offset/recon consistency，B8/2GPU/E8 | `outputs/.../training/Crossview/vggt/p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_final_p7struct_posscale`；PLY：`debug/plyview/461_1/vggt_p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_final`、`debug/plyview/493/vggt_p7_scalefree_denseh_globalrecon_highconf_frombest_e8_b8_final` |
| 正尺度无关 global-vector 约束 | 正结构/正尺度 benchmark 基本复现上一行：`RS-only rs_point_l1=295.072`，`Joint=294.660`；`RS-only rs_point_l1_scale_aligned=95.505`，`Joint=95.493`；`RS-only rs_height_mae_affine=8.933`，`Joint=15.106`。训练日志中新增 `rs_projection_global_vector_loss` 约 `0.09-0.12`，但 `global_dir_cosine` 验证均值仍约 `-0.246` | 直接监督 `slope * dir` 没有修复投影方向退化，也没有改善 remote point head。说明问题不是单个 direction/slope loss 形式，而是 aux 几何任务需要更强的阶段式 teacher、低维参数化或从 GT pointmap 拟合的稳定方向监督 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；新增 `LAMBDA_PROJ_GLOBAL_VECTOR=0.20`，其余保持 scale-free dense height/global recon，B8/2GPU/E6，从 robust-overlap best warm-start | `outputs/.../training/Crossview/vggt/p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_final_posscale`；PLY：`debug/plyview/461_1/vggt_p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_final`、`debug/plyview/493/vggt_p7_scalefree_globalvec_denseh_recon_frombest_e6_b8_final` |
| 从 GT pointmap 拟合 global direction/slope target | 正结构/正尺度 benchmark 仍基本复现：`RS-only rs_point_l1=295.072`，`Joint=294.660`；`RS-only rs_point_l1_scale_aligned=95.505`，`Joint=95.493`；`RS-only rs_height_mae_affine=8.933`，`Joint=15.106`。训练中 `rs_projection_global_target_pointmap_valid_batch_ratio=1.0`，说明拟合 target 每 batch 都有效；但 `global_vector_pred_norm` 长期约 `0.113`，随 GT slope 变化不明显，`global_dir_cosine` 多数时间在 0 附近或偏负 | 用 GT pointmap 拟合低维投影 target 后，监督域和尺度定义都更合理，但仍没有让 aux 分支学出稳定可解释投影，也没有改善 remote point head。当前不应继续简单加大 `global_vector`/`global_dir` 权重；下一步应考虑更直接的 aux teacher 预训练、冻结 point head 的机制预训，或把 global 投影参数改成由场景/remote 图像显式估计的低维模块 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；新增 `PROJ_GLOBAL_TARGET_FROM_POINTMAP=true`、`PROJ_GLOBAL_TARGET_MIN_PIXELS=512`、`LAMBDA_PROJ_GLOBAL_VECTOR=0.50`，保持 scale-free dense height/global recon，B8/2GPU/E6，从 robust-overlap best warm-start | `outputs/.../training/Crossview/vggt/p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_2gpu` | benchmark `.../remote_pointmetric20/vggt_p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_final_posscale`；PLY：`debug/plyview/461_1/vggt_p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_final`、`debug/plyview/493/vggt_p7_scalefree_pointmapfit_globaltarget_frombest_e6_b8_final` |
| P7 pre-aggregator view-type + gated residual | 正确结构加载后：`RS-only rs_point_l1=295.068`，`Joint rs_point_l1=294.675`；`RS-only rs_height_mae_affine=9.506` | 没有超过当前 best；轻量 residual 通路没有把 dense/aux 表征转化为更好的 remote pointmap。已修复 export/benchmark，使其可从 checkpoint key 自动打开 pre-aggregator/gated-residual/late-fusion 结构 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux_gated_residual`；从当前 best warm-start，加入 pre-aggregator view-type embedding 和 remote-to-aerial gated residual adapter | `outputs/.../training/Crossview/vggt/p7_frombest_gatedres_preagg_e3_b8_2gpu` | benchmark `.../vggt_p7_frombest_gatedres_preagg_final_p7arch`；PLY：`debug/plyview/461_1/vggt_p7_frombest_gatedres_preagg_final`、`493/...` |
| P7 token residual adapter | 正确 P7 架构加载后：`RS-only rs_point_l1=295.076`，`Joint rs_point_l1=294.665`；`RS-only rs_height_mae_affine=9.065` | 没有超过 robust-overlap best；简单 token residual 不能把 aux 表征有效转化为 remote 点云 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_parallel_token_aux_residual_lowtrunklr.yaml`；remote patch tokens 经小 MLP gated residual 后进入 point head | `outputs/.../training/Crossview/vggt/p7_proj_tokenres_g001_from_robustoverlap_e4_b8_2gpu_v2` | benchmark `.../vggt_p7_proj_tokenres_g001_from_robustoverlap_final_p7arch`；PLY：`debug/plyview/461_1/vggt_p7_proj_tokenres_g001_from_robustoverlap_final`、`493/...` |
| 8-view 普通视角扩展 | 正确 P7 架构 + 8-view benchmark：`RS-only rs_point_l1=295.077`，`Joint rs_point_l1=294.657`；aerial/pose 指标略好 | 增加普通视角数量不能修复 remote pointmap；该方向暂不继续 | `p7_vggt_p5b_shared_norm_projection_aux.sh`，`NUM_VIEWS=8`，从 robust-overlap best warm-start | `outputs/.../training/Crossview/vggt/p7_proj_views8_from_robustoverlap_e3_b4_2gpu` | benchmark `.../vggt_p7_proj_views8_from_robustoverlap_final_p7arch`；PLY：`debug/plyview/461_1/vggt_p7_proj_views8_from_robustoverlap_final`、`493/...` |
| 更强鲁棒裁剪 + overlap 强化 | 正确 P7 架构加载后：`RS-only rs_point_l1=295.077`，`Joint rs_point_l1=294.664`；`RS-only rs_height_mae_affine=9.030` | 比当前 best `292.558/292.122` 明显差；remote 标签噪声/离群点不是仅靠更激进 top-percent 和 overlap 权重能解决 | `p7_vggt_p5b_shared_norm_projection_aux.sh`，从 robust-overlap best warm-start，`REMOTE_POINTMAP_TOP_N_PERCENT=10`、`LAMBDA_REMOTE_OVERLAP_PM=8` | `outputs/.../training/Crossview/vggt/p7_proj_robusttop10_overlap8_from_best_e3_b8_2gpu` | benchmark `.../vggt_p7_proj_robusttop10_overlap8_from_best_final`；PLY：`debug/plyview/461_1/vggt_p7_proj_robusttop10_overlap8_from_best_final`、`493/...` |
| Split remote aggregator + FiLM late fusion + dense high-q height | `RS-only rs_point_l1=294.979`，`Joint rs_point_l1=294.979`；`RS-only rs_height_mae_affine=7.718`；Aerial `pointmaps_abs_rel=0.05693` | 没有超过当前 best `292.558/292.122`，但 height affine 是近期较好；FiLM gate 训练后仍极小，说明轻量 late fusion 实际影响不足。B8 OOM，B6 峰值显存约 77G，可作为结构反事实结果保留 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_shared_norm_projection_aux_split_film.yaml`；从 robust-overlap best warm-start，remote/aerial 分路 aggregator，FiLM late fusion，pointmap-derived dense high-q height soft aux | `outputs/.../training/Crossview/vggt/p7_frombest_splitfilm_denseh_highq_e3_b6_2gpu` | benchmark `.../vggt_p7_frombest_splitfilm_denseh_highq_final_p7arch`；PLY：`debug/plyview/461_1/vggt_p7_frombest_splitfilm_denseh_highq_final`、`493/...` |
| Z-distance consistency + high-z 轻权重续训 | `RS-only rs_point_l1=295.066`，`Joint rs_point_l1=294.665`；Aerial `pointmaps_abs_rel=0.05321` | 没有超过当前 best；z-distance/high-z 这类几何一致性弱约束没有解决 remote pointmap 飞点。已删除 best/last，仅保留 final ckpt 节省空间 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 robust-overlap best warm-start，增加 z-distance consistency 和轻量 high-z 约束 | `outputs/.../training/Crossview/vggt/p7_frombest_zdist02_highz006_e3_b8_2gpu` | benchmark `.../vggt_p7_frombest_zdist02_highz006_final`；aux-vs-pointhead PLY 使用当前 best 导出：`debug/plyview/461_1/vggt_p7_best_aux_vs_pointhead`、`493/...` |
| Pointmap-derived dense height + dense global-offset aux | E8：`RS-only rs_point_l1=295.072`，`Joint rs_point_l1=294.655`；E2 no-offset/joint 约 `294.667`，E2 dense-offset/joint 约 `294.667`；导出摘要中 aux offset world 均值约 `4-5m`，不再出现旧 offset 十几米级长尾 | 稠密 height/offset 监督可以稳定拟合，并修正了 aux offset 飞掉的一部分问题，但没有转化为 remote point head 的 benchmark 提升；问题不只是 aux 标签稀疏，当前 aux 任务与 point head 的协同仍弱 | `p7_vggt_p5b_shared_norm_projection_aux.sh`；从 robust-overlap best warm-start，`PROJ_DENSE_REL_HEIGHT_WEIGHT=0.45`，`PROJ_DENSE_GLOBAL_OFFSET_WEIGHT=0.20`，旧 sparse offset 主损失关闭 | `outputs/.../training/Crossview/vggt/p7_frombest_denseheight_densegoffset_e8_b8_2gpu`；短跑反事实：`p7_frombest_denseheight_main_nooffset_e2_b8_2gpu`、`p7_frombest_denseheight_densegoffset_e2_b8_2gpu` | benchmark `.../vggt_p7_frombest_denseheight_densegoffset_e8_final`；PLY：`debug/plyview/461_1/vggt_p7_frombest_denseheight_densegoffset_e8_final`、`493/...` |
| 从原始 VGGT 初始化的 dense height + dense global-offset 反事实 | `RS-only rs_point_l1=295.113`，`Joint rs_point_l1=294.665`；Aerial `pointmaps_abs_rel=0.06196`，Joint `0.05928`；训练稳定但普通视角和 remote 都弱于 warm-start | “从零微调可能跳出局部解”的假设没有得到支持；从零反而损失了已有普通视角/remote 表征。后续不应继续在同一 P7/P5B dense-aux 权重上加 epoch | `p7_vggt_p5b_shared_norm_projection_aux.sh`；`WARMSTART_CKPT=null`，从本地 VGGT checkpoint 初始化 P7/private remote head，其他监督同上，E12 | `outputs/.../training/Crossview/vggt/p7_scratch_denseheight_densegoffset_e12_b8_2gpu` | benchmark `.../vggt_p7_scratch_denseheight_densegoffset_e12_final`；PLY：`debug/plyview/461_1/vggt_p7_scratch_denseheight_densegoffset_e12_final`、`493/...` |
| DPT/depth-head 初始化 height aux | aux-only 和 joint 2-city 短训能正常下降，但没有在 493/461 可视化中表现出稳定 remote 点云突破；global-recon mini benchmark：`RS-only rs_point_l1=291.390`，`Joint rs_point_l1=291.100`，`RS-only centered=97.441`，`Joint centered=97.472` | 用原始 depth/head 权重初始化 height head 有一定合理性，但没有解决核心问题；height 分布可拟合，点云局部形状仍不稳。它说明“head 从零初始化”不是唯一瓶颈 | `p7_vggt_p5b_shared_norm_projection_aux.sh`，DPT/head init 相关 train params；2-city 快速验证 | `outputs/.../training/Crossview/vggt/p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu`、`p7_dptinit_remotehead_joint_2city_e8_b16_2gpu`、`p7_dptinit_remotehead_aggtail2_2city_e12_b16_2gpu`、`p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu` | PLY：`debug/plyview/493/vggt_p7_dptinit_*`；mini benchmark：`benchmarking/rs_guided_dense_mv/newyork/p7_dptinit_globalrecon_normgt_globalfast_2city_e8_b24_2gpu_mini_controls` |
| Seattle 493 projection-aux height 输出差异可视化 | final checkpoint 上 `valid_pixels=33272`，`rel_height_mae=13.584m`，`rel_height_norm_mae=0.145`，`global_dir_cosine=0.047`，`offset_mae=2.575`；height 均值/方差接近 GT，但 global direction 很差，offset 幅度明显偏小 | height head 不是完全塌缩，但投影几何参数没有形成可用的统一机制；这支持“损失/训练过程没有把 height、direction、reconstruction 组织成稳定任务”的判断 | `scripts/visualize_projection_aux_outputs.py --preset p5b_shared_norm --aux-source tokens --pred-normalized` | 非训练实验；checkpoint `p7_frombest_highz_normrecon_globalgt_e3_b8_2gpu/checkpoint-final.pth` | `outputs/.../debug/projection_aux_height/seattle_493_Google_Satellite/vggt_p7_frombest_highz_normrecon_globalgt_final` |
| Aux-only overfit 4 scenes: dense height + dense global-offset | 训练单 batch aux loss 可压低；训练场景 summary：`rel_height_world_abs_mean=0.004-0.014`，`offset_world_norm_mean=0.52-1.33`；test aux loss 仍高，约 `0.715` | 说明 aux head 在固定 token 上至少能记住部分训练场景，标签/导出链路不是完全错误；但泛化很差，且该方案主要学到全局/低维 offset，不证明 sparse offset 分支可用 | `p7_vggt_p5b_shared_norm_projection_aux.sh` + `vggt_p7_p5b_private_oldp7_p5bhead_frozen_trunk_remotehead_auxonly.yaml`；冻结 aggregator/point/depth/camera/remote point head，只训练 projection aux token/image/global heads；`overfit_num_sets=4`，train scenes 为 `chicago__location_1..4` | `outputs/.../training/Crossview/vggt/p7_auxonly_overfit4_denseheight_densegoffset_e80_b4_2gpu` | overfit PLY：`debug/plyview/overfit/chicago_location_{1..4}_google/vggt_p7_auxonly_overfit4_denseheight_densegoffset_e80_final`；对照 PLY：`461_1/...`、`493/...` 同名目录 |
| Aux-only overfit 4 scenes: dense height + direct offset | 训练 loss 没有稳定过拟合；final test `rs_projection_aux_loss=1.552`，`offset_dir_cosine=0.0365`；训练场景 summary：`rel_height_world_abs_mean=0.159-0.239`，`offset_world_norm_mean=0.63-2.52`，比 dense-global overfit 更差 | 强 direct-offset 监督没有解决 `aux_offset_remote.ply` 乱的问题，反而破坏 height/offset 平衡。当前 offset 参数化/解码头对方向场学习不稳，下一步不应继续简单调大 offset loss，而应改成重建点云一致性或低维几何参数化 | 同上冻结设置；关闭 dense global-offset，显式打开 direct offset/balanced/mag/dir，global dir/slope 仅 `0.01` 用于避免 DDP static graph unused 参数 | `outputs/.../training/Crossview/vggt/p7_auxonly_overfit4_denseheight_directoffset_e120_b4_2gpu` | overfit PLY：`debug/plyview/overfit/chicago_location_{1..4}_google/vggt_p7_auxonly_overfit4_denseheight_directoffset_e120_final` |
| Aux-only overfit 随机视角数量诊断 | dense-global：`rel_height_world_abs_mean` 均值约 `0.009-0.010`，`offset_world_norm_mean` 随 n 从 `0.49` 增到 `1.17`；direct-offset：`rel_height_world_abs_mean` 均值约 `0.204-0.230`，`offset_world_norm_mean` 随 n 从 `1.00` 增到 `1.92` | 随机视角和更多普通视角没有改变结论。dense-global 对 height 稳定但 offset 随 view 数增加变大；direct-offset 在训练场景上仍 height 过大、offset 更不稳。之前顺序 `00-03` 只是不够严格的导出诊断，不是训练采样方式 | 不重新训练；为 `chicago__location_1..4` 随机抽普通视角，分别构造 `n=2/4/8/16` 输入目录，remote 固定 Google satellite，测试同一 overfit checkpoint 对 view 数量和 view 选择的敏感性 | 输入：`debug/overfit_random_export_inputs/seed_0608/chicago_location_{1..4}_google_n{2,4,8,16}` | PLY：`debug/plyview/overfit_random/chicago_location_{1..4}_google_n{2,4,8,16}/vggt_p7_auxonly_denseglobal_e80` 和 `.../vggt_p7_auxonly_directoffset_e120` |
| VGGT-Omega projection-aux 反事实 | 启动前检查失败：`/root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt` 不存在；在 `/root/autodl-tmp/outputs/checkpoints` 和 `/root/autodl-tmp` 未找到 omega/vggt-1b 权重 | 当前不能作为有效实验；VGGT-Omega 需要本地 gated checkpoint。不要用随机初始化替代，否则结果不可解释 | `bash_scripts/train/Crossview/vggt_omega/p7_vggt_omega_projection_aux.sh`；计划用 dense height/dense global-offset 做短 smoke | 未启动 | 无 |

## 当前问题判断：损失和训练过程

目标仍然是 remote 点云恢复。projection_aux 多任务分解投影参数是必要的，因为它能约束模型不是只在 remote pointmap 上做混乱拟合，而是显式学到“remote 像素如何通过高度/投影方向还原 3D”的机制。但目前实验说明，aux head 能学到若干边缘信号，不等于它已经学成了能反哺 point head 的机制。

现在更像是损失和训练过程问题，原因有四个：

- 数据标签问题已经被多次反证到“不是主因”：pointmap 标签能训练收敛；dense height 从 `pixel_to_point_map` 派生后与旧 aux 标签在 common 区域基本一致；Seattle 493 的 GT/pred 可视化也显示 height 分布不是完全不可学。
- 模型容量也不像主瓶颈：aux-only overfit 能把 dense-global height 压得很低；DPT/head 初始化、remote head 训练、head-only repair、midtrunk/open aggregator 都能稳定训练，但不能把收益转成 remote pointmap 精度。
- 真正坏的是任务耦合：height 可拟合，但 global direction/offset/reconstructed point 经常不稳定；aux 重建点云差时，如果直接强行与 point head 一致，容易把 point head 往坏的 aux 方向拖。
- 现有损失各管一段，缺少阶段式约束：height、direction/slope、reconstructed_xyz、remote pointmap 同时训练时，模型可以分别降低局部 loss，但不一定形成同一个可反投影的几何解。

因此下一步不应继续简单调大 aux 权重或继续加同类 weak prior。更合理的训练过程是阶段式：

1. `Stage A`：只稳定机制参数。用 pointmap-derived dense height 监督 height；direction/slope 用从 GT pointmap 拟合出的全局方向监督；不要让 aux reconstructed point loss 回传到 point head。
2. `Stage B`：让 aux 反投影点云直接对 GT pointmap 做 scale-normalized loss，但只在 high-structure / overlap / high-confidence 区域启用，并且先只更新 aux head 和很小一部分 remote adapter。
3. `Stage C`：aux reconstructed point 已经可视化可用后，再加入 point head 与 aux reconstruction 的一致性。这个一致性必须是 stop-gradient 或 teacher-student 形式，初期只能让 point head 向稳定 aux teacher 靠，不能让差 aux 和 point head 互相拉扯。
4. `Stage D`：最后再联合微调 remote point head、aux head、少量 remote-specific adapter；ordinary branch 继续保护，避免 remote 梯度污染普通多视角重建。

具体损失建议：

- height 用 dense `rel_height`，Huber/Charbonnier，加 high-z 或梯度区域权重，但避免 low/background 主导。
- direction/slope 不再依赖稀疏 direct offset 场；优先从 GT pointmap 对全图拟合一个低维投影方向/斜率，作为每 scene/provider 的稳定监督。
- reconstructed_xyz loss 必须和 GT pointmap 做同样的中心化/尺度归一化后再算；正式 benchmark 虽在 `remote_pointmetric20` 下报告 metric 相关结果，但模型训练目标仍应按非绝对尺度处理，不能强迫 aux height 学 GT 的米制绝对尺度。
- consistency loss 初期使用 `detach(aux_recon)` 或 EMA aux teacher，权重小，并且只在 aux 可视化通过 sanity check 后打开。
- remote pointmap 主 loss 继续保留 top-percent/robust/overlap，作为最终点云质量锚点；aux 是机制约束，不应替代 pointmap 主监督。

## Aux 标签稠密化候选

目标不是简单放宽所有 mask，而是把 aux 监督拆成不同可信度层级：高可信标签继续严格监督，低可信或伪标签区域只提供软约束。这样可以增加梯度覆盖，同时避免把 noisy remote 标签直接放大。

可尝试的方向：

- Pointmap 派生 dense height：从所有有效 remote `pts3d` 直接投影出相对高度图，不再只依赖当前严格 aux mask；对异常高度使用 percentile clamp 或 Huber loss。
- Soft mask 而不是 hard mask：把 MoGe confidence、点云有效性、局部平滑度、overlap 一致性合成连续权重，低置信区域仍有小权重梯度。
- GT + MoGe 蒸馏融合：GT 有效处以 GT height 为主，GT 稀疏或不稳定处用 MoGe 相对高度/梯度补充，只监督局部形状和 ranking，不强行监督绝对尺度。
- Morphological mask 扩张：对高可信 aux mask 做小半径膨胀，用邻域插值补齐局部洞；权重随离可信像素距离衰减。
- Overlap 区域强锚点，非 overlap 区域弱形状：普通视角与 remote 重叠点用严格 point/height loss，非重叠区域只做相对高度、梯度、edge 一致性。
- Multi-scale aux：低分辨率全图 height/gradient 提供稠密形状趋势，高分辨率只在高可信区域监督细节，减少稀疏高频标签导致的局部噪声。

优先级建议：先做 `dense pointmap-derived height + soft mask + MoGe relative-gradient fallback`，因为它改动最小，且直接针对当前 aux 标签过稀的问题。

已完成的直接派生检查：

- 脚本：`scripts/visualize_dense_pointmap_height_aux.py`
- 输出：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/dense_pointmap_height_aux`
- 方法：用 `projection_aux` 的 `original_z - rel_height` 估计参考 `ground_z`，再对所有有效 `pixel_to_point_map.xyz[..., 2]` 计算 dense `rel_height = z - ground_z`。
- 结果：dense height 的建筑轮廓和高低趋势与原 aux height 一致，common 区域中位误差小；新增区域主要是原 aux 没覆盖到的道路、阴影和建筑边缘。
- 风险：新增区域低高度背景很多，如果训练时权重过高，会进一步强化 low/background calibration；应使用 soft mask、Huber/clamp 和较低权重。

已完成的 dense projection aux 反投影检查：

- 脚本：`scripts/reconstruct_remote_pointcloud_from_projection_aux.py --export_dense_pointmap_height`
- 输出：`/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/projection_aux_reconstruct_dense`
- 新增文件：`dense_pointmap_original_xyz.ply`、`dense_projected_from_global_centered.ply`、`dense_reconstructed_from_rel_global.ply`、`dense_pointmap_projection_aux_labels.npz`、`dense_pointmap_height_grid.png`
- 结果：461/493 的 dense height 与原 aux height 在 common 区域形状和数值一致，说明 height 标签不是主要错误源；dense rel-height + global direction/slope 可以几何自洽地重建回 pointmap。
- 关键问题：461/493 中 pointmap valid 和 aux valid 总比例接近，但 common 只有 `4.48%/2.26%`，大量有效像素互为 dense-only 或 aux-only。也就是说当前 aux 任务不是在 pointmap 主 loss 的同一批像素上解释几何，监督域错位会削弱多任务学习的协同。
- 训练含义：如果 aux 仍使用旧 mask，继续提高权重或加复杂 head 很可能只让 aux head 学到一个旁路任务；更合理的下一步是把 rel_height 监督直接从 pointmap 派生到 pointmap valid 区域，同时把旧 aux/global/offset 标签作为高可信 subset 或弱一致性约束。

## 导出与评估约定

所有新结构或阶段性好模型，至少导出两个固定场景：

- New York 461：`/root/autodl-tmp/test/scence/461_1`
- Seattle 493：`/root/autodl-tmp/test/scence/493`

注意：这里的 `493` 按当前测试目录使用 Seattle 场景，不是 `newyork__location_493`。如果从训练数据目录直接取 remote GT，可对应到 `seattle__location_493/Google_Satellite`。

正式 benchmark：

- 主路径仍为 `rs_guided_dense_mv/newyork/remote_pointmetric20`，这样可以和已有表格结果对齐。
- `remote_pointmetric20` 输出中需要同时记录：`rs_point_l1`、`rs_point_l1_centered`、`rs_point_l1_scale_aligned`、`rs_height_mae_affine`、`rs_height_rmse_affine`。
- `rs_point_l1_centered` 的计算是 `(pred - mean(pred))` 对比 `(gt - mean(gt))`，只消除平移，不消除尺度；因此它比 raw `rs_point_l1` 更能排除整体偏移，但仍不能当作尺度归一化指标。
- 筛选实验时优先看 `rs_point_l1_scale_aligned`、`rs_height_mae_affine` 和可视化局部形状；`rs_point_l1` 受整体平移/尺度估计影响较大，不能单独作为成败判断。`rs_point_l1_scale_aligned` 当前只做平移和正单尺度对齐，不做旋转，也不允许负尺度反射；如果后续怀疑方向/旋转误差，也可以再补 `rs_point_l1_sim3`。
- `scripts/run_rs_guided_inference_batch.py` 中 joint remote aligned 路径会构造 avg-dis 归一化空间；remote metric 路径还会用 `meters_per_pixel` 估计尺度。因此文档中报告 metric 数字可以保留，但训练 loss 不应按绝对米制尺度硬约束。

`scripts/export_pointcloud_ply.py` 当前默认行为：

- 每个 control mode 输出 mixed 点云，例如 `mapanything_pointcloud_same.ply`
- 同目录自动附带 remote-only 点云，例如 `mapanything_pointcloud_same_remote.ply`
- 不再需要额外传 `--export_view_filter remote`
- 如果传 `--export_projection_aux_reconstruction`，还会同目录输出 projection aux 诊断重建点云，例如 `mapanything_pointcloud_same_aux_offset_remote.ply` 和 `mapanything_pointcloud_same_aux_global_remote.ply`，用于和 remote point head 的 `mapanything_pointcloud_same_remote.ply` 对比。该重建在纯推理时没有 GT projection base，因此使用 point-head remote 点云作为投影基准，适合检查 aux 输出形状是否自洽，不应当直接当作真实物理反投影误差。
- P7 projection-aux/private remote head 现在优先从 checkpoint key 自动识别；新实验名不再必须包含 `p7_proj`。导出和 benchmark 日志里仍应检查 remote view 使用 `head=point`，并且 `use_remote_projection_aux_head=True`、`remote_projection_aux_source=tokens`。

标准导出命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_pointcloud_ply.py \
  --model vggt \
  --checkpoint_path /path/to/checkpoint-final.pth \
  --image_folder /root/autodl-tmp/test/scence/461_1 \
  --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/EXPERIMENT_NAME \
  --remote_view_names image.png \
  --export_remote_control_modes same blank \
  --export_projection_aux_reconstruction
```

正式 benchmark 模板：

```bash
OUTPUT_DIR='${root_experiments_dir}/mapanything/benchmarking/rs_guided_dense_mv/newyork/remote_pointmetric20/EXPERIMENT_NAME' \
CKPT_PATH=/path/to/checkpoint-final.pth \
NUM_VIEWS=4 BATCH_SIZE=1 CUDA_DEVICE=0 REMOTE_CONTROL_MODES='[same,blank,shuffled]' \
bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_crossview_p5b_unified.sh \
  vggt_use_remote_private_point_head=true
```

## Dense Height 的尺度处理

P7 projection_aux 的 dense height 不应该被理解成绝对米制高度。VGGT/MapAnything 当前重建本身是 up-to-scale 的：普通 pointmap loss 也通过 `avg_dis` / shared normalization 处理尺度。因此 dense height 训练必须在归一化空间里做，而不是直接要求预测 `GT rel_height` 的米制数值。

推荐做法：

1. 从 GT `pixel_to_point_map` 派生 dense `rel_height_gt_raw`。
2. 对每个 remote 样本计算一个稳定尺度 `s_gt`，优先用同一 valid mask 下 pointmap 的 `avg_dis`，或 rel-height 的有效分位数，例如 q0.9；后续保持一种定义，不混用。
3. 训练 target 使用 `rel_height_gt_norm = rel_height_gt_raw / s_gt`。
4. 模型输出的是 normalized height，例如 `rel_height_pred_norm`；可视化或反投影时再乘以当前样本的尺度 `s`。
5. 如果 reconstructed_xyz 要和 GT pointmap 算 loss，必须先把 predicted reconstructed_xyz 和 GT pointmap 做同样的中心化/尺度归一化，再计算 L1/Huber。

这意味着 height loss 监督的是“相对高度形状和比例”，不是绝对米制高度。对高楼/边缘区域可以加权，但加权后的 loss 仍应在 normalized height 空间中计算。

当前已有配置里 `remote_projection_rel_height_scale_mode` 支持 `gt_pointmap_norm` 和 `valid_quantile`。下一步实验建议固定为 `gt_pointmap_norm` 或 `valid_quantile(q=0.9)` 之一，并在 benchmark/可视化日志中输出实际 scale，避免不同实验的 height 数字不可比。

## 后续优先级

1. 稠密 height、MoGe shape、overlap 强化、top-percent 鲁棒裁剪、6/8-view、low/zero-covis、Google-only 都没有超过 robust-overlap best；不要再只做同类权重微调。
2. split remote aggregator + 轻量 FiLM late fusion 已验证没有改善 remote pointmap，且 gate 实际幅度很小；继续结构方向时应避免再加保守小 gate，除非改成更直接的 remote 专用解码或更强 backbone 反事实。
3. remote 标签仍可能存在 scene/region 级噪声，继续保留 top-percent/overlap 这类鲁棒监督；但单纯加大 MoGe、普通视角数、或轻量融合结构，目前证据不足。
