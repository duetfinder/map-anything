#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Export a unified world-space point cloud from an image folder.

Supported benchmark models from bash_scripts/benchmark/rs_guided_dense_mv:
- pi3
- pi3_modality_embedding
- pi3_modality_embedding_remote_head
- vggt
- vggt_omega
- da3
- mapanything
- mapanything_rs_joint

Include:

pi3：
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/pi3 \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_base/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_base \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_modality_embedding/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_modality_embedding \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_freeze_shared/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_freeze_shared \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding_remote_head \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_modality_embedding_remote_head/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_modality_embedding_remote_head \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3_modality_embedding_remote_head \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p7_pi3_remote_head_projection_aux_moge_balanced20x4_grad010_edge002_e30_b8_4gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/pi3_p7_remote_head_projection_aux_moge \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_zero_covis/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_zero_covis \
&& \
python scripts/export_pointcloud_ply.py \
    --model pi3 \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/pi3/p3_pi3_low_covis/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/p3_pi3_low_covis \

mapanything:
python scripts/export_pointcloud_ply.py \
    --model mapanything \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/mapanything

mapanything_rs_joint:
# P4 MapAnything RS-joint checkpoints are supported for export. Use
# --model mapanything_rs_joint and pass the trained checkpoint. If the input
# folder contains a satellite / map image, mark it with --remote_view_names
# or --remote_view_indices so that it is routed through the remote direct
# pointmap head. Unmarked views use the ordinary MapAnything aerial branch.
# Filename metadata used for --remote_view_names is stripped before calling
# MapAnything.infer(), whose input validator only accepts model-facing keys.
python scripts/export_pointcloud_ply.py \
    --model mapanything_rs_joint \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/mapanything/p4_mapanything_rs_joint_500_4gpu_all/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/mapanything_p4_rs_joint \
    --remote_view_names zimage.png

# Baseline comparison with the original MapAnything checkpoint.
python scripts/export_pointcloud_ply.py \
    --model mapanything \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/mapanything_base


vggt:
# Original VGGT, matching the wrapper path used by the benchmark. This detects
# /outputs/checkpoints/vggt/model.pt as a raw VGGT state_dict and loads it via
# VGGTWrapper.model.load_state_dict(...), not as a MapAnything training ckpt.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/vggt/model.pt \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt \
&& \
# p5b default mixed export: ordinary views use camera+depth, remote uses point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5b_mixed \
    --vggt_joint_remote_export \
    --vggt_export_mode mixed \
    --remote_view_names image.png \
&& \
# p5b diagnostic: the mixed PLY is exported by default, and a separate remote
# companion PLY is written next to it when a remote view is marked.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5b \
    --vggt_joint_remote_export \
    --vggt_export_mode mixed \
    --remote_view_names image.png \
&& \
# P7 dpt-init projection aux capacity probe. This writes the mixed point-head
# PLY, the companion remote-only PLY, and aux reconstructions:
# *_aux_offset_remote.ply / *_aux_global_remote.ply.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_dptinit_linearheight_auxonly_2city_highbucket_e6_b16_2gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_dptinit_linearheight_auxonly_2city_e6_final \
    --remote_view_names image.png \
    --export_projection_aux_reconstruction \
&& \
# p5b diagnostic: force every view through camera+depth.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5b_depth_all \
    --vggt_joint_remote_export \
    --vggt_export_mode depth_all \
    --remote_view_names image.png \
&& \
# p5b diagnostic: force every view through point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5b_vggt_joint_shared_all_shared_norm/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5b_point_all \
    --vggt_joint_remote_export \
    --vggt_export_mode point_all \
    --remote_view_names image.png \
&& \
# p5c
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5c_vggt_joint_shared_all_viewtype/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5c_mixed \
    --vggt_joint_remote_export \
    --vggt_export_mode mixed \
    --config_overrides machine=aws model=vggt model.model_config.use_view_type_bias=true \
    --remote_view_names image.png \
&& \
# p5d remote-private point head + consistency checkpoint.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5d_vggt_remote_point_head_consistency/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5d_mixed \
    --vggt_joint_remote_export \
    --vggt_use_remote_private_point_head \
    --vggt_export_mode mixed \
    --remote_view_names image.png \
&& \
# p5e default mixed export: ordinary views use camera+depth, remote uses point_head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5e_vggt_remote_head_attention_viewtype/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5e_mixed \
    --vggt_joint_remote_export \
    --vggt_ordinary_output_head depth \
    --vggt_remote_output_head point \
    --vggt_use_remote_private_point_head \
    --remote_view_names image.png

# p5f-lite: early view-type embedding + remote-to-aerial gated residual.
# If --export_remote_control_modes is set, one PLY is written per mode with
# suffixes such as *_same.ply and *_blank.ply. shuffled also requires
# --shuffled_remote_image_path.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p5f_vggt_lite_early_bias_gated_residual/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p5f_lite_mixed \
    --vggt_p5f_lite_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p6a conditional remote adapter: official raw VGGT base + split late cross-attn.
# P6A uses remote as a conditioning input. By default this exports ordinary-view
# points only, because protected split heads predict remote points in a separate
# split frame. Add --include_remote_points only for debugging that remote branch.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6a_vggt_raw_base_conditional_remote_adapter/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p6a_raw_ordinary \
    --vggt_p6a_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p6b joint remote alignment: ordinary views use camera+depth, remote views use
# the trained remote point path. Private-head and shared-head checkpoints are
# auto-detected from the checkpoint path; use --vggt_p6b_export for clarity.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p6b_vggt_joint_remote_alignment_private_head_w03_bs5_static_remoteonly/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p6b_private_mixed \
    --vggt_p6b_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 projection-aux: split late fusion plus remote projection auxiliary heads.
# Default export writes ordinary-view reconstruction under remote conditioning;
# add --include_remote_points only to inspect the remote branch itself.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_projection_aux_split_late_fusion/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_projection_aux_ordinary \
    --vggt_p7_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 remote-head projection-aux: p5d-style separate remote point head plus
# projection auxiliary multitask learning, without split/late fusion.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_vggt_remote_head_projection_aux_trunk/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_remote_head_projection_aux_mixed \
    --vggt_p7_remote_head_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 p5b shared-norm projection-aux: P5B shared point head plus projection
# auxiliary multitask supervision, without split aggregator, late fusion, or
# remote private point head.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_p5b_shared_norm_projection_aux_mixed \
    --vggt_p7_p5b_shared_norm_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 p5b diagnostic. The remote-only companion is written automatically.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_joint_pm4_aux_h075_lowover15_lowtrunklr2e6_warmbest_e8_b8_2gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_p5b_shared_norm_projection_aux \
    --vggt_p7_p5b_shared_norm_projection_aux_export \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 p5e private-viewtype projection-aux lowtrunkfull: best current mechanism /
# remote reconstruction candidate. The preset is auto-detected from the path.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull_mixed \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 p5e diagnostic. The remote-only companion is written automatically.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5e_private_viewtype_projection_aux_lowtrunkfull_warmp5efinal_e6_b9_2gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_p5e_private_viewtype_projection_aux_lowtrunkfull \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# New P7 structure/training-direction checks on the fixed 448 scene:
#
# p7 p5b parallel-token aux + recover-p5b diagnostic. This is the main
# parallel aux-head candidate to compare against P5B/P7-P5B. The mixed PLY and
# remote-only companion PLY are written into the same output directory.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_allcities_p5b_parallel_token_aux_recoverp5b_h035_warmpreserve_e8_b9_4gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/448 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_p5b_parallel_token_aux_recoverp5b_h035_b9_4gpu_final \
    --vggt_p7_p5b_shared_norm_projection_aux_export \
    --vggt_projection_aux_source tokens \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 oldP7 private remote-head trainable diagnostic. This is the least-degraded
# current trainable remote-head run, not a final quality candidate yet. The
# mixed PLY and remote-only companion PLY are written into the same directory.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_oldp7_train_remotehead_nonreentrant_raw001_paramanchor500k_lowlr3e6_h003_e2_b32_4gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/448 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/448/vggt_p7_oldp7_train_remotehead_nonreentrant_raw001_paramanchor500k_lowlr3e6_h003_e2_final \
    --vggt_p7_p5b_shared_norm_projection_aux_export \
    --vggt_projection_aux_source tokens \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

# p7 projection-MoGe aux: MoGe2 edge/gradient prior is applied to the
# projection-aux relative height branch, not to the remote pointmap branch.
# The checkpoint path is auto-detected as a P7 P5B-style parallel-token aux
# export. Each output directory contains mixed PLYs plus same-path remote-only
# companion PLYs.
python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/461_1 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_best \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_aux_balanced20x4_private_tokens_raw001_gradz005_projmg02_edge005_h003_warme2_e40_b28_4gpu/checkpoint-final.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_p7_proj_moge_aux_balanced20x4_private_tokens_projmg02_edge005_final \
    --remote_view_names image.png \
    --export_remote_control_modes same blank

python scripts/export_pointcloud_ply.py \
    --model vggt \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt/p7_proj_moge_pmheight_h5_prior0_balanced20x4_private_tokens_warmbest_e6_b8_4gpu/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/461_1 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/461_1/vggt_p7_proj_moge_pmheight_h5_prior0_best \
    --remote_view_names image.png \
    --export_remote_control_modes same



vggt_omega:
# Fine-tuned VGGT-Omega Crossview checkpoint. VGGT-Omega uses patch_size=16, so
# use resolution_set=512 or another 16-aligned fixed_size.
python scripts/export_pointcloud_ply.py \
    --model vggt_omega \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_omega_finetuned \
    --resolution_set 512 \
    --remote_view_names image.png \
python scripts/export_pointcloud_ply.py \
    --model vggt_omega \
    --checkpoint_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/training/Crossview/vggt_omega/p1_vggt_omega_joint_depth_512_all_2/checkpoint-best.pth \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_omega_finetuned_2 \
    --resolution_set 512 \
    --remote_view_names image.png \
&& \
# Raw released VGGT-Omega checkpoint before Crossview fine-tuning.
python scripts/export_pointcloud_ply.py \
    --model vggt_omega \
    --checkpoint_path /root/autodl-tmp/outputs/checkpoints/vggt_omega/vggt_omega_1b_512.pt \
    --image_folder /root/autodl-tmp/test/scence/493 \
    --output_path /root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/plyview/493/vggt_omega_raw \
    --resolution_set 512
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from time import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

from mapanything.utils.colmap_export import voxel_downsample_point_cloud
from mapanything.utils.geometry import depthmap_to_world_frame, normalize_multiple_pointclouds
from mapanything.utils.hf_utils.hf_helpers import (
    initialize_mapanything_local,
    initialize_mapanything_model,
)
from mapanything.utils.image import heif_support_enabled, load_images

DEFAULT_MODEL = "pi3"
DEFAULT_CONFIG_PATH = "configs/train.yaml"
DEFAULT_MAPANYTHING_HF_MODEL = "facebook/map-anything"
SUPPORTED_MODELS = [
    "pi3",
    "pi3_modality_embedding",
    "pi3_modality_embedding_remote_head",
    "vggt",
    "vggt_omega",
    "da3",
    "mapanything",
    "mapanything_rs_joint",
]
DEFAULT_CONFIG_OVERRIDES = {
    "pi3": [
        "machine=aws",
        "model=pi3",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "pi3_modality_embedding": [
        "machine=aws",
        "model=pi3_modality_embedding",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "pi3_modality_embedding_remote_head": [
        "machine=aws",
        "model=pi3_modality_embedding_remote_head",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "vggt": [
        "machine=aws",
        "model=vggt",
    ],
    "vggt_omega": [
        "machine=aws",
        "model=vggt_omega",
    ],
    "da3": [
        "machine=aws",
        "model=da3",
    ],
    "mapanything": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "mapanything_rs_joint": [
        "machine=aws",
        "model=mapanything_rs_joint",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
}
IDENTITY_MODELS = {
    "anycalib",
    "moge",
    "pi3",
    "pi3_modality_embedding",
    "pi3_modality_embedding_remote_head",
    "pi3x",
    "vggt",
    "vggt_omega",
}
CLASH_ENV = {
    "http_proxy": "http://127.0.0.1:7890",
    "https_proxy": "http://127.0.0.1:7890",
    "all_proxy": "socks5://127.0.0.1:7891",
}
REMOTE_INSTANCE_VALUE = "remote"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a supported benchmark model on an image folder and export the "
            "unified world-space point cloud as PLY."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help="Model to run. Matches the rs_guided_dense_mv benchmark model set.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing input images.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional local checkpoint (.pth/.pt/.safetensors). If omitted, the "
            "script uses the model's default HuggingFace weights."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mapanything_pointcloud.ply",
        help="Output PLY path, or a directory to receive mapanything_pointcloud.ply.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Hydra config path used for local-checkpoint initialization.",
    )
    parser.add_argument(
        "--config_json_path",
        type=str,
        default=None,
        help="Optional JSON containing model_str/model_config overrides.",
    )
    parser.add_argument(
        "--model_str",
        type=str,
        default=None,
        help="Optional model alias override for local-checkpoint initialization.",
    )
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=None,
        help="Optional Hydra override list. Defaults depend on --model.",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace model name for no-checkpoint runs. Currently used "
            "for mapanything; defaults to facebook/map-anything."
        ),
    )
    parser.add_argument(
        "--enable_clash_proxy",
        action="store_true",
        default=False,
        help=(
            "Set the same proxy env vars as 'source /etc/profile.d/clash.sh && proxy_on' "
            "before downloading HuggingFace weights."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load checkpoint with strict=True. Default is False for compatibility.",
    )
    parser.add_argument(
        "--vggt_joint_remote_export",
        action="store_true",
        default=False,
        help=(
            "Special handling for VGGT p5b/p5c RS-joint checkpoints: disable "
            "wrapper-side pretrained/custom init, enable remote point-head routing, "
            "and enable remote-view tagging for mixed export."
        ),
    )
    parser.add_argument(
        "--vggt_export_mode",
        type=str,
        default=None,
        choices=["mixed", "depth_all", "point_all", "ordinary_point_remote_depth"],
        help=(
            "Convenience VGGT output-head mode. mixed=ordinary depth and remote "
            "point; depth_all=all camera+depth; point_all=all point_head; "
            "ordinary_point_remote_depth swaps the mixed assignment."
        ),
    )
    parser.add_argument(
        "--vggt_ordinary_output_head",
        type=str,
        default=None,
        choices=["depth", "point"],
        help="Explicit output head for non-remote VGGT views.",
    )
    parser.add_argument(
        "--vggt_remote_output_head",
        type=str,
        default=None,
        choices=["auto", "depth", "point"],
        help="Explicit output head for remote VGGT views.",
    )
    parser.add_argument(
        "--vggt_use_remote_private_point_head",
        action="store_true",
        default=False,
        help=(
            "Enable VGGT remote private point_head when exporting p5d checkpoints."
        ),
    )
    parser.add_argument(
        "--vggt_p5f_lite_export",
        action="store_true",
        default=False,
        help=(
            "Enable the p5f-lite VGGT export preset: mixed ordinary/remote heads, "
            "remote private point head, pre-aggregator view-type embedding, and "
            "remote-to-aerial gated residual. This is also auto-enabled when "
            "checkpoint_path contains p5f_vggt_lite."
        ),
    )
    parser.add_argument(
        "--vggt_p6a_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P6A VGGT export preset: mixed ordinary/remote heads, "
            "remote private point head, split remote aggregator, late remote-to-aerial "
            "cross-attention, and protected ordinary heads. This is also auto-enabled "
            "when checkpoint_path contains p6a_vggt."
        ),
    )
    parser.add_argument(
        "--vggt_p6b_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P6B VGGT export preset: mixed ordinary/remote heads and "
            "P6B-specific remote point routing. Private-head, shared-head, and "
            "viewtype variants are auto-detected from checkpoint_path when possible. "
            "This is also auto-enabled when checkpoint_path contains p6b_vggt."
        ),
    )
    parser.add_argument(
        "--vggt_p7_projection_aux_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 VGGT projection-aux export preset: mixed ordinary/remote "
            "heads, split remote aggregator, late remote-to-aerial fusion, protected "
            "ordinary heads, and remote projection auxiliary heads. This is also "
            "auto-enabled when checkpoint_path contains p7_vggt_projection_aux or "
            "p7_projection_aux."
        ),
    )
    parser.add_argument(
        "--vggt_p7_remote_head_projection_aux_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 remote-head projection-aux export preset: mixed "
            "ordinary/remote heads, remote private point head, and remote projection "
            "auxiliary heads, without split aggregator or late fusion. This is also "
            "auto-enabled when checkpoint_path contains "
            "p7_vggt_remote_head_projection_aux or p7_remote_head_projection_aux."
        ),
    )
    parser.add_argument(
        "--vggt_p7_p5b_shared_norm_projection_aux_export",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 P5B shared-norm projection-aux export preset: mixed "
            "ordinary/remote heads, shared remote point head, and remote projection "
            "auxiliary heads, without split aggregator, late fusion, or remote "
            "private point head. This is also auto-enabled for checkpoint paths "
            "containing p7_vggt_p5b_shared_norm_projection_aux or "
            "p7_chicago_newyork_full_p5b_joint or p7_allcities_p5b_joint."
        ),
    )
    parser.add_argument(
        "--vggt_projection_aux_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for P7 remote projection auxiliary heads.",
    )
    parser.add_argument(
        "--vggt_projection_aux_source",
        type=str,
        default="auto",
        choices=["auto", "pointmap", "tokens", "dpt_init"],
        help=(
            "Feature source for P7 remote projection auxiliary heads. auto uses "
            "dpt_init or tokens when matching checkpoint keys are present, and "
            "pointmap otherwise."
        ),
    )
    parser.add_argument(
        "--vggt_projection_aux_detach_pointmap",
        action="store_true",
        default=False,
        help="Detach the pointmap input before the P7 projection auxiliary heads.",
    )
    parser.add_argument(
        "--vggt_projection_aux_use_rgb",
        action="store_true",
        default=False,
        help="Condition the P7 projection auxiliary pixel head on remote RGB plus pointmap.",
    )
    parser.add_argument(
        "--vggt_projection_aux_use_coord",
        action="store_true",
        default=False,
        help="Condition the P7 projection auxiliary pixel head on normalized image coordinates.",
    )
    parser.add_argument(
        "--vggt_projection_aux_positive_slope",
        action="store_true",
        default=False,
        help="Constrain the P7 projection auxiliary global slope prediction to be positive.",
    )
    parser.add_argument(
        "--vggt_projection_aux_image_stem_dim",
        type=int,
        default=0,
        help="Remote RGB image-stem channel dimension for P7 projection auxiliary heads.",
    )
    parser.add_argument(
        "--vggt_projection_aux_slope_init",
        type=float,
        default=0.1,
        help="Initial positive global slope for the P7 projection auxiliary head.",
    )
    parser.add_argument(
        "--vggt_projection_aux_num_blocks",
        type=int,
        default=0,
        help="Number of residual conv blocks in the P7 projection auxiliary pixel head.",
    )
    parser.add_argument(
        "--vggt_projection_aux_token_residual",
        action="store_true",
        default=False,
        help=(
            "Enable the P7 token-residual adapter on remote patch tokens before the "
            "remote point head. This is also auto-enabled for checkpoint paths "
            "containing p7_proj_tokenres."
        ),
    )
    parser.add_argument(
        "--vggt_projection_aux_token_residual_hidden_scale",
        type=float,
        default=0.25,
        help="Hidden scale for the P7 token-residual adapter.",
    )
    parser.add_argument(
        "--vggt_projection_aux_token_residual_gate_init",
        type=float,
        default=0.01,
        help="Gate init used to build the P7 token-residual adapter before loading a checkpoint.",
    )
    parser.add_argument(
        "--export_projection_aux_reconstruction",
        action="store_true",
        default=False,
        help=(
            "For P7 projection-aux checkpoints, export diagnostic remote point clouds "
            "reconstructed from the predicted projection aux rel-height/offset. The "
            "remote point-head output is used as the projection base because GT "
            "projection_center/projected_xyz are unavailable at inference time."
        ),
    )
    parser.add_argument(
        "--projection_aux_rel_height_scale_mode",
        type=str,
        default="pred_avg_dis",
        choices=["pred_avg_dis", "pred_z_std", "fixed", "gt_height_range"],
        help=(
            "Scale used to convert predicted aux rel-height back from loss space. "
            "pred_avg_dis matches avg_dis-style pointmap normalization approximately. "
            "gt_height_range min-max aligns predicted rel-height to GT rel-height "
            "quantiles from --projection_aux_gt_remote_dir for diagnostics."
        ),
    )
    parser.add_argument(
        "--projection_aux_gt_remote_dir",
        type=str,
        default=None,
        help=(
            "Remote GT provider directory containing projection_aux.npz. Required "
            "when --projection_aux_rel_height_scale_mode=gt_height_range."
        ),
    )
    parser.add_argument(
        "--projection_aux_use_gt_global_direction",
        action="store_true",
        default=False,
        help=(
            "Diagnostic only: use global_dir_xy from GT projection_aux.npz for "
            "projection-aux global reconstruction."
        ),
    )
    parser.add_argument(
        "--projection_aux_use_gt_global_slope",
        action="store_true",
        default=False,
        help=(
            "Diagnostic only: use global_slope from GT projection_aux.npz for "
            "projection-aux global reconstruction."
        ),
    )
    parser.add_argument(
        "--projection_aux_use_gt_projection_base",
        action="store_true",
        default=False,
        help=(
            "Diagnostic only: use projected_xyz_centered and projection_center_xy "
            "from GT projection_aux.npz as the reconstruction base. This matches "
            "the training reconstruction formula and should be used when validating "
            "aux geometry."
        ),
    )
    parser.add_argument(
        "--projection_aux_gt_height_range_low_quantile",
        type=float,
        default=0.01,
        help="Low quantile for GT-height-range diagnostic alignment.",
    )
    parser.add_argument(
        "--projection_aux_gt_height_range_high_quantile",
        type=float,
        default=0.99,
        help="High quantile for GT-height-range diagnostic alignment.",
    )
    parser.add_argument(
        "--projection_aux_rel_height_fixed_scale",
        type=float,
        default=1.0,
        help="Fixed rel-height scale used when --projection_aux_rel_height_scale_mode=fixed.",
    )
    parser.add_argument(
        "--projection_aux_xyz_align_mode",
        type=str,
        default="none",
        choices=["none", "gt_pointmap_unit_xy_zrange", "gt_pointmap_unit_xy_zrange_flipz"],
        help=(
            "Optional diagnostic alignment applied after projection-aux xyz reconstruction. "
            "gt_pointmap_unit_xy_zrange maps aux xyz into a scale-free coordinate system "
            "whose x/y span and z range are derived from GT pixel_to_point_map.npz. "
            "The flipz variant reverses the normalized z direction for viewers/datasets "
            "whose visual up direction is opposite to the exported z axis."
        ),
    )
    parser.add_argument(
        "--projection_aux_xyz_align_low_quantile",
        type=float,
        default=0.01,
        help="Low quantile for projection-aux xyz diagnostic alignment.",
    )
    parser.add_argument(
        "--projection_aux_xyz_align_high_quantile",
        type=float,
        default=0.99,
        help="High quantile for projection-aux xyz diagnostic alignment.",
    )
    parser.add_argument(
        "--projection_aux_offset_scale",
        type=float,
        default=32.0,
        help="Scale used to convert predicted aux offset_xy back from loss space.",
    )
    parser.add_argument(
        "--projection_aux_ground_quantile",
        type=float,
        default=0.2,
        help="Remote point-head z quantile used as the ground reference for aux height reconstruction.",
    )
    parser.add_argument(
        "--include_remote_points",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Mixed exports now include marked remote-view "
            "points by default, and a separate remote companion PLY is written next to "
            "the mixed PLY when a remote view is marked."
        ),
    )
    parser.add_argument(
        "--vggt_late_fusion_type",
        type=str,
        default="none",
        choices=["none", "film", "cross_attention"],
        help="Late remote-to-aerial fusion type used by P5h/P6A/P7 VGGT exports.",
    )
    parser.add_argument(
        "--vggt_late_gate_init",
        type=float,
        default=1e-3,
        help="Late fusion gate init used to build P6A export wrapper before loading checkpoint.",
    )
    parser.add_argument(
        "--vggt_max_remote_tokens",
        type=int,
        default=256,
        help="Maximum remote tokens for VGGT late cross-attention exports.",
    )
    parser.add_argument(
        "--vggt_cross_attention_heads",
        type=int,
        default=8,
        help="Number of heads for VGGT late remote-to-aerial cross-attention exports.",
    )
    parser.add_argument(
        "--force_remote_instance",
        action="store_true",
        default=False,
        help=(
            "Force every loaded view to use instance='remote'. Useful when exporting "
            "from RS-joint checkpoints that route remote views through a dedicated "
            "point head."
        ),
    )
    parser.add_argument(
        "--remote_view_indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "0-based indices of input images that should be treated as remote views "
            "for RS-joint export. Unspecified views remain ordinary views."
        ),
    )
    parser.add_argument(
        "--remote_view_names",
        nargs="*",
        default=None,
        help=(
            "Basenames of input images that should be treated as remote views for "
            "RS-joint export. If omitted, common satellite/remote filenames such as "
            "image.png, zimage.png, sate*.png, and *Satellite* are auto-detected."
        ),
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory-efficient inference when the model exposes model.infer().",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        help="Minibatch size used by model.infer in memory-efficient mode.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="fixed_mapping",
        choices=["fixed_mapping", "longest_side", "square", "fixed_size"],
        help="Resize mode passed to load_images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize size for longest_side/square modes.",
    )
    parser.add_argument(
        "--fixed_width",
        type=int,
        default=None,
        help="Resize width for fixed_size mode.",
    )
    parser.add_argument(
        "--fixed_height",
        type=int,
        default=None,
        help="Resize height for fixed_size mode.",
    )
    parser.add_argument(
        "--resolution_set",
        type=int,
        default=518,
        choices=[504, 512, 518],
        help="Resolution preset used by load_images when resize_mode=fixed_mapping.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Load every nth image from the folder.",
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        default=True,
        help="Apply non-ambiguous masks when the model exposes model.infer().",
    )
    parser.add_argument(
        "--no_apply_mask",
        action="store_false",
        dest="apply_mask",
        help="Disable non-ambiguous masking.",
    )
    parser.add_argument(
        "--mask_edges",
        action="store_true",
        default=True,
        help="Filter depth discontinuity / normal edges when the model exposes model.infer().",
    )
    parser.add_argument(
        "--no_mask_edges",
        action="store_false",
        dest="mask_edges",
        help="Disable edge masking.",
    )
    parser.add_argument(
        "--apply_confidence_mask",
        action="store_true",
        default=False,
        help="Apply confidence mask before exporting the point cloud.",
    )
    parser.add_argument(
        "--confidence_percentile",
        type=float,
        default=50.0,
        help="Percentile threshold used when apply_confidence_mask is enabled.",
    )
    parser.add_argument(
        "--voxel_downsample",
        action="store_true",
        default=False,
        help="Apply voxel downsampling before exporting. Requires open3d.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Explicit voxel size in world units. If unset, use voxel_fraction.",
    )
    parser.add_argument(
        "--voxel_fraction",
        type=float,
        default=0.01,
        help="Adaptive voxel size fraction used when voxel_size is not set.",
    )
    parser.add_argument(
        "--export_remote_control_modes",
        nargs="*",
        choices=["same", "blank", "shuffled"],
        default=None,
        help=(
            "Optional remote-control visualization modes. same uses the marked "
            "remote image, blank replaces marked remote views with a constant image, "
            "and shuffled replaces them with --shuffled_remote_image_path. When set, "
            "one PLY is exported per mode."
        ),
    )
    parser.add_argument(
        "--export_view_filter",
        type=str,
        default="all",
        choices=["all", "remote", "ordinary"],
        help=(
            "Deprecated compatibility option. The script now always writes the mixed "
            "PLY, plus a remote-only companion PLY when a remote view is marked."
        ),
    )
    parser.add_argument(
        "--blank_remote_value",
        type=float,
        default=0.5,
        help="Pixel value used for blank remote-control exports after identity conversion.",
    )
    parser.add_argument(
        "--shuffled_remote_image_path",
        type=str,
        default=None,
        help="Image path used to replace marked remote views in shuffled control exports.",
    )
    return parser.parse_args()


def resolve_load_size(args: argparse.Namespace):
    if args.resize_mode == "fixed_size":
        if args.fixed_width is None or args.fixed_height is None:
            raise ValueError(
                "--fixed_width and --fixed_height are required when --resize_mode fixed_size"
            )
        return (args.fixed_width, args.fixed_height)
    if args.resize_mode in {"longest_side", "square"}:
        if args.size is None:
            raise ValueError(
                f"--size is required when --resize_mode {args.resize_mode}"
            )
        return args.size
    return None


def is_raw_vggt_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    return checkpoint_path.name == "model.pt" and "checkpoints/vggt" in str(
        checkpoint_path
    )


def is_raw_vggt_omega_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt_omega" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    return checkpoint_path.name in {"vggt_omega_1b_512.pt", "model.pt"} and "checkpoints/vggt_omega" in str(
        checkpoint_path
    )


def is_p5f_lite_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p5f_vggt_lite" in checkpoint_path


def is_p6a_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p6a_vggt" in checkpoint_path


def is_p6b_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p6b_vggt" in checkpoint_path


def is_p7_split_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return "p7_vggt_projection_aux" in checkpoint_path or "p7_projection_aux" in checkpoint_path


def is_p7_remote_head_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return (
        "p7_vggt_remote_head_projection_aux" in checkpoint_path
        or "p7_remote_head_projection_aux" in checkpoint_path
    )


def is_p7_p5b_shared_norm_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return (
        "p7_vggt_p5b_shared_norm_projection_aux" in checkpoint_path
        or "p7_chicago_newyork_full_p5b_joint" in checkpoint_path
        or "p7_allcities_p5b_joint" in checkpoint_path
        or "p7_allcities_p5b_parallel_token_aux" in checkpoint_path
        or "p7_p5b_parallel_token_aux" in checkpoint_path
        or "p7_proj_moge_aux" in checkpoint_path
        or "p7_proj_moge_denseheight" in checkpoint_path
        or "p7_proj_moge_pmheight" in checkpoint_path
        or "p7_proj_moge_robustpm" in checkpoint_path
        or "p7_proj_denseh" in checkpoint_path
        or "p7_proj_headonly" in checkpoint_path
        or "p7_proj_robust" in checkpoint_path
        or "p7_proj_tokenres" in checkpoint_path
        or "p7_proj_views" in checkpoint_path
        or "overlappm" in checkpoint_path
        or checkpoint_has_projection_aux_head(args)
    )


def is_p7_p5e_private_viewtype_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = str(args.checkpoint_path).lower()
    return (
        "p7_allcities_p5e_private_viewtype_projection_aux" in checkpoint_path
        or "p7_vggt_p5e_private_viewtype_projection_aux" in checkpoint_path
    )


def is_p7_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    return (
        is_p7_split_projection_aux_checkpoint(args)
        or is_p7_remote_head_projection_aux_checkpoint(args)
        or is_p7_p5b_shared_norm_projection_aux_checkpoint(args)
        or is_p7_p5e_private_viewtype_projection_aux_checkpoint(args)
    )


def is_p6b_shared_head_checkpoint(args: argparse.Namespace) -> bool:
    if not is_p6b_checkpoint(args):
        return False
    return "shared_head" in str(args.checkpoint_path).lower()


def is_p6b_viewtype_checkpoint(args: argparse.Namespace) -> bool:
    if not is_p6b_checkpoint(args):
        return False
    return "viewtype" in str(args.checkpoint_path).lower()


def use_p5f_lite_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p5f_lite_export or is_p5f_lite_checkpoint(args)
    )


def use_p6a_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p6a_export or is_p6a_checkpoint(args)
    )


def use_p6b_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p6b_export or is_p6b_checkpoint(args)
    )


def use_p7_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p7_projection_aux_export or is_p7_split_projection_aux_checkpoint(args)
    )


def use_p7_remote_head_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p7_remote_head_projection_aux_export
        or is_p7_remote_head_projection_aux_checkpoint(args)
    )


def use_p7_p5b_shared_norm_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and (
        args.vggt_p7_p5b_shared_norm_projection_aux_export
        or is_p7_p5b_shared_norm_projection_aux_checkpoint(args)
    )


def use_p7_p5e_private_viewtype_projection_aux_export(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and is_p7_p5e_private_viewtype_projection_aux_checkpoint(args)


def checkpoint_has_remote_private_point_head(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint for remote_point_head: {exc}")
        return False
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_head = any(str(key).startswith("remote_point_head.") for key in state_dict.keys())
    del ckpt, state_dict
    return has_head


def checkpoint_has_projection_aux_head(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint for remote_projection_aux: {exc}")
        return False
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_head = any(
        str(key).startswith(
            (
                "remote_projection_aux_token_",
                "remote_projection_aux_image_stem.",
                "remote_projection_aux_head.",
                "remote_projection_aux_height_head.",
                "remote_projection_aux_offset_head.",
                "remote_projection_aux_dpt_global_head.",
            )
        )
        for key in state_dict.keys()
    )
    del ckpt, state_dict
    return has_head


def checkpoint_has_key_prefix(args: argparse.Namespace, *prefixes: str) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint keys: {exc}")
        return False
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_key = any(str(key).startswith(prefixes) for key in state_dict.keys())
    del ckpt, state_dict
    return has_key


def use_any_p7_projection_aux_export(args: argparse.Namespace) -> bool:
    return (
        use_p7_projection_aux_export(args)
        or use_p7_remote_head_projection_aux_export(args)
        or use_p7_p5b_shared_norm_projection_aux_export(args)
        or use_p7_p5e_private_viewtype_projection_aux_export(args)
    )


def resolve_vggt_late_fusion_type(args: argparse.Namespace) -> str:
    checkpoint_path = str(args.checkpoint_path or "").lower()
    if checkpoint_has_key_prefix(args, "remote_to_aerial_late_film."):
        return "film"
    if checkpoint_has_key_prefix(
        args,
        "remote_to_aerial_late_cross_attention.",
        "remote_to_aerial_late_query_norm.",
        "remote_to_aerial_late_key_value_norm.",
    ):
        return "cross_attention"
    if use_p7_projection_aux_export(args):
        if "no_fusion" in checkpoint_path:
            return "none"
        if "film" in checkpoint_path:
            return "film"
        if "crossattn" in checkpoint_path or "cross_attention" in checkpoint_path:
            return "cross_attention"
    return args.vggt_late_fusion_type


def resolve_vggt_projection_aux_source(args: argparse.Namespace) -> str:
    if args.vggt_projection_aux_source != "auto":
        return args.vggt_projection_aux_source
    checkpoint_path = str(args.checkpoint_path or "").lower()
    if (
        "dptinit" in checkpoint_path
        or "dpt_init" in checkpoint_path
        or checkpoint_has_key_prefix(
            args,
            "remote_projection_aux_height_head.",
            "remote_projection_aux_offset_head.",
            "remote_projection_aux_dpt_global_head.",
        )
    ):
        return "dpt_init"
    if (
        "parallel_token_aux" in checkpoint_path
        or "parallel_tokens_aux" in checkpoint_path
        or "private_tokens" in checkpoint_path
        or "p7_proj_moge_pmheight" in checkpoint_path
        or "p7_proj_moge_robustpm" in checkpoint_path
        or "p7_proj_denseh" in checkpoint_path
        or "p7_proj_headonly" in checkpoint_path
        or "p7_proj_robust" in checkpoint_path
        or "p7_proj_tokenres" in checkpoint_path
        or "p7_proj_views" in checkpoint_path
        or "overlappm" in checkpoint_path
        or checkpoint_has_token_projection_aux_head(args)
    ):
        return "tokens"
    return "pointmap"


def checkpoint_has_token_projection_aux_head(args: argparse.Namespace) -> bool:
    if args.model != "vggt" or not args.checkpoint_path:
        return False
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint for token projection aux: {exc}")
        return False
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    has_head = any(str(key).startswith("remote_projection_aux_token_") for key in state_dict.keys())
    del ckpt, state_dict
    return has_head


def use_vggt_projection_aux_token_residual(args: argparse.Namespace) -> bool:
    checkpoint_path_lower = str(args.checkpoint_path or "").lower()
    return args.model == "vggt" and (
        args.vggt_projection_aux_token_residual
        or "p7_proj_tokenres" in checkpoint_path_lower
        or "tokenres" in checkpoint_path_lower
        or "token_residual" in checkpoint_path_lower
        or checkpoint_has_key_prefix(args, "remote_projection_aux_token_residual")
    )


def use_vggt_pre_aggregator_view_type_bias(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and checkpoint_has_key_prefix(
        args, "pre_aggregator_view_type_embedding."
    )


def use_vggt_remote_to_aerial_gated_residual(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and checkpoint_has_key_prefix(
        args, "remote_to_aerial_gate", "remote_to_aerial_residual."
    )


def use_vggt_split_remote_aggregator(args: argparse.Namespace) -> bool:
    return args.model == "vggt" and resolve_vggt_late_fusion_type(args) != "none"


def use_vggt_remote_private_point_head(args: argparse.Namespace) -> bool:
    if args.model != "vggt":
        return False
    if args.vggt_use_remote_private_point_head:
        return True
    if checkpoint_has_remote_private_point_head(args):
        print("Auto-detected remote_point_head in checkpoint; enabling private remote point head.")
        return True
    if (
        use_p5f_lite_export(args)
        or use_p6a_export(args)
        or use_p7_projection_aux_export(args)
        or use_p7_remote_head_projection_aux_export(args)
        or use_p7_p5b_shared_norm_projection_aux_export(args)
        or use_p7_p5e_private_viewtype_projection_aux_export(args)
    ):
        return True
    if use_p6b_export(args):
        return not is_p6b_shared_head_checkpoint(args)
    return False


def is_pi3_projection_aux_checkpoint(args: argparse.Namespace) -> bool:
    if args.model != "pi3_modality_embedding_remote_head" or not args.checkpoint_path:
        return False
    return "p7_pi3_remote_head_projection_aux" in str(args.checkpoint_path).lower()


def resolve_pi3_projection_aux_output_scales(checkpoint_path) -> tuple[float, float]:
    checkpoint_path_lower = str(checkpoint_path or "").lower()
    rel_match = re.search(r"relscale([0-9]+(?:p[0-9]+)?)", checkpoint_path_lower)
    offset_match = re.search(r"offsetscale([0-9]+(?:p[0-9]+)?)", checkpoint_path_lower)
    rel_scale = float(rel_match.group(1).replace("p", ".")) if rel_match else 1.0
    offset_scale = float(offset_match.group(1).replace("p", ".")) if offset_match else 1.0
    return rel_scale, offset_scale


def resolve_vggt_output_heads(args: argparse.Namespace):
    if args.model != "vggt":
        return None, None

    ordinary_head = args.vggt_ordinary_output_head
    remote_head = args.vggt_remote_output_head

    if args.vggt_export_mode == "mixed" or use_p5f_lite_export(args) or use_p6a_export(args) or use_p6b_export(args) or use_any_p7_projection_aux_export(args):
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "point"
    elif args.vggt_export_mode == "depth_all":
        ordinary_head = ordinary_head or "depth"
        remote_head = remote_head or "depth"
    elif args.vggt_export_mode == "point_all":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "point"
    elif args.vggt_export_mode == "ordinary_point_remote_depth":
        ordinary_head = ordinary_head or "point"
        remote_head = remote_head or "depth"

    return ordinary_head, remote_head


def resolve_config_overrides(args: argparse.Namespace):
    if args.config_overrides is not None:
        overrides = list(args.config_overrides)
    else:
        overrides = list(DEFAULT_CONFIG_OVERRIDES[args.model])

    if is_raw_vggt_checkpoint(args):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=true",
                f"model.model_config.custom_ckpt_path={args.checkpoint_path}",
            ]
        )

    if is_raw_vggt_omega_checkpoint(args):
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=true",
                f"model.model_config.custom_ckpt_path={args.checkpoint_path}",
            ]
        )

    use_vggt_joint_remote_export = (
        args.vggt_joint_remote_export or use_p5f_lite_export(args) or use_p6a_export(args) or use_p6b_export(args) or use_any_p7_projection_aux_export(args)
    )
    if use_vggt_joint_remote_export:
        if args.model != "vggt":
            raise ValueError("VGGT joint remote export presets are only supported with --model vggt")
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.load_custom_ckpt=false",
                "model.model_config.use_point_head_for_remote=true",
            ]
        )

    if use_p5f_lite_export(args):
        overrides.extend(
            [
                "model.model_config.use_pre_aggregator_view_type_bias=true",
                "model.model_config.use_remote_to_aerial_gated_residual=true",
                "model.model_config.remote_to_aerial_residual_hidden_scale=0.25",
                "model.model_config.remote_to_aerial_gate_init=0.0",
            ]
        )

    if use_p6a_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=true",
                "model.model_config.use_split_remote_aggregator=true",
                f"model.model_config.remote_to_aerial_late_fusion_type={args.vggt_late_fusion_type}",
                "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                f"model.model_config.remote_to_aerial_late_fusion_gate_init={args.vggt_late_gate_init}",
                f"model.model_config.remote_to_aerial_cross_attention_heads={args.vggt_cross_attention_heads}",
                f"model.model_config.remote_to_aerial_max_remote_tokens={args.vggt_max_remote_tokens}",
                "model.model_config.protect_ordinary_heads_from_remote=true",
            ]
        )

    if use_p7_projection_aux_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=true",
                "model.model_config.use_split_remote_aggregator=true",
                f"model.model_config.remote_to_aerial_late_fusion_type={resolve_vggt_late_fusion_type(args)}",
                "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                f"model.model_config.remote_to_aerial_late_fusion_gate_init={args.vggt_late_gate_init}",
                f"model.model_config.remote_to_aerial_cross_attention_heads={args.vggt_cross_attention_heads}",
                f"model.model_config.remote_to_aerial_max_remote_tokens={args.vggt_max_remote_tokens}",
                "model.model_config.protect_ordinary_heads_from_remote=true",
                "model.model_config.use_remote_projection_aux_head=true",
                f"model.model_config.remote_projection_aux_hidden_dim={args.vggt_projection_aux_hidden_dim}",
                f"model.model_config.remote_projection_aux_source={resolve_vggt_projection_aux_source(args)}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                f"model.model_config.remote_projection_aux_use_rgb={str(args.vggt_projection_aux_use_rgb).lower()}",
                f"model.model_config.remote_projection_aux_use_coord={str(args.vggt_projection_aux_use_coord).lower()}",
                f"model.model_config.remote_projection_aux_positive_slope={str(args.vggt_projection_aux_positive_slope).lower()}",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                f"model.model_config.remote_projection_aux_num_blocks={args.vggt_projection_aux_num_blocks}",
                f"model.model_config.remote_projection_aux_image_stem_dim={args.vggt_projection_aux_image_stem_dim}",
            ]
        )

    if use_p7_remote_head_projection_aux_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=false",
                "model.model_config.use_pre_aggregator_view_type_bias=false",
                "model.model_config.use_remote_to_aerial_gated_residual=false",
                "model.model_config.use_split_remote_aggregator=false",
                "model.model_config.output_point_head_for_consistency=false",
                "model.model_config.use_remote_projection_aux_head=true",
                f"model.model_config.remote_projection_aux_hidden_dim={args.vggt_projection_aux_hidden_dim}",
                f"model.model_config.remote_projection_aux_source={resolve_vggt_projection_aux_source(args)}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                f"model.model_config.remote_projection_aux_use_rgb={str(args.vggt_projection_aux_use_rgb).lower()}",
                f"model.model_config.remote_projection_aux_use_coord={str(args.vggt_projection_aux_use_coord).lower()}",
                f"model.model_config.remote_projection_aux_positive_slope={str(args.vggt_projection_aux_positive_slope).lower()}",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                f"model.model_config.remote_projection_aux_num_blocks={args.vggt_projection_aux_num_blocks}",
                f"model.model_config.remote_projection_aux_image_stem_dim={args.vggt_projection_aux_image_stem_dim}",
            ]
        )

    if use_p7_p5b_shared_norm_projection_aux_export(args):
        use_pre_aggregator_bias = use_vggt_pre_aggregator_view_type_bias(args)
        use_remote_gated_residual = use_vggt_remote_to_aerial_gated_residual(args)
        use_split_remote_aggregator = use_vggt_split_remote_aggregator(args)
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=false",
                f"model.model_config.use_pre_aggregator_view_type_bias={str(use_pre_aggregator_bias).lower()}",
                f"model.model_config.use_remote_to_aerial_gated_residual={str(use_remote_gated_residual).lower()}",
                "model.model_config.remote_to_aerial_residual_hidden_scale=0.25",
                "model.model_config.remote_to_aerial_gate_init=0.0",
                f"model.model_config.use_split_remote_aggregator={str(use_split_remote_aggregator).lower()}",
                f"model.model_config.remote_to_aerial_late_fusion_type={resolve_vggt_late_fusion_type(args)}",
                "model.model_config.remote_to_aerial_late_fusion_hidden_scale=0.25",
                f"model.model_config.remote_to_aerial_late_fusion_gate_init={args.vggt_late_gate_init}",
                f"model.model_config.remote_to_aerial_cross_attention_heads={args.vggt_cross_attention_heads}",
                f"model.model_config.remote_to_aerial_max_remote_tokens={args.vggt_max_remote_tokens}",
                "model.model_config.protect_ordinary_heads_from_remote=false",
                "model.model_config.use_remote_projection_aux_head=true",
                "model.model_config.remote_projection_aux_hidden_dim=96",
                f"model.model_config.remote_projection_aux_source={resolve_vggt_projection_aux_source(args)}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                "model.model_config.remote_projection_aux_use_rgb=true",
                "model.model_config.remote_projection_aux_use_coord=true",
                "model.model_config.remote_projection_aux_positive_slope=true",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                "model.model_config.remote_projection_aux_num_blocks=6",
                "model.model_config.remote_projection_aux_image_stem_dim=32",
            ]
        )
        if use_vggt_projection_aux_token_residual(args):
            overrides.extend(
                [
                    "model.model_config.use_remote_projection_aux_token_residual=true",
                    f"model.model_config.remote_projection_aux_token_residual_hidden_scale={args.vggt_projection_aux_token_residual_hidden_scale}",
                    f"model.model_config.remote_projection_aux_token_residual_gate_init={args.vggt_projection_aux_token_residual_gate_init}",
                ]
            )

    if use_p7_p5e_private_viewtype_projection_aux_export(args):
        overrides.extend(
            [
                "model.model_config.use_view_type_bias=true",
                "model.model_config.use_pre_aggregator_view_type_bias=false",
                "model.model_config.use_remote_to_aerial_gated_residual=false",
                "model.model_config.use_split_remote_aggregator=false",
                "model.model_config.remote_to_aerial_late_fusion_type=none",
                "model.model_config.protect_ordinary_heads_from_remote=false",
                "model.model_config.output_point_head_for_consistency=false",
                "model.model_config.use_remote_projection_aux_head=true",
                "model.model_config.remote_projection_aux_hidden_dim=96",
                f"model.model_config.remote_projection_aux_source={resolve_vggt_projection_aux_source(args)}",
                f"model.model_config.remote_projection_aux_detach_pointmap={str(args.vggt_projection_aux_detach_pointmap).lower()}",
                "model.model_config.remote_projection_aux_use_rgb=true",
                "model.model_config.remote_projection_aux_use_coord=true",
                "model.model_config.remote_projection_aux_positive_slope=true",
                f"model.model_config.remote_projection_aux_slope_init={args.vggt_projection_aux_slope_init}",
                "model.model_config.remote_projection_aux_num_blocks=6",
                "model.model_config.remote_projection_aux_image_stem_dim=32",
            ]
        )

    if is_pi3_projection_aux_checkpoint(args):
        rel_scale, offset_scale = resolve_pi3_projection_aux_output_scales(args.checkpoint_path)
        overrides.extend(
            [
                "model.model_config.load_pretrained_weights=false",
                "model.model_config.use_remote_projection_aux_head=true",
                "model.model_config.remote_projection_aux_hidden_dim=96",
                "model.model_config.remote_projection_aux_use_rgb=true",
                "model.model_config.remote_projection_aux_use_coord=true",
                "model.model_config.remote_projection_aux_image_stem_dim=32",
                "model.model_config.remote_projection_aux_positive_slope=true",
                "model.model_config.remote_projection_aux_slope_init=0.1",
                "model.model_config.remote_projection_aux_num_blocks=6",
                f"model.model_config.remote_projection_aux_rel_height_output_scale={rel_scale}",
                f"model.model_config.remote_projection_aux_offset_output_scale={offset_scale}",
            ]
        )

    if use_p6b_export(args) and is_p6b_viewtype_checkpoint(args):
        overrides.append("model.model_config.use_view_type_bias=true")

    ordinary_head, remote_head = resolve_vggt_output_heads(args)
    if ordinary_head is not None:
        overrides.append(f"model.model_config.ordinary_output_head={ordinary_head}")
    if remote_head is not None:
        overrides.append(f"model.model_config.remote_output_head={remote_head}")
    if use_vggt_remote_private_point_head(args):
        overrides.append("model.model_config.use_remote_private_point_head=true")
        if not (
            use_p7_remote_head_projection_aux_export(args)
            or use_p7_p5e_private_viewtype_projection_aux_export(args)
        ):
            overrides.append("model.model_config.output_point_head_for_consistency=true")

    return overrides


def resolve_effective_model_name(args: argparse.Namespace) -> str:
    if args.model != "pi3" or not args.checkpoint_path:
        return args.model

    checkpoint_path_lower = str(args.checkpoint_path).lower()
    if (
        "pi3_modality_embedding_remote_head" in checkpoint_path_lower
        or "p7_pi3_remote_head_projection_aux" in checkpoint_path_lower
    ):
        print(
            "Auto-detected Pi3 variant from checkpoint path: "
            "pi3_modality_embedding_remote_head"
        )
        return "pi3_modality_embedding_remote_head"
    if (
        "pi3_modality_embedding" in checkpoint_path_lower
        or "p3_pi3_freeze_shared" in checkpoint_path_lower
    ):
        print(
            "Auto-detected Pi3 variant from checkpoint path: "
            "pi3_modality_embedding"
        )
        return "pi3_modality_embedding"
    return args.model


def maybe_enable_clash_proxy(enable_proxy: bool):
    if not enable_proxy:
        return
    clash_path = Path("/etc/profile.d/clash.sh")
    if not clash_path.exists():
        print("Clash helper not found at /etc/profile.d/clash.sh; skipping proxy setup")
        return
    os.environ.update(CLASH_ENV)
    print("Enabled Clash proxy environment for HuggingFace downloads")


def maybe_prepare_da3_pythonpath(model_name: str):
    if model_name != "da3":
        return
    da3_src = Path("/root/autodl-tmp/Models/Depth-Anything-3/src")
    if not da3_src.exists():
        raise FileNotFoundError(
            "DA3 requires /root/autodl-tmp/Models/Depth-Anything-3/src to exist"
        )
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))
        print(f"Added DA3 dependency path: {da3_src}")


def build_local_config(
    args: argparse.Namespace,
    config_overrides,
    effective_model_name: str,
) -> dict:
    local_config = {
        "path": args.config_path,
        "checkpoint_path": args.checkpoint_path,
        "config_overrides": config_overrides,
        "strict": args.strict,
        "model_str": args.model_str or effective_model_name,
    }
    if args.config_json_path is not None:
        local_config["config_json_path"] = args.config_json_path
    return local_config


def initialize_model(
    args: argparse.Namespace,
    device: str,
    config_overrides,
    effective_model_name: str,
):
    maybe_enable_clash_proxy(args.enable_clash_proxy)
    maybe_prepare_da3_pythonpath(effective_model_name)

    if args.checkpoint_path:
        if is_raw_vggt_checkpoint(args):
            print(
                "Detected raw VGGT checkpoint; loading it through "
                "model.model_config.custom_ckpt_path before the compatibility "
                "local-checkpoint load."
            )
        local_config = build_local_config(args, config_overrides, effective_model_name)
        print(f"Initializing model from local config: {local_config}")
        model = initialize_mapanything_local(local_config, device)
        print("Successfully loaded local checkpoint")
        return model

    if effective_model_name == "mapanything":
        hf_model_name = args.hf_model_name or DEFAULT_MAPANYTHING_HF_MODEL
        high_level_config = {
            "path": args.config_path,
            "hf_model_name": hf_model_name,
            "model_str": "mapanything",
            "config_overrides": config_overrides,
            "checkpoint_name": "model.safetensors",
            "config_name": "config.json",
        }
        print(f"Initializing model from HuggingFace defaults: {high_level_config}")
        model = initialize_mapanything_model(high_level_config, device)
        print("Successfully loaded HuggingFace weights")
        return model

    from mapanything.models import init_model_from_config

    print(
        f"Initializing model '{effective_model_name}' from default wrapper weights"
    )
    model = init_model_from_config(
        effective_model_name, device=device, machine="aws"
    ).eval()
    print("Successfully loaded default wrapper weights")
    return model


def convert_views_to_identity_if_needed(views, model_name: str):
    if model_name not in IDENTITY_MODELS:
        return views

    converted_views = []
    for view in views:
        norm_type = view["data_norm_type"][0]
        if norm_type == "identity":
            converted_views.append(view)
            continue

        if norm_type not in IMAGE_NORMALIZATION_DICT:
            raise ValueError(f"Unsupported norm_type for identity conversion: {norm_type}")

        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        mean = torch.as_tensor(
            img_norm.mean,
            dtype=view["img"].dtype,
            device=view["img"].device,
        ).view(1, -1, 1, 1)
        std = torch.as_tensor(
            img_norm.std,
            dtype=view["img"].dtype,
            device=view["img"].device,
        ).view(1, -1, 1, 1)

        converted_view = dict(view)
        converted_view["img"] = (view["img"] * std + mean).clamp(0, 1)
        converted_view["data_norm_type"] = ["identity"]
        converted_views.append(converted_view)

    return converted_views


def move_views_to_device(views, device: torch.device):
    moved_views = []
    for view in views:
        moved_view = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                moved_view[key] = value.to(device)
            else:
                moved_view[key] = value
        moved_views.append(moved_view)
    return moved_views


def list_loaded_image_names(image_folder: str, stride: int):
    supported_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_extensions += [".heic", ".heif"]
    supported_extensions = tuple(supported_extensions)

    names = []
    for idx, name in enumerate(sorted(os.listdir(image_folder))):
        if idx % stride != 0:
            continue
        if not name.lower().endswith(supported_extensions):
            continue
        names.append(name)
    return names


def annotate_view_source_names(views, image_folder: str, stride: int):
    source_names = list_loaded_image_names(image_folder, stride)
    if len(source_names) != len(views):
        print(
            "Warning: loaded image-name count does not match view count; "
            "remote name selection may be unreliable."
        )
    for idx, view in enumerate(views):
        if idx < len(source_names):
            view["source_name"] = source_names[idx]
    return views


def maybe_assign_remote_instances(views, args: argparse.Namespace):
    use_joint_remote_logic = (
        args.force_remote_instance
        or args.vggt_joint_remote_export
        or args.vggt_p5f_lite_export
        or args.vggt_p6a_export
        or args.vggt_p6b_export
        or args.vggt_p7_projection_aux_export
        or args.vggt_p7_remote_head_projection_aux_export
        or args.vggt_p7_p5b_shared_norm_projection_aux_export
        or is_p5f_lite_checkpoint(args)
        or is_p6a_checkpoint(args)
        or is_p6b_checkpoint(args)
        or is_p7_projection_aux_checkpoint(args)
        or args.model == "mapanything_rs_joint"
        or bool(args.remote_view_indices)
        or bool(args.remote_view_names)
    )
    if not use_joint_remote_logic:
        return views

    remote_indices = set(args.remote_view_indices or [])
    remote_names = {name for name in (args.remote_view_names or [])}
    explicit_remote_selection = bool(remote_indices) or bool(remote_names)

    if use_joint_remote_logic and not explicit_remote_selection and not args.force_remote_instance:
        inferred_names = infer_remote_view_names(views)
        if inferred_names:
            remote_names.update(inferred_names)
            print(
                "Auto-detected remote view names: "
                + ", ".join(sorted(inferred_names))
            )

    if args.force_remote_instance:
        remote_indices = set(range(len(views)))

    forced_views = []
    remote_assignments = []
    for idx, view in enumerate(views):
        forced_view = dict(view)
        source_name = forced_view.get("source_name")
        is_remote = idx in remote_indices or (
            source_name is not None and source_name in remote_names
        )
        if is_remote:
            forced_view["instance"] = [REMOTE_INSTANCE_VALUE]
            remote_assignments.append((idx, source_name or f"view_{idx}"))
        forced_views.append(forced_view)

    if remote_assignments:
        print("Assigned remote views:")
        for idx, source_name in remote_assignments:
            print(f"  - idx={idx} name={source_name}")
    else:
        print(
            "No views were marked as remote; export will use ordinary view logic. "
            "Pass --remote_view_indices/--remote_view_names for mixed RS export, or "
            "--force_remote_instance for remote-only debugging."
        )

    return forced_views


def infer_remote_view_names(views):
    inferred = set()
    exact_names = {
        "image.png",
        "zimage.png",
        "sate.png",
        "sate1.png",
        "satellite.png",
        "remote.png",
    }
    remote_tokens = (
        "satellite",
        "sate",
        "remote",
        "google_satellite",
        "bing_satellite",
        "esri_satellite",
        "yandex_satellite",
    )

    for view in views:
        source_name = view.get("source_name")
        if not source_name:
            continue
        lower_name = str(source_name).lower()
        if lower_name in exact_names or any(token in lower_name for token in remote_tokens):
            inferred.add(source_name)

    return inferred


def is_remote_view(view) -> bool:
    instance = view.get("instance")
    if isinstance(instance, (list, tuple)) and len(instance) > 0:
        instance = instance[0]
    return instance == REMOTE_INSTANCE_VALUE


def get_remote_view_indices(views):
    return [idx for idx, view in enumerate(views) if is_remote_view(view)]


def get_output_head_name(pred):
    return pred.get("vggt_output_head", pred.get("vggt_omega_output_head", "default"))


def should_export_view(view, view_filter: str) -> bool:
    if view_filter == "all":
        return True
    remote = is_remote_view(view)
    if view_filter == "remote":
        return remote
    if view_filter == "ordinary":
        return not remote
    raise ValueError(f"Unsupported export view filter: {view_filter}")


def should_skip_remote_points(args: argparse.Namespace) -> bool:
    return False


def copy_views(views):
    return [dict(view) for view in views]


def make_blank_remote_control_views(views, remote_indices, blank_value: float):
    control_views = copy_views(views)
    for idx in remote_indices:
        control_views[idx] = dict(control_views[idx])
        control_views[idx]["img"] = torch.full_like(
            control_views[idx]["img"], fill_value=float(blank_value)
        )
        control_views[idx]["source_name"] = f"blank::{control_views[idx].get('source_name', idx)}"
    return control_views


def load_shuffled_remote_view_like(args, remote_view, model_name: str):
    if args.shuffled_remote_image_path is None:
        raise ValueError(
            "--shuffled_remote_image_path is required when exporting shuffled remote controls"
        )
    if not Path(args.shuffled_remote_image_path).exists():
        raise FileNotFoundError(
            f"shuffled remote image not found: {args.shuffled_remote_image_path}"
        )

    _, _, height, width = remote_view["img"].shape
    loaded = load_images(
        [args.shuffled_remote_image_path],
        resize_mode="fixed_size",
        size=(width, height),
        resolution_set=args.resolution_set,
    )
    loaded = convert_views_to_identity_if_needed(loaded, model_name)
    replacement = loaded[0]
    replacement["source_name"] = Path(args.shuffled_remote_image_path).name
    return replacement


def make_shuffled_remote_control_views(views, remote_indices, args, model_name: str):
    control_views = copy_views(views)
    for idx in remote_indices:
        replacement = load_shuffled_remote_view_like(args, control_views[idx], model_name)
        control_view = dict(control_views[idx])
        control_view["img"] = replacement["img"]
        control_view["true_shape"] = replacement.get("true_shape", control_view.get("true_shape"))
        control_view["source_name"] = f"shuffled::{replacement.get('source_name', idx)}"
        control_views[idx] = control_view
    return control_views


def build_remote_control_view_variants(views, args: argparse.Namespace, model_name: str):
    modes = args.export_remote_control_modes
    if not modes:
        return [(None, views)]

    remote_indices = get_remote_view_indices(views)
    if not remote_indices:
        raise ValueError(
            "--export_remote_control_modes requires at least one marked remote view. "
            "Use --remote_view_names, --remote_view_indices, or --force_remote_instance."
        )

    variants = []
    for mode in modes:
        if mode == "same":
            variants.append((mode, views))
        elif mode == "blank":
            variants.append(
                (mode, make_blank_remote_control_views(views, remote_indices, args.blank_remote_value))
            )
        elif mode == "shuffled":
            variants.append(
                (mode, make_shuffled_remote_control_views(views, remote_indices, args, model_name))
            )
        else:
            raise ValueError(f"Unsupported remote control mode: {mode}")
    return variants


def strip_export_only_view_keys(views):
    stripped_views = []
    for view in views:
        stripped_view = dict(view)
        stripped_view.pop("source_name", None)
        stripped_views.append(stripped_view)
    return stripped_views


def run_model_inference(model, views, args: argparse.Namespace):
    views_for_model = strip_export_only_view_keys(views)
    if hasattr(model, "infer"):
        return model.infer(
            views_for_model,
            memory_efficient_inference=args.memory_efficient_inference,
            minibatch_size=args.minibatch_size,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=args.apply_mask,
            mask_edges=args.mask_edges,
            apply_confidence_mask=args.apply_confidence_mask,
            confidence_percentile=args.confidence_percentile,
        )

    model_device = next(model.parameters()).device
    return model(move_views_to_device(views_for_model, model_device))


def get_view_colors(pred, view):
    if "img_no_norm" in pred:
        image_np = pred["img_no_norm"][0].cpu().numpy()
    else:
        image_np = view["img"][0].permute(1, 2, 0).cpu().numpy()
    return np.clip(image_np * 255.0, 0, 255).astype(np.uint8)


def collect_world_space_point_cloud(
    outputs,
    views,
    view_filter="all",
    apply_confidence_mask=False,
    confidence_percentile=50.0,
    skip_remote_points=False,
):
    all_points = []
    all_colors = []
    per_view_stats = []

    for view_idx, pred in enumerate(outputs):
        view = views[view_idx] if view_idx < len(views) else {}
        if not should_export_view(view, view_filter):
            per_view_stats.append(
                {
                    "view_idx": view_idx,
                    "points": 0,
                    "head": get_output_head_name(pred),
                    "skipped": f"view_filter={view_filter}",
                }
            )
            continue

        if skip_remote_points and view_idx < len(views) and is_remote_view(views[view_idx]):
            per_view_stats.append(
                {
                    "view_idx": view_idx,
                    "points": 0,
                    "head": get_output_head_name(pred),
                    "skipped": "remote_split_frame",
                }
            )
            continue

        if "pts3d" in pred:
            pts3d_np = pred["pts3d"][0].cpu().numpy()
            export_mask = np.isfinite(pts3d_np).all(axis=-1)
            if apply_confidence_mask and "conf" in pred:
                conf_np = pred["conf"][0].cpu().numpy()
                if conf_np.ndim == 3 and conf_np.shape[-1] == 1:
                    conf_np = conf_np.squeeze(-1)
                valid_conf = conf_np[export_mask]
                if valid_conf.size > 0:
                    conf_threshold = np.percentile(valid_conf, confidence_percentile)
                    export_mask &= conf_np >= conf_threshold
        else:
            depthmap_torch = pred["depth_z"][0].squeeze(-1)
            intrinsics_torch = pred["intrinsics"][0]
            camera_pose_torch = pred["camera_poses"][0]

            pts3d_world, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsics_torch, camera_pose_torch
            )
            pts3d_np = pts3d_world.cpu().numpy()

            valid_mask_np = valid_mask.cpu().numpy()
            if "mask" in pred:
                export_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
                export_mask &= valid_mask_np
            else:
                export_mask = valid_mask_np

        colors_np = get_view_colors(pred, views[view_idx])
        selected_points = pts3d_np[export_mask]
        selected_colors = colors_np[export_mask]

        per_view_stats.append(
            {
                "view_idx": view_idx,
                "points": int(selected_points.shape[0]),
                "head": get_output_head_name(pred),
            }
        )

        if selected_points.shape[0] == 0:
            continue

        all_points.append(selected_points)
        all_colors.append(selected_colors)

    if not all_points:
        raise RuntimeError("No valid points remained after masking; cannot export PLY.")

    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0),
        per_view_stats,
    )


def resolve_output_path(output_path_str: str) -> Path:
    output_path = Path(output_path_str)
    if output_path.suffix.lower() == ".ply":
        return output_path
    if output_path.exists() and output_path.is_dir():
        return output_path / "mapanything_pointcloud.ply"
    if output_path.suffix == "":
        return output_path / "mapanything_pointcloud.ply"
    return output_path.with_suffix(".ply")


def resolve_variant_output_path(output_path_str: str, variant_name: str | None) -> Path:
    output_path = resolve_output_path(output_path_str)
    if variant_name is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_{variant_name}{output_path.suffix}")


def resolve_remote_companion_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_remote{output_path.suffix}")


def resolve_aux_reconstruction_output_path(output_path: Path, method: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_aux_{method}_remote{output_path.suffix}")


def resolve_aux_reconstruction_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_aux_reconstruction_summary.json")


def write_point_cloud_ply(
    points,
    colors,
    output_path: Path,
    args: argparse.Namespace,
    description: str,
) -> None:
    print(f"Total {description} points before downsampling: {points.shape[0]}")
    if args.voxel_downsample:
        points, colors = voxel_downsample_point_cloud(
            points,
            colors,
            voxel_fraction=args.voxel_fraction,
            voxel_size=args.voxel_size,
        )
        print(f"Total {description} points after downsampling: {points.shape[0]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=points, colors=colors).export(output_path)
    print(f"Saved {description} point cloud PLY to: {output_path}")


def print_per_view_stats(per_view_stats):
    for stat in per_view_stats:
        skipped = stat.get("skipped")
        if skipped:
            print(
                f"View {stat['view_idx']}: skipped ({skipped}, head={stat['head']})"
            )
        else:
            print(
                f"View {stat['view_idx']}: kept {stat['points']} points "
                f"(head={stat['head']})"
            )


def _tensor_view0_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        if value.ndim >= 1:
            value = value[0]
        return value.numpy()
    value = np.asarray(value)
    if value.ndim >= 1:
        value = value[0]
    return value


def estimate_projection_aux_rel_height_scale(base_pts: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace) -> float:
    if args.projection_aux_rel_height_scale_mode == "fixed":
        return float(args.projection_aux_rel_height_fixed_scale)
    if not valid_mask.any():
        return 1.0
    pts = base_pts[valid_mask]
    if args.projection_aux_rel_height_scale_mode == "pred_z_std":
        return float(max(np.std(pts[:, 2]), 1e-6))
    distances = np.linalg.norm(pts, axis=-1)
    return float(max(np.mean(distances), 1e-6))


def load_projection_aux_gt_rel_height(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if not args.projection_aux_gt_remote_dir:
        raise ValueError(
            "--projection_aux_gt_remote_dir is required when "
            "--projection_aux_rel_height_scale_mode=gt_height_range"
        )
    gt_path = Path(args.projection_aux_gt_remote_dir) / "projection_aux.npz"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt = np.load(gt_path)
    if "rel_height" not in gt or "valid_mask" not in gt:
        raise KeyError(f"{gt_path} must contain rel_height and valid_mask")
    rel_height = gt["rel_height"].astype(np.float32)
    valid_mask = gt["valid_mask"].astype(bool) & np.isfinite(rel_height)
    if not valid_mask.any():
        raise ValueError(f"No valid rel_height values in {gt_path}")
    return rel_height, valid_mask


def compute_gt_pointmap_norm_scale(args: argparse.Namespace) -> float:
    if not args.projection_aux_gt_remote_dir:
        return 1.0
    gt_path = Path(args.projection_aux_gt_remote_dir) / "pixel_to_point_map.npz"
    if not gt_path.exists():
        return 1.0
    gt = np.load(gt_path)
    xyz_key = "xyz" if "xyz" in gt else "pts3d" if "pts3d" in gt else None
    if xyz_key is None:
        return 1.0
    xyz = gt[xyz_key].astype(np.float32)
    valid = np.isfinite(xyz).all(axis=-1) & (np.linalg.norm(xyz, axis=-1) > 1e-6)
    if int(valid.sum()) < 16:
        return 1.0
    norm_output = normalize_multiple_pointclouds(
        [torch.from_numpy(xyz).unsqueeze(0)],
        [torch.from_numpy(valid).unsqueeze(0)],
        "avg_dis",
        ret_factor=True,
    )
    scale = float(norm_output[-1].detach().cpu().reshape(-1)[0])
    return scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def load_projection_aux_gt_global(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if not args.projection_aux_gt_remote_dir:
        raise ValueError(
            "--projection_aux_gt_remote_dir is required when using GT projection "
            "global direction or slope"
        )
    gt_path = Path(args.projection_aux_gt_remote_dir) / "projection_aux.npz"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt = np.load(gt_path)
    if "global_dir_xy" not in gt or "global_slope" not in gt:
        raise KeyError(f"{gt_path} must contain global_dir_xy and global_slope")
    global_dir = np.asarray(gt["global_dir_xy"], dtype=np.float32).reshape(-1)[:2]
    global_slope = float(np.asarray(gt["global_slope"], dtype=np.float32).reshape(-1)[0])
    return global_dir, global_slope


def load_projection_aux_gt_projection_base(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if not args.projection_aux_gt_remote_dir:
        raise ValueError("--projection_aux_gt_remote_dir is required when using GT projection base")
    gt_path = Path(args.projection_aux_gt_remote_dir) / "projection_aux.npz"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt = np.load(gt_path)
    required = ["projected_xyz_centered", "projection_center_xy"]
    missing = [key for key in required if key not in gt]
    if missing:
        raise KeyError(f"{gt_path} is missing {missing}")
    projected = gt["projected_xyz_centered"].astype(np.float32)
    center_xy = gt["projection_center_xy"].astype(np.float32).reshape(-1)[:2]
    return projected, center_xy


def load_projection_aux_gt_pointmap(args: argparse.Namespace) -> np.ndarray:
    if not args.projection_aux_gt_remote_dir:
        raise ValueError(
            "--projection_aux_gt_remote_dir is required when "
            "--projection_aux_xyz_align_mode=gt_pointmap_unit_xy_zrange"
        )
    gt_path = Path(args.projection_aux_gt_remote_dir) / "pixel_to_point_map.npz"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt = np.load(gt_path)
    if "xyz" not in gt:
        raise KeyError(f"{gt_path} must contain xyz")
    xyz = gt["xyz"].astype(np.float32)
    valid = np.isfinite(xyz).all(axis=-1)
    if int(valid.sum()) < 16:
        raise ValueError(f"No valid xyz values in {gt_path}")
    return xyz[valid]


def resize_scalar_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape[:2] == shape:
        return array
    y_idx = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(np.int64)
    x_idx = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(np.int64)
    return array[np.ix_(y_idx, x_idx)]


def resize_image_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape[:2] == shape:
        return array
    y_idx = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(np.int64)
    x_idx = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(np.int64)
    return array[np.ix_(y_idx, x_idx)]


def affine_align_scalar(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if int(valid.sum()) < 2:
        return pred.copy(), {"scale": None, "shift": None, "valid_pixels": int(valid.sum())}
    x = pred[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).mean())
    scale = 0.0 if denom < 1e-12 else float(((x - x_mean) * (y - y_mean)).mean() / denom)
    shift = y_mean - scale * x_mean
    return pred.astype(np.float32) * np.float32(scale) + np.float32(shift), {
        "scale": scale,
        "shift": shift,
        "valid_pixels": int(valid.sum()),
    }


def colorize_scalar(values: np.ndarray, mask: np.ndarray, *, symmetric: bool = False) -> Image.Image:
    values = values.astype(np.float32)
    valid = mask.astype(bool) & np.isfinite(values)
    vals = values[valid]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    elif symmetric:
        vmax = max(float(np.percentile(np.abs(vals), 98)), 1e-6)
        vmin = -vmax
    else:
        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
    norm = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    red = np.where(norm < 0.5, norm * 2.0, 1.0)
    green = np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 2.0)
    blue = np.where(norm < 0.5, 1.0, (1.0 - norm) * 2.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.where(valid[..., None], rgb, 0.15)
    return Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))


def make_height_panel(title: str, values: np.ndarray, mask: np.ndarray, *, symmetric: bool = False) -> Image.Image:
    image = colorize_scalar(values, mask, symmetric=symmetric).resize((256, 256), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 284), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), title, fill=(0, 0, 0))
    return canvas


def save_projection_aux_height_comparison_for_view(
    pred,
    view,
    args: argparse.Namespace,
    output_path: Path,
) -> dict | None:
    if not args.projection_aux_gt_remote_dir:
        return None
    if "remote_projection_rel_height_pred" not in pred:
        return None

    rel_pred_norm = _tensor_view0_to_numpy(pred["remote_projection_rel_height_pred"]).astype(np.float32)
    rel_gt, gt_mask = load_projection_aux_gt_rel_height(args)
    rel_gt = resize_scalar_nearest(rel_gt, rel_pred_norm.shape[:2]).astype(np.float32)
    gt_mask = resize_scalar_nearest(gt_mask.astype(np.uint8), rel_pred_norm.shape[:2]).astype(bool)
    mask = gt_mask & np.isfinite(rel_gt) & np.isfinite(rel_pred_norm)
    if int(mask.sum()) < 16:
        return None

    gt_scale = compute_gt_pointmap_norm_scale(args)
    rel_gt_norm = rel_gt / np.float32(gt_scale)
    rel_pred_affine, affine = affine_align_scalar(rel_pred_norm, rel_gt_norm, mask)
    norm_err = np.abs(rel_pred_norm - rel_gt_norm)
    affine_err = np.abs(rel_pred_affine - rel_gt_norm)

    out_dir = output_path.with_name(f"{output_path.stem}_aux_height_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        make_height_panel("GT height norm", rel_gt_norm, mask),
        make_height_panel("Pred height norm", rel_pred_norm, mask),
        make_height_panel("Norm abs err", norm_err, mask),
        make_height_panel("Pred affine", rel_pred_affine, mask),
        make_height_panel("Affine abs err", affine_err, mask),
    ]
    grid = Image.new("RGB", (3 * 256, 2 * 284), "white")
    for idx, panel in enumerate(panels):
        grid.paste(panel, ((idx % 3) * 256, (idx // 3) * 284))
    grid.save(out_dir / "rel_height_gt_pred_grid.png")
    colorize_scalar(rel_gt_norm, mask).save(out_dir / "rel_height_gt_norm.png")
    colorize_scalar(rel_pred_norm, mask).save(out_dir / "rel_height_pred_norm.png")
    colorize_scalar(norm_err, mask).save(out_dir / "rel_height_norm_abs_err.png")
    colorize_scalar(rel_pred_affine, mask).save(out_dir / "rel_height_pred_norm_affine.png")
    colorize_scalar(affine_err, mask).save(out_dir / "rel_height_norm_affine_abs_err.png")
    np.savez_compressed(
        out_dir / "rel_height_arrays.npz",
        rel_height_gt_norm=rel_gt_norm.astype(np.float32),
        rel_height_pred_norm=rel_pred_norm.astype(np.float32),
        rel_height_norm_abs_err=norm_err.astype(np.float32),
        rel_height_pred_norm_affine=rel_pred_affine.astype(np.float32),
        rel_height_norm_affine_abs_err=affine_err.astype(np.float32),
        valid_mask=mask.astype(bool),
    )
    summary = {
        "view_idx_source_name": str(view.get("source_name", "")),
        "gt_remote_dir": str(args.projection_aux_gt_remote_dir),
        "valid_pixels": int(mask.sum()),
        "gt_pointmap_norm_scale": float(gt_scale),
        "rel_height_norm_mae": float(norm_err[mask].mean()),
        "rel_height_norm_affine_mae": float(affine_err[mask].mean()),
        "rel_height_norm_affine": affine,
        "rel_height_gt_norm_mean": float(rel_gt_norm[mask].mean()),
        "rel_height_pred_norm_mean": float(rel_pred_norm[mask].mean()),
        "rel_height_gt_norm_std": float(rel_gt_norm[mask].std()),
        "rel_height_pred_norm_std": float(rel_pred_norm[mask].std()),
        "output_dir": str(out_dir),
    }
    with (out_dir / "rel_height_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Saved projection-aux height comparison to: {out_dir / 'rel_height_gt_pred_grid.png'}")
    return summary


def rel_height_from_gt_height_range(rel_pred: np.ndarray, valid_mask: np.ndarray, args: argparse.Namespace):
    gt_rel, gt_valid = load_projection_aux_gt_rel_height(args)
    gt_rel = resize_scalar_nearest(gt_rel, rel_pred.shape[:2])
    gt_valid = resize_scalar_nearest(gt_valid.astype(np.uint8), rel_pred.shape[:2]).astype(bool)
    common = valid_mask & gt_valid & np.isfinite(rel_pred) & np.isfinite(gt_rel)
    if int(common.sum()) < 16:
        common = valid_mask & np.isfinite(rel_pred)
    pred_vals = rel_pred[common]
    gt_vals = gt_rel[gt_valid & np.isfinite(gt_rel)]
    if pred_vals.size < 16 or gt_vals.size < 16:
        raise ValueError("Not enough valid pixels for gt_height_range rel-height alignment")
    q0 = float(np.clip(args.projection_aux_gt_height_range_low_quantile, 0.0, 1.0))
    q1 = float(np.clip(args.projection_aux_gt_height_range_high_quantile, 0.0, 1.0))
    if q1 <= q0:
        raise ValueError("projection aux GT height high quantile must be greater than low quantile")
    pred_low, pred_high = np.quantile(pred_vals, [q0, q1]).astype(np.float32)
    gt_low, gt_high = np.quantile(gt_vals, [q0, q1]).astype(np.float32)
    pred_range = float(max(pred_high - pred_low, 1e-6))
    scale = float((gt_high - gt_low) / pred_range)
    shift = float(gt_low - scale * float(pred_low))
    rel_height = rel_pred * np.float32(scale) + np.float32(shift)
    summary = {
        "mode": "gt_height_range",
        "gt_remote_dir": str(args.projection_aux_gt_remote_dir),
        "low_quantile": q0,
        "high_quantile": q1,
        "common_pixels": int(common.sum()),
        "pred_low": float(pred_low),
        "pred_high": float(pred_high),
        "gt_low": float(gt_low),
        "gt_high": float(gt_high),
        "scale": scale,
        "shift": shift,
    }
    return rel_height, summary


def align_aux_points_to_gt_unit_xy_zrange(points: np.ndarray, args: argparse.Namespace):
    if args.projection_aux_xyz_align_mode not in {
        "gt_pointmap_unit_xy_zrange",
        "gt_pointmap_unit_xy_zrange_flipz",
    }:
        return points, {"mode": "none"}

    gt_points = load_projection_aux_gt_pointmap(args)
    q0 = float(np.clip(args.projection_aux_xyz_align_low_quantile, 0.0, 1.0))
    q1 = float(np.clip(args.projection_aux_xyz_align_high_quantile, 0.0, 1.0))
    if q1 <= q0:
        raise ValueError("projection aux xyz align high quantile must be greater than low quantile")

    src_low, src_high = np.quantile(points, [q0, q1], axis=0).astype(np.float32)
    gt_low, gt_high = np.quantile(gt_points, [q0, q1], axis=0).astype(np.float32)

    src_xy_scale = float(max(src_high[0] - src_low[0], src_high[1] - src_low[1], 1e-6))
    src_z_range = float(max(src_high[2] - src_low[2], 1e-6))
    gt_xy_scale = float(max(gt_high[0] - gt_low[0], gt_high[1] - gt_low[1], 1e-6))
    gt_z_range = float(max(gt_high[2] - gt_low[2], 1e-6) / gt_xy_scale)

    aligned = points.astype(np.float32, copy=True)
    aligned[:, 0] = (aligned[:, 0] - src_low[0]) / src_xy_scale
    aligned[:, 1] = (aligned[:, 1] - src_low[1]) / src_xy_scale
    aligned[:, 2] = (aligned[:, 2] - src_low[2]) / src_z_range * gt_z_range
    if args.projection_aux_xyz_align_mode == "gt_pointmap_unit_xy_zrange_flipz":
        aligned[:, 2] = gt_z_range - aligned[:, 2]

    summary = {
        "mode": args.projection_aux_xyz_align_mode,
        "gt_remote_dir": str(args.projection_aux_gt_remote_dir),
        "low_quantile": q0,
        "high_quantile": q1,
        "src_low": [float(v) for v in src_low],
        "src_high": [float(v) for v in src_high],
        "gt_low": [float(v) for v in gt_low],
        "gt_high": [float(v) for v in gt_high],
        "src_xy_scale": src_xy_scale,
        "src_z_range": src_z_range,
        "gt_xy_scale": gt_xy_scale,
        "gt_unit_z_range": gt_z_range,
    }
    return aligned, summary


def projection_aux_pixel_grid(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32) if width > 1 else np.zeros(width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32) if height > 1 else np.zeros(height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def reconstruct_projection_aux_points_for_view(pred, view, args: argparse.Namespace):
    required = [
        "remote_projection_rel_height_pred",
        "remote_projection_offset_xy_pred",
        "remote_projection_global_dir_xy_pred",
        "remote_projection_global_slope_pred",
    ]
    missing = [key for key in required if key not in pred]
    if missing:
        return None, {"missing": missing}
    if "pts3d" not in pred:
        return None, {"missing": ["pts3d"]}

    pred_base_pts = _tensor_view0_to_numpy(pred["pts3d"]).astype(np.float32)
    base_pts = pred_base_pts
    rel_pred = _tensor_view0_to_numpy(pred["remote_projection_rel_height_pred"]).astype(np.float32)
    offset_pred = _tensor_view0_to_numpy(pred["remote_projection_offset_xy_pred"]).astype(np.float32)
    global_dir = _tensor_view0_to_numpy(pred["remote_projection_global_dir_xy_pred"]).astype(np.float32).reshape(-1)
    global_slope = float(np.asarray(_tensor_view0_to_numpy(pred["remote_projection_global_slope_pred"])).reshape(-1)[0])
    pred_global_dir_raw = global_dir.copy()
    pred_global_slope = global_slope

    if args.projection_aux_use_gt_global_direction or args.projection_aux_use_gt_global_slope:
        gt_global_dir, gt_global_slope = load_projection_aux_gt_global(args)
        if args.projection_aux_use_gt_global_direction:
            global_dir = gt_global_dir.astype(np.float32)
        if args.projection_aux_use_gt_global_slope:
            global_slope = float(gt_global_slope)

    if args.projection_aux_use_gt_projection_base:
        gt_projected, gt_center_xy = load_projection_aux_gt_projection_base(args)
        base_pts = resize_image_nearest(gt_projected, rel_pred.shape[:2]).astype(np.float32)
        center_xy = gt_center_xy.astype(np.float32).reshape(1, 1, 2)
    else:
        center_xy = np.zeros((1, 1, 2), dtype=np.float32)

    if base_pts.shape[:2] != rel_pred.shape[:2]:
        return None, {
            "error": (
                "projection aux shape mismatch: "
                f"pts={base_pts.shape}, rel={rel_pred.shape}, offset={offset_pred.shape}"
            )
        }

    valid_mask = (
        np.isfinite(base_pts).all(axis=-1)
        & np.isfinite(rel_pred)
        & np.isfinite(offset_pred).all(axis=-1)
    )
    if not valid_mask.any():
        return None, {"error": "no finite aux reconstruction points"}

    rel_scale = estimate_projection_aux_rel_height_scale(base_pts, valid_mask, args)
    rel_height_alignment = {"mode": args.projection_aux_rel_height_scale_mode}
    if args.projection_aux_rel_height_scale_mode == "gt_height_range":
        rel_height, rel_height_alignment = rel_height_from_gt_height_range(rel_pred, valid_mask, args)
    else:
        rel_height = rel_pred * rel_scale
    offset_xy = offset_pred * float(args.projection_aux_offset_scale)
    ground_z = float(np.quantile(base_pts[..., 2][valid_mask], args.projection_aux_ground_quantile))

    offset_recon = base_pts.copy()
    if not args.projection_aux_use_gt_projection_base:
        offset_recon[..., 2] = ground_z + rel_height
    offset_recon[..., :2] = base_pts[..., :2] + center_xy - offset_xy

    dir_norm = np.linalg.norm(global_dir)
    if not np.isfinite(dir_norm) or dir_norm < 1e-6:
        global_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        global_dir = (global_dir / dir_norm).astype(np.float32)
    global_recon = base_pts.copy()
    if not args.projection_aux_use_gt_projection_base:
        global_recon[..., 2] = ground_z + rel_height
    global_recon[..., :2] = (
        base_pts[..., :2]
        + center_xy
        - rel_height[..., None] * float(global_slope) * global_dir.reshape(1, 1, 2)
    )

    grid_xy = projection_aux_pixel_grid(rel_pred.shape[:2])
    grid_recon = np.zeros((*rel_pred.shape[:2], 3), dtype=np.float32)
    grid_recon[..., :2] = (
        grid_xy
        - rel_pred[..., None] * float(global_slope) * global_dir.reshape(1, 1, 2)
    )
    grid_recon[..., 2] = rel_pred
    grid_valid_mask = (
        np.isfinite(rel_pred)
        & np.isfinite(grid_recon).all(axis=-1)
    )

    colors_np = get_view_colors(pred, view)
    offset_points = offset_recon[valid_mask]
    global_points = global_recon[valid_mask]
    grid_global_points = grid_recon[grid_valid_mask]
    offset_points, offset_xyz_alignment = align_aux_points_to_gt_unit_xy_zrange(offset_points, args)
    global_points, global_xyz_alignment = align_aux_points_to_gt_unit_xy_zrange(global_points, args)
    grid_global_points, grid_global_xyz_alignment = align_aux_points_to_gt_unit_xy_zrange(
        grid_global_points,
        args,
    )

    summary = {
        "points": int(valid_mask.sum()),
        "grid_global_points": int(grid_valid_mask.sum()),
        "rel_height_scale": rel_scale,
        "rel_height_alignment": rel_height_alignment,
        "xyz_alignment": {
            "offset": offset_xyz_alignment,
            "global": global_xyz_alignment,
            "grid_global": grid_global_xyz_alignment,
        },
        "offset_scale": float(args.projection_aux_offset_scale),
        "ground_z": ground_z,
        "ground_quantile": float(args.projection_aux_ground_quantile),
        "use_gt_projection_base": bool(args.projection_aux_use_gt_projection_base),
        "projection_center_xy": [float(v) for v in center_xy.reshape(-1)],
        "global_dir_xy": [float(v) for v in global_dir.reshape(-1)],
        "global_slope": global_slope,
        "pred_global_dir_xy": [float(v) for v in pred_global_dir_raw.reshape(-1)],
        "pred_global_slope": float(pred_global_slope),
        "use_gt_global_direction": bool(args.projection_aux_use_gt_global_direction),
        "use_gt_global_slope": bool(args.projection_aux_use_gt_global_slope),
        "rel_height_pred_abs_mean": float(np.mean(np.abs(rel_pred[valid_mask]))),
        "rel_height_world_abs_mean": float(np.mean(np.abs(rel_height[valid_mask]))),
        "offset_pred_norm_mean": float(np.linalg.norm(offset_pred[valid_mask], axis=-1).mean()),
        "offset_world_norm_mean": float(np.linalg.norm(offset_xy[valid_mask], axis=-1).mean()),
    }
    return {
        "offset": (offset_points, colors_np[valid_mask]),
        "global": (global_points, colors_np[valid_mask]),
        "grid_global": (grid_global_points, colors_np[grid_valid_mask]),
    }, summary


def export_projection_aux_reconstruction(outputs, views, args: argparse.Namespace, output_path: Path):
    if not args.export_projection_aux_reconstruction:
        return

    remote_indices = get_remote_view_indices(views)
    if not remote_indices:
        print("No marked remote views; projection-aux reconstruction was not written.")
        return

    method_points = {"offset": [], "global": [], "grid_global": []}
    method_colors = {"offset": [], "global": [], "grid_global": []}
    summaries = []
    for view_idx in remote_indices:
        if view_idx >= len(outputs):
            continue
        height_summary = save_projection_aux_height_comparison_for_view(
            outputs[view_idx],
            views[view_idx],
            args,
            output_path,
        )
        result, summary = reconstruct_projection_aux_points_for_view(
            outputs[view_idx],
            views[view_idx],
            args,
        )
        summary["view_idx"] = int(view_idx)
        summary["source_name"] = str(views[view_idx].get("source_name", f"view_{view_idx}"))
        if height_summary is not None:
            summary["height_comparison"] = height_summary
        summaries.append(summary)
        if result is None:
            print(f"View {view_idx}: skipped projection-aux reconstruction ({summary})")
            continue
        for method, (points, colors) in result.items():
            method_points[method].append(points)
            method_colors[method].append(colors)
            print(
                f"View {view_idx}: reconstructed {points.shape[0]} projection-aux "
                f"points with method={method}"
            )

    for method in ["offset", "global", "grid_global"]:
        if not method_points[method]:
            continue
        points = np.concatenate(method_points[method], axis=0)
        colors = np.concatenate(method_colors[method], axis=0)
        write_point_cloud_ply(
            points,
            colors,
            resolve_aux_reconstruction_output_path(output_path, method),
            args,
            description=f"projection-aux {method} remote",
        )

    summary_path = resolve_aux_reconstruction_summary_path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"views": summaries}, f, indent=2)
    print(f"Saved projection-aux reconstruction summary to: {summary_path}")


def export_point_cloud_for_views(model, views, args: argparse.Namespace, output_path: Path, label: str | None):
    if label:
        print(f"Running inference for remote-control mode: {label}")
    else:
        print("Running inference...")
    start_time = time()
    with torch.inference_mode():
        outputs = run_model_inference(model, views, args)
    duration = time() - start_time
    print(f"Inference finished in {duration:.3f}s")

    if args.export_view_filter != "all":
        print(
            "--export_view_filter is deprecated and ignored; exporting mixed PLY "
            "plus remote companion PLY."
        )
    print("Collecting mixed world-space point cloud...")
    points, colors, per_view_stats = collect_world_space_point_cloud(
        outputs,
        views,
        view_filter="all",
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        skip_remote_points=False,
    )
    print_per_view_stats(per_view_stats)
    write_point_cloud_ply(points, colors, output_path, args, description="mixed")

    if not get_remote_view_indices(views):
        print("No marked remote views; remote companion PLY was not written.")
        return

    print("Collecting remote-only companion point cloud...")
    remote_points, remote_colors, remote_per_view_stats = collect_world_space_point_cloud(
        outputs,
        views,
        view_filter="remote",
        apply_confidence_mask=args.apply_confidence_mask,
        confidence_percentile=args.confidence_percentile,
        skip_remote_points=False,
    )
    print_per_view_stats(remote_per_view_stats)
    remote_output_path = resolve_remote_companion_output_path(output_path)
    write_point_cloud_ply(
        remote_points,
        remote_colors,
        remote_output_path,
        args,
        description="remote-only companion",
    )
    export_projection_aux_reconstruction(outputs, views, args, output_path)


def main() -> None:
    args = parse_args()
    if args.model == "vggt_omega" and args.resize_mode == "fixed_mapping" and args.resolution_set != 512:
        print(
            "VGGT-Omega uses patch_size=16; overriding fixed_mapping "
            f"resolution_set {args.resolution_set} -> 512."
        )
        args.resolution_set = 512
    effective_model_name = resolve_effective_model_name(args)
    if effective_model_name == "mapanything_rs_joint" and not args.checkpoint_path:
        raise ValueError(
            "--model mapanything_rs_joint requires --checkpoint_path because there "
            "is no default HuggingFace checkpoint for this local RS-joint variant."
        )
    if effective_model_name != args.model:
        print(
            f"Resolved export model '{args.model}' -> '{effective_model_name}' "
            "based on the checkpoint path"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    original_model_name = args.model
    args.model = effective_model_name
    config_overrides = resolve_config_overrides(args)
    model = initialize_model(args, device, config_overrides, effective_model_name)
    args.model = original_model_name

    load_size = resolve_load_size(args)
    print(f"Loading images from: {args.image_folder}")
    views = load_images(
        args.image_folder,
        resize_mode=args.resize_mode,
        size=load_size,
        resolution_set=args.resolution_set,
        stride=args.stride,
    )
    if len(views) == 0:
        raise ValueError(f"No images found in {args.image_folder}")
    print(f"Loaded {len(views)} views")
    views = annotate_view_source_names(views, args.image_folder, args.stride)

    model_name = getattr(model, "name", effective_model_name)
    views = convert_views_to_identity_if_needed(views, model_name)
    views = maybe_assign_remote_instances(views, args)

    variants = build_remote_control_view_variants(views, args, model_name)
    for variant_name, variant_views in variants:
        output_path = resolve_variant_output_path(args.output_path, variant_name)
        export_point_cloud_for_views(
            model=model,
            views=variant_views,
            args=args,
            output_path=output_path,
            label=variant_name,
        )


if __name__ == "__main__":
    main()
