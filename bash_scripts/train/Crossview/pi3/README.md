# Crossview Pi3 Training Scripts

这里收敛 `VIGOR Chicago / Crossview` 相关的 `Pi3` 训练脚本。

当前建议按阶段使用：

- `p0_pi3_data_sanity_500_pretrained_2gpu.sh`
  - `P0` 数据读取与短跑检查
- `p1_pi3_baseline_500_pretrained_2gpu.sh`
  - `P1` aerial-only baseline
- `p2_pi3_rs_only_debug_2gpu.sh`
  - `P2` RS-only smoke
- `p2a_pi3_rs_only_loss_ablation_2gpu.sh`
  - `P2a` RS-only loss ablation
- `p3_pi3_base.sh`
  - `P3` 当前 joint 主基线
- `p3_pi3_modality_embedding.sh`
  - `P3` 模态 embedding 版本
- `p3_pi3_freeze_shared.sh`
  - `P3` 模态 embedding + 冻结 shared decoder 版本
- `p3_pi3_modality_embedding_remote_head.sh`
  - `P3` 模态 embedding + remote head 版本
- `p3_pi3_low_covis.sh`
  - `P3` 低共视版本
- `p3_pi3_zero_covis.sh`
  - `P3` 零共视版本

说明：

- 旧路径 `bash_scripts/train/vigor_chicago/p*.sh` 仍保留为兼容包装。
- 新实验优先放到本目录，避免 `vigor_chicago` 目录继续堆积不同阶段与不同模型线。
- `P3` 命名已统一为 `p3_pi3_*`；旧的 `p3_pi3_joint_input_*` 名称不再作为规范入口。
