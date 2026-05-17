# Crossview MapAnything 训练脚本

该目录放置 `Crossview` 数据上的 `MapAnything` 训练入口。

当前脚本：

- [p4_mapanything_rs_joint_debug_1gpu.sh](p4_mapanything_rs_joint_debug_1gpu.sh)
  - 1 GPU smoke / debug 入口
  - 用于快速验证 `mapanything_rs_joint` 的 joint forward、loss、checkpoint 保存和评估链路

- [p4_mapanything_rs_joint_500_4gpu_all.sh](p4_mapanything_rs_joint_500_4gpu_all.sh)
  - 4 GPU 正式训练入口
  - 用于 `Crossview` / `RS-joint` 的 `MapAnything` 微调实验

说明：

- 这两个脚本由原先的 `bash_scripts/train/vigor_chicago/` 路径迁移而来
- 当前统一收敛到 `bash_scripts/train/Crossview/mapanything/`
- 相关训练记录见：
  - [../../../../VIGOR_CHICAGO_TRAINING_CN.md](../../../../VIGOR_CHICAGO_TRAINING_CN.md)
  - [../../../../MAPANYTHING_RS_JOINT_MODEL_CN.md](../../../../MAPANYTHING_RS_JOINT_MODEL_CN.md)
