# MapAnything 现有 Benchmark 总览（不含 RS-Aerial Reconstruction Benchmark）

本文档汇总 `Models/map-anything` 项目中，除 `RS-Aerial Reconstruction Benchmark` 之外，当前已经存在并可运行的 benchmark。

当前纳入统计的 benchmark 有 3 条主线：

- `Single-View Image Calibration Benchmark`
- `Dense Up to N View Reconstruction Benchmark`
- `RobustMVD Benchmark`

不纳入本文范围的 benchmark：

- `RS-Aerial Reconstruction Benchmark`
  - 相关设计与入口见：`RS_GUIDED_DENSE_MV_BENCHMARK_DESIGN_CN.md`
  - 相关实现目录见：`benchmarking/rs_guided_dense_mv/`

## 1. Benchmark 总表

| Benchmark 名称 | 目录 / 入口 | 任务目标 | 输入形态 | 主要数据集 | 指标来源 |
|---|---|---|---|---|---|
| Single-View Image Calibration Benchmark | `benchmarking/calibration/benchmark.py` | 从单张 RGB 图像恢复相机内参与 ray directions | 单视图 RGB | ETH3D、ScanNet++V2、TartanAirV2-WB | 项目内自定义指标 |
| Dense Up to N View Reconstruction Benchmark | `benchmarking/dense_n_view/benchmark.py` | 评估 2 到 N 视图条件下的稠密多视图重建能力 | 多视图 RGB，视图之间要求 covisibility 连通 | ETH3D、ScanNet++V2、TartanAirV2-WB | 项目内自定义指标 |
| RobustMVD Benchmark | `benchmarking/rmvd_mvs_benchmark/benchmark.py` | 接入 RobustMVD 官方评测协议，对 MapAnything 做标准 MVD 评测 | 单视图或多视图 RGB，可选附加 intrinsics / pose | KITTI、ScanNet | 外部 RMVD 官方评测框架 |

## 2. 指标表

### 2.1 Single-View Image Calibration Benchmark

该 benchmark 的核心目标不是直接评估深度或位姿，而是评估模型是否能仅凭图像恢复正确的相机光线方向，也就是隐式恢复相机内参。

| 指标名 | 含义 | 计算方式 / 解释 | 趋势 |
|---|---|---|---|
| `ray_dirs_err_deg` | 单位光线方向角误差 | 对 GT 与预测 ray direction 的 L2 差，转换为角度误差后，在像素上取均值，再在视图与样本上聚合 | 越小越好 |

补充说明：

- 指标由 `benchmarking/calibration/benchmark.py` 直接计算。
- 当前实现对每个 scene 保存 `per_scene_results`，并进一步聚合出按数据集和全 benchmark 的平均结果。
- 当前主指标只有 `ray_dirs_err_deg`，没有引入深度、位姿或尺度指标。

### 2.2 Dense Up to N View Reconstruction Benchmark

这是 MapAnything 项目中的标准多视图重建 benchmark。它评估的是在给定 2 到 N 个连通视图输入时，模型在几何、深度、位姿、相机方向和尺度上的综合表现。

| 指标名 | 含义 | 计算方式 / 解释 | 趋势 |
|---|---|---|---|
| `metric_scale_abs_rel` | metric scale 绝对相对误差 | 比较预测 metric scale 与 GT scale 的相对误差 | 越小越好 |
| `pointmaps_abs_rel` | 全局 pointmap 绝对相对误差 | 用 `m_rel_ae` 比较 GT 与预测 3D 点图 | 越小越好 |
| `pointmaps_inlier_thres_103` | pointmap 1.03 阈值内点率 | 用 `thresh_inliers(..., thresh=1.03)` 统计点图预测是否落入相对误差阈值内 | 越大越好 |
| `z_depth_abs_rel` | z-depth 绝对相对误差 | 在相机坐标系下比较 GT 与预测 z 深度 | 越小越好 |
| `z_depth_inlier_thres_103` | z-depth 1.03 阈值内点率 | 用 `thresh_inliers(..., thresh=1.03)` 统计深度内点比例 | 越大越好 |
| `ray_dirs_err_deg` | 光线方向角误差 | 比较预测与 GT ray directions 的角度误差 | 越小越好 |
| `pose_ate_rmse` | 位姿绝对轨迹误差 RMSE | 通过 `evaluate_ate` 计算多视图集合内的位姿 ATE | 越小越好 |
| `pose_auc_5` | 位姿误差 AUC@5 | 由相对旋转/平移误差构造 AUC，阈值上限 5，并转成百分比 | 越大越好 |

补充说明：

- 这些指标直接来自 `benchmarking/dense_n_view/benchmark.py` 中 `per_scene_results` 的初始化和后续计算逻辑。
- 当前实现会先把所有量统一到 `view0` 坐标系，并进行联合归一化；因此其中多项指标本质上是归一化坐标下的误差，不是直接米制误差。
- 该 benchmark 还有一个变体入口：`benchmarking/dense_n_view/benchmark_global_pm_only.py`。
  - 该变体只保留更轻量的全局 pointmap 相关指标：
  - `metric_scale_abs_rel`
  - `pointmaps_abs_rel`
  - `pointmaps_inlier_thres_103`
  - 它更像是 `dense_n_view` 的裁剪版，不建议单独视作新的 benchmark 体系。

### 2.3 RobustMVD Benchmark

该 benchmark 不是在仓库内部重新定义一套指标，而是将 MapAnything 包装成 RMVD 兼容模型，然后交给 `rmvd.create_evaluation(...)` 执行官方评测。

| 指标类 | 当前仓库中的定义状态 | 说明 |
|---|---|---|
| 深度 / MVD 官方指标 | 由外部 RMVD 框架定义 | `benchmarking/rmvd_mvs_benchmark/benchmark.py` 中没有手写指标公式，而是直接调用 RMVD 官方 evaluation |
| 对齐协议 | 仓库内有显式配置 | `evaluation_alignment` 可选 `none` 或 `median` |
| 输入条件协议 | 仓库内有显式配置 | `evaluation_conditioning` 可选 `img`、`img+intrinsics`、`img+intrinsics+pose` |
| 视图协议 | 仓库内有显式配置 | `evaluation_views` 可选 `single_view` 或 `multi_view` |

对这个 benchmark，需要特别说明：

- 本仓库负责的是“适配与调用”，不是“重新定义指标”。
- 代码入口为：`benchmarking/rmvd_mvs_benchmark/benchmark.py`
- 模型包装器为：`benchmarking/rmvd_mvs_benchmark/adaptors.py`
- 实际评测由：`rmvd.create_dataset(...)` 和 `rmvd.create_evaluation(...)` 完成。
- 因此，如果要写论文或设计文档，最稳妥的表述应该是：
  - “指标采用 RobustMVD 官方 MVD evaluation protocol”
  - 而不是声称这些指标是 `map-anything` 仓库内部定义的。

## 3. 数据集表

### 3.1 Single-View Calibration 与 Dense N-View 共用数据集族

这两条 benchmark 共用一套 WAI 格式 benchmark 数据，只是采样方式与 `num_views` 不同。

#### Calibration benchmark 数据集

对应配置：`configs/dataset/benchmark_sv_calib_518_many_ar_eth3d_snpp_tav2.yaml`

| 数据集 | 配置名 | 分辨率设置 | 采样说明 | 规模说明 |
|---|---|---|---|---|
| ETH3D | `ETH3DWAI` test split | `518_many_ar` | 每个 scene 采样 20 帧 | 13 scenes，合计 260 samples |
| ScanNet++V2 | `ScanNetPPWAI` test split | `518_many_ar` | 每个 scene 采样 20 帧 | 30 scenes，合计 600 samples |
| TartanAirV2-WB | `TartanAirV2WBWAI` test split | `518_many_ar` | 每个 scene 采样 20 帧 | 5 scenes，合计 100 samples |

#### Dense N-View benchmark 数据集

对应配置：`configs/dataset/benchmark_518_eth3d_snpp_tav2.yaml`

| 数据集 | 配置名 | 分辨率设置 | 采样说明 | 规模说明 |
|---|---|---|---|---|
| ETH3D | `ETH3DWAI` test split | `518_1_52_ar` | 每个 scene 采样 10 个 multi-view sets | 13 scenes，合计 130 sets |
| ScanNet++V2 | `ScanNetPPWAI` test split | `518_1_52_ar` | 每个 scene 采样 10 个 multi-view sets | 30 scenes，合计 300 sets |
| TartanAirV2-WB | `TartanAirV2WBWAI` test split | `518_1_00_ar` | 每个 scene 采样 10 个 multi-view sets | 5 scenes，合计 50 sets |

补充说明：

- `dense_n_view` 的 README 明确要求采样到的输入视图在预计算 covisibility 图上形成单个连通分量。
- Calibration benchmark 的 `num_views=1`，本质上是 many-aspect-ratio 的单视图内参恢复测试。
- Dense benchmark 的 `num_views` 可变，脚本层可以扩展到 2-view 和 N-view 设置。

### 3.2 RobustMVD Benchmark 数据集

对应说明：`benchmarking/rmvd_mvs_benchmark/README.md`

| 数据集 | 来源 | 说明 |
|---|---|---|
| KITTI | RMVD 官方数据准备流程 | 仓库 README 明确要求按 RMVD 官方说明准备，并放到 `external_benchmark_data_root_data_dir/kitti` |
| ScanNet | RMVD 官方数据准备流程 | 仓库 README 明确要求按 RMVD 官方说明准备，并放到 `external_benchmark_data_root_data_dir/scannet` |

补充说明：

- `configs/rmvd_benchmark.yaml` 中还保留了 `eval_dataset: eth3d` 的默认值，但 README 与生成脚本 `generate_benchmark_scripts.py` 当前实际覆盖的是 `kitti` 和 `scannet` 两个协议集。
- 生成脚本当前会组合以下评测维度：
  - 数据集：`kitti` / `scannet`
  - 视图数：`single_view` / `multi_view`
  - 条件输入：`image`、`image+intrinsics`、`image+intrinsics+pose`
  - 对齐方式：`none` / `median`

## 4. 结论性梳理

如果只从“当前项目里已有、且不是 RS-Aerial 的 benchmark”来划分，可以把它们理解成三层能力测试：

| 层级 | 对应 benchmark | 核心能力 |
|---|---|---|
| 单视图相机几何 | Calibration Benchmark | 单图恢复相机内参与 ray directions |
| 多视图统一重建 | Dense Up to N View Reconstruction Benchmark | 几何、深度、位姿、尺度的联合评估 |
| 外部标准协议对齐 | RobustMVD Benchmark | 在外部通用 MVD protocol 下和社区方法对表 |

如果后续你还想继续扩展这份文档，最自然的下一步有两个：

- 加一节“各 benchmark 与 RS-Aerial benchmark 的关系”，专门说明哪些指标可以复用，哪些不能直接复用。
- 再补一节“结果文件格式对照表”，把每个 benchmark 最终输出的 JSON 文件结构也统一列出来，便于之后写可视化或汇总脚本。
