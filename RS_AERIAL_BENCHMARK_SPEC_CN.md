# RS-Aerial Reconstruction Benchmark 规格说明

本文档只描述当前统一 benchmark 的正式定义、指标与运行方式，不再重复历史阶段划分。

## 1. Benchmark 任务定义，数据集处理与定义

### 1.1 任务定义

本 benchmark 评估的是 **RS-Aerial 联合三维重建能力**。对同一个 `location`，数据分为两类输入：

- `Aerial-only`：仅输入空中视角多帧透视图像；
- `Aerial+RS`：同时输入空中视角多帧透视图像与对应卫星图像。

benchmark 关注三类问题：

- 空中视角重建是否准确；
- 卫星视角高度恢复是否准确；
- 联合输入后，两类视图是否进入同一个全局三维空间。

### 1.2 数据来源

当前数据来自两个实验输出：

- 空中视角数据：[`../../CrossView/dataset_generation/crossview_reconstruction_pipeline/experiments/exp_001_reconstrc`](../../CrossView/dataset_generation/crossview_reconstruction_pipeline/experiments/exp_001_reconstrc)
- 卫星视角标签：[`../../CrossView/dataset_generation/crossview_reconstruction_pipeline/experiments/exp_005_map_points_generate`](../../CrossView/dataset_generation/crossview_reconstruction_pipeline/experiments/exp_005_map_points_generate)

空中视角场景已经整理成 `WAI` 格式，数据根目录位于：

- [`../../outputs/dataset/vigor_chicago_wai`](../../outputs/dataset/vigor_chicago_wai)

卫星视角与 benchmark 相关的元数据位于：

- [`../../outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial`](../../outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial)

### 1.3 关键标签定义

卫星视角的逐像素几何标签来自 `exp_005` 的 `pixel_to_point_map.npz`，其中：

- `xyz`：形状为 `(H, W, 3)` 的逐像素全局点云标签；
- 无效像素为 `NaN`；
- `valid_mask.npy`：与 `~np.isnan(height_map)` 对应的有效掩码；
- `height_map.npy`：逐像素高度标签。

这意味着卫星视角不是标准 pinhole 相机视图，但它仍然可以作为一个带逐像素几何监督的输入视图参与 benchmark。

### 1.4 数据组织

当前统一 benchmark 直接使用 paired scene：

- `aerial`：空中视角图像、位姿、内参、点图 GT；
- `remote`：卫星图像、逐像素全局点云 GT、有效掩码、高度图。

推荐的 scene 结构如下：

```text
outputs/dataset/vigor_chicago_rs_guided/
└── location_x/
    ├── aerial/
    │   ├── scene_meta.json
    │   ├── images/
    │   ├── depth/
    │   └── cameras/
    └── remote/
        └── Google_Satellite/
            ├── pixel_to_point_map.npz
            ├── valid_mask.npy
            ├── height_map.npy
            └── ...
```

### 1.5 数据集配置关系

当前项目里和 `VIGOR Chicago` 相关的配置与 metadata 有 4 层，它们不是重复关系，而是逐层组合关系：

#### 1.5.1 底层 WAI split 配置

目录：[`configs/dataset/vigor_chicago_wai/`](configs/dataset/vigor_chicago_wai)

这一层定义的是底层数据集类 `VigorChicagoWAI` 在不同 split 下如何实例化：

- [`configs/dataset/vigor_chicago_wai/default.yaml`](configs/dataset/vigor_chicago_wai/default.yaml)：组合 `train/val/test` 三套子配置；
- [`configs/dataset/vigor_chicago_wai/train/default.yaml`](configs/dataset/vigor_chicago_wai/train/default.yaml)：训练 split 的参数；
- [`configs/dataset/vigor_chicago_wai/val/default.yaml`](configs/dataset/vigor_chicago_wai/val/default.yaml)：验证 split 的参数；
- [`configs/dataset/vigor_chicago_wai/test/default.yaml`](configs/dataset/vigor_chicago_wai/test/default.yaml)：测试 split 的参数。

这一层主要回答：

- 用哪个 split；
- 用哪个 metadata 根目录；
- 用什么 `transform`、`resolution`、`num_views`、`covisibility_thres`。

#### 1.5.2 顶层训练配置

文件：[`configs/dataset/vigor_chicago_500_518.yaml`](configs/dataset/vigor_chicago_500_518.yaml)

这一层不是新的数据集类，而是把上面的底层 split 配置组装成一套训练实验配置。它定义的是：

- 训练时默认使用哪个 `train_dataset`；
- 验证/测试时使用哪个 `test_dataset`；
- 当前实验的 `num_views` 与分辨率；
- 当前 500-scene 版本下的 train/val/test 协议。

因此：

- `vigor_chicago_wai/*` 是底层模板；
- `vigor_chicago_500_518.yaml` 是训练入口配置。

#### 1.5.3 空中视角 metadata

目录：[`../../outputs/dataset/mapanything_metadata/vigor_chicago/`](../../outputs/dataset/mapanything_metadata/vigor_chicago/)

这一层是 `VigorChicagoWAI` 的 split metadata，服务于普通 aerial 数据读入。

当前数量为：

- `train`: 400 scenes
- `val`: 50 scenes
- `test`: 50 scenes

对应文件示例：

- `train/vigor_chicago_scene_list_train.npy`
- `val/vigor_chicago_scene_list_val.npy`
- `test/vigor_chicago_scene_list_test.npy`

#### 1.5.4 RS-Aerial paired metadata

目录：[`../../outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial/`](../../outputs/dataset/mapanything_metadata/vigor_chicago_rs_aerial/)

这一层是 benchmark 专用 metadata。它不是普通 aerial split，而是：

- 只保留同时具备 aerial 与 RS 标签的 paired scenes；
- 为 RS 视图保存额外 manifest；
- 让 benchmark 能同时构建 aerial loader 和 remote loader。

当前数量同样为：

- `train`: 400 paired scenes
- `val`: 50 paired scenes
- `test`: 50 paired scenes

对应文件示例：

- `train/vigor_chicago_rs_aerial_train.json`
- `train/vigor_chicago_rs_aerial_scene_list_train.npy`
- `val/vigor_chicago_rs_aerial_val.json`
- `val/vigor_chicago_rs_aerial_scene_list_val.npy`

#### 1.5.5 当前 unified benchmark 使用哪一层

当前统一 benchmark 配置文件是：[`configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml`](configs/dataset/benchmark_vigor_chicago_rs_aerial.yaml)

它会同时用到两层 metadata：

- aerial loader：底层仍然实例化 `VigorChicagoWAI`，读取 `vigor_chicago` 这份 metadata；
- paired scene 过滤与 remote loader：读取 `vigor_chicago_rs_aerial` 这份 paired metadata。

当前为了符合 benchmark 评测协议，统一 benchmark 已切换为使用训练数据集的 `val` split，也就是：

- aerial：`split='val'`
- remote：`split='val'`
- scene list：`vigor_chicago_rs_aerial/val/vigor_chicago_rs_aerial_scene_list_val.npy`

因此，当前 benchmark 的正式评测规模应为：

- `50` 个 `val` paired scenes

### 1.6 分辨率约定

当前统一 benchmark 统一将输入分辨率收敛到 `518 x 518`。图像、掩码、深度/点图标签的 resize 规则必须保持一致；几何量避免使用会生成“半空中点”的插值策略。

### 1.7 运行对象

当前 benchmark 采用统一入口：

- [`benchmarking/rs_guided_dense_mv/benchmark_unified.py`](benchmarking/rs_guided_dense_mv/benchmark_unified.py)

对外只保留一个统一 benchmark；内部通过结果结构区分：

- `aerial_only`
- `rs_only`
- `joint`
- `improvement`


## 2. 指标汇总

### 2.1 指标总表

下表汇总当前统一 benchmark 的正式指标。`Metric-only` 表示该指标只对显式输出 `metric_scaling_factor` 的模型报告；`All-model` 表示所有模型都可以报告。

| 指标 | 适用子任务 | 数据空间 | 对齐方式 | Metric-only | 含义 | 状态 |
|---|---|---|---|---|---|---|
| `pointmaps_abs_rel` | Aerial-only | 空中视角逐像素 3D 点图 | 先切到各自 `view0` 参考系，再对 aerial 集合联合归一化 | 否 | 评估空中视角多视图几何是否自洽 | 主指标 |
| `z_depth_abs_rel` | Aerial-only | 空中视角 z-depth | 同 `pointmaps_abs_rel` | 否 | 评估深度恢复质量 | 主指标 |
| `pose_ate_rmse` | Aerial-only | 轨迹平移 | 先切到 `view0` 参考系，再按联合尺度归一化 | 否 | 评估位姿轨迹恢复误差 | 主指标 |
| `pose_auc_5` | Aerial-only | 相对位姿 | 先切到 `view0` 参考系 | 否 | 评估相对位姿误差在阈值内的 AUC | 主指标 |
| `ray_dirs_err_deg` | Aerial-only | 射线方向 | 直接比较单位射线方向 | 否 | 评估相机射线几何是否正确 | 主指标 |
| `metric_scale_abs_rel` | Aerial-only | 尺度因子 | 比较预测尺度因子与 GT 尺度因子的相对误差 | 是 | 评估绝对尺度恢复是否准确 | 主指标 |
| `metric_point_l1` | Aerial-only | 真实尺度下的点图 | 在 metric 全局坐标中直接比较 | 是 | 评估真实空间中的点云偏差，单位可直接解释为米 | 主指标 |
| `rs_height_mae_affine` | RS-only | 卫星高度 | 先做每个 scene 的仿射 z 对齐，再计算误差 | 否 | 评估卫星视角下相对高度结构是否正确 | 主指标 |
| `rs_height_rmse_affine` | RS-only | 卫星高度 | 同上 | 否 | 对大误差更敏感 | 主指标 |
| `rs_height_mae` | RS-only | 卫星高度 | 仅对 metric-only 模型报告，通常不再作为通用主指标 | 是 | metric 版本的高度 MAE | 辅助 |
| `rs_height_rmse` | RS-only | 卫星高度 | 仅对 metric-only 模型报告 | 是 | metric 版本的高度 RMSE | 辅助 |
| `joint_global_pointmaps_abs_rel` | Joint | Aerial + RS 全局点图 | 将 aerial 与 RS 的全局点图合并后做全局联合归一化 | 否 | 评估联合输入后是否进入同一个全局空间 | 主指标 |
| `joint_global_point_l1` | Joint | Aerial + RS 全局点图 | metric 全局坐标下直接比较 | 是 | 评估联合输入后全局空间配准是否准确 | 主指标 |

### 2.2 有效集合记号

为避免歧义，统一定义：

- $\mathcal{M}_a$：Aerial-only 的有效像素集合；每个 aerial view 单独定义。
- $\mathcal{M}_r$：RS-only 的有效像素集合；来自 `remote_valid_mask`。
- $\mathcal{M}_{global}$：Joint 非 metric 指标的有效像素集合；由 aerial 与 RS 的有效像素并集组成。

其中 $\mathcal{M}_a$ 在实现中还会与 $\lVert \mathbf{P}^{gt} \rVert_2 > 0$ 的条件求交。

### 2.3 各指标公式

#### 2.3.1 `pointmaps_abs_rel`

对每个有效像素 $i \in \mathcal{M}_a$：

$$
\mathrm{pointmaps\_abs\_rel}
= \frac{1}{|\mathcal{M}_a|}
\sum_{i \in \mathcal{M}_a}
\frac{\lVert \mathbf{P}_i^{pr} - \mathbf{P}_i^{gt} \rVert_2}{\lVert \mathbf{P}_i^{gt} \rVert_2 + \varepsilon}
$$

这里 $\mathbf{P}_i^{gt}$ 和 $\mathbf{P}_i^{pr}$ 都是先切到各自 $view0$ 参考系后，再做 aerial 集合联合归一化得到的点图。

#### 2.3.2 `z_depth_abs_rel`

对每个有效像素 $i \in \mathcal{M}_a$：

$$
\mathrm{z\_depth\_abs\_rel}
= \frac{1}{|\mathcal{M}_a|}
\sum_{i \in \mathcal{M}_a}
\frac{|z_i^{pr} - z_i^{gt}|}{|z_i^{gt}| + \varepsilon}
$$

这里的 $z$ 是相机坐标系下的深度分量，计算前同样经过 aerial 集合的统一参考系和统一尺度处理。

#### 2.3.3 `pose_ate_rmse`

设第 $k$ 个视图的平移为 $\mathbf{t}_k^{pr}$ 与 $\mathbf{t}_k^{gt}$，则：

$$
\mathrm{pose\_ate\_rmse}
= \sqrt{\frac{1}{N} \sum_{k=1}^{N} \lVert \mathbf{t}_k^{pr} - \mathbf{t}_k^{gt} \rVert_2^2}
$$

这里的轨迹先被表达在统一的 $view0$ 参考系中。

#### 2.3.4 `pose_auc_5`

先计算相对旋转误差 $e_r$ 与相对平移误差 $e_t$，再统计阈值 $\tau$ 下的 inlier 曲线：

$$
\mathrm{pose\_auc\_5}
= \int_0^5 \mathrm{InlierRate}(\tau)\, d\tau
$$

实现中最终通常以百分数报告。

#### 2.3.5 `ray_dirs_err_deg`

单位射线方向 $\mathbf{r}_i^{pr}$ 与 $\mathbf{r}_i^{gt}$ 的角度误差：

$$
\theta_i = \arccos\left(\mathrm{clip}(\langle \mathbf{r}_i^{pr}, \mathbf{r}_i^{gt} \rangle, -1, 1)\right)
$$

$$
\mathrm{ray\_dirs\_err\_deg}
= \frac{1}{|\mathcal{M}_a|} \sum_{i \in \mathcal{M}_a} \theta_i
$$

#### 2.3.6 `metric_scale_abs_rel`

设 GT 与预测的尺度因子分别为 $s^{gt}$ 和 $s^{pr}$：

$$
\mathrm{metric\_scale\_abs\_rel}
= \frac{|s^{pr} - s^{gt}|}{|s^{gt}| + \varepsilon}
$$

这个指标只适用于显式输出 `metric_scaling_factor` 的模型。

#### 2.3.7 `metric_point_l1`

在真实 metric 全局坐标系下，直接比较点图：

$$
\mathrm{metric\_point\_l1}
= \frac{1}{|\mathcal{M}_a|}
\sum_{i \in \mathcal{M}_a}
\lVert \mathbf{P}_i^{pr} - \mathbf{P}_i^{gt} \rVert_1
$$

这个指标回答的是“平均每个点偏了多少米”。

#### 2.3.8 `rs_height_mae_affine` / `rs_height_rmse_affine`

先对每个 scene 做仿射 z 对齐：

$$
 z^{pr}_{aligned} = a z^{pr} + b
$$

其中 $a, b$ 由 $\mathcal{M}_r$ 上的最小二乘拟合得到。

然后：

$$
\mathrm{rs\_height\_mae\_affine}
= \frac{1}{|\mathcal{M}_r|}
\sum_{i \in \mathcal{M}_r} |z^{pr}_{aligned}(i) - z_i^{gt}|
$$

$$
\mathrm{rs\_height\_rmse\_affine}
= \sqrt{\frac{1}{|\mathcal{M}_r|}
\sum_{i \in \mathcal{M}_r}
\left(z^{pr}_{aligned}(i) - z_i^{gt}\right)^2}
$$

这组指标是 RS-only 的通用主指标，更适合 `Pi3` / `VGGT` / `MapAnything` 横向比较。

#### 2.3.9 `rs_height_mae` / `rs_height_rmse`

这是 metric 版本的 RS 高度指标，只适用于输出 metric scale 的模型：

$$
\mathrm{rs\_height\_mae}
= \frac{1}{|\mathcal{M}_r|}
\sum_{i \in \mathcal{M}_r} |z_i^{pr} - z_i^{gt}|
$$

$$
\mathrm{rs\_height\_rmse}
= \sqrt{\frac{1}{|\mathcal{M}_r|}
\sum_{i \in \mathcal{M}_r} (z_i^{pr} - z_i^{gt})^2}
$$

#### 2.3.10 `joint_global_pointmaps_abs_rel`

将 aerial 与 RS 的全局点图合并后，对 GT 集合和预测集合分别做全局联合归一化，然后计算逐像素相对误差：

$$
\mathrm{joint\_global\_pointmaps\_abs\_rel}
= \frac{1}{|\mathcal{M}_{global}|}
\sum_{i \in \mathcal{M}_{global}}
\frac{\lVert \mathbf{P}_i^{pr} - \mathbf{P}_i^{gt} \rVert_2}{\lVert \mathbf{P}_i^{gt} \rVert_2 + \varepsilon}
$$

这里的关键是：归一化使用的是 **联合全局点集合**，不是只用 aerial 点。

#### 2.3.11 `joint_global_point_l1`

metric 版本的联合全局误差：

$$
\mathrm{joint\_global\_point\_l1}
= \frac{1}{|\mathcal{M}_{global}|}
\sum_{i \in \mathcal{M}_{global}}
\lVert \mathbf{P}_i^{pr} - \mathbf{P}_i^{gt} \rVert_1
$$

这个指标只适用于 metric 模型。

### 2.4 指标解释原则

- `pointmaps_abs_rel`、`z_depth_abs_rel`、`pose_*`、`ray_dirs_err_deg`：评估相对几何，自身不要求 metric scale；
- `metric_scale_abs_rel`、`metric_point_l1`、`rs_height_mae`、`rs_height_rmse`、`joint_global_point_l1`：只适用于显式 metric scale 模型；
- `rs_height_mae_affine` / `rs_height_rmse_affine`、`joint_global_pointmaps_abs_rel`：对所有模型都可以作为统一比较指标。


## 3. benchmark 运行说明

### 3.1 统一入口

统一 benchmark 入口是：

- [`benchmarking/rs_guided_dense_mv/benchmark_unified.py`](benchmarking/rs_guided_dense_mv/benchmark_unified.py)

统一结果会输出到：

- [`../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/`](../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/)

### 3.2 单次运行

Pi3：

```bash
cd /root/autodl-tmp/Models/map-anything
source /etc/profile.d/clash.sh && proxy_on
NUM_VIEWS=4 BATCH_SIZE=1 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified.sh
```

VGGT：

```bash
cd /root/autodl-tmp/Models/map-anything
source /etc/profile.d/clash.sh && proxy_on
NUM_VIEWS=4 BATCH_SIZE=1 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/vggt_unified.sh
```

MapAnything：

```bash
cd /root/autodl-tmp/Models/map-anything
source /etc/profile.d/clash.sh && proxy_on
export MAPANYTHING_CKPT=/root/autodl-tmp/outputs/checkpoints/mapanything/map-anything_benchmark.pth
NUM_VIEWS=4 BATCH_SIZE=1 \
  bash bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified.sh
```

### 3.3 sweep 运行

三模型 sweep 脚本分别是：

- [`bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified_sweep.sh`](bash_scripts/benchmark/rs_guided_dense_mv/pi3_unified_sweep.sh)
- [`bash_scripts/benchmark/rs_guided_dense_mv/vggt_unified_sweep.sh`](bash_scripts/benchmark/rs_guided_dense_mv/vggt_unified_sweep.sh)
- [`bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified_sweep.sh`](bash_scripts/benchmark/rs_guided_dense_mv/mapanything_unified_sweep.sh)

默认 sweep 帧数：`2, 4, 8, 16, 24, 32, 40`。

### 3.4 结果聚合与可视化

统一聚合脚本：

- [`scripts/aggregate_rs_aerial_sweeps.py`](scripts/aggregate_rs_aerial_sweeps.py)

运行方式：

```bash
cd /root/autodl-tmp/Models/map-anything
python scripts/aggregate_rs_aerial_sweeps.py \
  --root /root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv
```

输出文件：

- [`../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_long.csv`](../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_long.csv)
- [`../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_summary.csv`](../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_summary.csv)
- [`../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_grid.png`](../../outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/aggregated/sweep_metrics_grid.png)

### 3.5 结果解读

建议优先看三类结果：

- `sweep_metrics_summary.csv`：适合做最终对比表；
- `sweep_metrics_grid.png`：适合看帧数变化趋势；
- `per_scene_results`：适合定位个别场景退化或提升。

### 3.6 当前运行约定

- `Aerial+RS` 曲线使用实线；
- `Aerial-only` 曲线使用虚线；
- 横轴固定为帧数；
- `metric-only` 指标在非 metric 模型上留空。

