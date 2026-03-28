# VIGOR Chicago RS Dataset

说明：本文档中的文件链接统一使用相对于当前 `.md` 文件的相对路径，不使用绝对路径。

## 1. 目的

为 VIGOR Chicago 的遥感数据建立统一的数据集根目录：

- 目标目录：`../../traindata/vigor_chicago_rs`

这样后续训练与 benchmark 都不再直接依赖：

- `../../outputs/experiments/exp_005_map_points_generate/vigor/chicago`
- `../../dataset/Vigor/map/chicago_subset_2000`

统一后的目标是：

- 遥感图像和几何标签在同一个 dataset root 下
- provider 级别样本自包含
- 后续 metadata / manifest 从统一根目录生成
- 训练和 benchmark 的路径依赖更清晰，更便于迁移和复现

## 2. 迁移来源与目标

当前迁移脚本为：

- [scripts/migrate_vigor_chicago_rs_dataset.py](scripts/migrate_vigor_chicago_rs_dataset.py)

默认迁移关系：

- 几何标签来源：`../../outputs/experiments/exp_005_map_points_generate/vigor/chicago`
- 遥感图像来源：`../../dataset/Vigor/map/chicago_subset_2000`
- 统一目标目录：`../../traindata/vigor_chicago_rs`

也就是说，脚本会把两套源数据整合为一个统一 root。

## 3. 标准目录结构

建议采用如下结构：

```text
traindata/vigor_chicago_rs/
  README.md
  dataset_meta.json
  providers.json
  location_manifest.json

  location_1/
    map_metadata.json
    map_parameters.txt
    Google_Satellite/
      image.png
      pixel_to_point_map.npz
      valid_mask.npy
      height_map.npy
      info.json
      density_map.npy                # optional
      height_map.exr                 # optional
      occupancy_map.png              # optional
      pixel_to_point_overlay.png     # optional
      point_cloud_top_view.png       # optional
      projected_xy.npy               # optional
      projected_z.npy                # optional
      statistics.json                # optional
    Bing_Satellite/
      ...
  location_2/
    ...
```

说明：

- `location_x/` 表示一个场景。
- `location_x/<provider>/` 表示该场景下某一个遥感 provider 的完整样本。
- `image.png` 是统一命名后的遥感图像。
- `pixel_to_point_map.npz` / `valid_mask.npy` / `height_map.npy` / `info.json` 是核心训练标签。
- `map_metadata.json` 与 `map_parameters.txt` 是 scene 级别的原始地图元信息，保存在 `location_x/` 根下。

## 4. 迁移规则

脚本默认执行以下映射：

### 4.1 scene 级别元信息

从：

- `../../dataset/Vigor/map/chicago_subset_2000/location_x/map_metadata.json`
- `../../dataset/Vigor/map/chicago_subset_2000/location_x/map_parameters.txt`

迁到：

- `../../traindata/vigor_chicago_rs/location_x/map_metadata.json`
- `../../traindata/vigor_chicago_rs/location_x/map_parameters.txt`

### 4.2 provider 级别遥感图像

从：

- `../../dataset/Vigor/map/chicago_subset_2000/location_x/<provider>.png`

迁到：

- `../../traindata/vigor_chicago_rs/location_x/<provider>/image.png`

### 4.3 provider 级别几何标签

从：

- `../../outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/pixel_to_point_map.npz`
- `../../outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/valid_mask.npy`
- `../../outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/height_map.npy`
- `../../outputs/experiments/exp_005_map_points_generate/vigor/chicago/location_x/<provider>/info.json`

迁到：

- `../../traindata/vigor_chicago_rs/location_x/<provider>/pixel_to_point_map.npz`
- `../../traindata/vigor_chicago_rs/location_x/<provider>/valid_mask.npy`
- `../../traindata/vigor_chicago_rs/location_x/<provider>/height_map.npy`
- `../../traindata/vigor_chicago_rs/location_x/<provider>/info.json`

### 4.4 provider 级别可选附加文件

如果存在，下列文件也会一起迁移：

- `density_map.npy`
- `height_map.exr`
- `occupancy_map.png`
- `pixel_to_point_overlay.png`
- `point_cloud_top_view.png`
- `projected_xy.npy`
- `projected_z.npy`
- `statistics.json`

## 5. 推荐迁移方式

脚本支持三种 materialization 模式：

- `symlink`
  - 默认模式。
  - 最适合先整理目录、不想复制大文件的阶段。
- `hardlink`
  - 适合源目录与目标目录在同一文件系统上，且希望目标像普通文件一样工作。
- `copy`
  - 适合最终定版，避免后续源目录变动影响目标数据集。

建议顺序：

1. 先 `--dry-run` 检查
2. 再用 `--mode symlink` 建立统一目录
3. 确认路径与 metadata 稳定后，再用 `--mode copy --overwrite` 实体化

当前实际状态：

- 数据根已经迁到 `../../traindata/vigor_chicago_rs`
- 当前 RS 数据已经用 `copy` 模式实体化，不再依赖软链接
- `location_1_1`、`location_13_2` 这类异常 scene 不纳入正式训练数据集

## 6. 使用示例

### 6.1 全量 dry-run

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --dry-run
```

### 6.2 全量迁移为软链接

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --mode symlink
```

### 6.3 覆盖已有目标目录内容

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --mode symlink --overwrite
```

### 6.4 全量实体化为真实文件

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --mode copy --skip-missing --overwrite
```

### 6.5 只迁移前 50 个 location

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --max-locations 50 --mode symlink
```

### 6.6 只迁移指定 provider

```bash
cd Models/map-anything
python scripts/migrate_vigor_chicago_rs_dataset.py --providers Google_Satellite Bing_Satellite --mode symlink
```

## 7. 迁移产物说明

脚本会在目标目录额外生成：

- `README.md`
- `dataset_meta.json`
- `providers.json`
- `location_manifest.json`

其中：

- `dataset_meta.json` 记录源路径、目标路径、provider 列表和迁移统计。
- `providers.json` 记录 provider 级别元信息。
- `location_manifest.json` 记录每个 location 实际迁移到的 provider 条目。

## 8. 对后续代码的影响

完成迁移后，下一步建议是：

1. 将 RS benchmark / train metadata 生成逻辑改为直接读取 `../../traindata/vigor_chicago_rs`
2. 逐步去掉对 `exp_005_map_points_generate` 和原始 `Vigor/map` 的直接运行时依赖
3. 当前训练和 benchmark 已统一依赖：
   - `../../traindata/vigor_chicago_wai`
   - `../../traindata/vigor_chicago_rs`
   - `../../traindata/mapanything_metadata/...`

## 9. 当前结论

当前遥感数据已经形成统一 dataset root：`../../traindata/vigor_chicago_rs`。

当前状态：

- 数据根已从 `outputs/dataset` 迁到 `traindata`
- RS 数据已用 `copy` 模式实体化为真实文件
- RS-aerial metadata 已重建到 `../../traindata/mapanything_metadata/vigor_chicago_rs_aerial`
- 训练与 benchmark 默认都应以 `traindata` 为准

本脚本和本文档现在不只是“设计草案”，而是当前真实数据组织方式的说明。
