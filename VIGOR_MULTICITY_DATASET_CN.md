# Crossview 四城数据集组织说明

本文档描述当前已经在本机整理完成的 Crossview 四城数据集组织方式。这里的 Crossview 数据来自 VIGOR 四个城市：`chicago`、`newyork`、`sanfrancisco`、`seattle`，但在正式训练与评测文件系统中统一使用 `Crossview_*` 命名，不再把 `vigor` 写进默认数据根目录名。

## 1. 当前正式目录

当前正式数据根与 metadata 根应为：

- `../../traindata/Crossview_wai`
- `../../traindata/Crossview_rs`
- `../../traindata/mapanything_metadata/Crossview`
- `../../traindata/mapanything_metadata/Crossview_rs_aerial`

它们分别对应：

- aerial WAI 数据
- RS 数据
- train / val / test 场景划分
- RS-aerial benchmark manifest

## 2. 核心命名规则

### 2.1 scene id

四城统一后，所有正式 scene 都使用：

- `city__location_x`

例如：

- `chicago__location_1`
- `newyork__location_1`
- `sanfrancisco__location_1`
- `seattle__location_1`

这样可以避免不同城市之间的 `location_1` 冲突。

### 2.2 城市过滤

训练和评测时可以通过 `cities` 参数指定要使用哪些城市。

例如：

- `cities=[chicago,newyork]`
- `cities=[seattle]`
- `cities=[]` 表示使用全部城市

当前 loader 已支持按 `scene_name` 中的城市前缀进行过滤。

## 3. Crossview_wai

路径：

- `../../traindata/Crossview_wai`

每个 scene 的标准结构：

```text
traindata/Crossview_wai/
  chicago__location_1/
    scene_meta.json
    images/
      location_1_00.jpg
      ...
    depth/
      location_1_00.exr
      ...
    covisibility/
      v0_gtdepth_native/
        pairwise_covisibility--NxN.npy
        generation_summary.json
```

说明：

- `scene_meta.json` 是 aerial loader 的入口。
- `images/` 和 `depth/` 是每帧 RGB 与深度。
- `covisibility/v0_gtdepth_native` 是按已有 GT depth 方法生成的 pairwise covisibility。
- `scene_meta.json` 中的 `scene_modalities.pairwise_covisibility` 会指向该 mmap 文件。

当前 covisibility 生成脚本：

- [scripts/generate_vigor_chicago_gt_covisibility.py](scripts/generate_vigor_chicago_gt_covisibility.py)

当前不是沿用旧 `location_x` 目录里的 covisibility，而是对新的四城 `city__location_x` scene 按同样方法重新生成。

## 4. Crossview_rs

路径：

- `../../traindata/Crossview_rs`

每个 scene / provider 的标准结构：

```text
traindata/Crossview_rs/
  chicago__location_1/
    map_metadata.json
    map_parameters.txt
    Google_Satellite/
      image.png
      pixel_to_point_map.npz
      info.json
    Bing_Satellite/
      image.png
      pixel_to_point_map.npz
      info.json
```

当前 RS 正式契约只要求 provider 目录下存在：

- `image.png`
- `pixel_to_point_map.npz`
- `info.json`

运行时从 `pixel_to_point_map.npz['xyz']` 派生：

- `valid_mask = np.isfinite(xyz).all(axis=-1)`
- `height_map = xyz[..., 2]`

因此当前不再要求磁盘上存在：

- `valid_mask.npy`
- `height_map.npy`

当前迁移脚本：

- [scripts/migrate_vigor_chicago_rs_dataset.py](scripts/migrate_vigor_chicago_rs_dataset.py)

## 5. Crossview split metadata

路径：

- `../../traindata/mapanything_metadata/Crossview`

结构：

```text
traindata/mapanything_metadata/Crossview/
  split_summary.json
  train/
    Crossview_scene_list_train.npy
  val/
    Crossview_scene_list_val.npy
  test/
    Crossview_scene_list_test.npy
```

说明：

- 当前只保留 `Crossview_scene_list_*.npy`，不再生成旧的 `vigor_scene_list_*.npy` 或 `vigor_chicago_scene_list_*.npy`。
- 这套 split 只包含正式四城 scene，也就是 `city__location_x`。
- 当传入多个城市时，`train_scenes / val_scenes / test_scenes` 表示“每个城市”的配额，而不是所有城市合计的总配额。
- 生成逻辑会先在每个城市内部独立切分，再把各城市结果合并，因此每个城市对 train / val / test 的贡献数量相同。

当前生成脚本：

- [scripts/prepare_vigor_chicago_splits.py](scripts/prepare_vigor_chicago_splits.py)

## 6. Crossview_rs_aerial benchmark metadata

路径：

- `../../traindata/mapanything_metadata/Crossview_rs_aerial`

结构：

```text
traindata/mapanything_metadata/Crossview_rs_aerial/
  summary.json
  train/
    Crossview_rs_aerial_scene_list_train.npy
    Crossview_rs_aerial_train.json
    Crossview_rs_aerial_missing_train.json
    chicago__location_1.json
    ...
  val/
    Crossview_rs_aerial_scene_list_val.npy
    Crossview_rs_aerial_val.json
    Crossview_rs_aerial_missing_val.json
    ...
  test/
    Crossview_rs_aerial_scene_list_test.npy
    Crossview_rs_aerial_test.json
    Crossview_rs_aerial_missing_test.json
    ...
```

兼容文件也会同步写出：

- `vigor_rs_aerial_*`
- `vigor_chicago_rs_aerial_*`

说明：

- 这不是第二份图像/pointmap 数据，只是 benchmark pairing manifest。
- 每个 manifest 条目会把 aerial scene 与某个 RS provider 样本配对起来。
- benchmark 读取的是这套 metadata，再回到 `Crossview_rs` 读取真实 RS 图像与 pointmap。

当前生成脚本：

- [scripts/prepare_rs_aerial_benchmark_metadata.py](scripts/prepare_rs_aerial_benchmark_metadata.py)

## 7. 当前训练 / benchmark 读取关系

### 7.1 aerial-only / joint 训练

`VigorChicagoWAI` 和 `VigorChicagoJointRSAerial` 直接读取：

- `Crossview_wai`
- `mapanything_metadata/Crossview`

其中 joint 训练还会额外读取：

- `Crossview_rs`

### 7.2 RS-only 训练

`VigorChicagoRS` 读取：

- `Crossview_rs`
- `mapanything_metadata/Crossview`

### 7.3 RS-aerial benchmark

`VigorChicagoRSAerial` 读取：

- `mapanything_metadata/Crossview_rs_aerial`

其中 manifest 再指向：

- `Crossview_wai`
- `Crossview_rs`

## 8. 当前准备流程

### 8.1 生成 WAI 数据

```bash
python scripts/convert_vigor_chicago_to_wai.py \
  --source_parent /root/autodl-tmp/outputs/experiments/exp_001_reconstrc \
  --cities chicago newyork sanfrancisco seattle \
  --target_root /root/autodl-tmp/traindata/Crossview_wai \
  --max_locations 500 \
  --overwrite
```

### 8.2 生成 covisibility

```bash
python scripts/generate_vigor_chicago_gt_covisibility.py \
  --dataset-root /root/autodl-tmp/traindata/Crossview_wai \
  --scene-regex '.*__location_[0-9]+' \
  --device cuda
```

### 8.3 生成 split metadata

```bash
python scripts/prepare_vigor_chicago_splits.py \
  --dataset_root /root/autodl-tmp/traindata/Crossview_wai \
  --metadata_root /root/autodl-tmp/traindata/mapanything_metadata/Crossview \
  --cities chicago newyork sanfrancisco seattle \
  --train_scenes 160 \
  --val_scenes 20 \
  --test_scenes 20
```

### 8.4 迁移 RS 数据

```bash
python scripts/migrate_vigor_chicago_rs_dataset.py \
  --geometry-parent /root/autodl-tmp/outputs/experiments/exp_005_map_points_generate/vigor \
  --map-parent /root/autodl-tmp/dataset/Vigor/map \
  --cities chicago newyork sanfrancisco seattle \
  --output-root /root/autodl-tmp/traindata/Crossview_rs \
  --mode symlink \
  --overwrite
```

### 8.5 生成 benchmark metadata

```bash
python scripts/prepare_rs_aerial_benchmark_metadata.py \
  --aerial-root /root/autodl-tmp/traindata/Crossview_wai \
  --aerial-split-root /root/autodl-tmp/traindata/mapanything_metadata/Crossview \
  --remote-root /root/autodl-tmp/traindata/Crossview_rs \
  --output-root /root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial \
  --providers Google_Satellite Bing_Satellite ESRI_Satellite Yandex_Satellite OSM_Standard \
  --splits train val test
```

## 9. 清理原则

当前正式四城数据集不再保留旧的 Chicago-only legacy 目录：

- `traindata/vigor_chicago_wai/location_x`
- `traindata/vigor_chicago_rs/location_x`

也不再把正式训练 / benchmark 默认指向：

- `traindata/vigor_chicago_wai`
- `traindata/vigor_chicago_rs`
- `traindata/mapanything_metadata/vigor_chicago`
- `traindata/mapanything_metadata/vigor_chicago_rs_aerial`

如果保留这些旧目录，只应视为迁移前遗留，不应再被新配置默认使用。

## 10. 相关文件

- [scripts/convert_vigor_chicago_to_wai.py](scripts/convert_vigor_chicago_to_wai.py)
- [scripts/generate_vigor_chicago_gt_covisibility.py](scripts/generate_vigor_chicago_gt_covisibility.py)
- [scripts/prepare_vigor_chicago_splits.py](scripts/prepare_vigor_chicago_splits.py)
- [scripts/migrate_vigor_chicago_rs_dataset.py](scripts/migrate_vigor_chicago_rs_dataset.py)
- [scripts/prepare_rs_aerial_benchmark_metadata.py](scripts/prepare_rs_aerial_benchmark_metadata.py)
- [mapanything/datasets/wai/vigor_chicago.py](mapanything/datasets/wai/vigor_chicago.py)
- [mapanything/datasets/wai/vigor_chicago_rs.py](mapanything/datasets/wai/vigor_chicago_rs.py)
- [mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py](mapanything/datasets/wai/vigor_chicago_joint_rs_aerial.py)
- [mapanything/datasets/wai/vigor_chicago_rs_aerial.py](mapanything/datasets/wai/vigor_chicago_rs_aerial.py)
