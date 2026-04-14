# Crossview / VIGOR 四城 RS Dataset

这份文档对应的是早期 Chicago-only RS 数据整理记录，已经不是当前正式的数据集组织说明。

当前请优先查看：

- [VIGOR_MULTICITY_DATASET_CN.md](VIGOR_MULTICITY_DATASET_CN.md)

当前正式默认目录已经统一为：

- `../../traindata/Crossview_wai`
- `../../traindata/Crossview_rs`
- `../../traindata/mapanything_metadata/Crossview`
- `../../traindata/mapanything_metadata/Crossview_rs_aerial`

并且当前正式 scene id 统一为：

- `city__location_x`

例如：

- `chicago__location_1`
- `newyork__location_1`

当前 RS 数据契约也已经切换为 pointmap-only：

- `image.png`
- `pixel_to_point_map.npz`
- `info.json`

`valid_mask` 与 `height_map` 都在运行时从 `pixel_to_point_map.npz['xyz']` 派生，不再要求单独落盘。
