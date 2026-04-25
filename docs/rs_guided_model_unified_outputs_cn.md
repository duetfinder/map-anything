# RS Guided Dense MV 模型统一输出整理

本文整理 `Models/map-anything/benchmarking/rs_guided_dense_mv` 评测里常用的 4 个模型：

- `pi3`
- `mapanything`
- `da3`（Depth Anything 3）
- `vggt`

重点不是重复统一输出字段名，而是说明：

1. 每个模型原生先预测了什么
2. 这些原生输出如何被 wrapper / `infer()` 后处理转换为 MapAnything 统一输出
3. 哪些字段是核心统一输出，哪些是常用扩展输出
4. 模型本体内部到底是“直接解码”了哪些量，哪些量是后续组合出来的

## 1. 统一输出接口总览

MapAnything 框架下，模型统一返回：

- `List[Dict]`
- 列表长度 = 输入 view 数
- 每个 `Dict` 对应一个 view 的预测结果

文档中明确列出的核心统一输出为：

| 统一字段 | 含义 |
|---|---|
| `pts3d` | 世界坐标系下的 3D 点 |
| `pts3d_cam` | 相机坐标系下的 3D 点 |
| `ray_directions` | 相机坐标系下的单位射线方向 |
| `depth_along_ray` | 沿射线方向的深度 |
| `cam_trans` | `cam2world` 平移 |
| `cam_quats` | `cam2world` 四元数 |
| `conf` | 像素级置信度 |

参考：

- [README.md](/root/autodl-tmp/Models/map-anything/README.md:504)
- [README.md](/root/autodl-tmp/Models/map-anything/README.md:475)

## 2. 常用扩展输出字段

对 `MapAnything.infer()` 而言，后处理还会补出一批常用字段：

| 扩展字段 | 典型来源 |
|---|---|
| `img_no_norm` | 由输入图像反归一化得到 |
| `depth_z` | 由 `pts3d_cam[..., 2:3]` 得到 |
| `intrinsics` | 由 `ray_directions` 反求 pinhole 内参 |
| `camera_poses` | 由 `cam_quats + cam_trans` 组装成 4x4 `cam2world` |
| `metric_scaling_factor` | 仅对带 scale head 的模型/模式存在 |
| `non_ambiguous_mask` | 模型原生 mask head 输出 |
| `non_ambiguous_mask_logits` | 模型原生 mask logits |
| `mask` | 后处理将 ambiguity mask、confidence mask、edge mask 合并后的最终 mask |

参考：

- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:2087)
- [mapanything/utils/inference.py](/root/autodl-tmp/Models/map-anything/mapanything/utils/inference.py:370)

## 3. 四个模型的统一输出生成方式对比

### 3.1 按模型看“先预测什么”

| 模型 | 当前配置入口 | 原生/近原生预测重点 | 统一输出的构造特点 |
|---|---|---|---|
| `pi3` | [configs/model/pi3.yaml](/root/autodl-tmp/Models/map-anything/configs/model/pi3.yaml:1) | 模型本体输出 `camera_poses`、`local_points`、`points`、`conf` | `local_points` 是直接解码；`points` 是模型内部用 `camera_poses` 变换 `local_points` 得到 |
| `mapanything` | [configs/model/mapanything.yaml](/root/autodl-tmp/Models/map-anything/configs/model/mapanything.yaml:1) | 取决于 `scene_rep_type`；当前默认是 `raydirs+depth+pose+confidence+mask+scale` | dense / pose / scale 三条分支分别解码，再在 `forward()` 末尾统一映射为公共字段 |
| `da3` | [configs/model/da3.yaml](/root/autodl-tmp/Models/map-anything/configs/model/da3.yaml:1) | 当前 wrapper 实际消费的是 `depth`、`depth_conf`、`extrinsics`、`intrinsics` | 不直接给世界系点图；且当前 benchmark 走的是 `use_ray_pose=True`，位姿/内参来自 ray 分支而不是 camera decoder 默认分支 |
| `vggt` | [configs/model/vggt.yaml](/root/autodl-tmp/Models/map-anything/configs/model/vggt.yaml:1) | 模型本体同时挂 `camera_head`、`depth_head`、`point_head` | 当前 wrapper 只使用 `camera_head + depth_head`，没有直接消费原生 `world_points` |

### 3.2 按统一字段看“是直接输出还是推导出来”

| 统一字段 | Pi3 | MapAnything | DA3 | VGGT |
|---|---|---|---|---|
| `pts3d` | 直接取 `results["points"]` | 视 `scene_rep_type` 而定；可直接预测，也可由射线/深度/位姿组合得到 | 由 `ray_directions + depth_along_ray + cam pose` 组合得到 | 由 `ray_directions + depth_along_ray + cam pose` 组合得到 |
| `pts3d_cam` | 直接取 `results["local_points"]` | 视 `scene_rep_type` 而定；可直接预测，也可由 `ray_directions * depth_along_ray` 得到 | 由 `depth_z + intrinsics -> depthmap_to_camera_frame` 得到 | 由 `depth_z + intrinsics -> depthmap_to_camera_frame` 得到 |
| `ray_directions` | 由 `pts3d_cam / ||pts3d_cam||` 得到 | 视模式而定；可能直接预测，也可能由 `pts3d_cam` 归一化得到 | 由 `intrinsics` 调 `get_rays_in_camera_frame()` 得到 | 由 `intrinsics` 调 `get_rays_in_camera_frame()` 得到 |
| `depth_along_ray` | 由 `||pts3d_cam||` 得到 | 视模式而定；可能直接预测，也可能由 `||pts3d_cam||` 得到 | 由 `depth_z + intrinsics` 转换得到 | 由 `depth_z + intrinsics` 转换得到 |
| `cam_trans` | 由 `camera_poses` 拆出来 | 若模式带 pose，则直接由 pose head 给出 | 由 `extrinsics` 先转成 `cam2world`，再拆平移 | 由 `extrinsic` 先转成 `cam2world`，再拆平移 |
| `cam_quats` | 由 `camera_poses` 旋转矩阵转四元数 | 若模式带 pose，则直接由 pose head 给出 | 由 `extrinsics` 先转成 `cam2world`，再矩阵转四元数 | 由 `extrinsic` 先转成 `cam2world`，再矩阵转四元数 |
| `conf` | 直接取 `results["conf"]` | 若模式带 confidence，则由 confidence head 给出 | 直接取 `results["depth_conf"]` | 直接取 `depth_conf` |
| `metric_scaling_factor` | 无 | 默认配置下有 scale head | 无 | 无 |
| `non_ambiguous_mask` | 无 | 默认配置下有 mask head | 无 | 无 |
| `mask` | 仅在走 `infer()` 且启用 masking 时由后处理生成 | `infer()` 后处理生成 | 通常 wrapper `forward()` 无；若经统一 `infer()` 后处理则可生成 | 通常 wrapper `forward()` 无；若经统一 `infer()` 后处理则可生成 |

## 4. 各模型详细拆解

## 4.1 Pi3

### 4.1.1 wrapper 输入与主干输出

Pi3 wrapper 直接把多视图图像堆成 `(B, V, C, H, W)`，调用底层 `Pi3` 模型：

- `results["camera_poses"]`
- `results["local_points"]`
- `results["points"]`
- `results["conf"]`

参考：

- [mapanything/models/external/pi3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/pi3/__init__.py:82)

### 4.1.2 统一字段是怎么来的

| 统一字段 | Pi3 中的来源/转换 |
|---|---|
| `pts3d` | 直接取 `results["points"][:, view_idx, ...]` |
| `pts3d_cam` | 直接取 `results["local_points"][:, view_idx, ...]` |
| `depth_along_ray` | `torch.norm(pts3d_cam, dim=-1, keepdim=True)` |
| `ray_directions` | `pts3d_cam / depth_along_ray` |
| `cam_trans` | `camera_poses[..., :3, 3]` |
| `cam_quats` | `mat_to_quat(camera_poses[..., :3, :3])` |
| `conf` | 直接取 `results["conf"][:, view_idx, ...]` |

参考：

- [mapanything/models/external/pi3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/pi3/__init__.py:94)

### 4.1.3 模型本体里这些量是怎么组织出来的

Pi3 模型本体并不是“两个独立点头分别直接输出 `local_points` 和 `points`”。

真实组织方式是：

1. 共享 encoder + decoder 产生 token 表示
2. `point_decoder + point_head` 直接解码局部相机坐标下的点
3. `camera_decoder + camera_head` 直接解码每个 view 的 `camera_poses`
4. `conf_decoder + conf_head` 直接解码 `conf`
5. `points` 不是独立点头回归，而是用 `camera_poses` 对 `local_points` 做齐次变换后得到

源码证据：

- `local_points` 由 `point_head(...)` 产生，之后把 `(xy, z)` 组装成 `(x, y, z)`，其中 `z = exp(z)`，`xy` 先乘 `z`，见 [pi3.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/pi3/models/pi3.py:219)
- `camera_poses` 由 `camera_head(...)` 产生，见 [pi3.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/pi3/models/pi3.py:235)
- `points` 通过 `camera_poses @ homogenize_points(local_points)` 得到，见 [pi3.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/pi3/models/pi3.py:241)

因此更准确的说法是：

- 直接解码的几何量是 `local_points`
- 直接解码的相机量是 `camera_poses`
- 全局 `points` 是模型内部基于这两者进一步计算出来的

### 4.1.4 这一类模型的特点

Pi3 属于“已经原生提供世界系点图和局部相机系点图”的模型，因此统一接口转换最直接。`pts3d` 不需要通过位姿再算一次，`pts3d_cam` 也不是由深度反投影恢复，而是直接来自模型输出。

## 4.2 MapAnything

### 4.2.1 当前默认配置走哪条输出路径

当前默认配置：

- `pred_head: dpt_pose_scale`
- 其 adaptor config 是 `raydirs_depth_pose_confidence_mask_scale`

这意味着默认 MapAnything 实际预测的是：

- `ray_directions`
- `depth`
- `pose`
- `confidence`
- `mask`
- `metric scale`

对应配置：

- [configs/model/mapanything.yaml](/root/autodl-tmp/Models/map-anything/configs/model/mapanything.yaml:1)
- [configs/model/pred_head/dpt_pose_scale.yaml](/root/autodl-tmp/Models/map-anything/configs/model/pred_head/dpt_pose_scale.yaml:1)
- [configs/model/pred_head/adaptor_config/raydirs_depth_pose_confidence_mask_scale.yaml](/root/autodl-tmp/Models/map-anything/configs/model/pred_head/adaptor_config/raydirs_depth_pose_confidence_mask_scale.yaml:1)

### 4.2.2 为什么 MapAnything 的统一输出要分模式讨论

`MapAnything` 本体不是只有一种输出形式。它内部先预测 `scene_rep_type`，再在 `forward()` 末尾统一组装成公共字段。代码里支持的主模式包括：

- `pointmap`
- `raymap+depth`
- `raydirs+depth+pose`
- `campointmap+pose`
- `pointmap+raydirs+depth+pose`

以及它们的 `+confidence` / `+mask` 变体。

参考：

- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:1670)

### 4.2.3 不同 `scene_rep_type` 到统一字段的映射

| `scene_rep_type` | 模型先预测什么 | `pts3d` 如何得到 | `pts3d_cam` 如何得到 | 备注 |
|---|---|---|---|---|
| `pointmap` | 直接预测世界系点图 | 直接取 dense output | 不一定有 | 只保证 `pts3d`，再乘 `metric_scaling_factor` |
| `raymap+depth` | 射线原点 + 射线方向 + 深度 | `ray_origins + ray_directions * depth_along_ray` | 不一定有 | 这是世界系 ray 表达 |
| `raydirs+depth+pose` | 相机系 ray dir + depth + pose | `convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(...)` | `ray_directions * depth_along_ray` | 默认配置对应这一类 |
| `campointmap+pose` | 相机系点图 + pose | 先从 `pts3d_cam` 求 ray/depth，再结合 pose 变到世界系 | 直接取 dense output | 典型“局部点图 + 位姿”路线 |
| `pointmap+raydirs+depth+pose` | 同时预测世界系点图、ray/深度、pose | 默认直接用预测的 `pts3d`；若 `use_factored_predictions_for_global_pointmaps=True`，则改用 ray/depth/pose 重建 | `ray_directions * depth_along_ray` | 同时拥有直接世界点和分解式世界点 |

参考：

- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:1670)

### 4.2.4 默认 benchmark 配置下的核心输出生成

对当前默认 `raydirs+depth+pose+confidence+mask+scale` 配置，统一字段生成链条是：

1. dense branch 预测 `ray_directions` 与 `depth_along_ray`
2. pose branch 预测 `cam_trans` 与 `cam_quats`
3. scale head 预测 `metric_scaling_factor`
4. 世界系点图通过
   `convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(...)`
   重建
5. 相机系点图通过
   `ray_directions * depth_along_ray`
   得到
6. confidence 与 mask 直接来自对应 head

参考：

- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:1733)
- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:1923)

### 4.2.5 模型本体里这些量是怎么组织出来的

`MapAnything` 本体内部不是一个单头输出结构，而是拆成了三条分支：

- `dense head`
- `pose head`
- `scale head`

从 `downstream_head()` 可以直接看出：

1. `downstream_dense_head(...)` 负责输出 dense scene representation
2. 如果 `pred_head_type == "dpt+pose"`，则 `pose_head(...) + pose_adaptor(...)` 单独输出位姿表示
3. `scale_head(...) + scale_adaptor(...)` 单独输出尺度因子

参考：

- [mapanything/models/mapanything/model.py](/root/autodl-tmp/Models/map-anything/mapanything/models/mapanything/model.py:1369)

对当前默认配置 `raydirs+depth+pose+confidence+mask+scale`，根据 config 和 `forward()` 末端拼装逻辑，可以确认：

- dense head 直接输出的是 `ray_directions + depth (+ confidence + mask logits)`
- pose head 直接输出的是 `cam_trans + cam_quats`
- scale head 直接输出的是 `metric_scaling_factor`
- `pts3d_cam` 不是独立 head 直接回归，而是 `ray_directions * depth_along_ray`
- `pts3d` 也不是独立 head 直接回归，而是由 `ray_directions + depth_along_ray + cam_trans + cam_quats` 组合得到

所以默认配置下，MapAnything 更像是“分解式几何预测模型”。

但如果切换 `scene_rep_type`，结论会变化：

- `pointmap` 模式里，`pts3d` 是 dense head 直接输出
- `campointmap+pose` 模式里，`pts3d_cam` 是 dense head 直接输出，`pts3d` 再由 pose 变换得到
- `pointmap+raydirs+depth+pose` 模式里，dense head 会同时给出世界点图和分解式几何量

也就是说，对 MapAnything 必须同时区分：

- 当前配置让 head 直接解码了什么
- 统一输出阶段又把这些中间量组合成了什么

### 4.2.6 `infer()` 后处理补出的扩展字段

`MapAnything.infer()` 不是简单返回 `forward()` 原始输出，它还会统一补字段：

| 扩展字段 | 派生方式 |
|---|---|
| `img_no_norm` | 输入图像按 `data_norm_type` 反归一化 |
| `depth_z` | `pts3d_cam[..., 2:3]` |
| `intrinsics` | `recover_pinhole_intrinsics_from_ray_directions(ray_directions)` |
| `camera_poses` | `cam_quats` 转旋转矩阵后，与 `cam_trans` 拼成 4x4 |
| `mask` | 在 `non_ambiguous_mask` 基础上，再按 confidence percentile 和 edge mask 叠加 |

参考：

- [mapanything/utils/inference.py](/root/autodl-tmp/Models/map-anything/mapanything/utils/inference.py:370)
- [mapanything/utils/inference.py](/root/autodl-tmp/Models/map-anything/mapanything/utils/inference.py:493)

## 4.3 DA3（Depth Anything 3）

### 4.3.1 wrapper 实际拿到的原始输出

DA3 wrapper 调 `DepthAnything3.from_pretrained(...)` 后，在 `forward()` 中读取：

- `results["extrinsics"]`
- `results["intrinsics"]`
- `results["depth"]`
- `results["depth_conf"]`

参考：

- [mapanything/models/external/da3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/da3/__init__.py:121)
- [mapanything/models/external/da3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/da3/__init__.py:134)

需要注意的是，wrapper 调底层 DA3 时显式传了：

- `use_ray_pose=True`

也就是说，当前 `mapanything` 仓库中的 DA3 benchmark 路线，不是 DA3 的默认 camera decoder 路线，而是 ray-based pose 路线。

参考：

- [mapanything/models/external/da3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/da3/__init__.py:121)

### 4.3.2 统一字段生成链条

DA3 不直接输出统一世界坐标系点图。它走的是“深度 + 内参 + 外参恢复”的路线：

| 统一字段 | DA3 中的来源/转换 |
|---|---|
| `cam_trans` | `results["extrinsics"]` 先经 `closed_form_inverse_se3()` 变成 `cam2world`，再取平移 |
| `cam_quats` | 同上，旋转部分经 `mat_to_quat()` |
| `pts3d_cam` | `depthmap_to_camera_frame(depth_z, intrinsics)` |
| `depth_along_ray` | `convert_z_depth_to_depth_along_ray(depth_z, intrinsics)` |
| `ray_directions` | `get_rays_in_camera_frame(intrinsics, H, W, normalize_to_unit_sphere=True)` |
| `pts3d` | `convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(ray_dirs, depth_along_ray, cam_trans, cam_quats)` |
| `conf` | 直接取 `results["depth_conf"]` |

参考：

- [mapanything/models/external/da3/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/da3/__init__.py:135)

### 4.3.3 模型本体里这些量是怎么组织出来的

现在可以基于 `Models/Depth-Anything-3` 源码把这一层说清楚。

### 4.3.3.1 DA3 本体的总结构

`DepthAnything3` API 下挂的核心网络是 `DepthAnything3Net`。它的主要组成是：

- `backbone`
- `head`
- 可选 `cam_dec`
- 可选 `cam_enc`
- 可选 `gs_head` / `gs_adapter`

源码与注释都明确写了：

- `head` 负责 dense prediction
- `cam_dec` / `cam_enc` 负责 camera estimation

参考：

- [api.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/api.py:48)
- [da3.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/da3.py:40)

### 4.3.3.2 默认 DA3 配置的 dense head 是什么

默认 `da3-giant` 配置里：

- `head` 是 `DualDPT`
- `cam_dec` 是 `CameraDec`
- `cam_enc` 是 `CameraEnc`

参考：

- [da3-giant.yaml](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/configs/da3-giant.yaml:1)

而 `DualDPT` 的 `head_names` 默认是：

- `("depth", "ray")`

这意味着它本体会直接输出两组 dense 量：

- 主头：`depth` 与 `depth_conf`
- 辅头：`ray` 与 `ray_conf`

参考：

- [dualdpt.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/dualdpt.py:30)

### 4.3.3.3 这些量是怎么直接解码出来的

`DualDPT` 不是先出一个统一 feature 再外部拆，而是内部就有：

1. 一条 main head，输出 `depth` 和 `depth_conf`
2. 一条 auxiliary head，输出 `ray` 和 `ray_conf`

对应代码：

- main head 最终返回 `self.head_main` 与 `f"{self.head_main}_conf"`，即 `depth` / `depth_conf`
- auxiliary head 最终返回 `self.head_aux` 与 `f"{self.head_aux}_conf"`，即 `ray` / `ray_conf`

参考：

- [dualdpt.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/dualdpt.py:242)

所以，至少在当前默认配置下，DA3 本体里“直接解码”的 dense 几何量其实是：

- `depth`
- `depth_conf`
- `ray`
- `ray_conf`

### 4.3.3.4 `extrinsics` / `intrinsics` 是怎么来的

这里要分两条路径。

第一条是默认 camera decoder 路线：

1. `cam_dec` 直接从特征解码一个 9 维 `pose_enc`
2. 其中包含 `T(3) + quat(4) + FoV(2)`
3. `pose_encoding_to_extri_intri()` 再把这 9 维量还原成相机外参与内参

参考：

- [cam_dec.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/cam_dec.py:19)
- [transform.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/utils/transform.py:41)

第二条是 ray pose 路线：

1. 先由 `DualDPT` 直接输出 `ray` 与 `ray_conf`
2. `get_extrinsic_from_camray(...)` 根据 ray map 反推出外参、焦距、主点
3. 再组装成 `pred_extrinsic` 和 `pred_intrinsic`

参考：

- [da3.py](/root/autodl-tmp/Models/Depth-Anything-3/src/depth_anything_3/model/da3.py:181)

### 4.3.3.5 当前 `mapanything` wrapper 实际走的是哪条

当前 `DA3Wrapper` 显式设置：

- `use_ray_pose=True`

因此在你当前 benchmark 里，DA3 的 `extrinsics` / `intrinsics` 不是由 `cam_dec` 直接解码出来的，而是：

- 先由 `DualDPT` 直接输出 `ray` / `ray_conf`
- 再通过 `get_extrinsic_from_camray(...)` 推回去

所以当前 benchmark 路线下，DA3 更准确的原生输出组织应该写成：

- 直接 dense 解码：`depth`、`depth_conf`、`ray`、`ray_conf`
- 再由 ray 分支恢复：`extrinsics`、`intrinsics`
- 再由 wrapper 恢复统一输出：`pts3d_cam`、`ray_directions`、`depth_along_ray`、`pts3d`、`cam_trans`、`cam_quats`

### 4.3.3.6 DA3 README 文档层面的原生输出

DA3 官方 README 对外承诺的 prediction 字段是：

- `depth`
- `conf`
- `extrinsics`
- `intrinsics`

参考：

- [Depth-Anything-3/README.md](/root/autodl-tmp/Models/Depth-Anything-3/README.md:108)

### 4.3.4 这一类模型的特点

DA3 更像“预测 dense depth/ray 与相机参数”的模型，而不是“直接输出统一点图”的模型。它的 `pts3d` 是 wrapper 后算出来的，不是底层网络直接回归出的世界系点图。

## 4.4 VGGT

### 4.4.1 wrapper 实际拿到的原始输出

VGGT wrapper 先调用 aggregator，再分两条支路：

- camera branch 预测 `pose_enc`
- 通过 `pose_encoding_to_extri_intri()` 得到 `extrinsic` 和 `intrinsic`
- depth head 预测 `depth_map` 与 `depth_conf`

参考：

- [mapanything/models/external/vggt/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/__init__.py:113)
- [mapanything/models/external/vggt/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/__init__.py:119)
- [mapanything/models/external/vggt/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/__init__.py:128)

### 4.4.2 统一字段生成链条

VGGT 与 DA3 非常相似，也是先拿深度和相机参数，再重建统一字段：

| 统一字段 | VGGT 中的来源/转换 |
|---|---|
| `cam_trans` | `extrinsic` 先经 `closed_form_inverse_se3()` 转成 `cam2world`，再取平移 |
| `cam_quats` | 同上，旋转矩阵转四元数 |
| `pts3d_cam` | `depthmap_to_camera_frame(depth_z, intrinsics)` |
| `depth_along_ray` | `convert_z_depth_to_depth_along_ray(depth_z, intrinsics)` |
| `ray_directions` | `get_rays_in_camera_frame(intrinsics, H, W, normalize_to_unit_sphere=True)` |
| `pts3d` | `convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(...)` |
| `conf` | 直接取 `depth_conf` |

参考：

- [mapanything/models/external/vggt/__init__.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/__init__.py:137)

### 4.4.3 模型本体里这些量是怎么组织出来的

VGGT 本体比当前 wrapper 使用得更多。

在模型定义里，它同时挂了：

- `camera_head`
- `point_head`
- `depth_head`
- `track_head`

参考：

- [mapanything/models/external/vggt/models/vggt.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/models/vggt.py:29)

其 `forward()` 原生会产出：

- `pose_enc`
- `depth`
- `depth_conf`
- `world_points`
- `world_points_conf`

参考：

- [mapanything/models/external/vggt/models/vggt.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/models/vggt.py:68)

这意味着：

1. VGGT 本体其实有一条原生 `point_head`，可以直接输出 `world_points`
2. 也有一条原生 `depth_head`，可以直接输出 `depth + depth_conf`
3. 还有一条 `camera_head`，先输出 `pose_enc`

其中：

- `camera_head` 直接解码的是 9 维 `pose encoding = [T(3), quat(4), FoV(2)]`，见 [camera_head.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/heads/camera_head.py:16)
- `pose_encoding_to_extri_intri()` 再把这 9 维量还原成 `extrinsics` 和 `intrinsics`，其中焦距由 FoV 恢复，主点固定在图像中心，见 [pose_enc.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/utils/pose_enc.py:68)
- `DPTHead` 会直接把 token 解码成 dense prediction + confidence，两者都来自同一个 head 输出通道经 `activate_head(...)` 分拆，见 [dpt_head.py](/root/autodl-tmp/Models/map-anything/mapanything/models/external/vggt/heads/dpt_head.py:291)

但当前 `mapanything` 仓库里的 `VGGTWrapper` 没有使用 `world_points` 这条原生点头，而是选择：

1. 用 `camera_head` 得到相机参数
2. 用 `depth_head` 得到 `depth_map`
3. 再自行重建 `pts3d_cam`、`ray_directions`、`depth_along_ray`、`pts3d`

所以当前 benchmark 里用到的 VGGT 路线，并不是“直接消费原生世界点图”，而是“消费原生深度和相机参数，再转成统一输出”。

### 4.4.4 这一类模型的特点

VGGT 与 DA3 的统一输出逻辑几乎同构。区别主要在于它的相机参数来自 `camera_head + pose_encoding_to_extri_intri()`，而不是直接从 DA3 API 返回。

## 5. 从“统一输出生成难度”看四类模型

可以把这 4 个模型粗分成 3 类：

| 类别 | 模型 | 统一输出构造复杂度 | 原因 |
|---|---|---|---|
| 局部点图+位姿变换型 | `pi3` | 最低 | 直接解码 `local_points` 与 `camera_poses`，再在模型内部得到 `points` |
| 深度+相机恢复型 | `da3`, `vggt` | 中等 | 先有深度、内参、外参，再通过几何函数恢复点图与射线 |
| 内部表征可切换型 | `mapanything` | 最高 | 不同 `scene_rep_type` 下原始输出语义不同，需要在 `forward()` 末尾统一组装 |

## 6. 对 rs_guided_dense_mv 评测最有用的结论

如果你的关注点是 benchmark 里这些模型“最终为什么都能喂给同一套评测”，最重要的是下面这几条：

1. `pi3` 虽然对外同时提供世界系点图和相机系点图，但其中 `points` 不是第二个独立点头直接解码，而是模型内部由 `camera_poses` 变换 `local_points` 得到。
2. `da3` 并不原生向当前 benchmark 提供统一世界系点图；它是通过深度、内参、外参在 wrapper 中重建出 `pts3d`、`pts3d_cam`、`ray_directions`、`depth_along_ray`、`cam_trans`、`cam_quats`。
3. `vggt` 模型本体其实具备原生 `world_points` 输出能力，但当前 wrapper 没有走这条分支，而是和 `da3` 类似，使用“深度 + 相机参数 -> 几何重建”的路线。
4. `mapanything` 更像一个“统一输出生成器”：其内部可预测多种 scene representation，但都会在 `forward()` 中折叠成同一套公共字段；默认配置下它直接解码的是分解式几何量，而不是直接全局点图。
5. 对 benchmark 最关键的公共字段仍然是：
   `pts3d`、`pts3d_cam`、`ray_directions`、`depth_along_ray`、`cam_trans`、`cam_quats`、`conf`。
6. 对分析和可视化很有帮助的扩展字段主要是：
   `depth_z`、`intrinsics`、`camera_poses`、`metric_scaling_factor`、`non_ambiguous_mask`、`mask`、`img_no_norm`。

## 7. 可继续补充的方向

如果后续你想把这份文档继续做成“训练/评测排错手册”，建议再补两张表：

1. “各模型输出字段的张量 shape 对照表”
2. “各模型统一字段的坐标系语义对照表”
   例如：`pts3d` 是否已经在预测 `view0` 参考系、世界系定义是谁、`cam2world` 是否总是 OpenCV convention 等。
