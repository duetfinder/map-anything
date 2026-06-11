# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything training wrapper for VGGT-Omega.
"""

import torch

from mapanything.models.external.vggt_omega.models import VGGTOmega
from mapanything.models.external.vggt_omega.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt_omega.utils.pose_enc import encoding_to_camera
from mapanything.models.external.vggt_omega.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
)


class _ProjectionAuxResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = torch.nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class VGGTOmegaWrapper(torch.nn.Module):
    """Expose VGGT-Omega through the MapAnything multi-view training interface."""

    def __init__(
        self,
        name,
        torch_hub_force_reload=False,
        load_pretrained_weights=False,
        load_custom_ckpt=False,
        custom_ckpt_path=None,
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        remote_instance_value="remote",
        ordinary_output_head="depth",
        remote_output_head="depth",
        use_remote_projection_aux_head=False,
        remote_projection_aux_hidden_dim=96,
        remote_projection_aux_use_rgb=True,
        remote_projection_aux_use_coord=True,
        remote_projection_aux_image_stem_dim=32,
        remote_projection_aux_positive_slope=True,
        remote_projection_aux_slope_init=0.1,
        remote_projection_aux_num_blocks=6,
        remote_projection_aux_rel_height_output_scale=1.0,
        remote_projection_aux_offset_output_scale=1.0,
        strict_ckpt_load=True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported VGGTOmegaWrapper options: {unknown}")
        if load_pretrained_weights and not load_custom_ckpt:
            raise ValueError(
                "VGGT-Omega does not support automatic pretrained download here. "
                "Set load_custom_ckpt=true and custom_ckpt_path to a local checkpoint."
            )

        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path
        self.remote_instance_value = remote_instance_value
        self.ordinary_output_head = ordinary_output_head
        self.remote_output_head = remote_output_head
        self.embed_dim = embed_dim
        self.use_remote_projection_aux_head = bool(use_remote_projection_aux_head)
        self.remote_projection_aux_hidden_dim = int(remote_projection_aux_hidden_dim)
        self.remote_projection_aux_use_rgb = bool(remote_projection_aux_use_rgb)
        self.remote_projection_aux_use_coord = bool(remote_projection_aux_use_coord)
        self.remote_projection_aux_image_stem_dim = int(remote_projection_aux_image_stem_dim)
        self.remote_projection_aux_positive_slope = bool(remote_projection_aux_positive_slope)
        self.remote_projection_aux_slope_init = float(remote_projection_aux_slope_init)
        self.remote_projection_aux_num_blocks = int(remote_projection_aux_num_blocks)
        self.remote_projection_aux_rel_height_output_scale = float(
            remote_projection_aux_rel_height_output_scale
        )
        self.remote_projection_aux_offset_output_scale = float(
            remote_projection_aux_offset_output_scale
        )

        self.model = VGGTOmega(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_depth=enable_depth,
            enable_alignment=enable_alignment,
        )

        if self.use_remote_projection_aux_head:
            hidden_dim = max(1, self.remote_projection_aux_hidden_dim)
            token_dim = 2 * self.embed_dim
            self.remote_projection_aux_token_norm = torch.nn.LayerNorm(token_dim)
            self.remote_projection_aux_token_proj = torch.nn.Linear(token_dim, hidden_dim)

            aux_pixel_in_channels = hidden_dim
            if self.remote_projection_aux_use_rgb:
                aux_pixel_in_channels += 3
            if self.remote_projection_aux_use_coord:
                aux_pixel_in_channels += 2
            if self.remote_projection_aux_image_stem_dim > 0:
                aux_pixel_in_channels += self.remote_projection_aux_image_stem_dim
                self.remote_projection_aux_image_stem = torch.nn.Sequential(
                    torch.nn.Conv2d(3, self.remote_projection_aux_image_stem_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                    torch.nn.Conv2d(
                        self.remote_projection_aux_image_stem_dim,
                        self.remote_projection_aux_image_stem_dim,
                        kernel_size=3,
                        padding=1,
                    ),
                    torch.nn.GELU(),
                )

            pixel_layers = [
                torch.nn.Conv2d(aux_pixel_in_channels, hidden_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
            ]
            pixel_layers.extend(
                _ProjectionAuxResidualBlock(hidden_dim)
                for _ in range(self.remote_projection_aux_num_blocks)
            )
            pixel_layers.append(torch.nn.Conv2d(hidden_dim, 3, kernel_size=1))
            self.remote_projection_aux_token_pixel_head = torch.nn.Sequential(*pixel_layers)
            self.remote_projection_aux_token_global_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 3),
            )
            if self.remote_projection_aux_positive_slope and self.remote_projection_aux_slope_init > 0:
                final_linear = self.remote_projection_aux_token_global_head[-1]
                slope_init = torch.tensor(
                    self.remote_projection_aux_slope_init,
                    dtype=final_linear.bias.dtype,
                )
                raw_slope_init = torch.log(torch.expm1(slope_init).clamp_min(1e-6))
                with torch.no_grad():
                    final_linear.weight[2].zero_()
                    final_linear.bias[2].copy_(raw_slope_init)

        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        if self.load_custom_ckpt:
            if self.custom_ckpt_path is None:
                raise ValueError("custom_ckpt_path must be set when load_custom_ckpt=true")
            print(f"Loading VGGT-Omega checkpoint from {self.custom_ckpt_path} ...")
            checkpoint = torch.load(self.custom_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)
            load_result = self.model.load_state_dict(state_dict, strict=bool(strict_ckpt_load))
            print(load_result)
            del checkpoint

    def _projection_aux_add_image_features(self, aux_chw, source_image):
        if self.remote_projection_aux_use_rgb:
            if source_image is None:
                raise RuntimeError("remote_projection_aux_use_rgb=True requires source_image")
            aux_chw = torch.cat(
                [aux_chw, source_image.to(device=aux_chw.device, dtype=aux_chw.dtype)],
                dim=1,
            )
        if self.remote_projection_aux_use_coord:
            _, _, height, width = aux_chw.shape
            y = torch.linspace(-1.0, 1.0, height, device=aux_chw.device, dtype=aux_chw.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=aux_chw.device, dtype=aux_chw.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(aux_chw.shape[0], -1, -1, -1)
            aux_chw = torch.cat([aux_chw, coord], dim=1)
        if self.remote_projection_aux_image_stem_dim > 0:
            if source_image is None:
                raise RuntimeError("remote_projection_aux_image_stem_dim>0 requires source_image")
            image_for_stem = source_image.to(device=aux_chw.device, dtype=aux_chw.dtype)
            aux_chw = torch.cat([aux_chw, self.remote_projection_aux_image_stem(image_for_stem)], dim=1)
        return aux_chw

    def _apply_remote_projection_aux_token_head(self, output, source_tokens, source_image, image_shape):
        if not self.use_remote_projection_aux_head:
            return output
        if source_tokens is None:
            raise RuntimeError("remote projection aux requires source_tokens")

        height, width = int(image_shape[0]), int(image_shape[1])
        patch_size = int(getattr(self.model.aggregator, "patch_size", 16))
        grid_h = max(1, height // patch_size)
        grid_w = max(1, width // patch_size)
        if source_tokens.shape[1] != grid_h * grid_w:
            grid_h = int(round(source_tokens.shape[1] ** 0.5))
            grid_w = source_tokens.shape[1] // max(1, grid_h)
            if grid_h * grid_w != source_tokens.shape[1]:
                raise RuntimeError(
                    "Cannot reshape VGGT-Omega projection aux tokens: "
                    f"num_tokens={source_tokens.shape[1]}, image_shape={image_shape}"
                )

        token_features = self.remote_projection_aux_token_proj(
            self.remote_projection_aux_token_norm(source_tokens.float())
        )
        aux_chw = (
            token_features.reshape(token_features.shape[0], grid_h, grid_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(dtype=source_tokens.dtype)
        )
        aux_chw = torch.nn.functional.interpolate(
            aux_chw,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        aux_chw = self._projection_aux_add_image_features(aux_chw, source_image)
        pixel_pred = self.remote_projection_aux_token_pixel_head(aux_chw)
        pixel_pred = pixel_pred.permute(0, 2, 3, 1).contiguous()
        output["remote_projection_rel_height_pred"] = (
            pixel_pred[..., 0] * self.remote_projection_aux_rel_height_output_scale
        )
        output["remote_projection_offset_xy_pred"] = (
            pixel_pred[..., 1:3] * self.remote_projection_aux_offset_output_scale
        )

        pooled = token_features.mean(dim=1)
        global_raw = self.remote_projection_aux_token_global_head(pooled.float())
        output["remote_projection_global_dir_xy_pred"] = torch.nn.functional.normalize(
            global_raw[:, :2], dim=-1, eps=1e-6
        )
        slope_pred = global_raw[:, 2:3]
        if self.remote_projection_aux_positive_slope:
            slope_pred = torch.nn.functional.softplus(slope_pred)
        output["remote_projection_global_slope_pred"] = slope_pred
        return output

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint state dict, got {type(checkpoint)}")

        prefixes = ("module.", "model.")
        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            state_dict[new_key] = value
        return state_dict

    def _is_remote_view(self, view):
        instance = view.get("instance")
        if isinstance(instance, (list, tuple)) and len(instance) > 0:
            instance = instance[0]
        return instance == self.remote_instance_value

    def _output_head_for_view(self, view):
        return self.remote_output_head if self._is_remote_view(view) else self.ordinary_output_head

    def _run_prediction_heads(self, aggregated_tokens_list, images, patch_token_start):
        if self.model.camera_head is None:
            raise RuntimeError("VGGTOmegaWrapper requires enable_camera=true")
        if self.model.dense_head is None:
            raise RuntimeError("VGGTOmegaWrapper requires enable_depth=true")

        pose_enc = self.model.camera_head(
            aggregated_tokens_list,
            patch_token_start=patch_token_start,
        )
        extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])
        depth_map, depth_conf = self.model.dense_head(
            aggregated_tokens_list,
            images=images,
            patch_token_start=patch_token_start,
        )
        return extrinsic, intrinsic, depth_map, depth_conf

    def forward(self, views):
        """
        Args:
            views: list of MapAnything view dictionaries. Each view must provide
                an ``img`` tensor with shape [B, 3, H, W] in [0, 1].

        Returns:
            A list of per-view prediction dictionaries consumed by MapAnything losses.
        """
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        data_norm_type = views[0]["data_norm_type"][0]
        if data_norm_type != "identity":
            raise AssertionError(
                "VGGT-Omega expects images in [0, 1] without dataset mean/std normalization"
            )
        if height % self.model.aggregator.patch_size != 0 or width % self.model.aggregator.patch_size != 0:
            raise ValueError(
                f"VGGT-Omega requires H/W divisible by patch_size={self.model.aggregator.patch_size}, "
                f"got {(height, width)}"
            )

        images = torch.stack([view["img"] for view in views], dim=1)
        if images.shape[0] != batch_size_per_view:
            raise RuntimeError("Unexpected batch-size mismatch while stacking views")

        device_type = "cuda" if images.is_cuda else "cpu"
        autocast_enabled = images.is_cuda
        with torch.autocast(device_type, dtype=self.dtype, enabled=autocast_enabled):
            aggregated_tokens_list, patch_token_start = self.model.aggregator(images)

        with torch.autocast(device_type, enabled=False):
            extrinsic, intrinsic, depth_map, depth_conf = self._run_prediction_heads(
                aggregated_tokens_list,
                images,
                patch_token_start,
            )

            res = []
            for view_idx in range(num_views):
                output_head = self._output_head_for_view(views[view_idx])
                if output_head != "depth":
                    raise RuntimeError(
                        "VGGT-Omega exposes only camera+depth outputs in this wrapper; "
                        f"got output_head={output_head!r}"
                    )

                curr_view_extrinsic = closed_form_inverse_se3(extrinsic[:, view_idx, ...])
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth_map[:, view_idx, ...].squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...]

                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z,
                    curr_view_intrinsic,
                )
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z,
                    curr_view_intrinsic,
                ).unsqueeze(-1)
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic,
                    height,
                    width,
                    normalize_to_unit_sphere=True,
                )
                curr_view_pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                    curr_view_ray_dirs,
                    curr_view_depth_along_ray,
                    curr_view_cam_translations,
                    curr_view_cam_quats,
                )

                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "depth_pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                        "vggt_omega_output_head": "depth",
                    }
                )

                if self._is_remote_view(views[view_idx]) and self.use_remote_projection_aux_head:
                    source_tokens = aggregated_tokens_list[-1][
                        :, view_idx, patch_token_start:, :
                    ]
                    self._apply_remote_projection_aux_token_head(
                        res[-1],
                        source_tokens,
                        views[view_idx]["img"],
                        (height, width),
                    )

        return res


__all__ = ["VGGTOmega", "VGGTOmegaWrapper"]
