# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for VGGT
"""

from copy import deepcopy

import torch
import torch.utils.checkpoint

from mapanything.models.external.vggt.models.vggt import VGGT
from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
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


class VGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        depth=24,
        num_heads=16,
        intermediate_layer_idx=[4, 11, 17, 23],
        load_custom_ckpt=False,
        custom_ckpt_path=None,
        use_point_head_for_remote=False,
        use_view_type_bias=False,
        remote_instance_value="remote",
        ordinary_output_head="depth",
        remote_output_head="auto",
        use_remote_private_point_head=False,
        output_point_head_for_consistency=False,
        use_pre_aggregator_view_type_bias=False,
        use_remote_to_aerial_gated_residual=False,
        remote_to_aerial_residual_hidden_scale=0.25,
        remote_to_aerial_gate_init=0.0,
        use_split_remote_aggregator=False,
        remote_to_aerial_late_fusion_type="none",
        remote_to_aerial_late_fusion_hidden_scale=0.25,
        remote_to_aerial_late_fusion_gate_init=0.0,
        remote_to_aerial_cross_attention_heads=8,
        remote_to_aerial_max_remote_tokens=256,
        protect_ordinary_heads_from_remote=False,
        use_remote_projection_aux_head=False,
        remote_projection_aux_hidden_dim=64,
        remote_projection_aux_detach_pointmap=False,
        remote_projection_aux_use_rgb=False,
        remote_projection_aux_use_coord=False,
        remote_projection_aux_image_stem_dim=0,
        remote_projection_aux_positive_slope=False,
        remote_projection_aux_slope_init=0.1,
        remote_projection_aux_num_blocks=0,
        remote_projection_aux_split_pixel_heads=False,
        use_remote_scene_matching_projection_head=False,
        remote_scene_matching_projection_dim=128,
        remote_scene_matching_projection_hidden_scale=0.25,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path
        self.use_point_head_for_remote = use_point_head_for_remote
        self.use_view_type_bias = use_view_type_bias
        self.remote_instance_value = remote_instance_value
        self.ordinary_output_head = ordinary_output_head
        self.remote_output_head = remote_output_head
        self.use_remote_private_point_head = use_remote_private_point_head
        self.output_point_head_for_consistency = output_point_head_for_consistency
        self.use_pre_aggregator_view_type_bias = use_pre_aggregator_view_type_bias
        self.use_remote_to_aerial_gated_residual = use_remote_to_aerial_gated_residual
        self.remote_to_aerial_residual_hidden_scale = remote_to_aerial_residual_hidden_scale
        self.remote_to_aerial_gate_init = remote_to_aerial_gate_init
        self.use_split_remote_aggregator = use_split_remote_aggregator
        self.remote_to_aerial_late_fusion_type = remote_to_aerial_late_fusion_type
        self.remote_to_aerial_late_fusion_hidden_scale = remote_to_aerial_late_fusion_hidden_scale
        self.remote_to_aerial_late_fusion_gate_init = remote_to_aerial_late_fusion_gate_init
        self.remote_to_aerial_cross_attention_heads = remote_to_aerial_cross_attention_heads
        self.remote_to_aerial_max_remote_tokens = remote_to_aerial_max_remote_tokens
        self.protect_ordinary_heads_from_remote = protect_ordinary_heads_from_remote
        self.use_remote_projection_aux_head = use_remote_projection_aux_head
        self.remote_projection_aux_hidden_dim = int(remote_projection_aux_hidden_dim)
        self.remote_projection_aux_detach_pointmap = remote_projection_aux_detach_pointmap
        self.remote_projection_aux_use_rgb = remote_projection_aux_use_rgb
        self.remote_projection_aux_use_coord = remote_projection_aux_use_coord
        self.remote_projection_aux_image_stem_dim = int(remote_projection_aux_image_stem_dim)
        self.remote_projection_aux_positive_slope = remote_projection_aux_positive_slope
        self.remote_projection_aux_slope_init = float(remote_projection_aux_slope_init)
        self.remote_projection_aux_num_blocks = int(remote_projection_aux_num_blocks)
        self.remote_projection_aux_split_pixel_heads = bool(remote_projection_aux_split_pixel_heads)
        self.use_remote_scene_matching_projection_head = use_remote_scene_matching_projection_head
        self.remote_scene_matching_projection_dim = int(remote_scene_matching_projection_dim)
        self.remote_scene_matching_projection_hidden_scale = float(
            remote_scene_matching_projection_hidden_scale
        )
        self.embed_dim = 1024
        self.latest_remote_to_aerial_stats = {}

        if load_pretrained_weights:
            # Load pre-trained weights
            if not torch_hub_force_reload:
                # Prefer an offline cache hit so training does not depend on
                # Hugging Face metadata requests once the model is already cached.
                print("Loading facebook/VGGT-1B from huggingface cache ...")
                try:
                    self.model = VGGT.from_pretrained(
                        "facebook/VGGT-1B",
                        local_files_only=True,
                    )
                except Exception as offline_error:
                    print(
                        "Local VGGT cache not usable, falling back to online download: "
                        f"{offline_error}"
                    )
                    self.model = VGGT.from_pretrained(
                        "facebook/VGGT-1B",
                    )
            else:
                # Initialize the 1B VGGT model
                print("Re-downloading facebook/VGGT-1B ...")
                self.model = VGGT.from_pretrained(
                    "facebook/VGGT-1B", force_download=True
                )
        else:
            # Load the VGGT class
            self.model = VGGT(
                depth=depth,
                num_heads=num_heads,
                intermediate_layer_idx=intermediate_layer_idx,
            )

        # Get the dtype for VGGT inference
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        if self.use_view_type_bias:
            token_dim = 2 * self.embed_dim
            self.aerial_view_type_embedding = torch.nn.Parameter(torch.zeros(token_dim))
            self.remote_view_type_embedding = torch.nn.Parameter(torch.zeros(token_dim))

        if self.use_pre_aggregator_view_type_bias:
            self.pre_aggregator_view_type_embedding = torch.nn.Embedding(2, self.embed_dim)
            torch.nn.init.zeros_(self.pre_aggregator_view_type_embedding.weight)

        if self.use_remote_to_aerial_gated_residual:
            token_dim = 2 * self.embed_dim
            hidden_dim = max(1, int(token_dim * remote_to_aerial_residual_hidden_scale))
            self.remote_to_aerial_residual = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim),
                torch.nn.Linear(token_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, token_dim),
            )
            self.remote_to_aerial_gate = torch.nn.Parameter(
                torch.tensor(float(remote_to_aerial_gate_init))
            )

        token_dim = 2 * self.embed_dim
        if self.remote_to_aerial_late_fusion_type not in {"none", "film", "cross_attention"}:
            raise ValueError(
                "remote_to_aerial_late_fusion_type must be one of "
                "{'none', 'film', 'cross_attention'}"
            )
        if self.remote_to_aerial_late_fusion_type == "film":
            hidden_dim = max(1, int(token_dim * remote_to_aerial_late_fusion_hidden_scale))
            self.remote_to_aerial_late_film = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim),
                torch.nn.Linear(token_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 2 * token_dim),
            )
            self.remote_to_aerial_late_gate = torch.nn.Parameter(
                torch.tensor(float(remote_to_aerial_late_fusion_gate_init))
            )
        elif self.remote_to_aerial_late_fusion_type == "cross_attention":
            self.remote_to_aerial_late_query_norm = torch.nn.LayerNorm(token_dim)
            self.remote_to_aerial_late_key_value_norm = torch.nn.LayerNorm(token_dim)
            self.remote_to_aerial_late_cross_attention = torch.nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=remote_to_aerial_cross_attention_heads,
                batch_first=True,
            )
            self.remote_to_aerial_late_gate = torch.nn.Parameter(
                torch.tensor(float(remote_to_aerial_late_fusion_gate_init))
            )

        if self.use_remote_scene_matching_projection_head:
            matching_hidden_dim = max(
                1, int(token_dim * self.remote_scene_matching_projection_hidden_scale)
            )
            self.remote_scene_matching_projection_head = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim),
                torch.nn.Linear(token_dim, matching_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(
                    matching_hidden_dim, self.remote_scene_matching_projection_dim
                ),
            )

        if self.use_remote_private_point_head:
            self.remote_point_head = deepcopy(self.model.point_head)

        if self.use_remote_projection_aux_head:
            hidden_dim = max(1, int(remote_projection_aux_hidden_dim))
            aux_pixel_in_channels = 3
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
            if self.remote_projection_aux_num_blocks > 0:
                pixel_layers = [
                    torch.nn.Conv2d(aux_pixel_in_channels, hidden_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                ]
                pixel_layers.extend(
                    _ProjectionAuxResidualBlock(hidden_dim)
                    for _ in range(self.remote_projection_aux_num_blocks)
                )
                if not self.remote_projection_aux_split_pixel_heads:
                    pixel_layers.append(torch.nn.Conv2d(hidden_dim, 3, kernel_size=1))
                self.remote_projection_aux_pixel_head = torch.nn.Sequential(*pixel_layers)
            else:
                pixel_layers = [
                    torch.nn.Conv2d(aux_pixel_in_channels, hidden_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                    torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                ]
                if not self.remote_projection_aux_split_pixel_heads:
                    pixel_layers.append(torch.nn.Conv2d(hidden_dim, 3, kernel_size=1))
                self.remote_projection_aux_pixel_head = torch.nn.Sequential(*pixel_layers)
            if self.remote_projection_aux_split_pixel_heads:
                self.remote_projection_aux_rel_height_head = torch.nn.Conv2d(hidden_dim, 1, kernel_size=1)
                self.remote_projection_aux_offset_head = torch.nn.Conv2d(hidden_dim, 2, kernel_size=1)
            self.remote_projection_aux_global_head = torch.nn.Sequential(
                torch.nn.LayerNorm(3),
                torch.nn.Linear(3, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 3),
            )
            if self.remote_projection_aux_positive_slope and self.remote_projection_aux_slope_init > 0:
                final_linear = self.remote_projection_aux_global_head[-1]
                slope_init = torch.tensor(
                    float(self.remote_projection_aux_slope_init), dtype=final_linear.bias.dtype
                )
                raw_slope_init = torch.log(torch.expm1(slope_init).clamp_min(1e-6))
                with torch.no_grad():
                    final_linear.weight[2].zero_()
                    final_linear.bias[2].copy_(raw_slope_init)

        # Load custom checkpoint if requested
        if self.load_custom_ckpt:
            print(f"Loading checkpoint from {self.custom_ckpt_path} ...")
            assert self.custom_ckpt_path is not None, (
                "custom_ckpt_path must be provided if load_custom_ckpt is set to True"
            )
            custom_ckpt = torch.load(self.custom_ckpt_path, map_location="cpu", weights_only=False)
            custom_state, is_wrapper_state = self._extract_custom_state_dict(custom_ckpt)
            if is_wrapper_state:
                custom_state = self._migrate_remote_projection_aux_split_heads(custom_state)
                print(self.load_state_dict(custom_state, strict=False))
            else:
                print(self.model.load_state_dict(custom_state, strict=True))
            del custom_ckpt  # in case it occupies memory
            if self.use_remote_private_point_head and not is_wrapper_state:
                self.remote_point_head.load_state_dict(
                    deepcopy(self.model.point_head.state_dict())
                )

    @staticmethod
    def _extract_custom_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break

        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint state dict, got {type(checkpoint)}")

        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            state_dict[new_key] = value

        is_wrapper_state = any(
            key.startswith(("model.", "remote_point_head.", "remote_projection_aux_"))
            for key in state_dict
        )
        if is_wrapper_state:
            return state_dict, True

        inner_state = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            inner_state[new_key] = value
        return inner_state, False

    def _migrate_remote_projection_aux_split_heads(self, state_dict):
        if not self.remote_projection_aux_split_pixel_heads:
            return state_dict
        rel_weight_key = "remote_projection_aux_rel_height_head.weight"
        offset_weight_key = "remote_projection_aux_offset_head.weight"
        if rel_weight_key in state_dict and offset_weight_key in state_dict:
            return state_dict

        final_weight_key = None
        final_index = -1
        for key, value in state_dict.items():
            if not key.startswith("remote_projection_aux_pixel_head.") or not key.endswith(".weight"):
                continue
            if not hasattr(value, "ndim") or value.ndim != 4 or value.shape[0] != 3:
                continue
            if tuple(value.shape[-2:]) != (1, 1):
                continue
            parts = key.split(".")
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            index = int(parts[1])
            if index > final_index:
                final_index = index
                final_weight_key = key

        if final_weight_key is None:
            return state_dict

        final_prefix = final_weight_key.rsplit(".", 1)[0]
        final_bias_key = f"{final_prefix}.bias"
        final_weight = state_dict[final_weight_key]
        state_dict[rel_weight_key] = final_weight[:1].clone()
        state_dict[offset_weight_key] = final_weight[1:3].clone()
        if final_bias_key in state_dict:
            final_bias = state_dict[final_bias_key]
            state_dict["remote_projection_aux_rel_height_head.bias"] = final_bias[:1].clone()
            state_dict["remote_projection_aux_offset_head.bias"] = final_bias[1:3].clone()
        return state_dict

    def _apply_remote_projection_aux_head(self, output, source_pointmap, source_image=None):
        if not self.use_remote_projection_aux_head:
            return output
        aux_input = source_pointmap
        if self.remote_projection_aux_detach_pointmap:
            aux_input = aux_input.detach()
        aux_chw = aux_input.permute(0, 3, 1, 2).contiguous()
        if self.remote_projection_aux_use_rgb:
            if source_image is None:
                raise RuntimeError("remote_projection_aux_use_rgb=True requires source_image")
            rgb = source_image.to(device=aux_chw.device, dtype=aux_chw.dtype)
            if self.remote_projection_aux_detach_pointmap:
                rgb = rgb.detach()
            aux_chw = torch.cat([aux_chw, rgb], dim=1)
        if self.remote_projection_aux_use_coord:
            _, _, height, width = aux_chw.shape
            y = torch.linspace(-1.0, 1.0, height, device=aux_chw.device, dtype=aux_chw.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=aux_chw.device, dtype=aux_chw.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(aux_chw.shape[0], -1, -1, -1)
            aux_chw = torch.cat([aux_chw, coord], dim=1)
        if self.remote_projection_aux_image_stem_dim > 0:
            if source_image is None:
                raise RuntimeError(
                    "remote_projection_aux_image_stem_dim>0 requires source_image"
                )
            image_for_stem = source_image.to(device=aux_chw.device, dtype=aux_chw.dtype)
            aux_chw = torch.cat([aux_chw, self.remote_projection_aux_image_stem(image_for_stem)], dim=1)
        pixel_features = self.remote_projection_aux_pixel_head(aux_chw)
        if self.remote_projection_aux_split_pixel_heads:
            rel_height_pred = self.remote_projection_aux_rel_height_head(pixel_features)
            offset_pred = self.remote_projection_aux_offset_head(pixel_features)
            pixel_pred = torch.cat([rel_height_pred, offset_pred], dim=1)
        else:
            pixel_pred = pixel_features
        pixel_pred = pixel_pred.permute(0, 2, 3, 1).contiguous()
        output["remote_projection_rel_height_pred"] = pixel_pred[..., 0]
        output["remote_projection_offset_xy_pred"] = pixel_pred[..., 1:3]

        finite_mask = torch.isfinite(aux_input).all(dim=-1, keepdim=True)
        safe_input = torch.where(finite_mask, aux_input, torch.zeros_like(aux_input))
        denom = finite_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        pooled = safe_input.sum(dim=(1, 2)) / denom
        global_raw = self.remote_projection_aux_global_head(pooled.float())
        dir_xy = torch.nn.functional.normalize(global_raw[:, :2], dim=-1, eps=1e-6)
        output["remote_projection_global_dir_xy_pred"] = dir_xy
        slope_pred = global_raw[:, 2:3]
        if self.remote_projection_aux_positive_slope:
            slope_pred = torch.nn.functional.softplus(slope_pred)
        output["remote_projection_global_slope_pred"] = slope_pred
        return output


    def _output_head_for_view(self, view):
        if self._is_remote_view(view):
            if self.remote_output_head == "auto":
                return "point" if self.use_point_head_for_remote else "depth"
            return self.remote_output_head
        return self.ordinary_output_head

    def _is_remote_view(self, view):
        instance = view.get("instance")
        if isinstance(instance, (list, tuple)) and len(instance) > 0:
            instance = instance[0]
        return instance == self.remote_instance_value

    def _view_type_ids(self, views, device):
        return torch.tensor(
            [1 if self._is_remote_view(view) else 0 for view in views],
            device=device,
            dtype=torch.long,
        )

    def _remote_view_mask(self, views, device):
        return torch.tensor(
            [self._is_remote_view(view) for view in views],
            device=device,
            dtype=torch.bool,
        )

    def _apply_view_type_bias(self, aggregated_tokens_list, views):
        if not self.use_view_type_bias:
            return aggregated_tokens_list

        remote_mask = self._remote_view_mask(
            views, device=aggregated_tokens_list[0].device
        )
        if not bool(remote_mask.any()):
            return [
                tokens + self.aerial_view_type_embedding.view(1, 1, 1, -1)
                for tokens in aggregated_tokens_list
            ]

        type_bias = torch.where(
            remote_mask.view(1, -1, 1, 1),
            self.remote_view_type_embedding.view(1, 1, 1, -1),
            self.aerial_view_type_embedding.view(1, 1, 1, -1),
        )
        return [tokens + type_bias for tokens in aggregated_tokens_list]

    def _apply_remote_to_aerial_gated_residual(self, aggregated_tokens_list, views):
        if not self.use_remote_to_aerial_gated_residual:
            return aggregated_tokens_list

        remote_mask = self._remote_view_mask(
            views, device=aggregated_tokens_list[0].device
        )
        if not bool(remote_mask.any()) or bool(remote_mask.all()):
            return aggregated_tokens_list

        aerial_mask = ~remote_mask
        patch_start_idx = getattr(self.model.aggregator, "patch_start_idx", 0)
        updated_tokens_list = []
        for tokens in aggregated_tokens_list:
            if tokens.shape[2] <= patch_start_idx:
                updated_tokens_list.append(tokens)
                continue

            remote_patch_tokens = tokens[:, remote_mask, patch_start_idx:, :]
            remote_context = remote_patch_tokens.mean(dim=(1, 2))
            remote_delta = self.remote_to_aerial_residual(remote_context)
            remote_delta = remote_delta.view(tokens.shape[0], 1, 1, tokens.shape[-1])
            gate = self.remote_to_aerial_gate.to(
                device=tokens.device, dtype=tokens.dtype
            )

            updated_tokens = tokens.clone()
            updated_tokens[:, aerial_mask, patch_start_idx:, :] = (
                updated_tokens[:, aerial_mask, patch_start_idx:, :] + gate * remote_delta
            )
            updated_tokens_list.append(updated_tokens)

        return updated_tokens_list

    def _aggregator_kwargs(self, views, images):
        if not self.use_pre_aggregator_view_type_bias:
            return {}
        return {
            "view_type_ids": self._view_type_ids(views, images.device),
            "view_type_embedding": self.pre_aggregator_view_type_embedding.weight,
        }

    def _run_aggregator(self, images, views):
        return self.model.aggregator(images, **self._aggregator_kwargs(views, images))

    def _combine_split_aggregated_tokens(
        self, aerial_tokens_list, remote_tokens_list, remote_mask, num_views
    ):
        combined_tokens_list = []
        aerial_mask = ~remote_mask
        for aerial_tokens, remote_tokens in zip(aerial_tokens_list, remote_tokens_list):
            combined_tokens = aerial_tokens.new_empty(
                aerial_tokens.shape[0],
                num_views,
                aerial_tokens.shape[2],
                aerial_tokens.shape[3],
            )
            combined_tokens[:, aerial_mask, :, :] = aerial_tokens
            combined_tokens[:, remote_mask, :, :] = remote_tokens
            combined_tokens_list.append(combined_tokens)
        return combined_tokens_list

    def _select_tokens_by_mask(self, aggregated_tokens_list, view_mask):
        return [tokens[:, view_mask, :, :] for tokens in aggregated_tokens_list]

    def _scatter_view_tensor(
        self, aerial_value, remote_value, aerial_mask, remote_mask, num_views
    ):
        ref_value = aerial_value if aerial_value is not None else remote_value
        if ref_value is None:
            return None

        output = ref_value.new_empty(ref_value.shape[0], num_views, *ref_value.shape[2:])
        if aerial_value is not None:
            output[:, aerial_mask, ...] = aerial_value
        if remote_value is not None:
            output[:, remote_mask, ...] = remote_value
        return output

    def _run_dpt_head(self, head, tokens_list, head_images, ps_idx):
        if not self.training:
            return head(tokens_list, head_images, ps_idx)

        def run_head(*tokens):
            return head(list(tokens), head_images, ps_idx)

        return torch.utils.checkpoint.checkpoint(
            run_head, *tokens_list, use_reentrant=True
        )

    def _run_prediction_heads(self, aggregated_tokens_list, images, ps_idx, views):
        remote_mask = self._remote_view_mask(views, device=images.device)
        has_remote = bool(remote_mask.any())
        has_aerial = bool((~remote_mask).any())
        protect_heads = (
            self.protect_ordinary_heads_from_remote
            and self.use_split_remote_aggregator
            and has_remote
            and has_aerial
        )

        need_shared_point_head = self.output_point_head_for_consistency or any(
            self._output_head_for_view(view) == "point"
            and not (
                self.use_remote_private_point_head and self._is_remote_view(view)
            )
            for view in views
        )
        need_remote_private_point_head = self.use_remote_private_point_head and any(
            self._output_head_for_view(view) == "point" and self._is_remote_view(view)
            for view in views
        )

        if not protect_heads:
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )
            depth_map, depth_conf = self._run_dpt_head(
                self.model.depth_head, aggregated_tokens_list, images, ps_idx
            )

            point_map = None
            point_conf = None
            if need_shared_point_head:
                point_map, point_conf = self._run_dpt_head(
                    self.model.point_head, aggregated_tokens_list, images, ps_idx
                )

            remote_point_map = None
            remote_point_conf = None
            if need_remote_private_point_head:
                remote_tokens_list = self._select_tokens_by_mask(
                    aggregated_tokens_list, remote_mask
                )
                remote_images = images[:, remote_mask, ...]
                remote_private_point_map, remote_private_point_conf = self._run_dpt_head(
                    self.remote_point_head, remote_tokens_list, remote_images, ps_idx
                )
                remote_point_map = self._scatter_view_tensor(
                    None, remote_private_point_map, ~remote_mask, remote_mask, len(views)
                )
                remote_point_conf = self._scatter_view_tensor(
                    None, remote_private_point_conf, ~remote_mask, remote_mask, len(views)
                )

            return (
                extrinsic,
                intrinsic,
                depth_map,
                depth_conf,
                point_map,
                point_conf,
                remote_point_map,
                remote_point_conf,
            )

        aerial_mask = ~remote_mask
        num_views = len(views)
        aerial_images = images[:, aerial_mask, ...]
        remote_images = images[:, remote_mask, ...]
        aerial_tokens_list = self._select_tokens_by_mask(
            aggregated_tokens_list, aerial_mask
        )
        remote_tokens_list = self._select_tokens_by_mask(
            aggregated_tokens_list, remote_mask
        )

        aerial_pose_enc = self.model.camera_head(aerial_tokens_list)[-1]
        remote_pose_enc = self.model.camera_head(remote_tokens_list)[-1]
        aerial_extrinsic, aerial_intrinsic = pose_encoding_to_extri_intri(
            aerial_pose_enc, images.shape[-2:]
        )
        remote_extrinsic, remote_intrinsic = pose_encoding_to_extri_intri(
            remote_pose_enc, images.shape[-2:]
        )
        extrinsic = self._scatter_view_tensor(
            aerial_extrinsic, remote_extrinsic, aerial_mask, remote_mask, num_views
        )
        intrinsic = self._scatter_view_tensor(
            aerial_intrinsic, remote_intrinsic, aerial_mask, remote_mask, num_views
        )

        aerial_depth_map, aerial_depth_conf = self._run_dpt_head(
            self.model.depth_head, aerial_tokens_list, aerial_images, ps_idx
        )
        remote_depth_map, remote_depth_conf = self._run_dpt_head(
            self.model.depth_head, remote_tokens_list, remote_images, ps_idx
        )
        depth_map = self._scatter_view_tensor(
            aerial_depth_map, remote_depth_map, aerial_mask, remote_mask, num_views
        )
        depth_conf = self._scatter_view_tensor(
            aerial_depth_conf, remote_depth_conf, aerial_mask, remote_mask, num_views
        )

        point_map = None
        point_conf = None
        if need_shared_point_head:
            aerial_point_map, aerial_point_conf = self._run_dpt_head(
                self.model.point_head, aerial_tokens_list, aerial_images, ps_idx
            )
            remote_point_map_shared, remote_point_conf_shared = self._run_dpt_head(
                self.model.point_head, remote_tokens_list, remote_images, ps_idx
            )
            point_map = self._scatter_view_tensor(
                aerial_point_map,
                remote_point_map_shared,
                aerial_mask,
                remote_mask,
                num_views,
            )
            point_conf = self._scatter_view_tensor(
                aerial_point_conf,
                remote_point_conf_shared,
                aerial_mask,
                remote_mask,
                num_views,
            )

        remote_point_map = None
        remote_point_conf = None
        if need_remote_private_point_head:
            remote_private_point_map, remote_private_point_conf = self.remote_point_head(
                remote_tokens_list, remote_images, ps_idx
            )
            remote_point_map = self._scatter_view_tensor(
                None, remote_private_point_map, aerial_mask, remote_mask, num_views
            )
            remote_point_conf = self._scatter_view_tensor(
                None, remote_private_point_conf, aerial_mask, remote_mask, num_views
            )

        return (
            extrinsic,
            intrinsic,
            depth_map,
            depth_conf,
            point_map,
            point_conf,
            remote_point_map,
            remote_point_conf,
        )

    def _aggregate_views(self, images, views):
        remote_mask = self._remote_view_mask(views, device=images.device)
        has_remote = bool(remote_mask.any())
        has_aerial = bool((~remote_mask).any())
        if not self.use_split_remote_aggregator or not (has_remote and has_aerial):
            return self._run_aggregator(images, views)

        aerial_mask = ~remote_mask
        aerial_images = images[:, aerial_mask, ...]
        remote_images = images[:, remote_mask, ...]
        aerial_views = [view for view in views if not self._is_remote_view(view)]
        remote_views = [view for view in views if self._is_remote_view(view)]

        aerial_tokens_list, ps_idx = self._run_aggregator(aerial_images, aerial_views)
        remote_tokens_list, remote_ps_idx = self._run_aggregator(remote_images, remote_views)
        if ps_idx != remote_ps_idx:
            raise RuntimeError(
                f"Split VGGT aggregators returned different patch starts: {ps_idx} vs {remote_ps_idx}"
            )
        return (
            self._combine_split_aggregated_tokens(
                aerial_tokens_list, remote_tokens_list, remote_mask, len(views)
            ),
            ps_idx,
        )

    def _downsample_remote_tokens(self, remote_tokens):
        max_tokens = int(self.remote_to_aerial_max_remote_tokens or 0)
        if max_tokens <= 0 or remote_tokens.shape[1] <= max_tokens:
            return remote_tokens
        stride = max(1, (remote_tokens.shape[1] + max_tokens - 1) // max_tokens)
        return remote_tokens[:, ::stride, :][:, :max_tokens, :]

    def _apply_late_remote_to_aerial_fusion(self, aggregated_tokens_list, views):
        self.latest_remote_to_aerial_stats = {}
        self.latest_remote_scene_matching = {}
        self.latest_remote_scene_matching_projected = None
        if self.remote_to_aerial_late_fusion_type == "none":
            return aggregated_tokens_list

        remote_mask = self._remote_view_mask(
            views, device=aggregated_tokens_list[0].device
        )
        if not bool(remote_mask.any()) or bool(remote_mask.all()):
            return aggregated_tokens_list

        aerial_mask = ~remote_mask
        patch_start_idx = getattr(self.model.aggregator, "patch_start_idx", 0)
        updated_tokens_list = []
        for tokens in aggregated_tokens_list:
            if tokens.shape[2] <= patch_start_idx:
                updated_tokens_list.append(tokens)
                continue

            aerial_patch_tokens = tokens[:, aerial_mask, patch_start_idx:, :]
            remote_patch_tokens = tokens[:, remote_mask, patch_start_idx:, :]
            batch_size, num_aerial, num_patches, token_dim = aerial_patch_tokens.shape
            remote_context_tokens = remote_patch_tokens.reshape(batch_size, -1, token_dim)
            aerial_context_tokens = aerial_patch_tokens.reshape(batch_size, -1, token_dim)

            matching = self.latest_remote_scene_matching
            matching.setdefault("aerial", []).append(aerial_context_tokens.mean(dim=1).float())
            matching.setdefault("remote", []).append(remote_context_tokens.mean(dim=1).float())

            if self.remote_to_aerial_late_fusion_type == "film":
                remote_context = remote_context_tokens.mean(dim=1)
                scale_bias = self.remote_to_aerial_late_film(remote_context)
                scale, bias = torch.chunk(scale_bias, chunks=2, dim=-1)
                delta = (
                    aerial_patch_tokens * scale.view(batch_size, 1, 1, token_dim)
                    + bias.view(batch_size, 1, 1, token_dim)
                )
            elif self.remote_to_aerial_late_fusion_type == "cross_attention":
                query_tokens = aerial_patch_tokens.reshape(batch_size, -1, token_dim)
                key_value_tokens = self._downsample_remote_tokens(remote_context_tokens)
                query_tokens = self.remote_to_aerial_late_query_norm(query_tokens)
                key_value_tokens = self.remote_to_aerial_late_key_value_norm(key_value_tokens)
                delta, _ = self.remote_to_aerial_late_cross_attention(
                    query_tokens,
                    key_value_tokens,
                    key_value_tokens,
                    need_weights=False,
                )
                delta = delta.reshape(batch_size, num_aerial, num_patches, token_dim)
            else:
                raise RuntimeError(
                    f"Unsupported late fusion type: {self.remote_to_aerial_late_fusion_type}"
                )

            gate = self.remote_to_aerial_late_gate.to(
                device=tokens.device, dtype=tokens.dtype
            )
            weighted_delta = gate * delta
            stats = self.latest_remote_to_aerial_stats
            stats.setdefault("late_gate_abs", []).append(gate.abs().float())
            stats.setdefault("late_delta_l2", []).append(
                delta.float().pow(2).mean().sqrt()
            )
            stats.setdefault("late_weighted_delta_l2", []).append(
                weighted_delta.float().pow(2).mean().sqrt()
            )
            updated_tokens = tokens.clone()
            updated_tokens[:, aerial_mask, patch_start_idx:, :] = (
                updated_tokens[:, aerial_mask, patch_start_idx:, :] + weighted_delta
            )
            updated_tokens_list.append(updated_tokens)

        return updated_tokens_list

    def _finalize_remote_scene_matching_descriptors(self):
        matching = getattr(self, "latest_remote_scene_matching", None) or {}
        aerial_values = matching.get("aerial") or []
        remote_values = matching.get("remote") or []
        if not aerial_values or not remote_values:
            self.latest_remote_scene_matching_projected = None
            return

        aerial_desc = torch.stack(aerial_values, dim=0).mean(dim=0)
        remote_desc = torch.stack(remote_values, dim=0).mean(dim=0)
        if hasattr(self, "remote_scene_matching_projection_head"):
            combined_desc = torch.cat([aerial_desc, remote_desc], dim=0)
            combined_desc = self.remote_scene_matching_projection_head(combined_desc)
            aerial_desc, remote_desc = combined_desc.chunk(2, dim=0)

        self.latest_remote_scene_matching_projected = {
            "aerial": aerial_desc,
            "remote": remote_desc,
        }

    def get_remote_scene_matching_descriptors(self):
        projected = getattr(self, "latest_remote_scene_matching_projected", None)
        if projected is not None:
            return projected

        matching = getattr(self, "latest_remote_scene_matching", None) or {}
        aerial_values = matching.get("aerial") or []
        remote_values = matching.get("remote") or []
        if not aerial_values or not remote_values:
            return None
        aerial_desc = torch.stack(aerial_values, dim=0).mean(dim=0)
        remote_desc = torch.stack(remote_values, dim=0).mean(dim=0)
        return {
            "aerial": aerial_desc,
            "remote": remote_desc,
        }

    def get_remote_to_aerial_regularization_terms(self):
        terms = {}
        if hasattr(self, "remote_to_aerial_late_gate"):
            gate = self.remote_to_aerial_late_gate.float()
            terms["remote_to_aerial_late_gate_l1"] = gate.abs()
            terms["remote_to_aerial_late_gate_l2"] = gate.pow(2)

        stats = getattr(self, "latest_remote_to_aerial_stats", None) or {}
        for key, values in stats.items():
            if values:
                terms[key] = torch.stack([value.float() for value in values]).mean()
        return terms

    def forward(self, views):
        """
        Forward pass wrapper for VGGT

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # VGGT expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        # Run the VGGT aggregator
        with torch.autocast("cuda", dtype=self.dtype):
            aggregated_tokens_list, ps_idx = self._aggregate_views(images, views)
            aggregated_tokens_list = self._apply_late_remote_to_aerial_fusion(
                aggregated_tokens_list, views
            )
            self._finalize_remote_scene_matching_descriptors()
            aggregated_tokens_list = self._apply_remote_to_aerial_gated_residual(
                aggregated_tokens_list, views
            )
            aggregated_tokens_list = self._apply_view_type_bias(
                aggregated_tokens_list, views
            )

        # Run the Camera + Pose Branch of VGGT
        with torch.autocast("cuda", enabled=False):
            (
                extrinsic,
                intrinsic,
                depth_map,
                depth_conf,
                point_map,
                point_conf,
                remote_point_map,
                remote_point_conf,
            ) = self._run_prediction_heads(aggregated_tokens_list, images, ps_idx, views)

            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics, intrinsics, depth map for the current view
                curr_view_extrinsic = extrinsic[:, view_idx, ...]
                curr_view_extrinsic = closed_form_inverse_se3(
                    curr_view_extrinsic
                )  # Convert to cam2world
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth_map[:, view_idx, ...]
                curr_view_depth_z = curr_view_depth_z.squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...]

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
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
                    }
                )

                if point_map is not None:
                    res[-1]["point_head_pts3d"] = point_map[:, view_idx, ...]
                    res[-1]["point_head_conf"] = point_conf[:, view_idx, ...]

                output_head = self._output_head_for_view(views[view_idx])
                if output_head == "point":
                    if (
                        remote_point_map is not None
                        and self._is_remote_view(views[view_idx])
                    ):
                        res[-1]["pts3d"] = remote_point_map[:, view_idx, ...]
                        res[-1]["conf"] = remote_point_conf[:, view_idx, ...]
                        res[-1]["remote_private_point_head_pts3d"] = remote_point_map[
                            :, view_idx, ...
                        ]
                        res[-1]["remote_private_point_head_conf"] = remote_point_conf[
                            :, view_idx, ...
                        ]
                    elif point_map is not None:
                        res[-1]["pts3d"] = point_map[:, view_idx, ...]
                        res[-1]["conf"] = point_conf[:, view_idx, ...]
                    else:
                        raise RuntimeError(
                            "VGGT point output requested but point_head was not run"
                        )
                    res[-1]["vggt_output_head"] = "point"
                else:
                    res[-1]["vggt_output_head"] = "depth"

                if self._is_remote_view(views[view_idx]):
                    self._apply_remote_projection_aux_head(
                        res[-1], res[-1]["pts3d"], views[view_idx]["img"]
                    )

        return res
