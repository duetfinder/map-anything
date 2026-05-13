# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything RS-joint variant.

This model keeps the original MapAnything aerial branch intact and adds:
- a remote image encoder initialized from the aerial encoder
- a remote DPT pointmap+confidence+mask head
- lightweight aerial/remote feature type biases before shared info sharing

Remote views are identified by setting view["instance"] == "remote".
"""

from copy import deepcopy
from typing import List, Sequence

import torch
import torch.nn as nn

from mapanything.models.mapanything.model import MapAnything
from uniception.models.prediction_heads.adaptors import (
    AdaptorInput,
    PointMapWithConfidenceAndMaskAdaptor,
)
from uniception.models.prediction_heads.base import PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import (
    DPTFeature,
    DPTRegressionProcessor,
)


class MapAnythingRSJoint(MapAnything):
    "MapAnything variant with a dedicated remote encoder and direct pointmap remote head."

    REMOTE_INSTANCE_VALUE = "remote"

    def __init__(
        self,
        *args,
        use_view_type_bias: bool = True,
        **kwargs,
    ):
        self.use_view_type_bias = use_view_type_bias
        super().__init__(*args, **kwargs)
        self._initialize_remote_modules()
        self._mirror_aerial_modules_into_remote()

    def _initialize_remote_modules(self):
        if self.pred_head_type != "dpt+pose":
            raise ValueError(
                "MapAnythingRSJoint currently expects the aerial branch to use a dpt+pose head."
            )

        self.remote_encoder = deepcopy(self.encoder)

        remote_feature_head_cfg = deepcopy(self.pred_head_config["feature_head"])
        self.remote_dpt_feature_head = DPTFeature(**remote_feature_head_cfg)

        remote_regressor_cfg = deepcopy(self.pred_head_config["regressor_head"])
        remote_regressor_cfg["output_dim"] = 5  # pointmap xyz + conf + mask
        self.remote_dpt_regressor_head = DPTRegressionProcessor(
            **remote_regressor_cfg
        )
        self.remote_dense_head = nn.Sequential(
            self.remote_dpt_feature_head,
            self.remote_dpt_regressor_head,
        )

        self.remote_dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(
            name="remote_pointmap+confidence+mask",
            pointmap_mode="exp",
            pointmap_vmin=float("-inf"),
            pointmap_vmax=float("inf"),
            confidence_type="exp",
            confidence_vmin=1,
            confidence_vmax=float("inf"),
        )
        self.remote_scene_rep_type = "pointmap+confidence+mask"

        self.aerial_view_type_embedding = nn.Parameter(
            torch.zeros(self.encoder.enc_embed_dim)
        )
        self.remote_view_type_embedding = nn.Parameter(
            torch.zeros(self.encoder.enc_embed_dim)
        )

    def _copy_matching_state(self, target_module: nn.Module, source_module: nn.Module):
        target_state = target_module.state_dict()
        source_state = source_module.state_dict()
        copied_state = {}
        for key, target_value in target_state.items():
            source_value = source_state.get(key)
            if source_value is not None and source_value.shape == target_value.shape:
                copied_state[key] = source_value.detach().clone()
        target_module.load_state_dict(copied_state, strict=False)

    def _mirror_aerial_modules_into_remote(self):
        self.remote_encoder.load_state_dict(deepcopy(self.encoder.state_dict()))
        self.remote_dpt_feature_head.load_state_dict(
            deepcopy(self.dpt_feature_head.state_dict())
        )
        self._copy_matching_state(
            self.remote_dpt_regressor_head, self.dpt_regressor_head
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        has_remote_weights = any(
            key.startswith("remote_encoder.")
            or key.startswith("remote_dpt_feature_head.")
            or key.startswith("remote_dpt_regressor_head.")
            for key in state_dict.keys()
        )
        if hasattr(self, "remote_encoder") and not has_remote_weights:
            self._mirror_aerial_modules_into_remote()
        return result

    @staticmethod
    def _normalize_instance_value(instance):
        if isinstance(instance, (list, tuple)) and len(instance) > 0:
            instance = instance[0]
        return instance

    def _is_remote_view(self, view) -> bool:
        instance = self._normalize_instance_value(view.get("instance"))
        return instance == self.REMOTE_INSTANCE_VALUE

    def _apply_view_type_bias(self, features, is_remote: bool):
        if not self.use_view_type_bias:
            return features
        bias = (
            self.remote_view_type_embedding
            if is_remote
            else self.aerial_view_type_embedding
        )
        return features + bias.view(1, -1, 1, 1)

    def _encode_n_views(self, views):
        all_encoder_features_across_views = []
        all_encoder_registers_across_views = []
        has_any_registers = False

        for view in views:
            data_norm_type = view["data_norm_type"][0]
            encoder_input = self.encoder_input_class(
                image=view["img"], data_norm_type=data_norm_type
            )
            encoder = self.remote_encoder if self._is_remote_view(view) else self.encoder
            encoder_output = encoder(encoder_input)
            features = self._apply_view_type_bias(
                encoder_output.features, is_remote=self._is_remote_view(view)
            )
            all_encoder_features_across_views.append(features)

            if self.use_register_tokens_from_encoder and encoder_output.registers is not None:
                all_encoder_registers_across_views.append(encoder_output.registers)
                has_any_registers = True
            else:
                all_encoder_registers_across_views.append(None)

        if not has_any_registers:
            all_encoder_registers_across_views = None

        return all_encoder_features_across_views, all_encoder_registers_across_views

    @property
    def encoder_input_class(self):
        from uniception.models.encoders import ViTEncoderInput

        return ViTEncoderInput

    @staticmethod
    def _select_stacked_view_slices(stacked, indices: Sequence[int], batch_size_per_view: int):
        if len(indices) == 0:
            raise ValueError("indices cannot be empty")
        chunks = [
            stacked[idx * batch_size_per_view : (idx + 1) * batch_size_per_view]
            for idx in indices
        ]
        return torch.cat(chunks, dim=0)

    def _select_dense_head_inputs(
        self, dense_head_inputs, indices: Sequence[int], batch_size_per_view: int
    ):
        if isinstance(dense_head_inputs, list):
            return [
                self._select_stacked_view_slices(x, indices, batch_size_per_view)
                for x in dense_head_inputs
            ]
        return self._select_stacked_view_slices(
            dense_head_inputs, indices, batch_size_per_view
        )

    def _run_remote_dense_head(self, dense_head_inputs, img_shape):
        remote_dense_outputs = self.remote_dense_head(
            PredictionHeadLayeredInput(
                list_features=dense_head_inputs,
                target_output_shape=img_shape,
            )
        )
        return self.remote_dense_adaptor(
            AdaptorInput(
                adaptor_feature=remote_dense_outputs.decoded_channels,
                output_shape_hw=img_shape,
            )
        )

    def _pack_remote_outputs(self, dense_final_outputs, num_views: int):
        output_pts3d = dense_final_outputs.value.permute(0, 2, 3, 1).contiguous()
        output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
        res = [{"pts3d": output_pts3d_per_view[i]} for i in range(num_views)]

        output_confidences = dense_final_outputs.confidence
        output_confidences = (
            output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
        )
        output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
        for i in range(num_views):
            res[i]["conf"] = output_confidences_per_view[i]

        output_masks = dense_final_outputs.mask
        output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous() > 0.5
        output_masks_per_view = output_masks.chunk(num_views, dim=0)
        output_mask_logits = dense_final_outputs.logits
        output_mask_logits = (
            output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
        )
        output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
        for i in range(num_views):
            res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
            res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

        return res

    def _pack_aerial_outputs(
        self,
        dense_final_outputs,
        pose_final_outputs,
        scale_final_output,
        num_views: int,
    ):
        if self.scene_rep_type not in [
            "raydirs+depth+pose",
            "raydirs+depth+pose+confidence",
            "raydirs+depth+pose+mask",
            "raydirs+depth+pose+confidence+mask",
        ]:
            raise ValueError(
                "MapAnythingRSJoint currently supports aerial scene_rep_type from the raydirs+depth+pose family."
            )

        output_dense_rep = dense_final_outputs.value.permute(0, 2, 3, 1).contiguous()
        output_ray_directions, output_depth_along_ray = output_dense_rep.split(
            [3, 1], dim=-1
        )
        output_cam_translations, output_cam_quats = pose_final_outputs.value.split(
            [3, 4], dim=-1
        )
        output_pts3d = self.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            output_ray_directions,
            output_depth_along_ray,
            output_cam_translations,
            output_cam_quats,
        )
        output_pts3d_cam = output_ray_directions * output_depth_along_ray

        output_ray_directions_per_view = output_ray_directions.chunk(num_views, dim=0)
        output_depth_along_ray_per_view = output_depth_along_ray.chunk(num_views, dim=0)
        output_cam_translations_per_view = output_cam_translations.chunk(num_views, dim=0)
        output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
        output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
        output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)

        res = []
        scale_hw = scale_final_output.unsqueeze(-1).unsqueeze(-1)
        for i in range(num_views):
            res.append(
                {
                    "pts3d": output_pts3d_per_view[i] * scale_hw,
                    "pts3d_cam": output_pts3d_cam_per_view[i] * scale_hw,
                    "ray_directions": output_ray_directions_per_view[i],
                    "depth_along_ray": output_depth_along_ray_per_view[i] * scale_hw,
                    "cam_trans": output_cam_translations_per_view[i] * scale_final_output,
                    "cam_quats": output_cam_quats_per_view[i],
                    "metric_scaling_factor": scale_final_output,
                }
            )

        if "confidence" in self.scene_rep_type:
            output_confidences = dense_final_outputs.confidence
            output_confidences = (
                output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
            for i in range(num_views):
                res[i]["conf"] = output_confidences_per_view[i]

        if "mask" in self.scene_rep_type:
            output_masks = dense_final_outputs.mask
            output_masks = (
                output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous() > 0.5
            )
            output_masks_per_view = output_masks.chunk(num_views, dim=0)
            output_mask_logits = dense_final_outputs.logits
            output_mask_logits = (
                output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
            for i in range(num_views):
                res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

        return res

    @property
    def convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(self):
        from mapanything.utils.geometry import (
            convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
        )

        return convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap

    def forward(self, views, memory_efficient_inference=False, minibatch_size=None):
        if not any(self._is_remote_view(view) for view in views):
            return super().forward(
                views,
                memory_efficient_inference=memory_efficient_inference,
                minibatch_size=minibatch_size,
            )

        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)

        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self._encode_n_views(views)
        )

        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )
        info_sharing_input = self.multi_view_transformer_input_class(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
            intermediate_info_sharing_multi_view_feat = None
        else:
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)

        dense_head_inputs_list = []
        if self.use_encoder_features_for_dpt:
            stacked_encoder_features = torch.cat(all_encoder_features_across_views, dim=0)
            dense_head_inputs_list.append(stacked_encoder_features)
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)
        else:
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)
            stacked_intermediate_features_3 = torch.cat(
                intermediate_info_sharing_multi_view_feat[2].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_3)
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)

        scale_head_inputs = final_info_sharing_multi_view_feat.additional_token_features

        aerial_indices = [idx for idx, view in enumerate(views) if not self._is_remote_view(view)]
        remote_indices = [idx for idx, view in enumerate(views) if self._is_remote_view(view)]

        aerial_res = {}
        remote_res = {}

        with torch.autocast("cuda", enabled=False):
            if aerial_indices:
                aerial_dense_inputs = self._select_dense_head_inputs(
                    dense_head_inputs_list, aerial_indices, batch_size_per_view
                )
                (
                    aerial_dense_final_outputs,
                    aerial_pose_final_outputs,
                    aerial_scale_final_output,
                ) = self.downstream_head(
                    dense_head_inputs=aerial_dense_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=memory_efficient_inference,
                    minibatch_size=minibatch_size,
                )
                packed_aerial = self._pack_aerial_outputs(
                    aerial_dense_final_outputs,
                    aerial_pose_final_outputs,
                    aerial_scale_final_output,
                    num_views=len(aerial_indices),
                )
                aerial_res = {
                    view_idx: pred for view_idx, pred in zip(aerial_indices, packed_aerial)
                }

            if remote_indices:
                remote_dense_inputs = self._select_dense_head_inputs(
                    dense_head_inputs_list, remote_indices, batch_size_per_view
                )
                remote_dense_final_outputs = self._run_remote_dense_head(
                    remote_dense_inputs, img_shape
                )
                packed_remote = self._pack_remote_outputs(
                    remote_dense_final_outputs, num_views=len(remote_indices)
                )
                remote_res = {
                    view_idx: pred for view_idx, pred in zip(remote_indices, packed_remote)
                }

        final_res = []
        for view_idx in range(num_views):
            if view_idx in remote_res:
                final_res.append(remote_res[view_idx])
            else:
                final_res.append(aerial_res[view_idx])
        return final_res

    @property
    def multi_view_transformer_input_class(self):
        from uniception.models.info_sharing.base import MultiViewTransformerInput

        return MultiViewTransformerInput
