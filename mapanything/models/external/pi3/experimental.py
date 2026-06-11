from copy import deepcopy

import torch
import torch.nn as nn

from mapanything.models.external.pi3.layers.transformer_head import LinearPts3d
from mapanything.models.external.pi3.models.pi3 import Pi3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat


class _ProjectionAuxResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class Pi3WithModalityEmbedding(Pi3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aerial_view_type_embedding = nn.Parameter(
            torch.zeros(self.dec_embed_dim)
        )
        self.remote_view_type_embedding = nn.Parameter(
            torch.zeros(self.dec_embed_dim)
        )

    def _apply_view_type_embeddings(self, hidden, view_type_ids):
        if view_type_ids is None:
            return hidden

        if view_type_ids.ndim == 1:
            flat_view_type_ids = view_type_ids
        else:
            flat_view_type_ids = view_type_ids.reshape(-1)
        flat_view_type_ids = flat_view_type_ids.to(hidden.device).long()

        type_embeddings = torch.stack(
            [self.aerial_view_type_embedding, self.remote_view_type_embedding], dim=0
        )
        per_view_embeddings = type_embeddings[flat_view_type_ids]
        return hidden + per_view_embeddings.unsqueeze(1).to(hidden.dtype)

    def _encode_images(self, imgs, view_type_ids=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        imgs = imgs.reshape(B * N, -1, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden = self._apply_view_type_embeddings(hidden, view_type_ids)
        hidden, pos = self.decode(hidden, N, H, W)
        return hidden, pos, B, N, H, W, patch_h, patch_w

    def forward(self, imgs, view_type_ids=None):
        hidden, pos, B, N, H, W, patch_h, patch_w = self._encode_images(
            imgs, view_type_ids=view_type_ids
        )

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                camera_poses,
                torch.cat(
                    [local_points, torch.ones_like(local_points[..., :1])], dim=-1
                ),
            )[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )


class Pi3WithModalityEmbeddingRemoteHead(Pi3WithModalityEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_remote_projection_aux_head = False
        self.remote_point_decoder = deepcopy(self.point_decoder)
        self.remote_point_head = LinearPts3d(
            patch_size=self.patch_size, dec_embed_dim=1024, output_dim=3
        )
        self.remote_conf_decoder = deepcopy(self.conf_decoder)
        self.remote_conf_head = LinearPts3d(
            patch_size=self.patch_size, dec_embed_dim=1024, output_dim=1
        )

    def init_remote_projection_aux_head(
        self,
        hidden_dim=96,
        use_rgb=True,
        use_coord=True,
        image_stem_dim=32,
        positive_slope=True,
        slope_init=0.1,
        num_blocks=6,
        rel_height_output_scale=1.0,
        offset_output_scale=1.0,
    ):
        self.use_remote_projection_aux_head = True
        self.remote_projection_aux_hidden_dim = int(hidden_dim)
        self.remote_projection_aux_use_rgb = bool(use_rgb)
        self.remote_projection_aux_use_coord = bool(use_coord)
        self.remote_projection_aux_image_stem_dim = int(image_stem_dim)
        self.remote_projection_aux_positive_slope = bool(positive_slope)
        self.remote_projection_aux_slope_init = float(slope_init)
        self.remote_projection_aux_num_blocks = int(num_blocks)
        self.remote_projection_aux_rel_height_output_scale = float(rel_height_output_scale)
        self.remote_projection_aux_offset_output_scale = float(offset_output_scale)

        hidden_dim = max(1, int(hidden_dim))
        token_dim = 2 * self.dec_embed_dim
        self.remote_projection_aux_token_norm = nn.LayerNorm(token_dim)
        self.remote_projection_aux_token_proj = nn.Linear(token_dim, hidden_dim)

        aux_pixel_in_channels = hidden_dim
        if self.remote_projection_aux_use_rgb:
            aux_pixel_in_channels += 3
        if self.remote_projection_aux_use_coord:
            aux_pixel_in_channels += 2
        if self.remote_projection_aux_image_stem_dim > 0:
            aux_pixel_in_channels += self.remote_projection_aux_image_stem_dim
            self.remote_projection_aux_image_stem = nn.Sequential(
                nn.Conv2d(3, self.remote_projection_aux_image_stem_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    self.remote_projection_aux_image_stem_dim,
                    self.remote_projection_aux_image_stem_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
            )

        pixel_layers = [
            nn.Conv2d(aux_pixel_in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        pixel_layers.extend(
            _ProjectionAuxResidualBlock(hidden_dim)
            for _ in range(self.remote_projection_aux_num_blocks)
        )
        pixel_layers.append(nn.Conv2d(hidden_dim, 3, kernel_size=1))
        self.remote_projection_aux_token_pixel_head = nn.Sequential(*pixel_layers)
        self.remote_projection_aux_token_global_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        if self.remote_projection_aux_positive_slope and self.remote_projection_aux_slope_init > 0:
            final_linear = self.remote_projection_aux_token_global_head[-1]
            slope_init_tensor = torch.tensor(
                self.remote_projection_aux_slope_init, dtype=final_linear.bias.dtype
            )
            raw_slope_init = torch.log(torch.expm1(slope_init_tensor).clamp_min(1e-6))
            with torch.no_grad():
                final_linear.weight[2].zero_()
                final_linear.bias[2].copy_(raw_slope_init)

    def mirror_aerial_modules_into_remote(self):
        self.remote_point_decoder.load_state_dict(
            deepcopy(self.point_decoder.state_dict())
        )
        self.remote_point_head.load_state_dict(deepcopy(self.point_head.state_dict()))
        self.remote_conf_decoder.load_state_dict(
            deepcopy(self.conf_decoder.state_dict())
        )
        self.remote_conf_head.load_state_dict(deepcopy(self.conf_head.state_dict()))

    def _projection_aux_add_image_features(self, aux_chw, source_image):
        if self.remote_projection_aux_use_rgb:
            if source_image is None:
                raise RuntimeError("remote projection aux RGB features require source_image")
            aux_chw = torch.cat([aux_chw, source_image.to(aux_chw.device, aux_chw.dtype)], dim=1)
        if self.remote_projection_aux_use_coord:
            _, _, height, width = aux_chw.shape
            y = torch.linspace(-1.0, 1.0, height, device=aux_chw.device, dtype=aux_chw.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=aux_chw.device, dtype=aux_chw.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(aux_chw.shape[0], -1, -1, -1)
            aux_chw = torch.cat([aux_chw, coord], dim=1)
        if self.remote_projection_aux_image_stem_dim > 0:
            if source_image is None:
                raise RuntimeError("remote projection aux image stem requires source_image")
            image_for_stem = source_image.to(aux_chw.device, aux_chw.dtype)
            aux_chw = torch.cat([aux_chw, self.remote_projection_aux_image_stem(image_for_stem)], dim=1)
        return aux_chw

    def _apply_remote_projection_aux_token_head(self, source_tokens, source_image, image_shape):
        height, width = int(image_shape[0]), int(image_shape[1])
        grid_h = max(1, height // int(self.patch_size))
        grid_w = max(1, width // int(self.patch_size))
        if source_tokens.shape[1] != grid_h * grid_w:
            grid_h = int(round(source_tokens.shape[1] ** 0.5))
            grid_w = source_tokens.shape[1] // max(1, grid_h)
            if grid_h * grid_w != source_tokens.shape[1]:
                raise RuntimeError(
                    "Cannot reshape Pi3 remote projection aux tokens: "
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
        rel_pred = pixel_pred[..., 0] * self.remote_projection_aux_rel_height_output_scale
        offset_pred = pixel_pred[..., 1:3] * self.remote_projection_aux_offset_output_scale

        pooled = token_features.mean(dim=1)
        global_raw = self.remote_projection_aux_token_global_head(pooled.float())
        dir_xy = torch.nn.functional.normalize(global_raw[:, :2], dim=-1, eps=1e-6)
        slope_pred = global_raw[:, 2:3]
        if self.remote_projection_aux_positive_slope:
            slope_pred = torch.nn.functional.softplus(slope_pred)
        return rel_pred, offset_pred, dir_xy, slope_pred

    def forward(self, imgs, view_type_ids=None):
        hidden, pos, B, N, H, W, patch_h, patch_w = self._encode_images(
            imgs, view_type_ids=view_type_ids
        )

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head(
                [point_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = self.conf_head(
                [conf_hidden[:, self.patch_start_idx :]], (H, W)
            ).reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(B, N, 4, 4)

            points = torch.einsum(
                "bnij, bnhwj -> bnhwi",
                camera_poses,
                torch.cat(
                    [local_points, torch.ones_like(local_points[..., :1])], dim=-1
                ),
            )[..., :3]

            if view_type_ids is not None:
                remote_mask = (
                    view_type_ids.reshape(-1).to(hidden.device).long() == 1
                )
                if remote_mask.any():
                    remote_hidden = hidden[remote_mask].float()
                    remote_pos = pos[remote_mask]
                    remote_point_hidden = self.remote_point_decoder(
                        remote_hidden, xpos=remote_pos
                    )
                    remote_points = self.remote_point_head(
                        [remote_point_hidden[:, self.patch_start_idx :]], (H, W)
                    )

                    remote_conf_hidden = self.remote_conf_decoder(
                        remote_hidden, xpos=remote_pos
                    ).float()
                    remote_conf = self.remote_conf_head(
                        [remote_conf_hidden[:, self.patch_start_idx :]], (H, W)
                    )

                    points_flat = points.reshape(B * N, H, W, 3).clone()
                    local_points_flat = local_points.reshape(B * N, H, W, 3).clone()
                    conf_flat = conf.reshape(B * N, H, W, 1).clone()
                    camera_poses_flat = camera_poses.reshape(B * N, 4, 4).clone()

                    points_flat[remote_mask] = remote_points
                    local_points_flat[remote_mask] = remote_points
                    conf_flat[remote_mask] = remote_conf
                    camera_poses_flat[remote_mask] = torch.eye(
                        4, device=camera_poses_flat.device, dtype=camera_poses_flat.dtype
                    )

                    points = points_flat.reshape(B, N, H, W, 3)
                    local_points = local_points_flat.reshape(B, N, H, W, 3)
                    conf = conf_flat.reshape(B, N, H, W, 1)
                    camera_poses = camera_poses_flat.reshape(B, N, 4, 4)

                    if self.use_remote_projection_aux_head:
                        imgs_flat = imgs.reshape(B * N, *imgs.shape[2:])
                        rel, offset, dir_xy, slope = self._apply_remote_projection_aux_token_head(
                            remote_hidden[:, self.patch_start_idx :],
                            imgs_flat[remote_mask],
                            (H, W),
                        )
                        rel_flat = torch.zeros(
                            B * N, H, W, device=rel.device, dtype=rel.dtype
                        )
                        offset_flat = torch.zeros(
                            B * N, H, W, 2, device=offset.device, dtype=offset.dtype
                        )
                        dir_flat = torch.zeros(
                            B * N, 2, device=dir_xy.device, dtype=dir_xy.dtype
                        )
                        slope_flat = torch.zeros(
                            B * N, 1, device=slope.device, dtype=slope.dtype
                        )
                        rel_flat[remote_mask] = rel
                        offset_flat[remote_mask] = offset
                        dir_flat[remote_mask] = dir_xy
                        slope_flat[remote_mask] = slope
                        remote_projection_aux = {
                            "rel_height": rel_flat.reshape(B, N, H, W),
                            "offset_xy": offset_flat.reshape(B, N, H, W, 2),
                            "global_dir_xy": dir_flat.reshape(B, N, 2),
                            "global_slope": slope_flat.reshape(B, N, 1),
                        }

        output = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
        if "remote_projection_aux" in locals():
            output["remote_projection_aux"] = remote_projection_aux
        return output


class _BasePi3ExperimentalWrapper(torch.nn.Module):
    MODEL_CLS = Pi3

    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        pos_type="rope100",
        decoder_size="large",
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
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        self.model = self.MODEL_CLS(pos_type=pos_type, decoder_size=decoder_size)
        if use_remote_projection_aux_head:
            if not hasattr(self.model, "init_remote_projection_aux_head"):
                raise ValueError(
                    f"{self.MODEL_CLS.__name__} does not support remote projection aux head"
                )
            self.model.init_remote_projection_aux_head(
                hidden_dim=remote_projection_aux_hidden_dim,
                use_rgb=remote_projection_aux_use_rgb,
                use_coord=remote_projection_aux_use_coord,
                image_stem_dim=remote_projection_aux_image_stem_dim,
                positive_slope=remote_projection_aux_positive_slope,
                slope_init=remote_projection_aux_slope_init,
                num_blocks=remote_projection_aux_num_blocks,
                rel_height_output_scale=remote_projection_aux_rel_height_output_scale,
                offset_output_scale=remote_projection_aux_offset_output_scale,
            )
        if load_pretrained_weights:
            if not torch_hub_force_reload:
                print(f"Loading {self.MODEL_CLS.__name__} from Pi3 huggingface cache ...")
                base_model = Pi3.from_pretrained("yyfz233/Pi3")
            else:
                base_model = Pi3.from_pretrained(
                    "yyfz233/Pi3", force_download=True
                )
            self.model.load_state_dict(base_model.state_dict(), strict=False)
            if hasattr(self.model, "mirror_aerial_modules_into_remote"):
                self.model.mirror_aerial_modules_into_remote()

        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

    @staticmethod
    def _normalize_instance(instance):
        if isinstance(instance, (list, tuple)) and len(instance) > 0:
            instance = instance[0]
        return instance

    def _build_view_type_ids(self, views, device):
        view_type_ids = []
        for view in views:
            instance = self._normalize_instance(view.get("instance"))
            view_type_ids.append(1 if instance == "remote" else 0)
        return torch.tensor(view_type_ids, device=device, dtype=torch.long)

    def forward(self, views):
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "Pi3 expects a normalized image but without the DINOv2 mean and std applied"
        )

        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)
        view_type_ids = self._build_view_type_ids(views, images.device)
        view_type_ids = view_type_ids.unsqueeze(0).repeat(batch_size_per_view, 1)

        with torch.autocast("cuda", dtype=self.dtype):
            results = self.model(images, view_type_ids=view_type_ids)

        with torch.autocast("cuda", enabled=False):
            res = []
            for view_idx in range(num_views):
                curr_view_extrinsic = results["camera_poses"][:, view_idx, ...]
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])

                curr_view_pts3d_cam = results["local_points"][:, view_idx, ...]
                curr_view_depth_along_ray = torch.norm(
                    curr_view_pts3d_cam, dim=-1, keepdim=True
                )
                curr_view_ray_dirs = curr_view_pts3d_cam / (
                    curr_view_depth_along_ray + 1e-8
                )
                curr_view_pts3d = results["points"][:, view_idx, ...]
                curr_view_confidence = results["conf"][:, view_idx, ...]

                curr_output = {
                    "pts3d": curr_view_pts3d,
                    "pts3d_cam": curr_view_pts3d_cam,
                    "ray_directions": curr_view_ray_dirs,
                    "depth_along_ray": curr_view_depth_along_ray,
                    "cam_trans": curr_view_cam_translations,
                    "cam_quats": curr_view_cam_quats,
                    "conf": curr_view_confidence,
                }
                if (
                    "remote_projection_aux" in results
                    and self._normalize_instance(views[view_idx].get("instance")) == "remote"
                ):
                    aux = results["remote_projection_aux"]
                    curr_output["remote_projection_rel_height_pred"] = aux["rel_height"][:, view_idx, ...]
                    curr_output["remote_projection_offset_xy_pred"] = aux["offset_xy"][:, view_idx, ...]
                    curr_output["remote_projection_global_dir_xy_pred"] = aux["global_dir_xy"][:, view_idx, ...]
                    curr_output["remote_projection_global_slope_pred"] = aux["global_slope"][:, view_idx, ...]
                res.append(curr_output)

        return res


class Pi3ModalityEmbeddingWrapper(_BasePi3ExperimentalWrapper):
    MODEL_CLS = Pi3WithModalityEmbedding


class Pi3ModalityEmbeddingRemoteHeadWrapper(_BasePi3ExperimentalWrapper):
    MODEL_CLS = Pi3WithModalityEmbeddingRemoteHead
