from copy import deepcopy

import torch
import torch.nn as nn

from mapanything.models.external.pi3.layers.transformer_head import LinearPts3d
from mapanything.models.external.pi3.models.pi3 import Pi3
from mapanything.models.external.vggt.utils.rotation import mat_to_quat


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
        self.remote_point_decoder = deepcopy(self.point_decoder)
        self.remote_point_head = LinearPts3d(
            patch_size=self.patch_size, dec_embed_dim=1024, output_dim=3
        )
        self.remote_conf_decoder = deepcopy(self.conf_decoder)
        self.remote_conf_head = LinearPts3d(
            patch_size=self.patch_size, dec_embed_dim=1024, output_dim=1
        )

    def mirror_aerial_modules_into_remote(self):
        self.remote_point_decoder.load_state_dict(
            deepcopy(self.point_decoder.state_dict())
        )
        self.remote_point_head.load_state_dict(deepcopy(self.point_head.state_dict()))
        self.remote_conf_decoder.load_state_dict(
            deepcopy(self.conf_decoder.state_dict())
        )
        self.remote_conf_head.load_state_dict(deepcopy(self.conf_head.state_dict()))

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

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )


class _BasePi3ExperimentalWrapper(torch.nn.Module):
    MODEL_CLS = Pi3

    def __init__(
        self,
        name,
        torch_hub_force_reload,
        load_pretrained_weights=True,
        pos_type="rope100",
        decoder_size="large",
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload

        self.model = self.MODEL_CLS(pos_type=pos_type, decoder_size=decoder_size)
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

                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                    }
                )

        return res


class Pi3ModalityEmbeddingWrapper(_BasePi3ExperimentalWrapper):
    MODEL_CLS = Pi3WithModalityEmbedding


class Pi3ModalityEmbeddingRemoteHeadWrapper(_BasePi3ExperimentalWrapper):
    MODEL_CLS = Pi3WithModalityEmbeddingRemoteHead
