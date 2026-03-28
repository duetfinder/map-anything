# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Joint VIGOR Chicago dataset that augments aerial multi-view samples with per-scene RS supervision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

from mapanything.datasets.wai.vigor_chicago import VigorChicagoWAI


class VigorChicagoJointRSAerial(VigorChicagoWAI):
    def __init__(
        self,
        *args,
        remote_ROOT,
        remote_provider='Google_Satellite',
        remote_resolution=(518, 518),
        remote_transform='imgnorm',
        skip_missing_remote=False,
        **kwargs,
    ):
        self.remote_ROOT = Path(remote_ROOT)
        self.remote_provider = remote_provider
        self.remote_resolution = tuple(remote_resolution)
        self.skip_missing_remote = skip_missing_remote

        if remote_transform == 'imgnorm':
            self.remote_transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported remote_transform: {remote_transform}')

        super().__init__(*args, **kwargs)

        available_scenes = []
        self.remote_scene_dirs = {}
        for scene_name in self.scenes:
            remote_scene_dir = self.remote_ROOT / scene_name / self.remote_provider
            required = [
                remote_scene_dir / 'image.png',
                remote_scene_dir / 'pixel_to_point_map.npz',
                remote_scene_dir / 'valid_mask.npy',
                remote_scene_dir / 'height_map.npy',
                remote_scene_dir / 'info.json',
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                if skip_missing_remote:
                    continue
                raise FileNotFoundError(
                    f'Missing RS files for {scene_name}/{self.remote_provider}: {missing}'
                )
            self.remote_scene_dirs[scene_name] = remote_scene_dir
            available_scenes.append(scene_name)

        self.scenes = available_scenes
        self.num_of_scenes = len(self.scenes)

    def _load_remote_sample(self, scene_name: str) -> dict:
        remote_scene_dir = self.remote_scene_dirs[scene_name]

        remote_image = Image.open(remote_scene_dir / 'image.png').convert('RGB')
        remote_image = remote_image.resize(
            (self.remote_resolution[1], self.remote_resolution[0]),
            resample=Image.BILINEAR,
        )
        remote_image = self.remote_transform(remote_image)

        remote_pointmap = np.load(remote_scene_dir / 'pixel_to_point_map.npz')['xyz'].astype(
            np.float32
        )
        remote_valid_mask = np.load(remote_scene_dir / 'valid_mask.npy').astype(bool)
        remote_height_map = np.load(remote_scene_dir / 'height_map.npy').astype(np.float32)

        with open(remote_scene_dir / 'info.json', 'r', encoding='utf-8') as f:
            remote_info = json.load(f)

        if remote_pointmap.shape[:2] != self.remote_resolution:
            pointmap_chw = torch.from_numpy(remote_pointmap).permute(2, 0, 1).unsqueeze(0)
            pointmap_chw = torch.nn.functional.interpolate(
                pointmap_chw,
                size=self.remote_resolution,
                mode='nearest',
            )
            remote_pointmap = pointmap_chw[0].permute(1, 2, 0).numpy()

            valid_mask_t = (
                torch.from_numpy(remote_valid_mask.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            valid_mask_t = torch.nn.functional.interpolate(
                valid_mask_t,
                size=self.remote_resolution,
                mode='nearest',
            )
            remote_valid_mask = valid_mask_t[0, 0].numpy() > 0.5

            height_map_t = (
                torch.from_numpy(remote_height_map.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            height_map_t = torch.nn.functional.interpolate(
                height_map_t,
                size=self.remote_resolution,
                mode='nearest',
            )
            remote_height_map = height_map_t[0, 0].numpy()

        return {
            'remote_scene_dir': str(remote_scene_dir),
            'remote_provider': self.remote_provider,
            'remote_projection_type': str(
                remote_info.get('projection_type', 'rs_global_projective')
            ),
            'remote_info_path': str(remote_scene_dir / 'info.json'),
            'remote_image': remote_image,
            'remote_pointmap': remote_pointmap,
            'remote_valid_mask': remote_valid_mask,
            'remote_height_map': remote_height_map,
        }

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        scene_name = views[0]['label']
        remote_sample = self._load_remote_sample(scene_name)

        for view in views:
            view.update(remote_sample)

        return views
