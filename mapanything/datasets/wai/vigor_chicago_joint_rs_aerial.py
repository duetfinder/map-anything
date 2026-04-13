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
from mapanything.datasets.wai.vigor_chicago_rs_common import (
    available_providers,
    load_pointmap_modalities,
    normalize_providers,
    preprocess_rs_modalities,
)


class VigorChicagoJointRSAerial(VigorChicagoWAI):
    def __init__(
        self,
        *args,
        remote_ROOT,
        remote_provider='Google_Satellite',
        remote_providers=None,
        remote_resolution=(518, 518),
        remote_transform='imgnorm',
        skip_missing_remote=False,
        remote_crop_mode='none',
        remote_crop_scale_range=(1.0, 1.0),
        remote_image_resize_mode='nearest',
        remote_label_resize_mode='nearest',
        **kwargs,
    ):
        self.remote_ROOT = Path(remote_ROOT)
        normalized_providers = normalize_providers(remote_providers)
        if normalized_providers is None and remote_provider is not None:
            normalized_providers = normalize_providers(remote_provider)
        self.remote_providers = normalized_providers
        self.remote_resolution = tuple(remote_resolution)
        self.skip_missing_remote = skip_missing_remote
        self.remote_crop_mode = remote_crop_mode
        self.remote_crop_scale_range = tuple(remote_crop_scale_range)
        self.remote_image_resize_mode = remote_image_resize_mode
        self.remote_label_resize_mode = remote_label_resize_mode

        if remote_transform == 'imgnorm':
            self.remote_transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported remote_transform: {remote_transform}')

        super().__init__(*args, **kwargs)

        available_scenes = []
        self.remote_scene_info = {}
        for scene_name in self.scenes:
            scene_root = self.remote_ROOT / scene_name
            candidate_providers = self.remote_providers or available_providers(scene_root)
            if not candidate_providers:
                if skip_missing_remote:
                    continue
                raise FileNotFoundError(f'No provider directories found under: {scene_root}')

            selected = None
            last_missing = None
            for provider_name in candidate_providers:
                remote_scene_dir = scene_root / provider_name
                required = [
                    remote_scene_dir / 'image.png',
                    remote_scene_dir / 'pixel_to_point_map.npz',
                    remote_scene_dir / 'info.json',
                ]
                missing = [str(path) for path in required if not path.exists()]
                if missing:
                    last_missing = missing
                    continue
                selected = {
                    'remote_scene_dir': remote_scene_dir,
                    'remote_provider': provider_name,
                }
                break

            if selected is None:
                if skip_missing_remote:
                    continue
                raise FileNotFoundError(
                    f'Missing RS files for {scene_name} under providers {candidate_providers}: {last_missing}'
                )

            self.remote_scene_info[scene_name] = selected
            available_scenes.append(scene_name)

        self.scenes = available_scenes
        self.num_of_scenes = len(self.scenes)

    def _load_remote_sample(self, scene_name: str) -> dict:
        remote_info = self.remote_scene_info[scene_name]
        remote_scene_dir = remote_info['remote_scene_dir']
        remote_provider = remote_info['remote_provider']

        remote_image = Image.open(remote_scene_dir / 'image.png').convert('RGB')
        remote_pointmap, remote_valid_mask, remote_height_map = load_pointmap_modalities(
            remote_scene_dir / 'pixel_to_point_map.npz'
        )

        with open(remote_scene_dir / 'info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)

        (
            remote_image,
            remote_pointmap,
            remote_valid_mask,
            remote_height_map,
            crop_box,
        ) = preprocess_rs_modalities(
            remote_image=remote_image,
            remote_pointmap=remote_pointmap,
            remote_valid_mask=remote_valid_mask,
            remote_height_map=remote_height_map,
            resolution=self.remote_resolution,
            crop_mode=self.remote_crop_mode,
            crop_scale_range=self.remote_crop_scale_range,
            image_resize_mode=self.remote_image_resize_mode,
            label_resize_mode=self.remote_label_resize_mode,
            rng=np.random.default_rng(0),
        )
        remote_image = self.remote_transform(remote_image)

        return {
            'remote_scene_dir': str(remote_scene_dir),
            'remote_provider': remote_provider,
            'remote_projection_type': str(
                info.get('projection_type', 'rs_global_projective')
            ),
            'remote_info_path': str(remote_scene_dir / 'info.json'),
            'remote_image': remote_image,
            'remote_pointmap': remote_pointmap,
            'remote_valid_mask': remote_valid_mask,
            'remote_height_map': remote_height_map,
            'remote_crop_box_xyxy': np.asarray(crop_box, dtype=np.int32),
        }

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        scene_name = views[0]['label']
        remote_sample = self._load_remote_sample(scene_name)

        for view in views:
            view.update(remote_sample)

        return views
