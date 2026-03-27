# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Direct VIGOR Chicago remote-sensing dataset backed by the unified outputs/dataset/vigor_chicago_rs root.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image


class VigorChicagoRS(torch.utils.data.Dataset):
    def __init__(
        self,
        ROOT,
        split=None,
        provider='Google_Satellite',
        dataset_metadata_dir=None,
        scene_list_path=None,
        overfit_num_sets=None,
        transform='imgnorm',
        resolution=(518, 518),
        skip_missing=False,
    ):
        self.root = Path(ROOT)
        self.split = split
        self.provider = provider
        self.dataset_metadata_dir = (
            Path(dataset_metadata_dir) if dataset_metadata_dir is not None else None
        )
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        self.overfit_num_sets = overfit_num_sets
        self.resolution = tuple(resolution)
        self.skip_missing = skip_missing

        if transform == 'imgnorm':
            self.transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported transform: {transform}')

        if self.scene_list_path is not None:
            scene_names = self._load_scene_list(self.scene_list_path)
        elif self.split is not None and self.dataset_metadata_dir is not None:
            split_path = (
                self.dataset_metadata_dir
                / self.split
                / f'vigor_chicago_scene_list_{self.split}.npy'
            )
            scene_names = self._load_scene_list(split_path)
        else:
            scene_names = sorted(
                [p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith('location_')],
                key=self._scene_sort_key,
            )

        samples = []
        for scene_name in scene_names:
            provider_dir = self.root / scene_name / provider
            if not provider_dir.exists():
                if skip_missing:
                    continue
                raise FileNotFoundError(f'Missing provider directory: {provider_dir}')

            required = {
                'remote_image_path': provider_dir / 'image.png',
                'remote_pointmap_path': provider_dir / 'pixel_to_point_map.npz',
                'remote_valid_mask_path': provider_dir / 'valid_mask.npy',
                'remote_height_map_path': provider_dir / 'height_map.npy',
                'remote_info_path': provider_dir / 'info.json',
            }
            missing = [str(path) for path in required.values() if not path.exists()]
            if missing:
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f'Missing required RS files for {scene_name}/{provider}: {missing}'
                )

            samples.append(
                {
                    'scene_name': scene_name,
                    'remote_provider': provider,
                    'remote_scene_dir': str(provider_dir),
                    **{k: str(v) for k, v in required.items()},
                }
            )

        if overfit_num_sets is not None:
            samples = samples[:overfit_num_sets]

        self.samples = samples

    @staticmethod
    def _scene_sort_key(name: str) -> tuple[int, str]:
        try:
            return (int(name.split('_')[-1]), name)
        except Exception:
            return (10**12, name)

    @staticmethod
    def _load_scene_list(path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f'Missing scene list: {path}')
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        remote_image = Image.open(sample['remote_image_path']).convert('RGB')
        remote_image = remote_image.resize(
            (self.resolution[1], self.resolution[0]), resample=Image.BILINEAR
        )
        remote_image = self.transform(remote_image)

        remote_pointmap = np.load(sample['remote_pointmap_path'])['xyz'].astype(np.float32)
        remote_valid_mask = np.load(sample['remote_valid_mask_path']).astype(bool)
        remote_height_map = np.load(sample['remote_height_map_path']).astype(np.float32)
        with open(sample['remote_info_path'], 'r', encoding='utf-8') as f:
            remote_info = json.load(f)

        if remote_pointmap.shape[:2] != self.resolution:
            pointmap_chw = torch.from_numpy(remote_pointmap).permute(2, 0, 1).unsqueeze(0)
            pointmap_chw = torch.nn.functional.interpolate(
                pointmap_chw,
                size=self.resolution,
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
                size=self.resolution,
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
                size=self.resolution,
                mode='nearest',
            )
            remote_height_map = height_map_t[0, 0].numpy()

        return {
            'scene_name': sample['scene_name'],
            'remote_provider': sample['remote_provider'],
            'remote_scene_dir': sample['remote_scene_dir'],
            'remote_image': remote_image,
            'remote_pointmap': remote_pointmap,
            'remote_valid_mask': remote_valid_mask,
            'remote_height_map': remote_height_map,
            'remote_info': remote_info,
            'remote_resolution': self.resolution,
        }
