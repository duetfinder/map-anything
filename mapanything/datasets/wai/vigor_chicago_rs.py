# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Direct VIGOR Chicago remote-sensing dataset backed by the unified RS root.

Supports:
- single-provider or multi-provider expansion
- deterministic crop augmentation for train/val/test via config
- explicit virtual dataset expansion via crop variants
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

from mapanything.datasets.base.easy_dataset import EasyDataset
from mapanything.datasets.wai.vigor_chicago_rs_common import (
    available_providers,
    filter_scene_names_by_cities,
    load_pointmap_modalities,
    load_scene_list,
    make_rng_seed,
    normalize_cities,
    normalize_providers,
    preprocess_rs_modalities,
    required_rs_paths,
    resolve_scene_list_path,
    scene_sort_key,
)


class VigorChicagoRS(EasyDataset, torch.utils.data.Dataset):
    def __init__(
        self,
        ROOT,
        split=None,
        provider='Google_Satellite',
        providers=None,
        dataset_metadata_dir=None,
        scene_list_path=None,
        overfit_num_sets=None,
        transform='imgnorm',
        data_norm_type='identity',
        resolution=(518, 518),
        skip_missing=False,
        provider_sampling_mode='expand',
        num_augmented_crops_per_sample=1,
        cities=None,
        crop_mode='none',
        crop_scale_range=(1.0, 1.0),
        image_resize_mode='nearest',
        label_resize_mode='nearest',
    ):
        self.root = Path(ROOT)
        self.split = split
        self.dataset_metadata_dir = (
            Path(dataset_metadata_dir) if dataset_metadata_dir is not None else None
        )
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        self.overfit_num_sets = overfit_num_sets
        self.resolution = tuple(resolution)
        self.skip_missing = skip_missing
        self.data_norm_type = data_norm_type
        self.num_views = 1
        self._resolutions = [tuple(resolution)]
        self._seed_offset = 0
        self.provider_sampling_mode = provider_sampling_mode
        self.num_augmented_crops_per_sample = max(1, int(num_augmented_crops_per_sample))
        self.cities = normalize_cities(cities)
        self.crop_mode = crop_mode
        self.crop_scale_range = tuple(crop_scale_range)
        self.image_resize_mode = image_resize_mode
        self.label_resize_mode = label_resize_mode
        normalized_providers = normalize_providers(providers)
        if normalized_providers is None and provider is not None:
            normalized_providers = normalize_providers(provider)
        self.providers = normalized_providers

        if provider_sampling_mode != 'expand':
            raise ValueError(
                f"Unsupported provider_sampling_mode: {provider_sampling_mode}. "
                "Currently only 'expand' is implemented."
            )

        if transform == 'imgnorm':
            self.transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported transform: {transform}')

        if self.scene_list_path is not None:
            scene_names = load_scene_list(self.scene_list_path)
        elif self.split is not None and self.dataset_metadata_dir is not None:
            split_path = resolve_scene_list_path(self.dataset_metadata_dir, self.split)
            scene_names = load_scene_list(split_path)
        else:
            scene_names = sorted(
                [
                    p.name
                    for p in self.root.iterdir()
                    if p.is_dir() and p.name.startswith('location_')
                ],
                key=scene_sort_key,
            )

        scene_names = filter_scene_names_by_cities(scene_names, self.cities)

        base_samples = []
        for scene_name in scene_names:
            scene_root = self.root / scene_name
            if not scene_root.exists():
                if skip_missing:
                    continue
                raise FileNotFoundError(f'Missing scene directory: {scene_root}')

            candidate_providers = self.providers or available_providers(scene_root)
            if not candidate_providers:
                if skip_missing:
                    continue
                raise FileNotFoundError(f'No provider directories found under: {scene_root}')

            for provider_name in candidate_providers:
                provider_dir = scene_root / provider_name
                if not provider_dir.exists():
                    if skip_missing:
                        continue
                    raise FileNotFoundError(f'Missing provider directory: {provider_dir}')

                required = required_rs_paths(provider_dir)
                missing = [str(path) for path in required.values() if not path.exists()]
                if missing:
                    if skip_missing:
                        continue
                    raise FileNotFoundError(
                        f'Missing required RS files for {scene_name}/{provider_name}: {missing}'
                    )

                base_samples.append(
                    {
                        'scene_name': scene_name,
                        'remote_provider': provider_name,
                        **{k: str(v) for k, v in required.items()},
                    }
                )

        if overfit_num_sets is not None:
            base_samples = base_samples[:overfit_num_sets]

        self.base_samples = base_samples

    def __len__(self):
        return len(self.base_samples) * self.num_augmented_crops_per_sample

    def _set_seed_offset(self, idx):
        self._seed_offset = idx

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]

        base_idx = idx // self.num_augmented_crops_per_sample
        aug_idx = idx % self.num_augmented_crops_per_sample
        sample = self.base_samples[base_idx]

        remote_image = Image.open(sample['remote_image_path']).convert('RGB')
        remote_pointmap, remote_valid_mask, remote_height_map = load_pointmap_modalities(
            Path(sample['remote_pointmap_path'])
        )

        rng = np.random.default_rng(
            make_rng_seed(self.split, base_idx, aug_idx, self._seed_offset)
        )
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
            resolution=self.resolution,
            crop_mode=self.crop_mode,
            crop_scale_range=self.crop_scale_range,
            image_resize_mode=self.image_resize_mode,
            label_resize_mode=self.label_resize_mode,
            rng=rng,
        )
        remote_image = self.transform(remote_image)

        view = {
            'img': remote_image,
            'data_norm_type': self.data_norm_type,
            'dataset': 'vigor_chicago_rs',
            'label': sample['scene_name'],
            'instance': sample['remote_provider'],
            'remote_pointmap': remote_pointmap,
            'remote_valid_mask': remote_valid_mask,
            'remote_height_map': remote_height_map,
            'remote_crop_box_xyxy': np.asarray(crop_box, dtype=np.int32),
            'remote_aug_variant': int(aug_idx),
        }

        return [view]
