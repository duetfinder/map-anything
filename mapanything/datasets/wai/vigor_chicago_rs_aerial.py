# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
RS-Aerial benchmark dataset that links an aerial WAI scene with one remote-sensing image and its
per-pixel global pointmap labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

from mapanything.datasets.wai.vigor_chicago_rs_common import (
    load_pointmap_modalities,
    make_rng_seed,
    normalize_cities,
    normalize_providers,
    preprocess_rs_modalities,
    scene_matches_cities,
)
from mapanything.utils.wai.core import load_data


class VigorChicagoRSAerial(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_root,
        split='train',
        provider='Google_Satellite',
        providers=None,
        overfit_num_sets=None,
        cities=None,
        load_aerial_scene_meta=True,
        transform='imgnorm',
        resolution=(518, 518),
        num_augmented_crops_per_sample=1,
        crop_mode='none',
        crop_scale_range=(1.0, 1.0),
        image_resize_mode='nearest',
        label_resize_mode='nearest',
        skip_missing=False,
    ):
        self.metadata_root = Path(metadata_root)
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.load_aerial_scene_meta = load_aerial_scene_meta
        self.resolution = tuple(resolution)
        self.num_augmented_crops_per_sample = max(1, int(num_augmented_crops_per_sample))
        self.crop_mode = crop_mode
        self.crop_scale_range = tuple(crop_scale_range)
        self.image_resize_mode = image_resize_mode
        self.label_resize_mode = label_resize_mode
        self.skip_missing = skip_missing
        self._seed_offset = 0
        normalized_providers = normalize_providers(providers)
        if normalized_providers is None and provider is not None:
            normalized_providers = normalize_providers(provider)
        self.providers = normalized_providers
        self.cities = normalize_cities(cities)

        split_dir = self.metadata_root / split
        aggregate_candidates = [
            split_dir / f'Crossview_rs_aerial_{split}.json',
            split_dir / f'vigor_rs_aerial_{split}.json',
            split_dir / f'vigor_chicago_rs_aerial_{split}.json',
        ]
        aggregate_manifest_path = next(
            (path for path in aggregate_candidates if path.exists()),
            aggregate_candidates[0],
        )
        if not aggregate_manifest_path.exists():
            raise FileNotFoundError(f'Missing aggregate manifest: {aggregate_manifest_path}')

        with open(aggregate_manifest_path, 'r', encoding='utf-8') as f:
            manifests = json.load(f)

        manifests = self._filter_manifests(manifests)
        if overfit_num_sets is not None:
            manifests = manifests[:overfit_num_sets]
        self.base_manifests = manifests

        if transform == 'imgnorm':
            self.transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported transform: {transform}')

    def _filter_manifests(self, manifests):
        manifests = [m for m in manifests if scene_matches_cities(m['scene_name'], self.cities)]

        if not self.providers:
            best_by_scene = {}
            for manifest in manifests:
                best_by_scene.setdefault(manifest['scene_name'], manifest)
            return [best_by_scene[scene] for scene in sorted(best_by_scene.keys())]

        provider_rank = {provider: rank for rank, provider in enumerate(self.providers)}
        manifests = [m for m in manifests if m['remote_provider'] in provider_rank]

        # Keep at most one remote sample per scene for benchmark pairing. If multiple providers
        # are allowed, choose the earliest provider in the requested order.
        best_by_scene = {}
        for manifest in manifests:
            scene_name = manifest['scene_name']
            rank = provider_rank[manifest['remote_provider']]
            current = best_by_scene.get(scene_name)
            if current is None or rank < current[0]:
                best_by_scene[scene_name] = (rank, manifest)
        return [best_by_scene[scene][1] for scene in sorted(best_by_scene.keys())]

    def __len__(self):
        return len(self.base_manifests) * self.num_augmented_crops_per_sample

    def _set_seed_offset(self, idx):
        self._seed_offset = idx

    def __getitem__(self, idx):
        base_idx = idx // self.num_augmented_crops_per_sample
        aug_idx = idx % self.num_augmented_crops_per_sample
        manifest = self.base_manifests[base_idx]

        remote_image_path = Path(manifest['remote_image_path'])
        remote_pointmap_path = Path(manifest['remote_pointmap_path'])
        remote_info_path = Path(manifest['remote_info_path'])

        required_paths = [
            remote_image_path,
            remote_pointmap_path,
            remote_info_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            if self.skip_missing:
                raise IndexError(f'Missing remote files for benchmark sample: {missing}')
            raise FileNotFoundError(f'Missing remote files for benchmark sample: {missing}')

        remote_image = Image.open(remote_image_path).convert('RGB')
        remote_pointmap, remote_valid_mask, remote_height_map = load_pointmap_modalities(
            remote_pointmap_path
        )
        with open(remote_info_path, 'r', encoding='utf-8') as f:
            remote_info = json.load(f)

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

        item = {
            'scene_name': manifest['scene_name'],
            'aerial_scene_dir': manifest['aerial_scene_dir'],
            'aerial_scene_meta_path': manifest['aerial_scene_meta'],
            'remote_provider': manifest['remote_provider'],
            'remote_projection_type': manifest['remote_projection_type'],
            'remote_image': remote_image,
            'remote_pointmap': remote_pointmap,
            'remote_valid_mask': remote_valid_mask,
            'remote_height_map': remote_height_map,
            'remote_info': remote_info,
            'remote_resolution': self.resolution,
            'remote_crop_box_xyxy': np.asarray(crop_box, dtype=np.int32),
            'remote_aug_variant': int(aug_idx),
        }

        if self.load_aerial_scene_meta:
            item['aerial_scene_meta'] = load_data(
                manifest['aerial_scene_meta'], 'scene_meta'
            )

        return item
