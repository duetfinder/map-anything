# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Joint VIGOR Chicago dataset that augments aerial multi-view samples with per-scene RS supervision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image

from mapanything.datasets.wai.vigor_chicago import VigorChicagoWAI
from mapanything.datasets.wai.vigor_chicago_rs_common import (
    available_providers,
    empty_moge_prior_modalities,
    load_pointmap_modalities,
    load_moge_prior_modalities,
    load_projection_aux_modalities,
    normalize_providers,
    preprocess_moge_prior_modalities,
    preprocess_projection_aux_modalities,
    preprocess_rs_modalities,
)


PROVIDER_COLUMN_PREFIX = "provider__"


def parse_bool_like(value) -> bool:
    if value is None:
        return False
    value = str(value).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


class VigorChicagoJointRSAerial(VigorChicagoWAI):
    def __init__(
        self,
        *args,
        remote_ROOT,
        remote_providers=None,
        remote_provider_map_csv=None,
        remote_dataset_metadata_dir=None,
        remote_provider_sampling_mode='first_available',
        remote_resolution=(518, 518),
        remote_transform='imgnorm',
        cities=None,
        skip_missing_remote=False,
        remote_crop_mode='none',
        remote_crop_scale_range=(1.0, 1.0),
        remote_num_views=1,
        remote_image_resize_mode='nearest',
        remote_label_resize_mode='nearest',
        **kwargs,
    ):
        self.remote_ROOT = Path(remote_ROOT)
        normalized_providers = normalize_providers(remote_providers)
        self.remote_providers = normalized_providers
        self.remote_provider_map_csv = (
            Path(remote_provider_map_csv)
            if remote_provider_map_csv not in (None, "None", "")
            else None
        )
        self.remote_dataset_metadata_dir = (
            Path(remote_dataset_metadata_dir)
            if remote_dataset_metadata_dir not in (None, "None", "")
            else None
        )
        self.remote_provider_sampling_mode = str(remote_provider_sampling_mode).lower()
        self.remote_resolution = tuple(remote_resolution)
        self.cities = cities
        self.skip_missing_remote = skip_missing_remote
        self.remote_crop_mode = remote_crop_mode
        self.remote_crop_scale_range = tuple(remote_crop_scale_range)
        self.remote_num_views = int(remote_num_views)
        self.remote_image_resize_mode = remote_image_resize_mode
        self.remote_label_resize_mode = remote_label_resize_mode

        if self.remote_num_views < 1:
            raise ValueError(f"remote_num_views must be >= 1, got {remote_num_views}")

        if self.remote_provider_sampling_mode not in {
            'first_available',
            'random',
            'expand',
        }:
            raise ValueError(
                'Unsupported remote_provider_sampling_mode: '
                f'{remote_provider_sampling_mode}'
            )

        if remote_transform == 'imgnorm':
            self.remote_transform = tvf.ToTensor()
        else:
            raise ValueError(f'Unsupported remote_transform: {remote_transform}')

        super().__init__(*args, cities=cities, **kwargs)

        self.remote_provider_map = {}
        if self.remote_provider_map_csv is not None:
            with self.remote_provider_map_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(
                        f"CSV file has no header: {self.remote_provider_map_csv}"
                    )
                if "scene_name" not in reader.fieldnames:
                    raise ValueError(
                        "remote_provider_map_csv must contain a scene_name column"
                    )
                provider_columns = [
                    name for name in reader.fieldnames
                    if str(name).startswith(PROVIDER_COLUMN_PREFIX)
                ]
                for row in reader:
                    scene_name = str(row["scene_name"]).strip()
                    if not scene_name:
                        continue
                    providers: list[str] = []
                    for column in provider_columns:
                        if parse_bool_like(row.get(column)):
                            providers.append(column[len(PROVIDER_COLUMN_PREFIX):])
                    providers = list(dict.fromkeys(providers))
                    if providers:
                        self.remote_provider_map[scene_name] = providers

        self.remote_manifest_by_scene = {}
        if self.remote_dataset_metadata_dir is not None:
            split_dir = self.remote_dataset_metadata_dir / str(self.split)
            for scene_name in self.scenes:
                manifest_path = split_dir / f"{scene_name.replace('/', '__')}.json"
                if not manifest_path.exists():
                    continue
                with manifest_path.open("r", encoding="utf-8") as f:
                    self.remote_manifest_by_scene[scene_name] = json.load(f)

        available_scenes = []
        self.remote_scene_candidates = {}
        self._expanded_scene_entries = []
        for scene_name in self.scenes:
            scene_root = self.remote_ROOT / scene_name
            scene_manifest = self.remote_manifest_by_scene.get(scene_name)
            if self.remote_providers:
                candidate_providers = self.remote_providers
            elif scene_manifest is not None and scene_manifest.get("remote_entries"):
                candidate_providers = [
                    str(entry["remote_provider"])
                    for entry in scene_manifest["remote_entries"]
                ]
            elif scene_name in self.remote_provider_map:
                candidate_providers = self.remote_provider_map[scene_name]
            else:
                candidate_providers = self.remote_providers or available_providers(scene_root)
            if not candidate_providers:
                if skip_missing_remote:
                    continue
                raise FileNotFoundError(f'No provider directories found under: {scene_root}')

            available_remote_entries = []
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
                available_remote_entries.append({
                    'remote_scene_dir': remote_scene_dir,
                    'remote_provider': provider_name,
                })

            if not available_remote_entries:
                if skip_missing_remote:
                    continue
                raise FileNotFoundError(
                    f'Missing RS files for {scene_name} under providers {candidate_providers}: {last_missing}'
                )

            self.remote_scene_candidates[scene_name] = available_remote_entries
            available_scenes.append(scene_name)
            if self.remote_provider_sampling_mode == 'expand':
                for remote_entry in available_remote_entries:
                    self._expanded_scene_entries.append((scene_name, remote_entry))

        self.scenes = available_scenes
        self._scene_name_to_idx = {
            scene_name: scene_idx for scene_idx, scene_name in enumerate(self.scenes)
        }
        if self.remote_provider_sampling_mode == 'expand':
            self.num_of_scenes = len(self._expanded_scene_entries)
        else:
            self.num_of_scenes = len(self.scenes)

    def _resolve_scene_name_and_remote_info(self, sampled_idx: int) -> tuple[str, dict]:
        if self.remote_provider_sampling_mode == 'expand':
            return self._expanded_scene_entries[sampled_idx]

        scene_name = self.scenes[sampled_idx]
        remote_candidates = self.remote_scene_candidates[scene_name]
        if self.remote_provider_sampling_mode == 'random':
            remote_info = remote_candidates[self._rng.integers(0, len(remote_candidates))]
        else:
            remote_info = remote_candidates[0]
        return scene_name, remote_info

    def _load_remote_sample(self, remote_info: dict, *, aug_variant: int = 0) -> dict:
        remote_scene_dir = remote_info['remote_scene_dir']
        remote_provider = remote_info['remote_provider']

        remote_image = Image.open(remote_scene_dir / 'image.png').convert('RGB')
        remote_pointmap, remote_valid_mask, remote_height_map = load_pointmap_modalities(
            remote_scene_dir / 'pixel_to_point_map.npz'
        )
        projection_aux_path = remote_scene_dir / 'projection_aux.npz'
        remote_projection_aux = load_projection_aux_modalities(projection_aux_path)
        moge_prior_path = remote_scene_dir / 'moge_prior.npz'
        remote_moge_prior = load_moge_prior_modalities(moge_prior_path)

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
            rng=self._rng,
        )
        remote_projection_aux = preprocess_projection_aux_modalities(
            remote_projection_aux,
            crop_box,
            self.remote_resolution,
            label_resize_mode=self.remote_label_resize_mode,
        )
        remote_moge_prior = preprocess_moge_prior_modalities(
            remote_moge_prior,
            crop_box,
            self.remote_resolution,
            label_resize_mode='bilinear',
        )
        remote_image = self.remote_transform(remote_image)

        sample = {
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
            'remote_aug_variant': int(aug_variant),
        }
        if remote_projection_aux is not None:
            sample['remote_projection_aux_path'] = str(projection_aux_path)
            sample.update(remote_projection_aux)
        if remote_moge_prior is None:
            remote_moge_prior = empty_moge_prior_modalities(self.remote_resolution)
        sample['remote_moge_prior_path'] = str(moge_prior_path) if moge_prior_path.exists() else ''
        sample.update(remote_moge_prior)
        return sample

    @staticmethod
    def _pack_remote_samples(remote_samples: list[dict]) -> dict:
        packed = {'remote_num_views': len(remote_samples)}
        for sample_idx, remote_sample in enumerate(remote_samples):
            suffix = "" if sample_idx == 0 else f"_{sample_idx}"
            for key, value in remote_sample.items():
                packed[f"{key}{suffix}"] = value
        return packed

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if self.remote_provider_sampling_mode == 'expand':
            scene_name, remote_info = self._resolve_scene_name_and_remote_info(sampled_idx)
            scene_base_idx = self._scene_name_to_idx[scene_name]
            views = super()._get_views(scene_base_idx, num_views_to_sample, resolution)
        else:
            _, remote_info = self._resolve_scene_name_and_remote_info(sampled_idx)
            views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        remote_samples = [
            self._load_remote_sample(remote_info, aug_variant=remote_idx)
            for remote_idx in range(self.remote_num_views)
        ]
        remote_payload = self._pack_remote_samples(remote_samples)

        for view in views:
            view.update(remote_payload)

        return views
