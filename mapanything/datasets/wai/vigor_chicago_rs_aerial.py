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

from mapanything.utils.wai.core import load_data


class VigorChicagoRSAerial(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_root,
        split="train",
        provider="Google_Satellite",
        overfit_num_sets=None,
        load_aerial_scene_meta=True,
        transform="imgnorm",
        resolution=(518, 518),
    ):
        self.metadata_root = Path(metadata_root)
        self.split = split
        self.provider = provider
        self.overfit_num_sets = overfit_num_sets
        self.load_aerial_scene_meta = load_aerial_scene_meta
        self.resolution = tuple(resolution)

        split_dir = self.metadata_root / split
        aggregate_manifest_path = split_dir / f"vigor_chicago_rs_aerial_{split}.json"
        if not aggregate_manifest_path.exists():
            raise FileNotFoundError(f"Missing aggregate manifest: {aggregate_manifest_path}")

        with open(aggregate_manifest_path, "r", encoding="utf-8") as f:
            manifests = json.load(f)

        manifests = [m for m in manifests if m["remote_provider"] == provider]
        if overfit_num_sets is not None:
            manifests = manifests[:overfit_num_sets]

        self.manifests = manifests

        if transform == "imgnorm":
            self.transform = tvf.ToTensor()
        else:
            raise ValueError(f"Unsupported transform: {transform}")

    def __len__(self):
        return len(self.manifests)

    def __getitem__(self, idx):
        manifest = self.manifests[idx]

        remote_image = Image.open(manifest["remote_image_path"]).convert("RGB")
        remote_image = remote_image.resize(
            (self.resolution[1], self.resolution[0]), resample=Image.BILINEAR
        )
        remote_image = self.transform(remote_image)

        remote_pointmap = np.load(manifest["remote_pointmap_path"])["xyz"].astype(np.float32)
        remote_valid_mask = np.load(manifest["remote_valid_mask_path"]).astype(bool)
        remote_height_map = np.load(manifest["remote_height_map_path"]).astype(np.float32)
        with open(manifest["remote_info_path"], "r", encoding="utf-8") as f:
            remote_info = json.load(f)

        if remote_pointmap.shape[:2] != self.resolution:
            pointmap_chw = torch.from_numpy(remote_pointmap).permute(2, 0, 1).unsqueeze(0)
            pointmap_chw = torch.nn.functional.interpolate(
                pointmap_chw,
                size=self.resolution,
                mode="nearest",
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
                mode="nearest",
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
                mode="nearest",
            )
            remote_height_map = height_map_t[0, 0].numpy()

        item = {
            "scene_name": manifest["scene_name"],
            "aerial_scene_dir": manifest["aerial_scene_dir"],
            "aerial_scene_meta_path": manifest["aerial_scene_meta"],
            "remote_provider": manifest["remote_provider"],
            "remote_projection_type": manifest["remote_projection_type"],
            "remote_image": remote_image,
            "remote_pointmap": remote_pointmap,
            "remote_valid_mask": remote_valid_mask,
            "remote_height_map": remote_height_map,
            "remote_info": remote_info,
            "remote_resolution": self.resolution,
        }

        if self.load_aerial_scene_meta:
            item["aerial_scene_meta"] = load_data(
                manifest["aerial_scene_meta"], "scene_meta"
            )

        return item
