# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
VIGOR Chicago dataset using WAI format data.
"""

import os

import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame


class VigorChicagoWAI(BaseDataset):
    """
    Custom WAI dataset built from VIGOR Chicago reconstruction outputs.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split="train",
        overfit_num_sets=None,
        scene_list_path: str | None = None,
        sample_specific_scene: bool = False,
        specific_scene_name: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, split=split, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.overfit_num_sets = overfit_num_sets
        self.scene_list_path = scene_list_path
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"vigor_chicago_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]

        if self.scene_list_path is not None:
            filtered_scene_list = np.load(self.scene_list_path, allow_pickle=True)
            allowed_scene_set = set(filtered_scene_list.tolist())
            self.scenes = [scene for scene in self.scenes if scene in allowed_scene_set]

        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[: self.overfit_num_sets]

        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = load_data(
            os.path.join(scene_root, "scene_meta.json"), "scene_meta"
        )
        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        pairwise_covis_meta = scene_meta.get("scene_modalities", {}).get(
            "pairwise_covisibility"
        )
        if pairwise_covis_meta:
            covisibility_map_path = os.path.join(
                scene_root, pairwise_covis_meta["scene_key"]
            )
        else:
            covisibility_map_dir = os.path.join(scene_root, "covisibility", "v0")
            covisibility_candidates = sorted(
                f for f in os.listdir(covisibility_map_dir) if f.endswith(".npy")
            )
            covisibility_map_name = next(
                (f for f in covisibility_candidates if "--" in f),
                covisibility_candidates[0],
            )
            covisibility_map_path = os.path.join(
                covisibility_map_dir, covisibility_map_name
            )

        pairwise_covisibility = load_data(covisibility_map_path, "mmap")

        view_indices = self._sample_view_indices(
            num_views_to_sample, num_views_in_scene, pairwise_covisibility
        )

        views = []
        for view_index in view_indices:
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            image = view_data["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,
                    camera_intrinsics=intrinsics,
                    dataset="VigorChicago",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            )

        return views
