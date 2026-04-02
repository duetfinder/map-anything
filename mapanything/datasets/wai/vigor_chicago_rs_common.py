from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def scene_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split('_')[-1]), name)
    except Exception:
        return (10**12, name)


def load_scene_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing scene list: {path}")
    return [str(x) for x in np.load(path, allow_pickle=True).tolist()]


def normalize_providers(providers):
    if providers is None:
        return None
    if isinstance(providers, str):
        providers = providers.strip()
        if not providers or providers.lower() == 'all':
            return None
        return [providers]
    if isinstance(providers, Iterable):
        result = [str(provider) for provider in providers]
        return result or None
    raise TypeError(f'Unsupported providers value: {providers!r}')


def available_providers(scene_root: Path) -> list[str]:
    return sorted([path.name for path in scene_root.iterdir() if path.is_dir()])


def required_rs_paths(provider_dir: Path) -> dict[str, Path]:
    return {
        'remote_image_path': provider_dir / 'image.png',
        'remote_pointmap_path': provider_dir / 'pixel_to_point_map.npz',
        'remote_valid_mask_path': provider_dir / 'valid_mask.npy',
        'remote_height_map_path': provider_dir / 'height_map.npy',
        'remote_info_path': provider_dir / 'info.json',
    }


def pil_resample_mode(mode: str):
    mode = str(mode).lower()
    if mode == 'nearest':
        return Image.NEAREST
    if mode == 'bilinear':
        return Image.BILINEAR
    if mode == 'bicubic':
        return Image.BICUBIC
    raise ValueError(f'Unsupported PIL resize mode: {mode}')


def torch_resample_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {'nearest', 'bilinear', 'bicubic'}:
        raise ValueError(f'Unsupported torch resize mode: {mode}')
    return mode


def make_rng_seed(split, base_idx: int, aug_idx: int, seed_offset: int = 0) -> int:
    split = '' if split is None else str(split)
    split_hash = sum(ord(ch) for ch in split) % 1000003
    seed = (
        split_hash * 73856093
        + int(base_idx) * 19349663
        + int(aug_idx) * 83492791
        + int(seed_offset) * 2654435761
    ) & 0xFFFFFFFF
    return seed


def sample_crop_box(shape_hw, resolution_hw, crop_mode: str, crop_scale_range, rng):
    height, width = int(shape_hw[0]), int(shape_hw[1])
    target_h, target_w = int(resolution_hw[0]), int(resolution_hw[1])
    crop_mode = str(crop_mode).lower()
    if crop_mode == 'none':
        return (0, 0, width, height)

    if crop_scale_range is None:
        crop_scale_range = (1.0, 1.0)
    scale_min = float(crop_scale_range[0])
    scale_max = float(crop_scale_range[1])
    if not (0 < scale_min <= scale_max <= 1.0):
        raise ValueError(f'Invalid crop_scale_range: {crop_scale_range}')

    crop_size = int(round(min(height, width) * rng.uniform(scale_min, scale_max)))
    crop_size = max(1, min(crop_size, min(height, width)))

    if crop_mode == 'random_scale_center':
        top = max(0, (height - crop_size) // 2)
        left = max(0, (width - crop_size) // 2)
    elif crop_mode == 'random_scale_offset':
        max_top = max(0, height - crop_size)
        max_left = max(0, width - crop_size)
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    else:
        raise ValueError(f'Unsupported crop_mode: {crop_mode}')

    return (left, top, left + crop_size, top + crop_size)


def preprocess_rs_modalities(
    *,
    remote_image,
    remote_pointmap,
    remote_valid_mask,
    remote_height_map,
    resolution,
    crop_mode='none',
    crop_scale_range=(1.0, 1.0),
    image_resize_mode='nearest',
    label_resize_mode='nearest',
    rng=None,
):
    resolution = tuple(int(x) for x in resolution)
    if rng is None:
        rng = np.random.default_rng(0)

    box = sample_crop_box(
        shape_hw=remote_pointmap.shape[:2],
        resolution_hw=resolution,
        crop_mode=crop_mode,
        crop_scale_range=crop_scale_range,
        rng=rng,
    )

    left, top, right, bottom = box
    remote_image = remote_image.crop(box)
    remote_image = remote_image.resize(
        (resolution[1], resolution[0]),
        resample=pil_resample_mode(image_resize_mode),
    )

    remote_pointmap = remote_pointmap[top:bottom, left:right]
    remote_valid_mask = remote_valid_mask[top:bottom, left:right]
    remote_height_map = remote_height_map[top:bottom, left:right]

    label_resize_mode = torch_resample_mode(label_resize_mode)
    if remote_pointmap.shape[:2] != resolution:
        pointmap_chw = torch.from_numpy(remote_pointmap.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        if label_resize_mode == 'nearest':
            pointmap_chw = torch.nn.functional.interpolate(pointmap_chw, size=resolution, mode=label_resize_mode)
        else:
            pointmap_chw = torch.nn.functional.interpolate(pointmap_chw, size=resolution, mode=label_resize_mode, align_corners=False)
        remote_pointmap = pointmap_chw[0].permute(1, 2, 0).numpy()

        valid_mask_t = torch.from_numpy(remote_valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        valid_mask_t = torch.nn.functional.interpolate(valid_mask_t, size=resolution, mode='nearest')
        remote_valid_mask = valid_mask_t[0, 0].numpy() > 0.5

        height_map_t = torch.from_numpy(remote_height_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if label_resize_mode == 'nearest':
            height_map_t = torch.nn.functional.interpolate(height_map_t, size=resolution, mode=label_resize_mode)
        else:
            height_map_t = torch.nn.functional.interpolate(height_map_t, size=resolution, mode=label_resize_mode, align_corners=False)
        remote_height_map = height_map_t[0, 0].numpy()

    return remote_image, remote_pointmap, remote_valid_mask, remote_height_map, box
