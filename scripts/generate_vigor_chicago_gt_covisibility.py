#!/usr/bin/env python3

"""
Generate MapAnything-style covisibility matrices for VIGOR Chicago scenes
using ground-truth depth maps already stored in WAI format.

This mirrors the native WAI covisibility computation logic, but avoids the
argconf/process-state pipeline and reads the scene depth modality directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.wai_processing.utils.covis_utils import (
    compute_frustum_intersection,
    load_scene_data,
    project_points_to_views,
    sample_depths_at_reprojections,
)
from mapanything.utils.wai.core import load_data, store_data


class AttrDict(dict):
    """Minimal dict/object hybrid for WAI utility compatibility."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate native-style covisibility from GT depth for VIGOR Chicago WAI scenes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/dataset/vigor_chicago_wai"),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Specific scene name(s) to process. Can be repeated.",
    )
    parser.add_argument(
        "--scene-regex",
        type=str,
        default=None,
        help="Optional regex to filter scene names.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Only process the first N matching scenes.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="covisibility/v0_gtdepth_native",
        help="Relative output directory inside each scene.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Long-side resize used by native covisibility preprocessing.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--disable-frustum-check",
        action="store_true",
        help="Disable frustum intersection pruning before pairwise reprojection.",
    )
    parser.add_argument("--depth-assoc-error-thres", type=float, default=0.5)
    parser.add_argument("--depth-assoc-error-temp", type=float, default=0.5)
    parser.add_argument("--depth-assoc-rel-error-thres", type=float, default=0.005)
    parser.add_argument(
        "--denominator-mode",
        choices=["full", "valid_target_depth"],
        default="full",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing covisibility output in the target directory.",
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, scene_name: str) -> AttrDict:
    return AttrDict(
        {
            "root": str(args.dataset_root),
            "scene_filters": [scene_name],
            "frame_modalities": ["depth"],
            "key_remap": {"depth": "depth"},
            "target_size": args.target_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "perform_frustum_check": not args.disable_frustum_check,
            "depth_assoc_error_thres": args.depth_assoc_error_thres,
            "depth_assoc_error_temp": args.depth_assoc_error_temp,
            "depth_assoc_rel_error_thres": args.depth_assoc_rel_error_thres,
            "denominator_mode": args.denominator_mode,
        }
    )


def list_scenes(args: argparse.Namespace) -> list[str]:
    scene_names = sorted(
        [path.name for path in args.dataset_root.iterdir() if path.is_dir()],
        key=natural_key,
    )
    if args.scene:
        requested = set(args.scene)
        scene_names = [name for name in scene_names if name in requested]
    if args.scene_regex:
        pattern = re.compile(args.scene_regex)
        scene_names = [name for name in scene_names if pattern.fullmatch(name)]
    if args.max_scenes is not None:
        scene_names = scene_names[: args.max_scenes]
    return scene_names


def compute_pairwise_covisibility(cfg: AttrDict, scene_name: str) -> tuple[torch.Tensor, dict]:
    timings: dict[str, float] = {}

    load_start = time.time()
    scene_data = load_scene_data(cfg, scene_name, cfg.device)
    timings["load_scene_data_sec"] = time.time() - load_start

    depths = scene_data["depths"]
    depth_h, depth_w = scene_data["depth_dims"]
    valid_depth_masks = scene_data["valid_depth_masks"]
    intrinsics = scene_data["intrinsics"]
    cam2worlds = scene_data["cam2worlds"]
    world_pts3d = scene_data["world_pts3d"]
    num_frames = int(depths.shape[0])

    frustum_start = time.time()
    frustum_intersection = compute_frustum_intersection(
        cfg,
        depths,
        valid_depth_masks,
        intrinsics,
        cam2worlds,
        cfg.device,
    )
    timings["frustum_sec"] = time.time() - frustum_start

    pairwise_covisibility = torch.zeros((num_frames, num_frames), device="cpu")

    compute_start = time.time()
    for idx in range(num_frames):
        if cfg.perform_frustum_check and frustum_intersection is not None:
            ov_inds = frustum_intersection[idx].argwhere()[:, 0].to(cfg.device)
        else:
            ov_inds = torch.arange(num_frames).to(cfg.device)
        if len(ov_inds) == 0:
            continue

        overlap_score = torch.zeros((num_frames,), device="cpu")
        reprojected_pts, valid_mask, _ = project_points_to_views(
            idx,
            ov_inds,
            depth_h,
            depth_w,
            valid_depth_masks,
            cam2worlds,
            world_pts3d,
            intrinsics,
            cfg.device,
        )
        if valid_mask.any():
            depth_lu, expected_depth = sample_depths_at_reprojections(
                reprojected_pts,
                depths,
                ov_inds,
                depth_h,
                depth_w,
                cfg.device,
            )
            reprojection_error = torch.abs(expected_depth - depth_lu)
            depth_assoc_thres = (
                cfg.depth_assoc_error_thres
                + cfg.depth_assoc_rel_error_thres * expected_depth
                - math.log(0.5) * cfg.depth_assoc_error_temp
            )
            valid_depth_projection = (reprojection_error < depth_assoc_thres) & valid_mask
            if cfg.denominator_mode == "valid_target_depth":
                comp_covisibility_score = valid_depth_projection.sum([1, 2]) / (
                    valid_depth_masks[ov_inds].sum([1, 2]).clamp(1)
                )
                comp_covisibility_score = comp_covisibility_score.clamp(0, 1)
            else:
                comp_covisibility_score = valid_depth_projection.sum([1, 2]) / (
                    depth_h * depth_w
                )
            overlap_score[ov_inds.cpu()] = comp_covisibility_score.cpu()

        pairwise_covisibility[idx] = overlap_score

    timings["pairwise_compute_sec"] = time.time() - compute_start
    timings["num_frames"] = num_frames
    timings["depth_height"] = int(depth_h)
    timings["depth_width"] = int(depth_w)
    timings["frustum_true_pairs"] = int(frustum_intersection.sum().item())
    return pairwise_covisibility, timings


def update_scene_meta(scene_root: Path, out_path: str, mmap_name: str) -> None:
    meta_path = scene_root / "scene_meta.json"
    scene_meta = load_data(meta_path, "scene_meta")
    scene_modalities = scene_meta.setdefault("scene_modalities", {})
    scene_modalities["pairwise_covisibility"] = {
        "scene_key": f"{out_path}/{mmap_name}",
        "format": "mmap",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2)


def process_scene(args: argparse.Namespace, scene_name: str) -> dict[str, object]:
    scene_root = args.dataset_root / scene_name
    out_dir = scene_root / args.out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    target_files = list(out_dir.glob("*.npy"))
    if target_files and not args.overwrite:
        return {
            "scene_name": scene_name,
            "skipped": True,
            "reason": "output_exists",
            "out_dir": str(out_dir),
        }

    for old_file in out_dir.glob("*.npy"):
        old_file.unlink()

    cfg = build_cfg(args, scene_name)
    start = time.time()
    pairwise_covisibility, timings = compute_pairwise_covisibility(cfg, scene_name)
    mmap_name = store_data(out_dir / "pairwise_covisibility.npy", pairwise_covisibility, "mmap")
    update_scene_meta(scene_root, args.out_path, mmap_name)

    result = {
        "scene_name": scene_name,
        "skipped": False,
        "out_dir": str(out_dir),
        "mmap_name": mmap_name,
        "total_sec": time.time() - start,
        **timings,
    }
    with open(out_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"Requested device '{args.device}' is unavailable, falling back to CPU.")
        args.device = "cpu"

    scene_names = list_scenes(args)
    if not scene_names:
        raise ValueError("No scenes matched the provided filters.")

    print(f"Processing {len(scene_names)} scene(s) under {args.dataset_root}")
    results = []
    for scene_name in scene_names:
        print(f"[start] {scene_name}")
        result = process_scene(args, scene_name)
        results.append(result)
        if result["skipped"]:
            print(f"[skip] {scene_name}: {result['reason']}")
        else:
            print(
                f"[done] {scene_name}: total={result['total_sec']:.3f}s "
                f"load={result['load_scene_data_sec']:.3f}s "
                f"frustum={result['frustum_sec']:.3f}s "
                f"pairwise={result['pairwise_compute_sec']:.3f}s"
            )

    summary = {
        "dataset_root": str(args.dataset_root),
        "out_path": args.out_path,
        "device": args.device,
        "num_scenes": len(results),
        "results": results,
    }
    summary_path = args.dataset_root / f"gt_covisibility_summary_{Path(args.out_path).name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
