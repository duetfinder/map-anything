#!/usr/bin/env python3
"""Generate lightweight MoGe-2 gradient priors for clean Crossview remote metadata."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything.models.external.moge import MoGeWrapper  # noqa: E402


@dataclass(frozen=True)
class RemoteEntry:
    split: str
    scene_name: str
    provider: str
    image_path: Path
    pointmap_path: Path
    remote_scene_dir: Path


def parse_csv(value: str | None) -> set[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def city_of(scene_name: str) -> str:
    return str(scene_name).split("__", 1)[0] if "__" in str(scene_name) else "chicago"


def split_limit(args: argparse.Namespace, split: str) -> int | None:
    specific = getattr(args, f"max_{split}_entries", None)
    if specific is not None:
        return specific
    return args.max_entries_per_split


def load_allowed_scenes(args: argparse.Namespace) -> set[str] | None:
    if args.scene_list_path is None:
        return None
    scene_list = np.load(args.scene_list_path, allow_pickle=True).tolist()
    return {str(scene_name) for scene_name in scene_list}


def load_entries(args: argparse.Namespace, split: str) -> list[RemoteEntry]:
    metadata_path = (
        args.metadata_dir
        / split
        / f"Crossview_rs_aerial_{split}.json"
    )
    with metadata_path.open("r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    cities = parse_csv(args.cities)
    providers = parse_csv(args.providers)
    allowed_scenes = load_allowed_scenes(args)
    entries: list[RemoteEntry] = []
    seen_dirs: set[Path] = set()
    limit = split_limit(args, split)

    for item in raw_entries:
        scene_name = str(item["scene_name"])
        provider = str(item["remote_provider"])
        if allowed_scenes is not None and scene_name not in allowed_scenes:
            continue
        if cities is not None and city_of(scene_name) not in cities:
            continue
        if providers is not None and provider not in providers:
            continue
        remote_scene_dir = Path(item["remote_scene_dir"])
        if remote_scene_dir in seen_dirs:
            continue
        image_path = Path(item["remote_image_path"])
        pointmap_path = Path(item["remote_pointmap_path"])
        if not image_path.exists() or not pointmap_path.exists():
            continue
        entries.append(
            RemoteEntry(
                split=split,
                scene_name=scene_name,
                provider=provider,
                image_path=image_path,
                pointmap_path=pointmap_path,
                remote_scene_dir=remote_scene_dir,
            )
        )
        seen_dirs.add(remote_scene_dir)
        if limit is not None and len(entries) >= limit:
            break

    return entries


def robust_affine_align(depth: np.ndarray, gt_z: np.ndarray, valid: np.ndarray, min_valid: int):
    mask = valid & np.isfinite(depth) & np.isfinite(gt_z)
    if int(mask.sum()) < min_valid:
        return None

    fit_mask = mask
    coeff = None
    for _ in range(3):
        d = depth[fit_mask].astype(np.float64)
        z = gt_z[fit_mask].astype(np.float64)
        if d.size < min_valid:
            break
        design = np.stack([d, np.ones_like(d)], axis=1)
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)
        aligned = coeff[0] * depth + coeff[1]
        residual = np.abs(aligned[mask] - gt_z[mask])
        cutoff = np.percentile(residual, 90.0)
        fit_mask = mask & (np.abs(aligned - gt_z) <= cutoff)

    if coeff is None or not np.isfinite(coeff).all():
        return None
    aligned = (coeff[0] * depth + coeff[1]).astype(np.float32)
    residual = np.abs(aligned[mask] - gt_z[mask])
    return aligned, float(coeff[0]), float(coeff[1]), {
        "valid_count": int(mask.sum()),
        "residual_mean": float(np.mean(residual)),
        "residual_p50": float(np.percentile(residual, 50.0)),
        "residual_p90": float(np.percentile(residual, 90.0)),
        "residual_p95": float(np.percentile(residual, 95.0)),
    }


def gradient_prior(
    height: np.ndarray,
    *,
    edge_percentile: float,
    prior_weight_base: float,
) -> dict[str, np.ndarray | float]:
    height = height.astype(np.float32)
    finite = np.isfinite(height)
    safe_height = np.where(finite, height, np.nanmedian(height[finite]) if finite.any() else 0.0)

    gx = np.zeros_like(safe_height, dtype=np.float32)
    gy = np.zeros_like(safe_height, dtype=np.float32)
    gx[:, 1:-1] = safe_height[:, 2:] - safe_height[:, :-2]
    gx[:, 0] = safe_height[:, 1] - safe_height[:, 0]
    gx[:, -1] = safe_height[:, -1] - safe_height[:, -2]
    gy[1:-1, :] = safe_height[2:, :] - safe_height[:-2, :]
    gy[0, :] = safe_height[1, :] - safe_height[0, :]
    gy[-1, :] = safe_height[-1, :] - safe_height[-2, :]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    finite_mag = mag[np.isfinite(mag)]
    edge_threshold = float(np.percentile(finite_mag, edge_percentile)) if finite_mag.size else 1.0
    edge_threshold = max(edge_threshold, 1e-6)
    grad_xy = np.stack([gx, gy], axis=-1)
    grad_xy = grad_xy / np.maximum(mag[..., None], 1e-6)
    edge_mask = mag >= edge_threshold
    mag_norm = np.clip(mag / edge_threshold, 0.0, 1.0).astype(np.float32)
    prior_weight = (float(prior_weight_base) * mag_norm).astype(np.float32)
    prior_weight[~finite] = 0.0
    edge_mask[~finite] = False
    grad_xy[~finite] = 0.0
    mag[~finite] = 0.0

    return {
        "moge_grad_xy": grad_xy.astype(np.float16),
        "moge_grad_mag": mag.astype(np.float16),
        "moge_edge_mask": edge_mask.astype(bool),
        "moge_prior_weight": prior_weight.astype(np.float16),
        "moge_edge_threshold": np.float32(edge_threshold),
    }


def infer_moge_depth(model: MoGeWrapper, image_path: Path, resolution: int, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image_small = image.resize((resolution, resolution), resample)
    image_tensor = to_tensor(image_small).unsqueeze(0).to(device)
    views = [{"img": image_tensor, "data_norm_type": ["identity"]}]

    with torch.inference_mode():
        output = model(views)[0]
        depth = output["depth_z"][..., 0].float()
        depth = F.interpolate(
            depth[:, None],
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return depth.detach().cpu().numpy().astype(np.float32)


def process_entry(
    model: MoGeWrapper,
    entry: RemoteEntry,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[bool, str]:
    output_path = entry.remote_scene_dir / args.output_name
    if output_path.exists() and not args.overwrite:
        return False, "exists"

    pointmap_npz = np.load(entry.pointmap_path)
    gt_xyz = pointmap_npz["xyz"].astype(np.float32)
    valid = np.isfinite(gt_xyz).all(axis=-1)
    gt_z = gt_xyz[..., 2].astype(np.float32)

    depth = infer_moge_depth(model, entry.image_path, args.moge_resolution, device)
    aligned = robust_affine_align(depth, gt_z, valid, args.min_valid)
    if aligned is None:
        return False, "insufficient_valid"
    aligned_height, scale, shift, stats = aligned
    if args.max_residual_p95 is not None and stats["residual_p95"] > args.max_residual_p95:
        if output_path.exists() and args.overwrite:
            output_path.unlink()
        return False, f"residual_p95_too_high={stats['residual_p95']:.4g}"

    prior = gradient_prior(
        aligned_height,
        edge_percentile=args.edge_percentile,
        prior_weight_base=args.prior_weight_base,
    )
    payload = {
        **prior,
        "moge_confidence_mask": np.isfinite(aligned_height),
        "moge_affine_scale": np.float32(scale),
        "moge_affine_shift": np.float32(shift),
        "moge_valid_count": np.int32(stats["valid_count"]),
        "moge_residual_mean": np.float32(stats["residual_mean"]),
        "moge_residual_p50": np.float32(stats["residual_p50"]),
        "moge_residual_p90": np.float32(stats["residual_p90"]),
        "moge_residual_p95": np.float32(stats["residual_p95"]),
    }
    if args.save_aligned_height:
        payload["moge_aligned_height"] = aligned_height.astype(np.float16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return True, (
        f"ok scale={scale:.4g} shift={shift:.4g} "
        f"p95={stats['residual_p95']:.4g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--cities", default="all")
    parser.add_argument("--providers", default="Google_Satellite,Bing_Satellite")
    parser.add_argument("--scene-list-path", type=Path, default=None)
    parser.add_argument("--output-name", default="moge_prior.npz")
    parser.add_argument("--moge-resolution", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-entries-per-split", type=int, default=None)
    parser.add_argument("--max-train-entries", type=int, default=None)
    parser.add_argument("--max-val-entries", type=int, default=None)
    parser.add_argument("--max-test-entries", type=int, default=None)
    parser.add_argument("--min-valid", type=int, default=128)
    parser.add_argument("--edge-percentile", type=float, default=95.0)
    parser.add_argument("--prior-weight-base", type=float, default=0.2)
    parser.add_argument("--max-residual-p95", type=float, default=None)
    parser.add_argument("--save-aligned-height", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for MoGe prior generation with the requested device")
    device = torch.device(args.device)
    model = MoGeWrapper(name="moge-2", model_string="Ruicheng/moge-2-vitl").to(device).eval()

    total_written = 0
    total_seen = 0
    for split in args.splits:
        entries = load_entries(args, split)
        print(f"[{split}] selected {len(entries)} entries", flush=True)
        split_written = 0
        for idx, entry in enumerate(entries, 1):
            total_seen += 1
            try:
                wrote, message = process_entry(model, entry, args, device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                raise
            split_written += int(wrote)
            total_written += int(wrote)
            if idx == 1 or idx % 10 == 0 or wrote:
                print(
                    f"[{split} {idx:04d}/{len(entries):04d}] "
                    f"{entry.scene_name}/{entry.provider}: {message}",
                    flush=True,
                )
        print(f"[{split}] wrote {split_written}/{len(entries)} priors", flush=True)
    print(f"done: wrote {total_written}/{total_seen} priors", flush=True)


if __name__ == "__main__":
    main()
