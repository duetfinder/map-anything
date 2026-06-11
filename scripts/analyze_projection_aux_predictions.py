#!/usr/bin/env python3
"""Per-scene diagnostics for P7 remote projection auxiliary predictions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything.datasets.wai.vigor_chicago_rs_common import (  # noqa: E402
    _resize_label_array,
    load_projection_aux_modalities,
    preprocess_projection_aux_modalities,
)
from mapanything.utils.image import load_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/traindata/Crossview_rs"))
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--bad-scenes-txt", type=Path, default=None)
    parser.add_argument("--provider", default="Google_Satellite", help="Comma-separated provider names or 'all'.")
    parser.add_argument("--scene-prefix", default=None, help="Optional scene prefix, e.g. newyork__ or chicago__.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-list-path", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=Path("/root/autodl-tmp/traindata/mapanything_metadata"))
    parser.add_argument("--metadata-kind", default="Crossview_rs_aerial")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ignore-metadata", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--label-resize-mode", default="nearest", choices=["nearest", "bilinear", "bicubic"])
    parser.add_argument("--config-path", default="configs/train.yaml")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--preset", default="remote_head", choices=["remote_head", "split"])
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--image-stem-dim", type=int, default=32)
    parser.add_argument("--aux-num-blocks", type=int, default=6)
    parser.add_argument("--split-pixel-heads", action="store_true")
    parser.add_argument("--detach-aux", action="store_true")
    parser.add_argument("--use-rgb-aux", action="store_true", default=True)
    parser.add_argument("--no-use-rgb-aux", dest="use_rgb_aux", action="store_false")
    parser.add_argument("--use-coord-aux", action="store_true", default=True)
    parser.add_argument("--no-use-coord-aux", dest="use_coord_aux", action="store_false")
    parser.add_argument("--positive-slope-aux", action="store_true", default=True)
    parser.add_argument("--no-positive-slope-aux", dest="positive_slope_aux", action="store_false")
    parser.add_argument("--slope-init", type=float, default=0.1)
    parser.add_argument("--offset-scale", type=float, default=32.0)
    parser.add_argument(
        "--rel-height-scale-mode",
        default="pointmap_norm_median",
        choices=["pointmap_norm_median", "pointmap_norm_mean", "fixed", "none"],
        help="How to scale GT rel_height into prediction/loss space for diagnostics.",
    )
    parser.add_argument("--rel-height-scale", type=float, default=1.0, help="Used when --rel-height-scale-mode=fixed.")
    parser.add_argument("--zero-eps", type=float, default=1e-6)
    parser.add_argument("--sort-key", default="badness_score")
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_export_module():
    path = REPO_ROOT / "scripts" / "export_pointcloud_ply.py"
    spec = importlib.util.spec_from_file_location("export_pointcloud_ply", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_export_args(args: argparse.Namespace, remote_dir: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model="vggt",
        checkpoint_path=str(args.checkpoint_path),
        image_folder=str(remote_dir),
        output_path=str(output_dir),
        config_path=args.config_path,
        config_json_path=None,
        model_str=None,
        config_overrides=None,
        hf_model_name=None,
        enable_clash_proxy=False,
        strict=args.strict,
        vggt_joint_remote_export=False,
        vggt_export_mode=None,
        vggt_ordinary_output_head=None,
        vggt_remote_output_head=None,
        vggt_use_remote_private_point_head=False,
        vggt_p5f_lite_export=False,
        vggt_p6a_export=False,
        vggt_p6b_export=False,
        vggt_p7_projection_aux_export=args.preset == "split",
        vggt_p7_remote_head_projection_aux_export=args.preset == "remote_head",
        vggt_projection_aux_hidden_dim=args.hidden_dim,
        vggt_projection_aux_detach_pointmap=args.detach_aux,
        vggt_projection_aux_use_rgb=args.use_rgb_aux,
        vggt_projection_aux_use_coord=args.use_coord_aux,
        vggt_projection_aux_positive_slope=args.positive_slope_aux,
        vggt_projection_aux_slope_init=args.slope_init,
        vggt_projection_aux_num_blocks=args.aux_num_blocks,
        vggt_projection_aux_split_pixel_heads=args.split_pixel_heads,
        vggt_projection_aux_image_stem_dim=args.image_stem_dim,
        include_remote_points=False,
        vggt_late_fusion_type="cross_attention",
        vggt_late_gate_init=1e-3,
        vggt_max_remote_tokens=256,
        vggt_cross_attention_heads=8,
        force_remote_instance=False,
        remote_view_indices=None,
        remote_view_names=["image.png"],
        memory_efficient_inference=False,
        minibatch_size=1,
        resize_mode="fixed_size",
        size=None,
        fixed_width=args.resolution,
        fixed_height=args.resolution,
        resolution_set=518,
        stride=1,
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=50.0,
        voxel_downsample=False,
        voxel_size=None,
        voxel_fraction=0.01,
        export_remote_control_modes=None,
        blank_remote_value=0.5,
        shuffled_remote_image_path=None,
    )


def to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def load_scene_list(path: Path) -> set[str]:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return {str(x) for x in arr.tolist()}
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def resolve_metadata_scene_list(args: argparse.Namespace) -> set[str] | None:
    if args.ignore_metadata:
        return None
    if args.scene_list_path is not None:
        return load_scene_list(args.scene_list_path)
    candidates = [
        args.metadata_dir / args.metadata_kind / args.split / f"{args.metadata_kind}_scene_list_{args.split}.npy",
        args.metadata_dir / args.metadata_kind / args.split / f"Crossview_rs_aerial_scene_list_{args.split}.npy",
        args.metadata_dir / args.metadata_kind / args.split / f"vigor_rs_aerial_scene_list_{args.split}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_scene_list(candidate)
    return None


def iter_remote_dirs(args: argparse.Namespace) -> list[Path]:
    allowed_scenes = resolve_metadata_scene_list(args)
    providers = None if args.provider == "all" else {p.strip() for p in args.provider.split(",") if p.strip()}
    remote_dirs: list[Path] = []
    for scene_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        if args.scene_prefix and not scene.startswith(args.scene_prefix):
            continue
        if allowed_scenes is not None and scene not in allowed_scenes:
            continue
        for provider_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            if providers is not None and provider_dir.name not in providers:
                continue
            if (provider_dir / "image.png").exists() and (provider_dir / "projection_aux.npz").exists():
                remote_dirs.append(provider_dir)
                if args.limit is not None and len(remote_dirs) >= args.limit:
                    return remote_dirs
    return remote_dirs


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float | None:
    valid = mask & np.isfinite(arr)
    return float(arr[valid].mean()) if valid.any() else None


def masked_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float | None:
    err = np.abs(pred - gt)
    if err.ndim == 3:
        err = err.mean(axis=-1)
    valid = mask & np.isfinite(err)
    return float(err[valid].mean()) if valid.any() else None


def masked_quantile(arr: np.ndarray, mask: np.ndarray, q: float) -> float | None:
    valid = mask & np.isfinite(arr)
    return float(np.quantile(arr[valid], q)) if valid.any() else None


def vec_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:2]
    b = np.asarray(b, dtype=np.float32).reshape(-1)[:2]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return None
    return float(np.dot(a, b) / (denom + 1e-8))


def pointmap_norm_scale(projected_xyz: np.ndarray | None, mask: np.ndarray, mode: str) -> float:
    if mode == "none":
        return 1.0
    if mode == "fixed":
        raise RuntimeError("fixed scale should be handled by caller")
    if projected_xyz is None:
        return 1.0
    norms = np.linalg.norm(np.asarray(projected_xyz, dtype=np.float32), axis=-1)
    valid = mask & np.isfinite(norms) & (norms > 1e-6)
    if not valid.any():
        return 1.0
    if mode == "pointmap_norm_mean":
        return float(norms[valid].mean())
    return float(np.median(norms[valid]))


def add_basic_stats(row: dict[str, object], prefix: str, pred_mag: np.ndarray, gt_mag: np.ndarray, mask: np.ndarray, zero_eps: float) -> None:
    row[f"{prefix}_mae"] = masked_mae(pred_mag, gt_mag, mask)
    row[f"{prefix}_pred_mean"] = masked_mean(pred_mag, mask)
    row[f"{prefix}_gt_mean"] = masked_mean(gt_mag, mask)
    row[f"{prefix}_gt_p50"] = masked_quantile(gt_mag, mask, 0.50)
    row[f"{prefix}_gt_p80"] = masked_quantile(gt_mag, mask, 0.80)
    row[f"{prefix}_gt_p95"] = masked_quantile(gt_mag, mask, 0.95)
    row[f"{prefix}_pred_p80"] = masked_quantile(pred_mag, mask, 0.80)
    row[f"{prefix}_pred_p95"] = masked_quantile(pred_mag, mask, 0.95)

    valid = mask & np.isfinite(gt_mag) & np.isfinite(pred_mag)
    if not valid.any():
        return
    q50 = float(np.quantile(gt_mag[valid], 0.50))
    q80 = float(np.quantile(gt_mag[valid], 0.80))
    buckets = {
        "zero": valid & (gt_mag <= zero_eps),
        "low50": valid & (gt_mag <= q50),
        "low80": valid & (gt_mag <= q80),
        "mid50_80": valid & (gt_mag > q50) & (gt_mag <= q80),
        "high20": valid & (gt_mag > q80),
    }
    for name, bucket in buckets.items():
        row[f"{prefix}_{name}_count"] = int(bucket.sum())
        row[f"{prefix}_{name}_ratio"] = float(bucket.sum() / max(int(valid.sum()), 1))
        row[f"{prefix}_{name}_pred_mean"] = masked_mean(pred_mag, bucket)
        row[f"{prefix}_{name}_gt_mean"] = masked_mean(gt_mag, bucket)
        row[f"{prefix}_{name}_mae"] = masked_mae(pred_mag, gt_mag, bucket)


def run_one(export, model, export_args, args: argparse.Namespace, remote_dir: Path) -> dict[str, object]:
    image_path = remote_dir / "image.png"
    aux_path = remote_dir / "projection_aux.npz"
    views = load_images(
        [str(image_path)],
        resize_mode="fixed_size",
        size=(args.resolution, args.resolution),
        resolution_set=518,
    )
    model_name = export.resolve_effective_model_name(export_args)
    views = export.convert_views_to_identity_if_needed(views, model_name)
    views[0]["instance"] = ["remote"]
    with torch.inference_mode():
        pred = export.run_model_inference(model, views, export_args)[0]

    aux = load_projection_aux_modalities(aux_path)
    raw_aux_npz = np.load(aux_path)
    raw_shape = aux["remote_projection_valid_mask"].shape
    gt = preprocess_projection_aux_modalities(
        aux,
        box=(0, 0, raw_shape[1], raw_shape[0]),
        resolution=(args.resolution, args.resolution),
        label_resize_mode=args.label_resize_mode,
    )

    projected_xyz = None
    if "projected_xyz_centered" in raw_aux_npz.files:
        projected_xyz = _resize_label_array(
            raw_aux_npz["projected_xyz_centered"].astype(np.float32),
            (args.resolution, args.resolution),
            mode=args.label_resize_mode,
        )

    mask = gt["remote_projection_valid_mask"].astype(bool)
    rel_gt_raw = np.asarray(gt["remote_projection_rel_height"], dtype=np.float32)
    offset_gt_raw = np.asarray(gt["remote_projection_offset_xy"], dtype=np.float32)
    rel_pred = to_numpy(pred["remote_projection_rel_height_pred"])[0].astype(np.float32)
    offset_pred = to_numpy(pred["remote_projection_offset_xy_pred"])[0].astype(np.float32)
    dir_gt = np.asarray(gt["remote_projection_global_dir_xy"], dtype=np.float32).reshape(-1)[:2]
    slope_gt = np.asarray(gt["remote_projection_global_slope"], dtype=np.float32).reshape(-1)[:1]
    dir_pred = to_numpy(pred["remote_projection_global_dir_xy_pred"])[0].reshape(-1)[:2]
    slope_pred = to_numpy(pred["remote_projection_global_slope_pred"])[0].reshape(-1)[:1]

    if args.rel_height_scale_mode == "fixed":
        rel_scale = float(args.rel_height_scale)
    else:
        rel_scale = pointmap_norm_scale(projected_xyz, mask, args.rel_height_scale_mode)
    rel_scale = max(rel_scale, 1e-6)
    offset_scale = max(float(args.offset_scale), 1e-6)

    rel_gt = np.abs(rel_gt_raw) / rel_scale
    offset_gt = np.linalg.norm(offset_gt_raw, axis=-1) / offset_scale
    rel_pred_mag = np.abs(rel_pred)
    offset_pred_mag = np.linalg.norm(offset_pred, axis=-1)

    valid_count = int(mask.sum())
    row: dict[str, object] = {
        "scene": remote_dir.parent.name,
        "provider": remote_dir.name,
        "remote_dir": str(remote_dir),
        "valid_pixels": valid_count,
        "valid_ratio": float(mask.mean()),
        "rel_height_scale": float(rel_scale),
        "offset_scale": float(offset_scale),
        "global_dir_cosine": vec_cosine(dir_gt, dir_pred),
        "global_slope_gt": float(slope_gt[0]) if slope_gt.size else None,
        "global_slope_pred": float(slope_pred[0]) if slope_pred.size else None,
    }
    for optional in ("remote_projection_building_mask", "remote_projection_tilt_projected_mask"):
        if optional in gt:
            opt_mask = np.asarray(gt[optional]).astype(bool)
            row[optional.replace("remote_projection_", "") + "_ratio"] = float((opt_mask & mask).sum() / max(valid_count, 1))

    add_basic_stats(row, "rel_height", rel_pred_mag, rel_gt, mask, args.zero_eps)
    add_basic_stats(row, "offset", offset_pred_mag, offset_gt, mask, args.zero_eps)

    high = mask & np.isfinite(offset_gt) & (offset_gt > (row.get("offset_gt_p80") or np.inf))
    if high.any():
        pred_vec = offset_pred[high].mean(axis=0)
        gt_vec = (offset_gt_raw / offset_scale)[high].mean(axis=0)
        row["offset_high20_vector_cosine"] = vec_cosine(gt_vec, pred_vec)
    else:
        row["offset_high20_vector_cosine"] = None

    low_pred = row.get("offset_low80_pred_mean") or 0.0
    low_gt = row.get("offset_low80_gt_mean") or 0.0
    high_pred = row.get("offset_high20_pred_mean") or 0.0
    high_gt = row.get("offset_high20_gt_mean") or 0.0
    row["offset_low80_over_gt"] = float(low_pred / (low_gt + 1e-8)) if low_gt is not None else None
    row["offset_low80_excess"] = float(max(0.0, low_pred - low_gt)) if low_gt is not None else None
    row["offset_high20_under_gt"] = float(max(0.0, high_gt - high_pred)) if high_gt is not None else None
    dir_penalty = max(0.0, 0.9 - float(row.get("global_dir_cosine") or 0.0))
    row["badness_score"] = float(
        10.0 * (row["offset_low80_excess"] or 0.0)
        + 5.0 * (row["offset_high20_under_gt"] or 0.0)
        + dir_penalty
    )
    return row


def main() -> int:
    args = parse_args()
    remote_dirs = iter_remote_dirs(args)
    if not remote_dirs:
        raise RuntimeError("No remote directories matched the filters.")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    export = load_export_module()
    tmp_output = args.output_csv.parent / "_projection_aux_pred_tmp"
    export_args = make_export_args(args, remote_dirs[0], tmp_output)
    config_overrides = export.resolve_config_overrides(export_args)
    model_name = export.resolve_effective_model_name(export_args)
    model = export.initialize_model(export_args, args.device, config_overrides, model_name)
    model.eval()

    rows = []
    for idx, remote_dir in enumerate(remote_dirs, start=1):
        print(f"[{idx}/{len(remote_dirs)}] {remote_dir}", flush=True)
        try:
            rows.append(run_one(export, model, export_args, args, remote_dir))
        except Exception as exc:  # keep long scans from dying on one corrupted sample
            rows.append({"scene": remote_dir.parent.name, "provider": remote_dir.name, "remote_dir": str(remote_dir), "error": repr(exc)})

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if "error" not in r]
    print(f"wrote {args.output_csv} rows={len(rows)} valid={len(valid_rows)}", flush=True)
    if args.sort_key and valid_rows:
        sortable = [r for r in valid_rows if r.get(args.sort_key) is not None]
        sortable.sort(key=lambda r: float(r[args.sort_key]), reverse=True)
        top = sortable[: args.top_k]
        print(f"top {len(top)} by {args.sort_key}:", flush=True)
        for row in top[:10]:
            print(
                f"  {row['scene']}/{row['provider']} {args.sort_key}={float(row[args.sort_key]):.4f} "
                f"off_low80_pred/gt={row.get('offset_low80_pred_mean')}/{row.get('offset_low80_gt_mean')} "
                f"off_high20_pred/gt={row.get('offset_high20_pred_mean')}/{row.get('offset_high20_gt_mean')}",
                flush=True,
            )
        if args.bad_scenes_txt is not None:
            args.bad_scenes_txt.parent.mkdir(parents=True, exist_ok=True)
            args.bad_scenes_txt.write_text("\n".join(str(r["scene"]) for r in top) + "\n", encoding="utf-8")
            print(f"wrote {args.bad_scenes_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
