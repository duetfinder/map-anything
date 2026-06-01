#!/usr/bin/env python3
"""Visualize P7 remote projection auxiliary predictions against GT labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mapanything.datasets.wai.vigor_chicago_rs_common import (  # noqa: E402
    load_projection_aux_modalities,
    preprocess_projection_aux_modalities,
)
from mapanything.utils.image import load_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--remote-dir",
        required=True,
        type=Path,
        help="Directory containing image.png and projection_aux.npz, e.g. traindata/Crossview_rs/chicago__location_1/Google_Satellite.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--label-resize-mode", default="nearest", choices=["nearest", "bilinear", "bicubic"])
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "remote_head", "split"],
        help="Export/model preset used to instantiate VGGT p7 checkpoint.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_export_module():
    path = REPO_ROOT / "scripts" / "export_pointcloud_ply.py"
    spec = importlib.util.spec_from_file_location("export_pointcloud_ply", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_export_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model="vggt",
        checkpoint_path=str(args.checkpoint_path),
        image_folder=str(args.remote_dir),
        output_path=str(args.output_dir),
        config_path=None,
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
        vggt_projection_aux_hidden_dim=64,
        vggt_projection_aux_detach_pointmap=False,
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


def normalize_scalar(arr, mask=None, vmin=None, vmax=None, symmetric=False):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= mask.astype(bool)
    vals = arr[finite]
    if vals.size == 0:
        vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    elif symmetric:
        m = float(np.percentile(np.abs(vals), 98))
        m = max(m, 1e-6)
        vmin, vmax = -m, m
    else:
        if vmin is None:
            vmin = float(np.percentile(vals, 2))
        if vmax is None:
            vmax = float(np.percentile(vals, 98))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1), float(vmin), float(vmax)


def colorize(arr, mask=None, symmetric=False):
    norm, vmin, vmax = normalize_scalar(arr, mask=mask, symmetric=symmetric)
    # Blue-white-red, simple and dependency-free.
    r = np.where(norm < 0.5, norm * 2.0, 1.0)
    g = np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 2.0)
    b = np.where(norm < 0.5, 1.0, (1.0 - norm) * 2.0)
    rgb = np.stack([r, g, b], axis=-1)
    if mask is not None:
        rgb = np.where(mask[..., None].astype(bool), rgb, 0.15)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), vmin, vmax


def make_panel(title, arr, mask=None, symmetric=False):
    img_arr, vmin, vmax = colorize(arr, mask=mask, symmetric=symmetric)
    img = Image.fromarray(img_arr).resize((256, 256), Image.BILINEAR)
    canvas = Image.new("RGB", (256, 286), "white")
    canvas.paste(img, (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), f"{title}", fill=(0, 0, 0))
    draw.text((6, 268), f"[{vmin:.3g}, {vmax:.3g}]", fill=(0, 0, 0))
    return canvas


def save_grid(panels, path: Path, cols=3):
    rows = int(np.ceil(len(panels) / cols))
    w, h = panels[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for idx, panel in enumerate(panels):
        grid.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    grid.save(path)


def masked_mae(pred, gt, mask):
    err = np.abs(pred - gt)
    if err.ndim == 3:
        err = err.mean(axis=-1)
    valid = mask.astype(bool) & np.isfinite(err)
    return float(err[valid].mean()) if valid.any() else None


def main() -> int:
    args = parse_args()
    image_path = args.remote_dir / "image.png"
    aux_path = args.remote_dir / "projection_aux.npz"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not aux_path.exists():
        raise FileNotFoundError(aux_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = load_export_module()
    export_args = make_export_args(args)
    config_overrides = export.resolve_config_overrides(export_args)
    model_name = export.resolve_effective_model_name(export_args)
    model = export.initialize_model(export_args, args.device, config_overrides, model_name)
    model.eval()

    views = load_images(
        [str(image_path)],
        resize_mode="fixed_size",
        size=(args.resolution, args.resolution),
        resolution_set=518,
    )
    views = export.convert_views_to_identity_if_needed(views, model_name)
    views[0]["instance"] = ["remote"]
    with torch.inference_mode():
        outputs = export.run_model_inference(model, views, export_args)
    pred = outputs[0]

    aux = load_projection_aux_modalities(aux_path)
    raw_shape = aux["remote_projection_valid_mask"].shape
    gt = preprocess_projection_aux_modalities(
        aux,
        box=(0, 0, raw_shape[1], raw_shape[0]),
        resolution=(args.resolution, args.resolution),
        label_resize_mode=args.label_resize_mode,
    )

    mask = gt["remote_projection_valid_mask"].astype(bool)
    rel_gt = gt["remote_projection_rel_height"]
    offset_gt = gt["remote_projection_offset_xy"]
    rel_pred = to_numpy(pred["remote_projection_rel_height_pred"])[0]
    offset_pred = to_numpy(pred["remote_projection_offset_xy_pred"])[0]
    dir_gt = np.asarray(gt["remote_projection_global_dir_xy"], dtype=np.float32).reshape(-1)[:2]
    slope_gt = np.asarray(gt["remote_projection_global_slope"], dtype=np.float32).reshape(-1)[:1]
    dir_pred = to_numpy(pred["remote_projection_global_dir_xy_pred"])[0].reshape(-1)[:2]
    slope_pred = to_numpy(pred["remote_projection_global_slope_pred"])[0].reshape(-1)[:1]

    offset_from_field = rel_pred[..., None] * float(slope_pred[0]) * dir_pred.reshape(1, 1, 2)

    panels = [
        make_panel("rel_height GT", rel_gt, mask),
        make_panel("rel_height Pred", rel_pred, mask),
        make_panel("rel_height AbsErr", np.abs(rel_pred - rel_gt), mask),
        make_panel("offset_x GT", offset_gt[..., 0], mask, symmetric=True),
        make_panel("offset_x Pred", offset_pred[..., 0], mask, symmetric=True),
        make_panel("offset_x AbsErr", np.abs(offset_pred[..., 0] - offset_gt[..., 0]), mask),
        make_panel("offset_y GT", offset_gt[..., 1], mask, symmetric=True),
        make_panel("offset_y Pred", offset_pred[..., 1], mask, symmetric=True),
        make_panel("offset_y AbsErr", np.abs(offset_pred[..., 1] - offset_gt[..., 1]), mask),
        make_panel("offset |GT|", np.linalg.norm(offset_gt, axis=-1), mask),
        make_panel("offset |Pred|", np.linalg.norm(offset_pred, axis=-1), mask),
        make_panel("consistency |err|", np.linalg.norm(offset_pred - offset_from_field, axis=-1), mask),
    ]
    save_grid(panels, args.output_dir / "projection_aux_gt_pred_grid.png")

    summary = {
        "remote_dir": str(args.remote_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "valid_pixels": int(mask.sum()),
        "rel_height_mae": masked_mae(rel_pred, rel_gt, mask),
        "offset_mae": masked_mae(offset_pred, offset_gt, mask),
        "consistency_mae": masked_mae(offset_pred, offset_from_field, mask),
        "global_dir_gt": dir_gt.tolist(),
        "global_dir_pred": dir_pred.tolist(),
        "global_dir_cosine": float(np.dot(dir_gt, dir_pred) / (np.linalg.norm(dir_gt) * np.linalg.norm(dir_pred) + 1e-8)),
        "global_slope_gt": slope_gt.tolist(),
        "global_slope_pred": slope_pred.tolist(),
    }
    with (args.output_dir / "projection_aux_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"wrote {args.output_dir / 'projection_aux_gt_pred_grid.png'}")
    print(f"wrote {args.output_dir / 'projection_aux_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
