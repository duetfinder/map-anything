#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from mapanything.datasets.wai.vigor_chicago_joint_rs_aerial import VigorChicagoJointRSAerial
from mapanything.models import init_model
from mapanything.utils.geometry import closed_form_pose_inverse, geotrf, quaternion_to_rotation_matrix
from mapanything.utils.inference import loss_of_one_batch_multi_view


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GT/pred point clouds for a P3 VIGOR Chicago checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--remote-provider", default="Google_Satellite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument(
        "--aerial-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_wai"),
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/mapanything_metadata/Crossview"),
    )
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_rs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/vigor_chicago_joint_predictions"),
    )
    return parser.parse_args()


def homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    out = np.full_like(flat, np.nan, dtype=np.float32)
    if valid.any():
        homo = np.concatenate([flat[valid].astype(np.float32), np.ones((valid.sum(), 1), dtype=np.float32)], axis=1)
        out[valid] = (transform.astype(np.float32) @ homo.T).T[:, :3]
    return out.reshape(points.shape)


def subsample_valid_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    flat = points.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    flat = flat[valid].astype(np.float32)
    if flat.shape[0] == 0:
        return flat
    if flat.shape[0] > max_points:
        idx = rng.choice(flat.shape[0], size=max_points, replace=False)
        flat = flat[idx]
    return flat


def write_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_colored_clouds(clouds, out_path: Path) -> None:
    all_points = []
    all_colors = []
    for _, points, color in clouds:
        if points.shape[0] == 0:
            continue
        all_points.append(points)
        all_colors.append(np.tile(np.asarray(color, dtype=np.uint8)[None], (points.shape[0], 1)))
    if not all_points:
        raise ValueError(f"No valid points to write: {out_path}")
    write_ply(np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), out_path)


def cloud_stats(points: np.ndarray) -> dict:
    if points.shape[0] == 0:
        return {"num_points": 0}
    centroid = points.mean(axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    mean_radius = np.linalg.norm(points - centroid[None], axis=1).mean()
    return {
        "num_points": int(points.shape[0]),
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "mean_radius": float(mean_radius),
    }


def symmetric_nn_l2(src: np.ndarray, dst: np.ndarray, num_samples: int, rng: np.random.Generator) -> dict:
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return {"symmetric_mean_l2": float("nan")}
    if src.shape[0] > num_samples:
        src = src[rng.choice(src.shape[0], size=num_samples, replace=False)]
    if dst.shape[0] > num_samples:
        dst = dst[rng.choice(dst.shape[0], size=num_samples, replace=False)]
    src_t = torch.from_numpy(src)
    dst_t = torch.from_numpy(dst)
    dists = torch.cdist(src_t, dst_t, p=2)
    src_to_dst = dists.min(dim=1).values.cpu().numpy()
    dst_to_src = dists.min(dim=0).values.cpu().numpy()
    return {
        "src_to_dst_mean_l2": float(src_to_dst.mean()),
        "dst_to_src_mean_l2": float(dst_to_src.mean()),
        "symmetric_mean_l2": float(0.5 * (src_to_dst.mean() + dst_to_src.mean())),
    }


def build_pred_camera0(pred0: dict) -> torch.Tensor:
    cam_pose = torch.eye(4, device=pred0["cam_quats"].device).unsqueeze(0)
    cam_pose[:, :3, :3] = quaternion_to_rotation_matrix(pred0["cam_quats"].float())
    cam_pose[:, :3, 3] = pred0["cam_trans"].float()
    return cam_pose


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VigorChicagoJointRSAerial(
        split=args.split,
        resolution=(args.resolution, args.resolution),
        principal_point_centered=False,
        seed=777,
        transform="imgnorm",
        data_norm_type="identity",
        ROOT=str(args.aerial_root),
        dataset_metadata_dir=str(args.metadata_root),
        overfit_num_sets=None,
        variable_num_views=False,
        num_views=args.num_views,
        covisibility_thres=0.05,
        remote_ROOT=str(args.remote_root),
        remote_providers=[args.remote_provider],
        remote_resolution=(args.resolution, args.resolution),
        remote_transform="imgnorm",
        skip_missing_remote=False,
    )

    if args.scene_name is None:
        scene_index = int(args.scene_index)
    else:
        if args.scene_name not in dataset.scenes:
            raise ValueError(f"Scene {args.scene_name} not found in split {args.split}")
        scene_index = dataset.scenes.index(args.scene_name)
    scene_name = dataset.scenes[scene_index]

    sample = dataset[scene_index]
    batch = default_collate([sample])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = init_model("pi3", OmegaConf.create({"name": "pi3", "load_pretrained_weights": True}), torch_hub_force_reload=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        result = loss_of_one_batch_multi_view(batch, model, criterion=None, device=device, use_amp=False)

    preds = [result[f"pred{i + 1}"] for i in range(len(result) - 1) if f"pred{i + 1}" in result]
    if len(preds) < args.num_views + 1:
        raise ValueError(f"Expected at least {args.num_views + 1} predictions, got {len(preds)}")

    rng = np.random.default_rng(0)
    gt_view0_pose = to_numpy(sample[0]["camera_pose"]).astype(np.float32)
    gt_inv_view0 = np.linalg.inv(gt_view0_pose).astype(np.float32)

    gt_aerial_world = np.concatenate([subsample_valid_points(to_numpy(view["pts3d"]), args.max_points, rng) for view in sample], axis=0)
    gt_remote_global = subsample_valid_points(to_numpy(sample[0]["remote_pointmap"]), args.max_points, rng)
    gt_aerial_view0 = subsample_valid_points(homogeneous_transform(gt_aerial_world, gt_inv_view0), args.max_points * max(1, len(sample)), rng)
    gt_remote_view0 = subsample_valid_points(homogeneous_transform(gt_remote_global, gt_inv_view0), args.max_points, rng)

    pred_aerial_global = np.concatenate([
        subsample_valid_points(to_numpy(pred["pts3d"])[0], args.max_points, rng) for pred in preds[:-1]
    ], axis=0)
    pred_remote_global = subsample_valid_points(to_numpy(preds[-1]["pts3d"])[0], args.max_points, rng)

    pred_camera0 = build_pred_camera0(preds[0])
    pred_inv_view0 = to_numpy(closed_form_pose_inverse(pred_camera0))[0].astype(np.float32)

    pred_aerial_view0_predref = subsample_valid_points(homogeneous_transform(pred_aerial_global, pred_inv_view0), args.max_points * max(1, len(sample)), rng)
    pred_remote_view0_predref = subsample_valid_points(homogeneous_transform(pred_remote_global, pred_inv_view0), args.max_points, rng)

    pred_aerial_view0_gtref = subsample_valid_points(homogeneous_transform(pred_aerial_global, gt_inv_view0), args.max_points * max(1, len(sample)), rng)
    pred_remote_view0_gtref = subsample_valid_points(homogeneous_transform(pred_remote_global, gt_inv_view0), args.max_points, rng)

    out_dir = args.output_dir / args.checkpoint.parent.name / args.split / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "gt_global_combo": out_dir / "gt_global_combo.ply",
        "gt_view0_combo": out_dir / "gt_view0_combo.ply",
        "pred_global_combo": out_dir / "pred_global_combo.ply",
        "pred_view0_gtref_combo": out_dir / "pred_view0_gtref_combo.ply",
        "pred_view0_predref_combo": out_dir / "pred_view0_predref_combo.ply",
        "remote_global_gt_vs_pred": out_dir / "remote_global_gt_vs_pred.ply",
        "remote_view0_gt_vs_pred_gtref": out_dir / "remote_view0_gt_vs_pred_gtref.ply",
    }

    write_colored_clouds([
        ("gt_aerial_world", gt_aerial_world, (255, 80, 80)),
        ("gt_remote_global", gt_remote_global, (80, 170, 255)),
    ], paths["gt_global_combo"])
    write_colored_clouds([
        ("gt_aerial_view0", gt_aerial_view0, (255, 80, 80)),
        ("gt_remote_view0", gt_remote_view0, (80, 170, 255)),
    ], paths["gt_view0_combo"])
    write_colored_clouds([
        ("pred_aerial_global", pred_aerial_global, (255, 180, 60)),
        ("pred_remote_global", pred_remote_global, (80, 255, 120)),
    ], paths["pred_global_combo"])
    write_colored_clouds([
        ("pred_aerial_view0_gtref", pred_aerial_view0_gtref, (255, 180, 60)),
        ("pred_remote_view0_gtref", pred_remote_view0_gtref, (80, 255, 120)),
    ], paths["pred_view0_gtref_combo"])
    write_colored_clouds([
        ("pred_aerial_view0_predref", pred_aerial_view0_predref, (255, 180, 60)),
        ("pred_remote_view0_predref", pred_remote_view0_predref, (80, 255, 120)),
    ], paths["pred_view0_predref_combo"])
    write_colored_clouds([
        ("gt_remote_global", gt_remote_global, (80, 170, 255)),
        ("pred_remote_global", pred_remote_global, (80, 255, 120)),
    ], paths["remote_global_gt_vs_pred"])
    write_colored_clouds([
        ("gt_remote_view0", gt_remote_view0, (80, 170, 255)),
        ("pred_remote_view0_gtref", pred_remote_view0_gtref, (80, 255, 120)),
    ], paths["remote_view0_gt_vs_pred_gtref"])

    summary = {
        "scene_name": scene_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "load_state_dict": str(msg),
        "gt_camera0_pose": gt_view0_pose.tolist(),
        "pred_camera0_pose": to_numpy(pred_camera0)[0].tolist(),
        "stats": {
            "gt_aerial_world": cloud_stats(gt_aerial_world),
            "gt_remote_global": cloud_stats(gt_remote_global),
            "pred_aerial_global": cloud_stats(pred_aerial_global),
            "pred_remote_global": cloud_stats(pred_remote_global),
        },
        "nn_metrics": {
            "gt_global_internal": symmetric_nn_l2(gt_aerial_world, gt_remote_global, 4096, rng),
            "pred_global_internal": symmetric_nn_l2(pred_aerial_global, pred_remote_global, 4096, rng),
            "pred_vs_gt_remote_global": symmetric_nn_l2(pred_remote_global, gt_remote_global, 4096, rng),
            "pred_vs_gt_remote_view0_gtref": symmetric_nn_l2(pred_remote_view0_gtref, gt_remote_view0, 4096, rng),
            "pred_view0_predref_internal": symmetric_nn_l2(pred_aerial_view0_predref, pred_remote_view0_predref, 4096, rng),
        },
        "ply_paths": {k: str(v) for k, v in paths.items()},
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
