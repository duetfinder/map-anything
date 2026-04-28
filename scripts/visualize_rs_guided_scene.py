#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import rerun as rr
    import rerun.blueprint as rrb
except Exception as exc:
    raise ImportError(
        "Failed to import the Rerun SDK. Install the correct package with "
        "`python -m pip install rerun-sdk`."
    ) from exc

if not hasattr(rr, "init"):
    raise ImportError(
        "Imported `rerun`, but it is not the Rerun SDK required by this viewer. "
        "Your environment likely has the unrelated `rerun` package installed. "
        "Fix with:\n"
        "  python -m pip uninstall rerun\n"
        "  python -m pip install rerun-sdk"
    )

from mapanything.utils.viz import script_add_rerun_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one scene bundle or a directory of scene bundles in Rerun."
    )
    parser.add_argument("bundle_path", type=Path, help="Path to scene_bundle.pt or a directory containing multiple scene_bundle.pt files.")
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Optional comma-separated frame indices to visualize. Defaults to all frames in the bundle.",
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        default="rs_guided_scene",
        help="Rerun recording id.",
    )
    script_add_rerun_args(parser)
    return parser.parse_args()


def parse_frame_filter(value):
    if value is None or not str(value).strip():
        return None
    text = str(value).replace("[", "").replace("]", "")
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    return {int(part) for part in parts}


def flatten_dict(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flatten_dict(next_prefix, item, out)
    else:
        out[prefix] = value


def sanitize_anyvalues(payload):
    sanitized = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def build_overview_markdown(metadata, metrics):
    scene_name = metadata.get("scene_name", "unknown")
    provider = metadata.get("provider", "unknown")
    frame_indices = metadata.get("frame_indices", [])
    joint = metrics.get("joint", {})
    aerial = metrics.get("aerial_only", {})
    rs_only = metrics.get("rs_only", {})
    scale_info = (
        metrics.get("debug", {})
        .get("remote_metric_scale", {})
    )

    def fmt(value):
        if value is None:
            return "nan"
        try:
            value = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6f}"

    rows = [
        ("Scene", str(scene_name)),
        ("Provider", str(provider)),
        ("Frames", str(frame_indices)),
        ("Aerial pointmaps_abs_rel", fmt(aerial.get("pointmaps_abs_rel"))),
        ("Aerial pose_ate_rmse", fmt(aerial.get("pose_ate_rmse"))),
        ("Aerial pose_auc_5", fmt(aerial.get("pose_auc_5"))),
        ("Joint pointmaps_abs_rel", fmt(joint.get("pointmaps_abs_rel"))),
        ("Joint pose_ate_rmse", fmt(joint.get("pose_ate_rmse"))),
        ("Joint global_pointmaps_abs_rel", fmt(joint.get("joint_global_pointmaps_abs_rel"))),
        ("RS height_mae", fmt(rs_only.get("rs_height_mae"))),
        ("RS height_rmse", fmt(rs_only.get("rs_height_rmse"))),
        ("Remote metric scale_factor", fmt(scale_info.get("scale_factor"))),
        ("Remote metric pred_spacing", fmt(scale_info.get("pred_spacing_xy"))),
        ("Remote metric meters_per_pixel", fmt(scale_info.get("meters_per_pixel"))),
    ]

    lines = [
        "# RS Guided 可视化概览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "说明：",
            "- `modes/remote_metric`：`view0` 参考系 + 遥感分辨率恢复尺度",
            "- `modes/benchmark`：与 benchmark 几何定义一致的归一化结果（joint remote 已按 `view0 + avg_dis` 对齐）",
        ]
    )
    return "\n".join(lines)


def init_rerun(args, bundle_path):
    rr.init(args.recording_id, default_enabled=True)
    fallback_path = bundle_path.with_suffix(".rrd")
    if args.stdout:
        rr.stdout()
        return
    if args.save:
        rr.save(str(args.save))
        print(
            f"Saving RRD to {args.save}\n"
            "Open it in a browser with:\n"
            f"  rerun {args.save} --web-viewer --bind 0.0.0.0"
        )
        return
    if args.serve:
        if hasattr(rr, "serve"):
            rr.serve(open_browser=not args.headless)
            return
        rr.save(str(fallback_path))
        print(
            "This rerun-sdk version does not expose rr.serve(). "
            f"Saving an RRD instead: {fallback_path}\n"
            "Open it in a browser with:\n"
            f"  rerun {fallback_path} --web-viewer --bind 0.0.0.0"
        )
        return
    elif args.connect:
        if hasattr(rr, "connect"):
            rr.connect(args.url)
            return
        print(
            "This rerun-sdk version does not expose rr.connect(). "
            "Falling back to saving an RRD file."
        )
        rr.save(str(fallback_path))
        print(
            f"Saved stream target: {fallback_path}\n"
            "Open it in a browser with:\n"
            f"  rerun {fallback_path} --web-viewer --bind 0.0.0.0"
        )
        return
    else:
        if args.headless:
            rr.save(str(fallback_path))
            print(
                f"Headless mode without rr.serve/rr.connect: saving RRD to {fallback_path}\n"
                "Open it in a browser with:\n"
                f"  rerun {fallback_path} --web-viewer --bind 0.0.0.0"
            )
            return
        rr.spawn(memory_limit="75%")


def log_camera(path, pose, intrinsics, image_rgb):
    pose = np.asarray(pose, dtype=np.float32)
    pose = np.linalg.inv(pose)
    height, width = image_rgb.shape[:2]
    rr.log(
        path,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=True,
    )
    rr.log(f"{path}", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        f"{path}/image",
        rr.Pinhole(
            image_from_camera=intrinsics,
            width=width,
            height=height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
        static=True,
    )
    rr.log(f"{path}/image/rgb", rr.Image(image_rgb), static=True)


def log_points(path, points, colors=None, valid_mask=None):
    mask = np.isfinite(points).all(axis=-1)
    if valid_mask is not None:
        mask = mask & valid_mask.astype(bool)
    selected_points = points[mask]
    if selected_points.size == 0:
        return
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[: mask.ndim] == mask.shape:
            selected_colors = colors[mask]
        elif colors.ndim == 2 and colors.shape[0] == selected_points.shape[0]:
            selected_colors = colors
        else:
            flat_colors = colors.reshape(-1, colors.shape[-1])
            flat_mask = mask.reshape(-1)
            if flat_colors.shape[0] == flat_mask.shape[0]:
                selected_colors = flat_colors[flat_mask]
            else:
                raise ValueError(
                    f"Color shape {colors.shape} is incompatible with point shape {points.shape}"
                )
        rr.log(path, rr.Points3D(positions=selected_points, colors=selected_colors), static=True)
    else:
        rr.log(path, rr.Points3D(positions=selected_points), static=True)


def resolve_aligned_view(view, key):
    aligned = view.get("benchmark_aligned", {})
    return aligned.get(key)


def resolve_metric_view(view):
    aligned = view.get("benchmark_aligned", {})
    return aligned.get("remote_metric_joint")


def normalize_root_prefix(root_prefix):
    if not root_prefix:
        return ""
    root_prefix = str(root_prefix).strip("/")
    return f"/{root_prefix}"


def scene_path(root_prefix, relative_path):
    base = normalize_root_prefix(root_prefix)
    rel = str(relative_path).strip("/")
    if not base:
        return f"/{rel}" if rel else "/"
    if not rel:
        return base
    return f"{base}/{rel}"


def build_scene_blueprint(root_prefix, scene_name):
    root = normalize_root_prefix(root_prefix) or "/"
    return rrb.Tabs(
        rrb.TextDocumentView(origin=scene_path(root, "overview"), name="Overview"),
        rrb.Spatial3DView(origin=scene_path(root, "modes/benchmark"), name="Benchmark 3D"),
        rrb.Spatial3DView(origin=scene_path(root, "modes/remote_metric"), name="RemoteMetric 3D"),
        rrb.Spatial2DView(origin=scene_path(root, "remote"), name="Remote 2D"),
        active_tab=0,
        name=scene_name,
    )


def build_blueprint(scene_specs):
    if len(scene_specs) == 1:
        _, scene_name = scene_specs[0]
        root_prefix = scene_specs[0][0]
        return rrb.Blueprint(
            build_scene_blueprint(root_prefix, scene_name),
            collapse_panels=True,
        )

    scene_tabs = [build_scene_blueprint(root_prefix, scene_name) for root_prefix, scene_name in scene_specs]
    return rrb.Blueprint(
        rrb.Tabs(*scene_tabs, active_tab=0, name="Scenes"),
        collapse_panels=True,
    )


def log_scene_bundle(bundle, selected_frames=None, root_prefix=""):
    metadata = bundle["metadata"]
    metrics_flat = {}
    flatten_dict("", bundle["metrics"], metrics_flat)
    rr.log(scene_path(root_prefix, "metrics"), rr.AnyValues(**sanitize_anyvalues(metrics_flat)), static=True)
    rr.log(scene_path(root_prefix, "scene_info"), rr.AnyValues(**sanitize_anyvalues(metadata)), static=True)
    rr.log(
        scene_path(root_prefix, "overview"),
        rr.TextDocument(build_overview_markdown(metadata, bundle["metrics"])),
        static=True,
    )

    global_clouds = bundle.get("global_pointclouds", {})
    aligned_global = global_clouds.get("benchmark_aligned", {})
    if "remote_metric_joint" in aligned_global:
        rr.log(
            scene_path(root_prefix, "modes/remote_metric/global/gt"),
            rr.Points3D(
                positions=aligned_global["remote_metric_joint"]["gt_points"],
                colors=aligned_global["remote_metric_joint"].get("gt_colors"),
            ),
            static=True,
        )
        rr.log(
            scene_path(root_prefix, "modes/remote_metric/global/pred"),
            rr.Points3D(
                positions=aligned_global["remote_metric_joint"]["pred_points"],
                colors=aligned_global["remote_metric_joint"].get("pred_colors"),
            ),
            static=True,
        )
        rr.log(
            scene_path(root_prefix, "modes/remote_metric/scale"),
            rr.AnyValues(**sanitize_anyvalues(aligned_global["remote_metric_joint"]["scale_info"])),
            static=True,
        )
    if "aerial_only" in aligned_global:
        rr.log(
            scene_path(root_prefix, "modes/benchmark/aerial_only/global/gt"),
            rr.Points3D(
                positions=aligned_global["aerial_only"]["gt_points"],
                colors=aligned_global["aerial_only"].get("gt_colors"),
            ),
            static=True,
        )
        rr.log(
            scene_path(root_prefix, "modes/benchmark/aerial_only/global/pred"),
            rr.Points3D(
                positions=aligned_global["aerial_only"]["pred_points"],
                colors=aligned_global["aerial_only"].get("pred_colors"),
            ),
            static=True,
        )
    if "joint" in aligned_global:
        rr.log(
            scene_path(root_prefix, "modes/benchmark/joint/global/gt"),
            rr.Points3D(
                positions=aligned_global["joint"]["gt_points"],
                colors=aligned_global["joint"].get("gt_colors"),
            ),
            static=True,
        )
        rr.log(
            scene_path(root_prefix, "modes/benchmark/joint/global/pred"),
            rr.Points3D(
                positions=aligned_global["joint"]["pred_points"],
                colors=aligned_global["joint"].get("pred_colors"),
            ),
            static=True,
        )

    for view in bundle["views"]:
        frame_index = int(view["frame_index"])
        if selected_frames is not None and frame_index not in selected_frames:
            continue

        suffix = f"frame_{frame_index:03d}"
        image_rgb = view["image_rgb"]
        intrinsics = view["camera_intrinsics"]
        aerial_aligned = resolve_aligned_view(view, "aerial_only")
        joint_aligned = resolve_aligned_view(view, "joint")
        metric_aligned = resolve_metric_view(view)

        if aerial_aligned is not None:
            log_camera(scene_path(root_prefix, f"modes/benchmark/aerial_only/views/gt/{suffix}"), aerial_aligned["gt_pose"], intrinsics, image_rgb)
            log_camera(scene_path(root_prefix, f"modes/benchmark/aerial_only/views/pred/{suffix}"), aerial_aligned["pred_pose"], intrinsics, image_rgb)
            log_points(
                scene_path(root_prefix, f"modes/benchmark/aerial_only/points/gt/{suffix}"),
                aerial_aligned["gt_points"],
                colors=image_rgb,
                valid_mask=aerial_aligned.get("valid_mask"),
            )
            log_points(
                scene_path(root_prefix, f"modes/benchmark/aerial_only/points/pred/{suffix}"),
                aerial_aligned["pred_points"],
                colors=image_rgb,
                valid_mask=aerial_aligned.get("valid_mask"),
            )
        if joint_aligned is not None:
            log_camera(scene_path(root_prefix, f"modes/benchmark/joint/views/gt/{suffix}"), joint_aligned["gt_pose"], intrinsics, image_rgb)
            log_camera(scene_path(root_prefix, f"modes/benchmark/joint/views/pred/{suffix}"), joint_aligned["pred_pose"], intrinsics, image_rgb)
            log_points(
                scene_path(root_prefix, f"modes/benchmark/joint/points/gt/{suffix}"),
                joint_aligned["gt_points"],
                colors=image_rgb,
                valid_mask=joint_aligned.get("valid_mask"),
            )
            log_points(
                scene_path(root_prefix, f"modes/benchmark/joint/points/pred/{suffix}"),
                joint_aligned["pred_points"],
                colors=image_rgb,
                valid_mask=joint_aligned.get("valid_mask"),
            )

        if metric_aligned is not None:
            log_camera(scene_path(root_prefix, f"modes/remote_metric/views/gt/{suffix}"), metric_aligned["gt_pose"], intrinsics, image_rgb)
            log_camera(scene_path(root_prefix, f"modes/remote_metric/views/pred/{suffix}"), metric_aligned["pred_pose"], intrinsics, image_rgb)
            log_points(
                scene_path(root_prefix, f"modes/remote_metric/points/gt/{suffix}"),
                metric_aligned["gt_points"],
                colors=image_rgb,
                valid_mask=metric_aligned.get("valid_mask"),
            )
            log_points(
                scene_path(root_prefix, f"modes/remote_metric/points/pred/{suffix}"),
                metric_aligned["pred_points"],
                colors=image_rgb,
                valid_mask=metric_aligned.get("valid_mask"),
            )

    remote = bundle["remote"]
    rr.log(scene_path(root_prefix, "remote/image"), rr.Image(remote["image_rgb"]), static=True)
    joint_remote_aligned = remote.get("benchmark_aligned", {}).get("joint")
    remote_metric_aligned = remote.get("remote_metric_aligned")
    if joint_remote_aligned is not None:
        log_points(
            scene_path(root_prefix, "modes/benchmark/joint/remote/gt"),
            joint_remote_aligned["gt_points"],
            colors=joint_remote_aligned.get("colors", remote["image_rgb"]),
            valid_mask=joint_remote_aligned.get("valid_mask"),
        )
        log_points(
            scene_path(root_prefix, "modes/benchmark/joint/remote/pred"),
            joint_remote_aligned["pred_points"],
            colors=joint_remote_aligned.get("colors", remote["image_rgb"]),
            valid_mask=joint_remote_aligned.get("valid_mask"),
        )
    if remote_metric_aligned is not None:
        if "gt_pose" in remote_metric_aligned and "camera_intrinsics" in remote_metric_aligned:
            log_camera(
                scene_path(root_prefix, "modes/remote_metric/remote/camera/gt"),
                remote_metric_aligned["gt_pose"],
                remote_metric_aligned["camera_intrinsics"],
                remote["image_rgb"],
            )
        if "pred_pose" in remote_metric_aligned and "camera_intrinsics" in remote_metric_aligned:
            log_camera(
                scene_path(root_prefix, "modes/remote_metric/remote/camera/pred"),
                remote_metric_aligned["pred_pose"],
                remote_metric_aligned["camera_intrinsics"],
                remote["image_rgb"],
            )
        log_points(
            scene_path(root_prefix, "modes/remote_metric/remote/gt"),
            remote_metric_aligned["gt_points"],
            colors=remote["image_rgb"],
            valid_mask=remote_metric_aligned.get("valid_mask"),
        )
        log_points(
            scene_path(root_prefix, "modes/remote_metric/remote/pred"),
            remote_metric_aligned["pred_points"],
            colors=remote["image_rgb"],
            valid_mask=remote_metric_aligned.get("valid_mask"),
        )
        if remote.get("meters_per_pixel") is not None:
            rr.log(
                scene_path(root_prefix, "modes/remote_metric/remote/info"),
                rr.AnyValues(meters_per_pixel=float(remote["meters_per_pixel"])),
                static=True,
            )

    return metadata


def main():
    args = parse_args()
    selected_frames = parse_frame_filter(args.frames)

    init_rerun(args, args.bundle_path)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    scene_specs = []
    if args.bundle_path.is_dir():
        bundle_paths = sorted(args.bundle_path.rglob("scene_bundle.pt"))
        if not bundle_paths:
            raise FileNotFoundError(f"No scene_bundle.pt found under directory: {args.bundle_path}")
        for bundle_path in bundle_paths:
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            metadata = log_scene_bundle(bundle, selected_frames=selected_frames, root_prefix=f"scenes/{bundle['metadata']['scene_name']}")
            scene_specs.append((f"scenes/{metadata['scene_name']}", metadata["scene_name"]))
            print(f"Loaded scene bundle: {bundle_path}")
            print(f"Scene: {metadata['scene_name']}")
            print(f"Frames: {metadata['frame_indices']}")
    else:
        bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)
        metadata = log_scene_bundle(bundle, selected_frames=selected_frames, root_prefix="")
        scene_specs.append(("", metadata["scene_name"]))
        print(f"Loaded scene bundle: {args.bundle_path}")
        print(f"Scene: {metadata['scene_name']}")
        print(f"Frames: {metadata['frame_indices']}")

    rr.send_blueprint(build_blueprint(scene_specs))


if __name__ == "__main__":
    main()
