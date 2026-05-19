import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VIEW_RE = re.compile(r"_(\d+)v$")


def as_float(value):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def clean_label(name):
    name = VIEW_RE.sub("", name)
    name = name.removeprefix("pi3_crossview_p3_")
    return name


def parse_run_dir(path):
    match = VIEW_RE.search(path.name)
    if not match:
        return None
    return clean_label(path.name), int(match.group(1))


def flatten(prefix, data, out):
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten(flat_key, value, out)
        else:
            out[flat_key] = value


def discover_runs(root):
    runs = []
    for result_json in sorted(root.glob("*/rs_aerial_benchmark_results.json")):
        parsed = parse_run_dir(result_json.parent)
        if parsed is None:
            continue
        variant, views = parsed
        runs.append((variant, views, result_json))
    return sorted(runs, key=lambda item: (item[0], item[1]))


def collect_tables(runs):
    summary_rows = []
    location_rows = []
    for variant, views, result_json in runs:
        result = json.loads(result_json.read_text())
        row = {
            "variant": variant,
            "views": views,
            "run_dir": str(result_json.parent),
        }
        if "metadata" in result:
            flatten("metadata", result["metadata"], row)
        for section in ("aerial_only", "rs_only", "joint"):
            source = result.get(section, {}).get("average")
            if source is not None:
                flatten(section, source, row)
        for section, values in result.get("improvement", {}).items():
            source = values.get("average") if isinstance(values, dict) else None
            if source is not None:
                flatten(f"improvement.{section}", source, row)
        summary_rows.append(row)

        per_scene = result.get("per_scene_results")
        if per_scene is None:
            per_scene_path = result_json.parent / "rs_aerial_per_scene_results.json"
            per_scene = json.loads(per_scene_path.read_text()) if per_scene_path.exists() else {}
        for location, metrics in per_scene.items():
            loc_row = {
                "variant": variant,
                "views": views,
                "location": location,
                "run_dir": str(result_json.parent),
            }
            flatten("", metrics, loc_row)
            aerial_pointmaps = as_float(loc_row.get("aerial_only.pointmaps_abs_rel"))
            joint_pointmaps = as_float(loc_row.get("joint.pointmaps_abs_rel"))
            loc_row["pointmaps_abs_rel_gain"] = aerial_pointmaps - joint_pointmaps
            loc_row["pointmaps_abs_rel_abs_delta"] = abs(aerial_pointmaps - joint_pointmaps)
            location_rows.append(loc_row)
    return summary_rows, location_rows


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_summary(summary_rows, metric):
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["variant"], []).append((row["views"], as_float(row.get(metric))))
    for values in grouped.values():
        values.sort()
    return grouped


def plot_metric_trends(summary_rows, out_dir):
    plots = [
        ("aerial_only.pointmaps_abs_rel", "Aerial pointmaps abs rel"),
        ("joint.pointmaps_abs_rel", "Joint pointmaps abs rel"),
        ("joint.joint_global_pointmaps_abs_rel", "Joint global pointmaps abs rel"),
        ("aerial_only.pose_ate_rmse", "Aerial pose ATE RMSE"),
        ("joint.pose_ate_rmse", "Joint pose ATE RMSE"),
        ("rs_only.rs_height_mae_affine", "RS height MAE affine"),
        ("joint.rs_height_mae_affine", "Joint RS height MAE affine"),
        ("improvement.aerial_vs_aerial_only.pointmaps_abs_rel", "Joint - aerial pointmaps abs rel"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), plots):
        for variant, values in group_summary(summary_rows, metric).items():
            xs = [v for v, y in values if not math.isnan(y)]
            ys = [y for v, y in values if not math.isnan(y)]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=variant)
        ax.set_title(title)
        ax.set_xlabel("views")
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.savefig(out_dir / "metric_trends_by_views.png", dpi=180)
    plt.close(fig)


def plot_location_boxplots(location_rows, out_dir):
    metrics = [
        ("joint.pointmaps_abs_rel", "Joint pointmaps abs rel"),
        ("joint.joint_global_pointmaps_abs_rel", "Joint global pointmaps abs rel"),
        ("joint.rs_height_mae_affine", "Joint RS height MAE affine"),
        ("improvement.aerial_vs_aerial_only.pointmaps_abs_rel", "Joint - aerial pointmaps abs rel"),
    ]
    variants = sorted({row["variant"] for row in location_rows})
    views = sorted({row["views"] for row in location_rows})
    for metric, title in metrics:
        fig, axes = plt.subplots(
            len(variants),
            1,
            figsize=(13, max(3, 2.4 * len(variants))),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, variant in zip(axes, variants):
            data = []
            labels = []
            for view in views:
                values = [
                    as_float(row.get(metric))
                    for row in location_rows
                    if row["variant"] == variant and row["views"] == view
                ]
                values = [value for value in values if not math.isnan(value)]
                if values:
                    data.append(values)
                    labels.append(str(view))
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=True)
            ax.set_title(variant)
            ax.set_ylabel(title)
            ax.grid(True, axis="y", alpha=0.25)
        axes[-1].set_xlabel("views")
        fig.suptitle(f"Location distribution: {title}")
        filename = metric.replace(".", "_").replace("-", "minus")
        fig.savefig(out_dir / f"location_distribution_{filename}.png", dpi=180)
        plt.close(fig)


def plot_location_heatmaps(location_rows, out_dir):
    metrics = [
        ("joint.pointmaps_abs_rel", "Joint pointmaps abs rel"),
        ("joint.rs_height_mae_affine", "Joint RS height MAE affine"),
    ]
    variants = sorted({row["variant"] for row in location_rows})
    views = sorted({row["views"] for row in location_rows})
    locations = sorted({row["location"] for row in location_rows})

    for metric, title in metrics:
        fig, axes = plt.subplots(
            1,
            len(variants),
            figsize=(4.5 * len(variants), max(8, 0.32 * len(locations))),
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        images = []
        for ax, variant in zip(axes, variants):
            grid = np.full((len(locations), len(views)), np.nan)
            for row in location_rows:
                if row["variant"] != variant:
                    continue
                i = locations.index(row["location"])
                j = views.index(row["views"])
                grid[i, j] = as_float(row.get(metric))
            image = ax.imshow(grid, aspect="auto", interpolation="nearest")
            images.append(image)
            ax.set_title(variant)
            ax.set_xticks(range(len(views)), labels=views)
            ax.set_xlabel("views")
            ax.set_yticks(range(len(locations)), labels=[loc.split("__")[-1] for loc in locations])
        axes[0].set_ylabel("location")
        fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.8, label=title)
        fig.suptitle(f"Location metric heatmap: {title}")
        filename = metric.replace(".", "_")
        fig.savefig(out_dir / f"location_heatmap_{filename}.png", dpi=180)
        plt.close(fig)


def write_pointmaps_gain_rankings(location_rows, out_dir):
    rows = []
    for row in location_rows:
        gain = as_float(row.get("pointmaps_abs_rel_gain"))
        if math.isnan(gain):
            continue
        rows.append(
            {
                "variant": row["variant"],
                "views": row["views"],
                "location": row["location"],
                "aerial_only.pointmaps_abs_rel": row.get("aerial_only.pointmaps_abs_rel"),
                "joint.pointmaps_abs_rel": row.get("joint.pointmaps_abs_rel"),
                "pointmaps_abs_rel_gain": gain,
                "pointmaps_abs_rel_abs_delta": abs(gain),
                "run_dir": row.get("run_dir"),
            }
        )
    rows.sort(key=lambda item: item["pointmaps_abs_rel_gain"], reverse=True)
    write_csv(out_dir / "pointmaps_abs_rel_gain_ranking.csv", rows)

    location_summary = {}
    for row in rows:
        location_summary.setdefault(row["location"], []).append(row["pointmaps_abs_rel_gain"])
    summary_rows = []
    for location, values in location_summary.items():
        arr = np.asarray(values, dtype=float)
        summary_rows.append(
            {
                "location": location,
                "mean_pointmaps_abs_rel_gain": float(np.nanmean(arr)),
                "median_pointmaps_abs_rel_gain": float(np.nanmedian(arr)),
                "max_pointmaps_abs_rel_gain": float(np.nanmax(arr)),
                "min_pointmaps_abs_rel_gain": float(np.nanmin(arr)),
                "negative_gain_count": int(np.sum(arr < 0)),
                "total_count": int(np.sum(~np.isnan(arr))),
            }
        )
    summary_rows.sort(key=lambda item: item["mean_pointmaps_abs_rel_gain"], reverse=True)
    write_csv(out_dir / "pointmaps_abs_rel_gain_by_location.csv", summary_rows)


def plot_pointmaps_gain_heatmap(location_rows, out_dir):
    variants = sorted({row["variant"] for row in location_rows})
    views = sorted({row["views"] for row in location_rows})
    locations = sorted({row["location"] for row in location_rows})

    max_abs = 0.0
    for row in location_rows:
        value = as_float(row.get("pointmaps_abs_rel_gain"))
        if not math.isnan(value):
            max_abs = max(max_abs, abs(value))
    if max_abs == 0:
        max_abs = 1.0

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(4.5 * len(variants), max(8, 0.32 * len(locations))),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    images = []
    for ax, variant in zip(axes, variants):
        grid = np.full((len(locations), len(views)), np.nan)
        for row in location_rows:
            if row["variant"] != variant:
                continue
            i = locations.index(row["location"])
            j = views.index(row["views"])
            grid[i, j] = as_float(row.get("pointmaps_abs_rel_gain"))
        image = ax.imshow(
            grid,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu",
            vmin=-max_abs,
            vmax=max_abs,
        )
        images.append(image)
        ax.set_title(variant)
        ax.set_xticks(range(len(views)), labels=views)
        ax.set_xlabel("views")
        ax.set_yticks(range(len(locations)), labels=[loc.split("__")[-1] for loc in locations])
    axes[0].set_ylabel("location")
    fig.colorbar(
        images[-1],
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="pointmaps abs rel gain (aerial_only - joint)",
    )
    fig.suptitle("Location gain heatmap: pointmaps abs rel (positive = satellite helps)")
    fig.savefig(out_dir / "location_heatmap_pointmaps_abs_rel_gain.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.result_root / "visualized_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.result_root)
    if not runs:
        raise SystemExit(f"No rs_aerial_benchmark_results.json files found under {args.result_root}")

    summary_rows, location_rows = collect_tables(runs)
    write_csv(out_dir / "sweep_summary.csv", summary_rows)
    write_csv(out_dir / "location_metrics.csv", location_rows)
    write_pointmaps_gain_rankings(location_rows, out_dir)

    plot_metric_trends(summary_rows, out_dir)
    plot_location_boxplots(location_rows, out_dir)
    plot_location_heatmaps(location_rows, out_dir)
    plot_pointmaps_gain_heatmap(location_rows, out_dir)

    print(f"Found {len(runs)} runs")
    print(f"Wrote {out_dir / 'sweep_summary.csv'}")
    print(f"Wrote {out_dir / 'location_metrics.csv'}")
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
