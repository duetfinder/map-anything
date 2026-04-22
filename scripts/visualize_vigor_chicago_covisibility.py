#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pairwise covisibility statistics for Crossview_wai scenes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_wai"),
        help="Root directory of Crossview_wai.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/mapanything_experiments/mapanything/debug/covisibility_visualization"),
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Specific scene name(s) to visualize in detail. Can be repeated.",
    )
    parser.add_argument(
        "--scene-regex",
        type=str,
        default=None,
        help="Optional regex for selecting detailed scenes.",
    )
    parser.add_argument(
        "--auto-scenes",
        type=int,
        default=3,
        help="If no --scene is given, also visualize top/bottom K scenes ranked by mean covisibility.",
    )
    parser.add_argument(
        "--covis-subpath",
        type=str,
        default="covisibility/v0_gtdepth_native",
        help="Relative subdirectory under each scene that stores covisibility npy files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold used for summary ratios such as non-trivial overlap.",
    )
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Include diagonal entries in distribution statistics. Off by default.",
    )
    return parser.parse_args()


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def find_covisibility_file(scene_root: Path, covis_subpath: str) -> Path | None:
    default_dir = scene_root / covis_subpath
    matches = sorted(default_dir.glob("pairwise_covisibility--*.npy"))
    if matches:
        return matches[0]

    meta_path = scene_root / "scene_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            scene_meta = json.load(f)
        covis_meta = scene_meta.get("scene_modalities", {}).get("pairwise_covisibility", {})
        scene_key = covis_meta.get("scene_key")
        if scene_key:
            candidate = scene_root / scene_key
            if candidate.exists():
                return candidate
    return None


def parse_city(scene_name: str) -> str:
    if "__" in scene_name:
        return scene_name.split("__", 1)[0]
    return "unknown"


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    return np.asarray(matrix, dtype=np.float32)


def flatten_entries(matrix: np.ndarray, include_diagonal: bool) -> np.ndarray:
    mask = np.ones_like(matrix, dtype=bool)
    if not include_diagonal:
        np.fill_diagonal(mask, False)
    return matrix[mask]


def compute_scene_stats(
    scene_name: str,
    matrix: np.ndarray,
    covis_path: Path,
    threshold: float,
    include_diagonal: bool,
) -> dict[str, object]:
    values = flatten_entries(matrix, include_diagonal=include_diagonal)
    if values.size == 0:
        raise ValueError(f"Scene {scene_name} produced no covisibility entries.")

    out_strength = matrix.mean(axis=1)
    in_strength = matrix.mean(axis=0)
    if not include_diagonal:
        n = matrix.shape[0]
        if n > 1:
            out_strength = (matrix.sum(axis=1) - np.diag(matrix)) / (n - 1)
            in_strength = (matrix.sum(axis=0) - np.diag(matrix)) / (n - 1)

    return {
        "scene_name": scene_name,
        "city": parse_city(scene_name),
        "num_frames": int(matrix.shape[0]),
        "num_entries": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "nonzero_ratio": float((values > 0).mean()),
        "above_threshold_ratio": float((values >= threshold).mean()),
        "out_mean_min": float(out_strength.min()),
        "out_mean_max": float(out_strength.max()),
        "in_mean_min": float(in_strength.min()),
        "in_mean_max": float(in_strength.max()),
        "matrix_path": str(covis_path),
    }


def collect_scene_data(
    dataset_root: Path,
    covis_subpath: str,
    threshold: float,
    include_diagonal: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    scene_stats: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}

    scene_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))
    for scene_root in scene_dirs:
        covis_path = find_covisibility_file(scene_root, covis_subpath)
        if covis_path is None:
            continue
        matrix = load_matrix(covis_path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Invalid covisibility matrix shape for {scene_root.name}: {matrix.shape}")
        matrices[scene_root.name] = matrix
        scene_stats.append(
            compute_scene_stats(
                scene_name=scene_root.name,
                matrix=matrix,
                covis_path=covis_path,
                threshold=threshold,
                include_diagonal=include_diagonal,
            )
        )

    if not scene_stats:
        raise ValueError(f"No covisibility matrices found under {dataset_root}")
    return scene_stats, matrices


def save_scene_summary_csv(scene_stats: list[dict[str, object]], out_dir: Path) -> Path:
    csv_path = out_dir / "scene_summary.csv"
    fieldnames = list(scene_stats[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scene_stats)
    return csv_path


def plot_overall_distribution(
    scene_stats: list[dict[str, object]],
    matrices: dict[str, np.ndarray],
    out_dir: Path,
    include_diagonal: bool,
) -> Path:
    values = [flatten_entries(matrices[item["scene_name"]], include_diagonal) for item in scene_stats]
    merged = np.concatenate(values, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(merged, bins=50, color="#1f77b4", alpha=0.9, edgecolor="white")
    axes[0].set_title("Overall Covisibility Distribution")
    axes[0].set_xlabel("Covisibility")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25, linestyle="--")

    sorted_vals = np.sort(merged)
    cdf = np.linspace(0.0, 1.0, len(sorted_vals), endpoint=True)
    axes[1].plot(sorted_vals, cdf, color="#d62728", linewidth=2)
    axes[1].set_title("Overall Covisibility CDF")
    axes[1].set_xlabel("Covisibility")
    axes[1].set_ylabel("CDF")
    axes[1].grid(alpha=0.25, linestyle="--")

    diag_text = "included" if include_diagonal else "excluded"
    fig.suptitle(f"Crossview_wai covisibility summary ({diag_text} diagonal)")
    fig.tight_layout()

    out_path = out_dir / "overall_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scene_ranking(scene_stats: list[dict[str, object]], out_dir: Path) -> Path:
    ranked = sorted(scene_stats, key=lambda item: item["mean"])
    if len(ranked) > 20:
        selected = ranked[:10] + ranked[-10:]
    else:
        selected = ranked

    labels = [item["scene_name"] for item in selected]
    means = [item["mean"] for item in selected]
    colors = ["#d62728" if i < len(selected) // 2 else "#2ca02c" for i in range(len(selected))]

    fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(selected))))
    y = np.arange(len(selected))
    ax.barh(y, means, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean covisibility")
    ax.set_title("Scene ranking by mean covisibility")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()

    out_path = out_dir / "scene_mean_ranking.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_frame_count_distribution(scene_stats: list[dict[str, object]], out_dir: Path) -> Path:
    frame_counts = np.array([item["num_frames"] for item in scene_stats], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame_counts, bins=min(20, max(5, len(np.unique(frame_counts)))), color="#9467bd", edgecolor="white")
    ax.set_title("Frames per scene")
    ax.set_xlabel("Number of frames")
    ax.set_ylabel("Number of scenes")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()

    out_path = out_dir / "frame_count_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_city_mean_comparison(scene_stats: list[dict[str, object]], out_dir: Path) -> Path:
    city_to_means: dict[str, list[float]] = {}
    for item in scene_stats:
        city_to_means.setdefault(str(item["city"]), []).append(float(item["mean"]))

    cities = sorted(city_to_means.keys(), key=natural_key)
    means = [float(np.mean(city_to_means[city])) for city in cities]
    medians = [float(np.median(city_to_means[city])) for city in cities]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cities))
    width = 0.35
    ax.bar(x - width / 2, means, width=width, label="mean", color="#1f77b4", alpha=0.9)
    ax.bar(x + width / 2, medians, width=width, label="median", color="#ff7f0e", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cities)
    ax.set_ylabel("Covisibility")
    ax.set_title("City-level mean and median covisibility")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "city_mean_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_city_summary_csv(scene_stats: list[dict[str, object]], out_dir: Path) -> Path:
    city_rows: list[dict[str, object]] = []
    city_names = sorted({str(item["city"]) for item in scene_stats}, key=natural_key)

    for city in city_names:
        rows = [item for item in scene_stats if item["city"] == city]
        means = np.array([float(item["mean"]) for item in rows], dtype=np.float32)
        frame_counts = np.array([int(item["num_frames"]) for item in rows], dtype=np.int32)
        above_threshold = np.array([float(item["above_threshold_ratio"]) for item in rows], dtype=np.float32)
        city_rows.append(
            {
                "city": city,
                "num_scenes": len(rows),
                "mean_of_scene_means": float(means.mean()),
                "median_of_scene_means": float(np.median(means)),
                "min_scene_mean": float(means.min()),
                "max_scene_mean": float(means.max()),
                "mean_num_frames": float(frame_counts.mean()),
                "median_num_frames": float(np.median(frame_counts)),
                "mean_above_threshold_ratio": float(above_threshold.mean()),
            }
        )

    csv_path = out_dir / "city_summary.csv"
    fieldnames = list(city_rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(city_rows)
    return csv_path


def plot_city_distributions(
    scene_stats: list[dict[str, object]],
    matrices: dict[str, np.ndarray],
    out_dir: Path,
    include_diagonal: bool,
) -> list[Path]:
    city_out_dir = out_dir / "cities"
    city_out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    city_names = sorted({str(item["city"]) for item in scene_stats}, key=natural_key)
    for city in city_names:
        city_rows = [item for item in scene_stats if item["city"] == city]
        city_values = np.concatenate(
            [flatten_entries(matrices[str(item["scene_name"])], include_diagonal) for item in city_rows],
            axis=0,
        )
        city_means = np.array([float(item["mean"]) for item in city_rows], dtype=np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].hist(city_values, bins=50, color="#1f77b4", alpha=0.9, edgecolor="white")
        axes[0].set_title(f"{city} covisibility distribution")
        axes[0].set_xlabel("Covisibility")
        axes[0].set_ylabel("Count")
        axes[0].grid(alpha=0.25, linestyle="--")

        ranked = sorted(city_rows, key=lambda item: item["mean"])
        labels = [str(item["scene_name"]) for item in ranked]
        values = [float(item["mean"]) for item in ranked]
        y = np.arange(len(ranked))
        axes[1].barh(y, values, color="#2ca02c", alpha=0.9)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlabel("Mean covisibility")
        axes[1].set_title(
            f"{city} scene ranking (n={len(city_rows)}, mean={city_means.mean():.3f}, median={np.median(city_means):.3f})"
        )
        axes[1].grid(axis="x", alpha=0.25, linestyle="--")

        fig.tight_layout()
        out_path = city_out_dir / f"{city}__distribution_ranking.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def select_detail_scenes(
    scene_stats: list[dict[str, object]],
    requested_scenes: list[str],
    scene_regex: str | None,
    auto_scenes: int,
) -> list[str]:
    scene_names = {item["scene_name"] for item in scene_stats}
    selected: list[str] = []

    for scene in requested_scenes:
        if scene not in scene_names:
            raise ValueError(f"Requested scene not found: {scene}")
        selected.append(scene)

    if scene_regex:
        pattern = re.compile(scene_regex)
        selected.extend([name for name in scene_names if pattern.fullmatch(name)])

    if not selected and auto_scenes > 0:
        ranked = sorted(scene_stats, key=lambda item: item["mean"])
        selected.extend([item["scene_name"] for item in ranked[:auto_scenes]])
        selected.extend([item["scene_name"] for item in ranked[-auto_scenes:]])

    return sorted(set(selected), key=natural_key)


def plot_scene_detail(
    scene_name: str,
    matrix: np.ndarray,
    out_dir: Path,
    threshold: float,
    include_diagonal: bool,
) -> list[Path]:
    scene_out_dir = out_dir / "scenes"
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    values = flatten_entries(matrix, include_diagonal=include_diagonal)
    n = matrix.shape[0]

    if include_diagonal or n == 1:
        out_mean = matrix.mean(axis=1)
        in_mean = matrix.mean(axis=0)
    else:
        out_mean = (matrix.sum(axis=1) - np.diag(matrix)) / (n - 1)
        in_mean = (matrix.sum(axis=0) - np.diag(matrix)) / (n - 1)

    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].set_title(f"{scene_name} heatmap")
    axes[0].set_xlabel("Target view")
    axes[0].set_ylabel("Source view")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(values, bins=30, color="#1f77b4", alpha=0.9, edgecolor="white")
    axes[1].axvline(values.mean(), color="#d62728", linestyle="--", linewidth=2, label=f"mean={values.mean():.3f}")
    axes[1].axvline(threshold, color="#2ca02c", linestyle=":", linewidth=2, label=f"threshold={threshold:.2f}")
    axes[1].set_title(f"{scene_name} entry distribution")
    axes[1].set_xlabel("Covisibility")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    heatmap_path = scene_out_dir / f"{scene_name}__heatmap_hist.png"
    fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(heatmap_path)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(n)
    ax.plot(x, out_mean, marker="o", linewidth=1.8, markersize=4, label="outgoing mean")
    ax.plot(x, in_mean, marker="s", linewidth=1.5, markersize=3.5, label="incoming mean")
    ax.axhline(threshold, color="#2ca02c", linestyle=":", linewidth=2, label=f"threshold={threshold:.2f}")
    ax.set_title(f"{scene_name} per-view mean covisibility")
    ax.set_xlabel("View index")
    ax.set_ylabel("Mean covisibility")
    ax.set_ylim(0.0, min(1.0, max(out_mean.max(), in_mean.max(), threshold) * 1.1 + 0.02))
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()

    curve_path = scene_out_dir / f"{scene_name}__per_view_means.png"
    fig.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs.append(curve_path)

    return outputs


def save_summary_json(
    scene_stats: list[dict[str, object]],
    out_dir: Path,
    threshold: float,
    include_diagonal: bool,
    detail_scenes: list[str],
) -> Path:
    all_means = np.array([item["mean"] for item in scene_stats], dtype=np.float32)
    city_names = sorted({str(item["city"]) for item in scene_stats}, key=natural_key)
    city_summary = {}
    for city in city_names:
        rows = [item for item in scene_stats if item["city"] == city]
        city_means = np.array([float(item["mean"]) for item in rows], dtype=np.float32)
        city_summary[city] = {
            "num_scenes": len(rows),
            "mean_of_scene_means": float(city_means.mean()),
            "median_of_scene_means": float(np.median(city_means)),
            "min_scene_mean": float(city_means.min()),
            "max_scene_mean": float(city_means.max()),
        }
    summary = {
        "num_scenes": len(scene_stats),
        "cities": city_summary,
        "threshold": threshold,
        "include_diagonal": include_diagonal,
        "mean_of_scene_means": float(all_means.mean()),
        "median_of_scene_means": float(np.median(all_means)),
        "min_scene_mean": float(all_means.min()),
        "max_scene_mean": float(all_means.max()),
        "detail_scenes": detail_scenes,
        "lowest_mean_scenes": sorted(scene_stats, key=lambda item: item["mean"])[:5],
        "highest_mean_scenes": sorted(scene_stats, key=lambda item: item["mean"])[-5:],
    }
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_stats, matrices = collect_scene_data(
        dataset_root=args.dataset_root,
        covis_subpath=args.covis_subpath,
        threshold=args.threshold,
        include_diagonal=args.include_diagonal,
    )
    scene_stats = sorted(scene_stats, key=lambda item: natural_key(item["scene_name"]))

    csv_path = save_scene_summary_csv(scene_stats, out_dir)
    city_csv_path = save_city_summary_csv(scene_stats, out_dir)
    overall_path = plot_overall_distribution(scene_stats, matrices, out_dir, args.include_diagonal)
    ranking_path = plot_scene_ranking(scene_stats, out_dir)
    frame_count_path = plot_frame_count_distribution(scene_stats, out_dir)
    city_compare_path = plot_city_mean_comparison(scene_stats, out_dir)
    city_plot_paths = plot_city_distributions(scene_stats, matrices, out_dir, args.include_diagonal)

    detail_scenes = select_detail_scenes(
        scene_stats=scene_stats,
        requested_scenes=args.scene,
        scene_regex=args.scene_regex,
        auto_scenes=args.auto_scenes,
    )

    detail_outputs: dict[str, list[str]] = {}
    for scene_name in detail_scenes:
        outputs = plot_scene_detail(
            scene_name=scene_name,
            matrix=matrices[scene_name],
            out_dir=out_dir,
            threshold=args.threshold,
            include_diagonal=args.include_diagonal,
        )
        detail_outputs[scene_name] = [str(path) for path in outputs]

    summary_path = save_summary_json(
        scene_stats=scene_stats,
        out_dir=out_dir,
        threshold=args.threshold,
        include_diagonal=args.include_diagonal,
        detail_scenes=detail_scenes,
    )

    print(f"Saved scene summary CSV: {csv_path}")
    print(f"Saved city summary CSV: {city_csv_path}")
    print(f"Saved overall distribution plot: {overall_path}")
    print(f"Saved scene ranking plot: {ranking_path}")
    print(f"Saved frame count plot: {frame_count_path}")
    print(f"Saved city comparison plot: {city_compare_path}")
    print(f"Saved summary JSON: {summary_path}")
    for path in city_plot_paths:
        print(f"[city] {path}")
    if detail_outputs:
        for scene_name, paths in detail_outputs.items():
            print(f"[scene] {scene_name}")
            for path in paths:
                print(f"  - {path}")


if __name__ == "__main__":
    main()
