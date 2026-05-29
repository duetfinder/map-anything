#!/usr/bin/env python3
"""Summarize RS-guided dense MV benchmark outputs for remote-conditioning experiments."""

import argparse
import csv
import json
import math
from pathlib import Path

LOWER_IS_BETTER = {
    "pointmaps_abs_rel",
    "z_depth_abs_rel",
    "pose_ate_rmse",
    "ray_dirs_err_deg",
    "metric_scale_abs_rel",
    "metric_point_l1",
    "rs_height_mae_affine",
    "rs_height_rmse_affine",
    "joint_global_pointmaps_abs_rel",
}
HIGHER_IS_BETTER = {"pose_auc_5"}
PRIMARY_AERIAL_METRICS = [
    "pointmaps_abs_rel",
    "z_depth_abs_rel",
    "ray_dirs_err_deg",
    "pose_ate_rmse",
    "pose_auc_5",
]
PRIMARY_RS_METRICS = ["rs_height_mae_affine", "rs_height_rmse_affine"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize same/blank/shuffled RS-guided benchmark diagnostics."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to rs_aerial_benchmark_results.json or their parent directories.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference result JSON/dir used for ordinary_damage diagnostics.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--primary_metric",
        default="pointmaps_abs_rel",
        choices=sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER),
        help="Primary aerial metric used for pass-rate diagnostics.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional directory for summary artifacts. Writes summary.csv, "
            "summary.json, summary.md, and summary.png when matplotlib is available."
        ),
    )
    return parser.parse_args()


def resolve_result_path(path_like):
    path = Path(path_like)
    if path.is_dir():
        path = path / "rs_aerial_benchmark_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark result not found: {path}")
    return path


def load_result(path_like):
    path = resolve_result_path(path_like)
    with path.open("r") as f:
        return path, json.load(f)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def metric_delta(new_value, base_value, metric_name):
    if not finite(new_value) or not finite(base_value):
        return float("nan")
    if metric_name in HIGHER_IS_BETTER:
        return float(new_value - base_value)
    return float(base_value - new_value)


def raw_difference(new_value, base_value):
    if not finite(new_value) or not finite(base_value):
        return float("nan")
    return float(new_value - base_value)


def degradation_delta(candidate_value, reference_value, metric_name):
    if not finite(candidate_value) or not finite(reference_value):
        return float("nan")
    if metric_name in HIGHER_IS_BETTER:
        return float(reference_value - candidate_value)
    return float(candidate_value - reference_value)


def avg_metrics(result, section, mode=None):
    if section == "remote_controls":
        return result.get("remote_controls", {}).get("joint_aerial", {}).get(mode, {}).get("average", {})
    return result.get(section, {}).get("average", {})


def per_scene_metrics(result, section, mode=None):
    if section == "remote_controls":
        return result.get("remote_controls", {}).get("joint_aerial", {}).get(mode, {}).get("per_scene", {})
    return result.get(section, {}).get("per_scene", {})


def compute_pass_rates(result, primary_metric):
    aerial = per_scene_metrics(result, "aerial_only")
    controls = result.get("remote_controls", {}).get("joint_aerial", {})
    same = controls.get("same", {}).get("per_scene", {})
    blank = controls.get("blank", {}).get("per_scene", {})
    shuffled = controls.get("shuffled", {}).get("per_scene", {})

    def is_better(lhs, rhs):
        lhs_value = lhs.get(primary_metric) if isinstance(lhs, dict) else None
        rhs_value = rhs.get(primary_metric) if isinstance(rhs, dict) else None
        if not finite(lhs_value) or not finite(rhs_value):
            return None
        if primary_metric in HIGHER_IS_BETTER:
            return lhs_value > rhs_value
        return lhs_value < rhs_value

    def rate_against(other):
        decisions = []
        for scene, same_metrics in same.items():
            if scene not in other:
                continue
            decision = is_better(same_metrics, other[scene])
            if decision is not None:
                decisions.append(decision)
        if not decisions:
            return float("nan")
        return float(sum(decisions) / len(decisions))

    return {
        f"pass_rate_same_better_than_aerial__{primary_metric}": rate_against(aerial),
        f"pass_rate_same_better_than_blank__{primary_metric}": rate_against(blank),
        f"pass_rate_same_better_than_shuffled__{primary_metric}": rate_against(shuffled),
    }


def summarize_one(name, path, result, reference_result, primary_metric):
    aerial = avg_metrics(result, "aerial_only")
    joint = avg_metrics(result, "joint")
    same = avg_metrics(result, "remote_controls", "same") or joint
    blank = avg_metrics(result, "remote_controls", "blank")
    shuffled = avg_metrics(result, "remote_controls", "shuffled")
    rs_only = avg_metrics(result, "rs_only")

    row = {
        "name": name,
        "path": str(path),
        "paired_scene_count": result.get("metadata", {}).get("paired_scene_count"),
        "primary_metric": primary_metric,
    }

    for metric in PRIMARY_AERIAL_METRICS:
        row[f"aerial_only__{metric}"] = aerial.get(metric, float("nan"))
        row[f"joint_same__{metric}"] = same.get(metric, float("nan"))
        row[f"joint_blank__{metric}"] = blank.get(metric, float("nan"))
        row[f"joint_shuffled__{metric}"] = shuffled.get(metric, float("nan"))
        row[f"same_gain__{metric}"] = metric_delta(
            same.get(metric), aerial.get(metric), metric
        )
        row[f"specific_gain_blank__{metric}"] = metric_delta(
            same.get(metric), blank.get(metric), metric
        )
        row[f"specific_gain_shuffled__{metric}"] = metric_delta(
            same.get(metric), shuffled.get(metric), metric
        )
        if reference_result is not None:
            ref_aerial = avg_metrics(reference_result, "aerial_only")
            row[f"ordinary_damage_vs_reference__{metric}"] = degradation_delta(
                aerial.get(metric), ref_aerial.get(metric), metric
            )

    for metric in PRIMARY_RS_METRICS:
        row[f"rs_only__{metric}"] = rs_only.get(metric, float("nan"))
        row[f"joint_rs__{metric}"] = joint.get(metric, float("nan"))
        row[f"remote_damage__{metric}"] = raw_difference(
            joint.get(metric), rs_only.get(metric)
        )

    row["joint_global_pointmaps_abs_rel"] = joint.get(
        "joint_global_pointmaps_abs_rel", float("nan")
    )
    row.update(compute_pass_rates(result, primary_metric))
    return row


def write_csv(rows, output_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(output_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows, output_path):
    columns = summary_columns(rows)
    with Path(output_path).open("w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_value(row.get(col, "")) for col in columns) + " |\n")


def summary_columns(rows=None):
    columns = [
        "name",
        "paired_scene_count",
        "aerial_only__pointmaps_abs_rel",
        "joint_same__pointmaps_abs_rel",
        "same_gain__pointmaps_abs_rel",
        "specific_gain_blank__pointmaps_abs_rel",
        "specific_gain_shuffled__pointmaps_abs_rel",
        "ordinary_damage_vs_reference__pointmaps_abs_rel",
        "pass_rate_same_better_than_aerial__pointmaps_abs_rel",
        "pass_rate_same_better_than_blank__pointmaps_abs_rel",
        "pass_rate_same_better_than_shuffled__pointmaps_abs_rel",
        "same_gain__z_depth_abs_rel",
        "same_gain__ray_dirs_err_deg",
        "remote_damage__rs_height_mae_affine",
        "joint_global_pointmaps_abs_rel",
    ]
    if rows is None:
        return columns
    return [col for col in columns if any(col in row for row in rows)]


def write_plot(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PNG summary because matplotlib is unavailable: {exc}")
        return False

    metrics = [
        "same_gain__pointmaps_abs_rel",
        "specific_gain_blank__pointmaps_abs_rel",
        "specific_gain_shuffled__pointmaps_abs_rel",
        "pass_rate_same_better_than_shuffled__pointmaps_abs_rel",
    ]
    labels = [
        "same gain",
        "same vs blank",
        "same vs shuffled",
        "pass > shuffled",
    ]
    names = [row.get("name", f"run{i}") for i, row in enumerate(rows)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(8, len(rows) * 1.6), 9), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric, label in zip(axes, metrics, labels):
        values = [row.get(metric, float("nan")) for row in rows]
        ax.bar(names, values)
        ax.axhline(0.0, color="black", linewidth=0.8)
        if metric.startswith("pass_rate"):
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
            ax.axhline(0.6, color="green", linewidth=0.8, linestyle=":")
            ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("RS-guided dense MV benchmark summary")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_output_dir(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "summary.csv")
    with (output_dir / "summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    write_markdown(rows, output_dir / "summary.md")
    write_plot(rows, output_dir / "summary.png")
    print(f"Wrote summary artifacts to {output_dir}")


def print_table(rows):
    columns = summary_columns(rows)
    print("	".join(columns))
    for row in rows:
        print("	".join(format_value(row.get(col, "")) for col in columns))


def main():
    args = parse_args()
    reference_result = None
    if args.reference is not None:
        _, reference_result = load_result(args.reference)

    rows = []
    for result_arg in args.results:
        path, result = load_result(result_arg)
        name = path.parent.name
        rows.append(
            summarize_one(
                name=name,
                path=path,
                result=result,
                reference_result=reference_result,
                primary_metric=args.primary_metric,
            )
        )

    print_table(rows)
    if args.output_dir:
        write_output_dir(rows, args.output_dir)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    if args.output_json:
        with Path(args.output_json).open("w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
