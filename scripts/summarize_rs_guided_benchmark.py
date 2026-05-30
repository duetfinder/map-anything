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
            return "-"
        abs_value = abs(value)
        if abs_value != 0 and (abs_value < 1e-4 or abs_value >= 1e4):
            return f"{value:.3e}"
        return f"{value:.6g}"
    if value is None:
        return "-"
    return str(value)


def markdown_escape(value):
    return format_value(value).replace("|", "\\|")


def write_markdown_table(f, columns, rows):
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(markdown_escape(row.get(col, "")) for col in columns) + " |\n")
    f.write("\n")


def concise_verdict(row):
    same_gain = row.get("same_gain__pointmaps_abs_rel", float("nan"))
    blank_gain = row.get("specific_gain_blank__pointmaps_abs_rel", float("nan"))
    shuffled_gain = row.get("specific_gain_shuffled__pointmaps_abs_rel", float("nan"))
    pass_aerial = row.get("pass_rate_same_better_than_aerial__pointmaps_abs_rel", float("nan"))
    if finite(same_gain) and same_gain > 0 and finite(blank_gain) and blank_gain > 0 and finite(shuffled_gain) and shuffled_gain > 0:
        return "same improves aerial and beats controls"
    if finite(blank_gain) and blank_gain > 0 and finite(shuffled_gain) and shuffled_gain > 0:
        return "remote-specific signal exists, but aerial path is not improved"
    if finite(pass_aerial) and pass_aerial >= 0.5:
        return "mixed; scene-level gains exist but averages are weak"
    return "no reliable same-remote gain"


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


def write_markdown(rows, output_path):
    overview_columns = [
        "name",
        "paired_scene_count",
        "same_gain__pointmaps_abs_rel",
        "specific_gain_blank__pointmaps_abs_rel",
        "specific_gain_shuffled__pointmaps_abs_rel",
        "pass_rate_same_better_than_aerial__pointmaps_abs_rel",
        "verdict",
    ]
    aerial_columns = [
        "name",
        "aerial_only__pointmaps_abs_rel",
        "joint_same__pointmaps_abs_rel",
        "joint_blank__pointmaps_abs_rel",
        "joint_shuffled__pointmaps_abs_rel",
        "same_gain__z_depth_abs_rel",
        "same_gain__ray_dirs_err_deg",
    ]
    control_columns = [
        "name",
        "pass_rate_same_better_than_aerial__pointmaps_abs_rel",
        "pass_rate_same_better_than_blank__pointmaps_abs_rel",
        "pass_rate_same_better_than_shuffled__pointmaps_abs_rel",
        "specific_gain_blank__pointmaps_abs_rel",
        "specific_gain_shuffled__pointmaps_abs_rel",
    ]
    rs_columns = [
        "name",
        "rs_only__rs_height_mae_affine",
        "joint_rs__rs_height_mae_affine",
        "remote_damage__rs_height_mae_affine",
        "rs_only__rs_height_rmse_affine",
        "joint_rs__rs_height_rmse_affine",
        "remote_damage__rs_height_rmse_affine",
        "joint_global_pointmaps_abs_rel",
    ]

    rows_with_verdict = []
    for row in rows:
        row_copy = dict(row)
        row_copy["verdict"] = concise_verdict(row)
        rows_with_verdict.append(row_copy)

    with Path(output_path).open("w") as f:
        f.write("# RS-guided Dense MV Summary\n\n")
        f.write("Positive `same_gain` means same remote improves over aerial-only for lower-is-better metrics. ")
        f.write("Positive `specific_gain_*` means same remote beats that control. ")
        f.write("Positive `remote_damage` means joint RS is worse than RS-only; negative means better.\n\n")

        f.write("## Overview\n\n")
        write_markdown_table(f, [c for c in overview_columns if any(c in r for r in rows_with_verdict)], rows_with_verdict)

        f.write("## Aerial Reconstruction\n\n")
        write_markdown_table(f, [c for c in aerial_columns if any(c in r for r in rows)], rows)

        f.write("## Remote-Control Specificity\n\n")
        write_markdown_table(f, [c for c in control_columns if any(c in r for r in rows)], rows)

        if any(any(c in row for c in rs_columns) for row in rows):
            f.write("## RS Branch / Global Metrics\n\n")
            write_markdown_table(f, [c for c in rs_columns if any(c in r for r in rows)], rows)

        if len(rows) <= 6:
            f.write("## Per-Experiment Details\n\n")
            for row in rows:
                f.write(f"### {markdown_escape(row.get('name', 'run'))}\n\n")
                detail_rows = [
                    {"metric": key, "value": value}
                    for key, value in row.items()
                    if key not in {"name", "path"}
                ]
                write_markdown_table(f, ["metric", "value"], detail_rows)


def _plot_values_for_metric(rows, metric):
    values = []
    for row in rows:
        value = row.get(metric, float("nan"))
        values.append(value if finite(value) else float("nan"))
    return values


def _bar_colors(values):
    colors = []
    for value in values:
        if not finite(value):
            colors.append("#b8b8b8")
        elif value >= 0:
            colors.append("#2c7fb8")
        else:
            colors.append("#d95f0e")
    return colors


def write_plot(rows, output_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PNG summary because matplotlib is unavailable: {exc}")
        return False

    metrics = [
        ("same_gain__pointmaps_abs_rel", "Same remote gain vs aerial", 0.0, None),
        ("specific_gain_blank__pointmaps_abs_rel", "Specific gain vs blank", 0.0, None),
        ("specific_gain_shuffled__pointmaps_abs_rel", "Specific gain vs shuffled", 0.0, None),
        ("pass_rate_same_better_than_aerial__pointmaps_abs_rel", "Pass rate: same > aerial", 0.5, (0, 1)),
        ("pass_rate_same_better_than_blank__pointmaps_abs_rel", "Pass rate: same > blank", 0.5, (0, 1)),
        ("remote_damage__rs_height_mae_affine", "RS height damage: joint - RS-only", 0.0, None),
    ]
    metrics = [item for item in metrics if any(item[0] in row for row in rows)]
    if not metrics:
        print("Skipping PNG summary because no plottable metrics were found")
        return False

    names = [str(row.get("name", f"run{i}")) for i, row in enumerate(rows)]
    short_names = [name if len(name) <= 42 else name[:19] + "..." + name[-20:] for name in names]

    n_metrics = len(metrics)
    fig_height = max(7.0, 1.7 * n_metrics + 0.28 * max(1, len(rows)))
    fig, axes = plt.subplots(n_metrics, 1, figsize=(13, fig_height), constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]

    y_positions = list(range(len(rows)))
    for ax, (metric, title, reference, xlim) in zip(axes, metrics):
        values = _plot_values_for_metric(rows, metric)
        ax.barh(y_positions, values, color=_bar_colors(values), height=0.62)
        ax.axvline(reference, color="#222222", linewidth=0.9)
        if metric.startswith("pass_rate"):
            ax.axvline(0.6, color="#238b45", linewidth=0.9, linestyle=":")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10, pad=6)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        if xlim is not None:
            ax.set_xlim(*xlim)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        for y, value in zip(y_positions, values):
            if finite(value):
                ax.text(value, y, "  " + format_value(value), va="center", fontsize=7)

    fig.suptitle("RS-guided dense MV benchmark summary", fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
