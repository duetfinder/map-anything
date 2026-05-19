import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS = {
    'pi3': 'Pi3',
    'vggt': 'VGGT',
    'mapanything': 'MapAnything',
    'da3': 'DA3',
    'pi3_chicago500_finetuned_p1': 'Pi3 (finetuned P1)',
    'pi3_chicago500_finetuned_p3': 'Pi3 (finetuned P3)',
}
CROSSVIEW_P3_MODELS = {
    'pi3_crossview_p3_base': 'Pi3 CrossView P3 Base',
    'pi3_crossview_p3_modality_embedding': 'Pi3 CrossView P3 Modality Embedding',
    'pi3_crossview_p3_modality_embedding_freeze_shared': 'Pi3 CrossView P3 Freeze Shared',
    'pi3_crossview_p3_modality_embedding_remote_head': 'Pi3 CrossView P3 Remote Head',
    'pi3_crossview_p3_zero_covis': 'Pi3 CrossView P3 Zero Covis',
}
MODEL_SETS = {
    'default': MODELS,
    'crossview_p3': CROSSVIEW_P3_MODELS,
}
VIEWS = [2, 4, 8, 16, 24, 32, 40]
AERIAL_METRICS = [
    'pointmaps_abs_rel',
    'z_depth_abs_rel',
    'pose_ate_rmse',
    'pose_auc_5',
    'ray_dirs_err_deg',
    'metric_scale_abs_rel',
    'metric_point_l1',
]
RS_METRIC_METRICS = [
    'rs_height_mae',
    'rs_height_rmse',
]
RS_NONMETRIC_METRICS = [
    'rs_height_mae_affine',
    'rs_height_rmse_affine',
]
JOINT_METRICS = [
    'joint_global_pointmaps_abs_rel',
]
ALL_METRICS = AERIAL_METRICS + RS_METRIC_METRICS + RS_NONMETRIC_METRICS + JOINT_METRICS


def load_result(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def infer_model_set(root: Path):
    for model_key in CROSSVIEW_P3_MODELS:
        if any((root / f'{model_key}_{num_views}v' / 'rs_aerial_benchmark_results.json').exists() for num_views in VIEWS):
            return CROSSVIEW_P3_MODELS
    return MODELS


def collect_rows(root: Path, models):
    rows = []
    for model_key, model_name in models.items():
        for num_views in VIEWS:
            if model_key.endswith('_unified'):
                dirname = f'{model_key}_{num_views}v'
            elif model_key.startswith('pi3_crossview_p3_'):
                dirname = f'{model_key}_{num_views}v'
            else:
                dirname = f'{model_key}_unified_{num_views}v'
            result_path = root / dirname / 'rs_aerial_benchmark_results.json'
            result = load_result(result_path)
            if result is None:
                print(f'Warning: Result file not found for {model_name} with {num_views} views at {result_path}')
                continue
            for mode_key, linestyle in [('aerial_only', '--'), ('joint', '-')]:
                metrics = result[mode_key]['average']
                row = {
                    'model_key': model_key,
                    'model': model_name,
                    'num_views': num_views,
                    'mode': mode_key,
                    'linestyle': linestyle,
                }
                for metric in AERIAL_METRICS:
                    row[metric] = metrics.get(metric)
                if model_key != 'mapanything':
                    row['metric_scale_abs_rel'] = None
                    row['metric_point_l1'] = None

                if mode_key == 'joint':
                    row['joint_global_pointmaps_abs_rel'] = result['joint']['average'].get('joint_global_pointmaps_abs_rel')
                    row['rs_height_mae'] = result['joint']['average'].get('rs_height_mae') if model_key == 'mapanything' else None
                    row['rs_height_rmse'] = result['joint']['average'].get('rs_height_rmse') if model_key == 'mapanything' else None
                    row['rs_height_mae_affine'] = result['joint']['average'].get('rs_height_mae_affine')
                    row['rs_height_rmse_affine'] = result['joint']['average'].get('rs_height_rmse_affine')
                else:
                    row['joint_global_pointmaps_abs_rel'] = None
                    row['rs_height_mae'] = None
                    row['rs_height_rmse'] = None
                    row['rs_height_mae_affine'] = None
                    row['rs_height_rmse_affine'] = None
                rows.append(row)
    return rows


def save_tables(rows, out_dir: Path):
    df = pd.DataFrame(rows)
    df = df.sort_values(['model', 'mode', 'num_views'])
    df.to_csv(out_dir / 'sweep_metrics_long.csv', index=False)

    summary_rows = []
    for model in df['model'].dropna().unique():
        for mode in ['aerial_only', 'joint']:
            sub = df[(df['model'] == model) & (df['mode'] == mode)].sort_values('num_views')
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                entry = {'model': model, 'mode': mode, 'num_views': int(row['num_views'])}
                for metric in ALL_METRICS:
                    entry[metric] = row.get(metric)
                summary_rows.append(entry)
    pd.DataFrame(summary_rows).to_csv(out_dir / 'sweep_metrics_summary.csv', index=False)


def plot_metrics(rows, out_dir: Path):
    df = pd.DataFrame(rows)
    colors = {
        'Pi3': '#1f77b4',
        'VGGT': "#d42727",
        'MapAnything': '#2ca02c',
        'DA3': '#9467bd',
        'Pi3 (finetuned P1)': "#fc7c0b",
        'Pi3 (finetuned P3)': "#0c31e9",
        'Pi3 CrossView P3 Base': '#1f77b4',
        'Pi3 CrossView P3 Modality Embedding': '#2ca02c',
        'Pi3 CrossView P3 Freeze Shared': '#ff7f0e',
        'Pi3 CrossView P3 Remote Head': '#9467bd',
        'Pi3 CrossView P3 Zero Covis': '#d62728',
    }
    mode_labels = {
        'aerial_only': 'Aerial-only',
        'joint': 'Aerial+RS',
    }
    ncols = 2
    nrows = (len(ALL_METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, ALL_METRICS):
        for model in df['model'].dropna().unique():
            model_df = df[df['model'] == model]
            if model_df.empty:
                continue
            for mode in ['aerial_only', 'joint']:
                sub = model_df[model_df['mode'] == mode].sort_values('num_views')
                if sub.empty:
                    continue
                y = sub[metric]
                if y.isna().all():
                    continue
                ax.plot(
                    sub['num_views'],
                    y,
                    color=colors.get(model),
                    linestyle='--' if mode == 'aerial_only' else '-',
                    marker='o',
                    label=f'{model} / {mode_labels[mode]}',
                )
        ax.set_title(metric + '↑' if 'pose_auc' in metric else (metric + '↓'))    
        ax.set_xlabel('num_views')
        ax.set_xticks(VIEWS)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(ALL_METRICS):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'sweep_metrics_grid.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=Path('/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork'),type=Path)
    parser.add_argument('--out-dir',default=Path('/root/autodl-tmp/outputs/mapanything_experiments/mapanything/benchmarking/rs_guided_dense_mv/newyork/aggregated'), type=Path)
    parser.add_argument('--model-set', choices=['auto', *MODEL_SETS.keys()], default='auto')
    args = parser.parse_args()

    out_dir = args.out_dir or args.root / 'aggregated'
    out_dir.mkdir(parents=True, exist_ok=True)

    models = infer_model_set(args.root) if args.model_set == 'auto' else MODEL_SETS[args.model_set]
    rows = collect_rows(args.root, models)
    if not rows:
        raise SystemExit('No benchmark result files found.')

    save_tables(rows, out_dir)
    plot_metrics(rows, out_dir)
    print(f'Wrote {out_dir / "sweep_metrics_long.csv"}')
    print(f'Wrote {out_dir / "sweep_metrics_summary.csv"}')
    print(f'Wrote {out_dir / "sweep_metrics_grid.png"}')


if __name__ == '__main__':
    main()
