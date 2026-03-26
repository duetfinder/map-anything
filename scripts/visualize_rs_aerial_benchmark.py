import argparse
import csv
import json
from pathlib import Path


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return 'nan'
        return f'{v:.6f}'
    return str(v)


def flatten(prefix, d, out):
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            flatten(key, v, out)
        else:
            out[key] = v


def build_markdown_table(title, metric_dict):
    lines = [f'## {title}', '', '| Metric | Value |', '| --- | ---: |']
    for key, value in metric_dict.items():
        lines.append(f'| `{key}` | {fmt(value)} |')
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_json', type=Path)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text())
    out_dir = args.out_dir or args.result_json.parent / 'visualized'
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = {
        'Aerial-only Average': result['aerial_only']['average'],
        'RS-only Average': result['rs_only']['average'],
        'Joint Average': result['joint']['average'],
        'Improvement over Aerial-only': result['improvement']['aerial_vs_aerial_only']['average'],
        'Improvement over RS-only': result['improvement']['rs_vs_rs_only']['average'],
    }

    md_parts = ['# RS-Aerial Benchmark Summary', '']
    md_parts.append('## Metadata')
    md_parts.append('')
    md_parts.append('| Key | Value |')
    md_parts.append('| --- | --- |')
    for key, value in result['metadata'].items():
        md_parts.append(f'| `{key}` | {value} |')
    md_parts.append('')
    for title, metrics in sections.items():
        md_parts.append(build_markdown_table(title, metrics))

    per_scene = result['per_scene_results']
    compact_rows = []
    for scene, scene_data in per_scene.items():
        row = {'scene': scene}
        flatten('', scene_data, row)
        compact_rows.append(row)

    fieldnames = sorted({k for row in compact_rows for k in row.keys()})
    with open(out_dir / 'per_scene_compact.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact_rows)

    ranking_rows = []
    for scene, scene_data in per_scene.items():
        ranking_rows.append({
            'scene': scene,
            'aerial_pointmaps_abs_rel': scene_data['aerial_only'].get('pointmaps_abs_rel'),
            'joint_pointmaps_abs_rel': scene_data['joint'].get('pointmaps_abs_rel'),
            'aerial_gain_pointmaps_abs_rel': scene_data['improvement']['aerial_vs_aerial_only'].get('pointmaps_abs_rel'),
            'rs_height_mae': scene_data['rs_only'].get('rs_height_mae'),
            'joint_rs_height_mae': scene_data['joint'].get('rs_height_mae'),
            'rs_gain_height_mae': scene_data['improvement']['rs_vs_rs_only'].get('rs_height_mae'),
            'joint_global_point_l1': scene_data['joint'].get('joint_global_point_l1'),
        })
    ranking_rows.sort(key=lambda x: (float('inf') if x['aerial_gain_pointmaps_abs_rel'] != x['aerial_gain_pointmaps_abs_rel'] else x['aerial_gain_pointmaps_abs_rel']))

    with open(out_dir / 'scene_ranking.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()) if ranking_rows else [])
        if ranking_rows:
            writer.writeheader()
            writer.writerows(ranking_rows)

    (out_dir / 'summary.md').write_text('\n'.join(md_parts))
    print(f'Wrote {out_dir / "summary.md"}')
    print(f'Wrote {out_dir / "per_scene_compact.csv"}')
    print(f'Wrote {out_dir / "scene_ranking.csv"}')


if __name__ == '__main__':
    main()
