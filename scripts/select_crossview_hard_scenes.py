#!/usr/bin/env python3
"""Select hard Crossview scenes from an RS-aerial benchmark result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_AVAILABLE_SCENES = Path(
    "/root/autodl-tmp/traindata/mapanything_metadata/Crossview_rs_aerial/val/"
    "Crossview_rs_aerial_scene_list_val.npy"
)


def metric_value(scene_result: dict, metric: str) -> float:
    value = scene_result.get("aerial_only", {}).get(metric)
    return float(value) if value is not None else float("nan")


def normalize_scene_name(scene: str, city: str | None, available: set[str]) -> str | None:
    if scene in available:
        return scene
    if city and "__" not in scene:
        candidate = f"{city}__{scene}"
        if candidate in available:
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--available-scenes", type=Path, default=DEFAULT_AVAILABLE_SCENES)
    parser.add_argument("--out-npy", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--metric", default="pointmaps_abs_rel")
    parser.add_argument(
        "--assume-city",
        default=None,
        help="Optional city prefix for legacy scene labels like location_451.",
    )
    args = parser.parse_args()

    data = json.loads(args.source_result.read_text(encoding="utf-8"))
    available = set(np.load(args.available_scenes, allow_pickle=True).tolist())

    rows = []
    skipped = []
    for raw_scene, scene_result in data.get("per_scene_results", {}).items():
        scene = normalize_scene_name(raw_scene, args.assume_city, available)
        if scene is None:
            skipped.append(raw_scene)
            continue
        value = metric_value(scene_result, args.metric)
        if not np.isfinite(value):
            skipped.append(raw_scene)
            continue
        rows.append(
            {
                "scene": scene,
                "source_scene": raw_scene,
                "aerial_pointmaps_abs_rel": value,
            }
        )

    rows.sort(key=lambda item: item["aerial_pointmaps_abs_rel"], reverse=True)
    selected = rows[: args.top_k]
    if len(selected) < args.top_k:
        raise SystemExit(
            f"Only found {len(selected)} valid scenes, fewer than requested top_k={args.top_k}. "
            f"Skipped {len(skipped)} scenes."
        )

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, np.asarray([item["scene"] for item in selected], dtype=object))
    args.out_json.write_text(
        json.dumps(
            {
                "source_result": str(args.source_result),
                "available_scenes": str(args.available_scenes),
                "metric": args.metric,
                "top_k": args.top_k,
                "assume_city": args.assume_city,
                "selected": selected,
                "skipped_count": len(skipped),
                "skipped_examples": skipped[:20],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_npy}")
    print(f"Wrote {args.out_json}")
    for idx, item in enumerate(selected, 1):
        print(f"{idx:02d} {item['scene']} {item['aerial_pointmaps_abs_rel']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
