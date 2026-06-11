#!/usr/bin/env python3
"""Visualize projection-aux offset labels saved by dense reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def finite_percentile(values: np.ndarray, q: float, default: float = 1.0) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to dense_pointmap_projection_aux_labels.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output images. Defaults to the npz parent directory.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Subsample stride for the quiver field.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=95.0,
        help="Magnitude/height percentile used for color clipping.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Matplotlib quiver scale. Larger values draw shorter arrows. Defaults to data-dependent value.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input)
    valid = data["dense_valid_mask"].astype(bool)
    offset = data["dense_offset_from_global"].astype(np.float32)
    rel_height = data["dense_rel_height"].astype(np.float32)

    finite_offset = np.isfinite(offset).all(axis=-1)
    mask = valid & finite_offset
    mag = np.linalg.norm(offset, axis=-1)
    mag_vis = np.where(mask, mag, np.nan)

    height_vmax = finite_percentile(np.maximum(rel_height[valid], 0.0), args.clip_percentile)
    mag_vmax = finite_percentile(mag[mask], args.clip_percentile)

    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sample = mask & (xx % args.stride == 0) & (yy % args.stride == 0)
    if not sample.any():
        raise RuntimeError("No valid vectors after applying stride/mask.")

    xs = xx[sample]
    ys = yy[sample]
    us = offset[..., 0][sample]
    vs = offset[..., 1][sample]
    cs = mag[sample]

    quiver_scale = args.scale
    if quiver_scale is None:
        median_mag = finite_percentile(cs, 50.0)
        quiver_scale = max(median_mag * 16.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    bg = ax.imshow(
        rel_height,
        cmap="magma",
        vmin=0.0,
        vmax=height_vmax,
        interpolation="nearest",
    )
    q = ax.quiver(
        xs,
        ys,
        us,
        -vs,
        cs,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0022,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.5,
    )
    ax.set_title(f"Dense projection offset field, stride={args.stride}")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(bg, ax=ax, fraction=0.035, pad=0.01, label="relative height")
    fig.colorbar(q, ax=ax, fraction=0.035, pad=0.05, label="offset magnitude")
    fig.tight_layout()
    quiver_path = output_dir / "dense_offset_vector_field.png"
    fig.savefig(quiver_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    im = ax.imshow(mag_vis, cmap="viridis", vmin=0.0, vmax=mag_vmax, interpolation="nearest")
    ax.set_title("Dense projection offset magnitude")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    mag_path = output_dir / "dense_offset_magnitude.png"
    fig.savefig(mag_path)
    plt.close(fig)

    summary = {
        "input": str(args.input),
        "stride": int(args.stride),
        "valid_ratio": float(mask.mean()),
        "vector_count": int(sample.sum()),
        "offset_mag_mean": float(np.nanmean(mag_vis)),
        "offset_mag_p50": finite_percentile(mag[mask], 50.0),
        "offset_mag_p90": finite_percentile(mag[mask], 90.0),
        "offset_mag_p95": mag_vmax,
        "offset_mag_max": float(np.nanmax(mag_vis)),
        "quiver_scale": float(quiver_scale),
        "quiver_path": str(quiver_path),
        "magnitude_path": str(mag_path),
    }
    summary_path = output_dir / "dense_offset_vector_field_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
