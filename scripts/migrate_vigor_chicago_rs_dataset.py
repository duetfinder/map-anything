#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

SCENE_LEVEL_MAP_FILES = [
    "map_metadata.json",
    "map_parameters.txt",
]

CORE_PROVIDER_FILES = [
    "pixel_to_point_map.npz",
    "info.json",
]

OPTIONAL_PROVIDER_FILES = [
    "density_map.npy",
    "height_map.exr",
    "occupancy_map.png",
    "pixel_to_point_overlay.png",
    "point_cloud_top_view.png",
    "projected_xy.npy",
    "projected_z.npy",
    "statistics.json",
]


@dataclass
class MigrationStats:
    num_locations_requested: int = 0
    num_locations_written: int = 0
    num_locations_skipped: int = 0
    num_provider_entries_written: int = 0
    num_provider_entries_skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate VIGOR remote-sensing data into a unified dataset root."
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/experiments/exp_005_map_points_generate/vigor/chicago"),
        help="Single-city source root containing per-location/provider geometry products.",
    )
    parser.add_argument(
        "--geometry-parent",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/experiments/exp_005_map_points_generate/vigor"),
        help="Parent directory containing one geometry root per city.",
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/Vigor/map/chicago_subset_2000"),
        help="Single-city source root containing per-location satellite map images.",
    )
    parser.add_argument(
        "--map-parent",
        type=Path,
        default=Path("/root/autodl-tmp/dataset/Vigor/map"),
        help="Parent directory containing one map root per city.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/traindata/Crossview_rs"),
        help="Target unified dataset root.",
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize files in the target root.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional provider allowlist. Default: all providers found under geometry roots.",
    )
    parser.add_argument(
        "--locations",
        nargs="*",
        default=None,
        help="Optional explicit location names. Default: all locations under each geometry root.",
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Optional cap after sorting locations numerically within each city.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target files.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip provider entries with missing source files instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )
    return parser.parse_args()


def natural_location_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_")[-1]), name)
    except Exception:
        return (10**12, name)


def list_locations(root: Path) -> list[str]:
    return sorted(
        [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("location_")],
        key=natural_location_key,
    )


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_parent(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if not dry_run:
            remove_existing(dst)
    ensure_parent(dst, dry_run)
    if dry_run:
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def collect_provider_names(location_geom_dir: Path, provider_allowlist: set[str] | None) -> list[str]:
    providers = sorted([p.name for p in location_geom_dir.iterdir() if p.is_dir()])
    if provider_allowlist is not None:
        providers = [p for p in providers if p in provider_allowlist]
    return providers


def build_required_sources(
    map_location_dir: Path,
    geom_provider_dir: Path,
    provider: str,
) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = [(map_location_dir / f"{provider}.png", "image.png")]
    items.extend((geom_provider_dir / name, name) for name in CORE_PROVIDER_FILES)
    return items


def build_optional_provider_sources(geom_provider_dir: Path) -> list[tuple[Path, str]]:
    return [(geom_provider_dir / name, name) for name in OPTIONAL_PROVIDER_FILES]


def migrate_location(
    *,
    city: str | None,
    scene_name: str,
    location: str,
    geometry_root: Path,
    map_root: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
    skip_missing: bool,
    dry_run: bool,
    provider_allowlist: set[str] | None,
    stats: MigrationStats,
) -> dict:
    map_location_dir = map_root / location
    geom_location_dir = geometry_root / location
    target_location_dir = output_root / scene_name

    if not geom_location_dir.exists() or not map_location_dir.exists():
        missing = []
        if not geom_location_dir.exists():
            missing.append(str(geom_location_dir))
        if not map_location_dir.exists():
            missing.append(str(map_location_dir))
        if skip_missing:
            stats.num_locations_skipped += 1
            return {
                "city": city,
                "location": location,
                "scene_name": scene_name,
                "status": "skipped_missing_location",
                "missing": missing,
                "providers": [],
            }
        raise FileNotFoundError(f"Missing location inputs for {scene_name}: {missing}")

    if not dry_run:
        target_location_dir.mkdir(parents=True, exist_ok=True)

    for map_file in SCENE_LEVEL_MAP_FILES:
        src = map_location_dir / map_file
        dst = target_location_dir / map_file
        if src.exists():
            materialize(src, dst, mode, overwrite, dry_run)

    providers_summary = []
    for provider in collect_provider_names(geom_location_dir, provider_allowlist):
        geom_provider_dir = geom_location_dir / provider
        target_provider_dir = target_location_dir / provider
        required = build_required_sources(map_location_dir, geom_provider_dir, provider)
        missing = [str(src) for src, _ in required if not src.exists()]
        if missing:
            if skip_missing:
                stats.num_provider_entries_skipped += 1
                providers_summary.append(
                    {"provider": provider, "status": "skipped_missing", "missing": missing}
                )
                continue
            raise FileNotFoundError(f"Missing required files for {scene_name}/{provider}: {missing}")

        if not dry_run:
            target_provider_dir.mkdir(parents=True, exist_ok=True)
        for src, rel_dst in required:
            materialize(src, target_provider_dir / rel_dst, mode, overwrite, dry_run)
        for src, rel_dst in build_optional_provider_sources(geom_provider_dir):
            if src.exists():
                materialize(src, target_provider_dir / rel_dst, mode, overwrite, dry_run)

        with open(geom_provider_dir / "info.json", "r", encoding="utf-8") as f:
            info = json.load(f)

        providers_summary.append(
            {
                "provider": provider,
                "status": "ok",
                "projection_type": "rs_global_projective",
                "scene_name": scene_name,
                "city": city,
                "image_path": f"{scene_name}/{provider}/image.png",
                "pointmap_path": f"{scene_name}/{provider}/pixel_to_point_map.npz",
                "info_path": f"{scene_name}/{provider}/info.json",
                "meters_per_pixel": info.get("meters_per_pixel"),
                "coverage_meters": info.get("coverage_meters"),
                "ground_z_approx": info.get("ground_z_approx"),
            }
        )
        stats.num_provider_entries_written += 1

    return {
        "city": city,
        "location": location,
        "scene_name": scene_name,
        "providers": providers_summary,
    }


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict, dry_run: bool) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n", dry_run)


def build_readme() -> str:
    lines = [
        "# VIGOR RS Dataset",
        "",
        "This directory stores the unified remote-sensing dataset for VIGOR.",
        "",
        "Layout:",
        "- `<city>__location_x/` groups one VIGOR scene with a city prefix to avoid collisions.",
        "- `<city>__location_x/<provider>/image.png` stores the remote image.",
        "- `<city>__location_x/<provider>/pixel_to_point_map.npz` stores the sparse per-pixel global XYZ map.",
        "- `<city>__location_x/<provider>/info.json` stores provider metadata used to interpret the pointmap.",
        "- valid mask and height are derived at load time from `pixel_to_point_map.npz`.",
        "- `dataset_meta.json` summarizes the migrated dataset.",
        "- `providers.json` lists provider-level metadata.",
        "",
        "This dataset root is designed to replace direct training-time dependencies on:",
        "- `outputs/experiments/exp_005_map_points_generate/vigor/<city>`",
        "- `dataset/Vigor/map/<city>_subset_2000`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    provider_allowlist = set(args.providers) if args.providers else None

    if args.cities:
        city_roots = [
            (city, args.geometry_parent / city, args.map_parent / f"{city}_subset_2000")
            for city in args.cities
        ]
    else:
        inferred_city = args.geometry_root.name.replace("_subset_2000", "")
        city_roots = [(inferred_city, args.geometry_root, args.map_root)]

    requested_pairs = []
    for city, geometry_root, map_root in city_roots:
        locations = args.locations if args.locations else list_locations(geometry_root)
        if args.max_locations is not None:
            locations = locations[: args.max_locations]
        requested_pairs.extend((city, geometry_root, map_root, location) for location in locations)

    stats = MigrationStats(num_locations_requested=len(requested_pairs))

    location_summaries = []
    for city, geometry_root, map_root, location in requested_pairs:
        scene_name = f"{city}__{location}" if args.cities else location
        location_summary = migrate_location(
            city=city if args.cities else None,
            scene_name=scene_name,
            location=location,
            geometry_root=geometry_root,
            map_root=map_root,
            output_root=output_root,
            mode=args.mode,
            overwrite=args.overwrite,
            skip_missing=args.skip_missing,
            dry_run=args.dry_run,
            provider_allowlist=provider_allowlist,
            stats=stats,
        )
        location_summaries.append(location_summary)
        if location_summary.get("status") != "skipped_missing_location":
            stats.num_locations_written += 1

    provider_names = sorted(
        {
            provider["provider"]
            for location in location_summaries
            for provider in location["providers"]
            if provider["status"] == "ok"
        }
    )

    dataset_meta = {
        "name": "vigor_rs",
        "geometry_root_source": str(args.geometry_root),
        "geometry_parent_source": str(args.geometry_parent),
        "map_root_source": str(args.map_root),
        "map_parent_source": str(args.map_parent),
        "cities": args.cities,
        "output_root": str(output_root),
        "materialization_mode": args.mode,
        "num_locations_requested": stats.num_locations_requested,
        "num_locations_written": stats.num_locations_written,
        "num_locations_skipped": stats.num_locations_skipped,
        "num_provider_entries_written": stats.num_provider_entries_written,
        "num_provider_entries_skipped": stats.num_provider_entries_skipped,
        "providers": provider_names,
        "layout_version": 3,
        "location_summaries_path": "location_manifest.json",
    }
    providers_meta = {
        "providers": provider_names,
        "default_provider": "Google_Satellite" if "Google_Satellite" in provider_names else None,
        "projection_type": "rs_global_projective",
    }
    location_manifest = {"locations": location_summaries}

    write_text(output_root / "README.md", build_readme(), args.dry_run)
    write_json(output_root / "dataset_meta.json", dataset_meta, args.dry_run)
    write_json(output_root / "providers.json", providers_meta, args.dry_run)
    write_json(output_root / "location_manifest.json", location_manifest, args.dry_run)

    print(json.dumps(dataset_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
