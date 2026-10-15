#!/usr/bin/env python3
"""Assemble the Hugging Face dataset tree (reglab/wisconsin-dairy-cafo) from a
legacy local data folder, so it can be uploaded and `download_data.py` yields a
working install.

    python scripts/stage_hf_upload.py --src data --analysis-outputs analysis_outputs
    # then inspect hf_upload_staging/ and upload:
    hf upload-large-folder reglab/wisconsin-dairy-cafo hf_upload_staging --repo-type dataset

The target layout is the one defined by config.config_params.DataPaths. Nothing
is uploaded by this script. Files already identical in size at the destination
are skipped, so it can be re-run.
"""
import argparse
import shutil
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config.config_params as cfg

SHP_PARTS = ["shp", "shx", "dbf", "prj", "cpg"]


def copy(src, dst, report):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        report["missing"].append(str(src))
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        report["skipped"].append(dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    report["copied"].append(dst)


def recompress_raster(src, dst, report):
    """Lossless deflate + tiling; the raw float32 slope grid is 1.1 GB uncompressed."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        report["missing"].append(str(src))
        return
    if dst.exists():
        report["skipped"].append(dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as r:
        profile = r.profile.copy()
        profile.update(driver="GTiff", compress="deflate", predictor=3, zlevel=9,
                       tiled=True, blockxsize=512, blockysize=512)
        profile.pop("interleave", None)
        with rasterio.open(dst, "w", **profile) as w:
            for _, window in r.block_windows(1):
                w.write(r.read(1, window=window), 1, window=window)
    report["copied"].append(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data", help="legacy local data folder")
    ap.add_argument("--analysis-outputs", default="analysis_outputs",
                    help="legacy folder holding image_bound_map.geojson")
    ap.add_argument("--out", default="hf_upload_staging")
    args = ap.parse_args()

    src, ao, out = Path(args.src), Path(args.analysis_outputs), Path(args.out)
    P = cfg.DataPaths(out)  # destination layout
    rel = lambda p: p.relative_to(out)
    report = {"copied": [], "skipped": [], "missing": []}

    # Files whose location is unchanged from the existing HF repo
    legacy_in = src / "input_data"
    for dst in [P.wdnr_cafos, P.ewg_afos, P.milk_producers, P.permit_animal_records,
                P.full_state_annotations, P.ewg_region_annotations, P.all_labeled_images,
                P.all_cf_clusters, P.four_band_clusters, P.three_band_clusters,
                P.open_water, P.impaired_lakes, P.impaired_rivers, P.water_table_depth,
                P.clusters_dir / "all_cf_clusters.geojson"]:
        copy(legacy_in / dst.relative_to(P.input_dir), dst, report)
    for part in SHP_PARTS + ["xml"]:
        name = f"County_Boundaries_24K.{part}"
        copy(legacy_in / "geospatial/county_boundaries" / name,
             P.county_boundaries.parent / name, report)
    for name in ["wi_cafo_facilities.geojson", "wi_cafo_facilities.parquet", "wi_cafo_facilities.csv"]:
        copy(src / name, out / name, report)

    # New files
    copy(ao / "image_bound_map.geojson", P.image_bound_map, report)
    for part in SHP_PARTS:
        copy(src / "WI_urban_areas" / f"tl_2020_55_place20.{part}",
             P.urban_areas.with_suffix(f".{part}"), report)
    copy(src / "model_predictions/inception/inception_dets.geojson", P.inception_dets, report)
    copy(src / "model_predictions/poultry/all_poultry_predictions_confidence_0.1.geojson",
         P.poultry_dets, report)
    for layer in P.SNAPMAPS_LAYERS:
        for suffix in ["", ".crs"]:
            copy(src / "snapmaps_feather" / f"{layer}.feather{suffix}",
                 Path(str(P.snapmaps_layer(layer)) + suffix), report)
    recompress_raster(legacy_in / "water_data/DEM_30m/WI_slope.tif", P.slope_tif, report)

    # Dataset card
    copy(cfg.REPO_ROOT / "docs/hf_dataset_card.md", out / "README.md", report)
    # (README.md differs in size from any stale copy, so force-refresh it)
    shutil.copy2(cfg.REPO_ROOT / "docs/hf_dataset_card.md", out / "README.md")

    print(f"copied {len(report['copied'])}, up to date {len(report['skipped'])}, "
          f"missing {len(report['missing'])}")
    for m in report["missing"]:
        print("  MISSING source:", m)
    for c in report["copied"]:
        print("  +", rel(c))
    still = P.missing_files()
    print(f"\nrequired-file check against staging tree: {len(still)} missing")
    for m in still:
        # reference_data / manual_review files come from git, not HF
        if out in m.parents:
            print("  MISSING in staging:", rel(m))


if __name__ == "__main__":
    main()
