#!/usr/bin/env python3
"""
Generate all paper results for the WI-CAFO dairy facility detection study.

Usage:
    python generate_paper_results.py
    python generate_paper_results.py --skip-pixel-stats      # use cached pixel P/R (default)
    python generate_paper_results.py --skip-snapmaps         # skip snapmaps loading
    python generate_paper_results.py --skip-imagery          # skip NAIP tile figures

Outputs are saved to paper_results/ with subdirectories for each analysis section.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import config.config_params as cfg
import create_figures as cf
import plotting_utils as pu
import cluster
import cluster_analysis_functions as caf
import estimate_animal_units as est_au
import analyze_model_outputs as amo
import process_snapmaps
import spatial_stats as sps
import permit_status_regression as psr
import regional_robustness_analysis as rra


def apply_paper_style():
    """Set consistent matplotlib rcParams for all paper figures.

    Reads font, size, and color constants from config.config_params so every
    figure produced by this pipeline shares the same visual language.
    """
    plt.rcParams.update({
        # Font
        "font.family": cfg.FIG_FONT_FAMILY,
        f"font.{cfg.FIG_FONT_FAMILY}": cfg.FIG_FONT,
        # Axes labels / titles
        "axes.labelsize": cfg.FIG_LABEL_SIZE,
        "axes.titlesize": cfg.FIG_TITLE_SIZE,
        "axes.linewidth": cfg.FIG_LINEWIDTH,
        # Tick marks
        "xtick.labelsize": cfg.FIG_TICK_SIZE,
        "ytick.labelsize": cfg.FIG_TICK_SIZE,
        # Legend
        "legend.fontsize": cfg.FIG_LEGEND_SIZE,
        "legend.framealpha": 0.8,
        # Resolution
        "savefig.dpi": cfg.FIG_DPI,
        "figure.dpi": 100,
    })


# ============================================================================
# Section 1: Path setup
# ============================================================================

def setup_paths():
    """Load config and create output directory structure."""
    with open("config/config.yml", "r") as f:
        configs = yaml.safe_load(f)

    paths = {
        "analysis_output_path": Path(configs["analysis_output_path"]),
        "data_path": Path(configs["data_path"]),
        "land_parcel_path": Path(configs["land_parcel_path"]),
        "model_prediction_path": Path(configs["model_prediction_path"]),
        "cluster_path": Path(configs["cluster_path"]),
        "fig_path": Path(configs["fig_path"]),
    }

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = configs["gcp_cred_path"]

    output_base = Path("paper_results")
    subdirs = {
        "validation": output_base / "01_model_validation",
        "segmentation": output_base / "02_segmentation_quality",
        "error": output_base / "03_error_analysis",
        "unpermitted": output_base / "04_unpermitted_analysis",
        "risk": output_base / "05_risk_assessment",
        "sensitivity": output_base / "06_space_param_sensitivity",
        "spatial": output_base / "07_spatial_clustering",
        "tables": output_base / "tables",
        "pub_dataset": output_base / "publication_dataset",
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    return paths, subdirs


# ============================================================================
# Section 2: Data loading
# ============================================================================

def load_all_data(paths, read_snapmaps=True):
    """Load all datasets using cf.load_data() and estimate animal units."""
    print("Loading data...")
    (
        image_bound_map, counties, parcels, milk_producers, WDNR_CAFOs,
        ewg_afos, cf_annotations_exc_EWG, all_cf_clusters, cf_clusters_EWG,
        cf_clusters_exc_EWG, inception_detections, poultry_detections,
        four_band_clusters, four_band_clusters_exc_EWG, three_band_clusters,
        three_band_clusters_exc_EWG, all_waters, impaired_waters,
        water_table_depth, snapmaps,
    ) = cf.load_data(
        paths["analysis_output_path"],
        paths["data_path"],
        paths["land_parcel_path"],
        paths["cluster_path"],
        paths["model_prediction_path"],
        read_snapmaps=read_snapmaps,
    )

    # Compute AU estimates if needed
    if "animal_unit_estimate" not in all_cf_clusters.columns:
        all_cf_clusters = est_au.sample_and_calc_au(
            all_cf_clusters, include_area_uncertainty=False
        )
    if "animal_unit_estimate" not in four_band_clusters.columns:
        four_band_clusters = est_au.sample_and_calc_au(
            four_band_clusters, include_area_uncertainty=True
        )

    return {
        "image_bound_map": image_bound_map,
        "counties": counties,
        "parcels": parcels,
        "milk_producers": milk_producers,
        "WDNR_CAFOs": WDNR_CAFOs,
        "ewg_afos": ewg_afos,
        "cf_annotations_exc_EWG": cf_annotations_exc_EWG,
        "all_cf_clusters": all_cf_clusters,
        "cf_clusters_EWG": cf_clusters_EWG,
        "cf_clusters_exc_EWG": cf_clusters_exc_EWG,
        "inception_detections": inception_detections,
        "poultry_detections": poultry_detections,
        "four_band_clusters": four_band_clusters,
        "four_band_clusters_exc_EWG": four_band_clusters_exc_EWG,
        "three_band_clusters": three_band_clusters,
        "three_band_clusters_exc_EWG": three_band_clusters_exc_EWG,
        "all_waters": all_waters,
        "impaired_waters": impaired_waters,
        "water_table_depth": water_table_depth,
        "snapmaps": snapmaps,
    }


# ============================================================================
# Section 3: Model validation figures
# ============================================================================

def generate_model_validation(data, subdirs):
    """Permit comparison and EWG validation. 6 figures → 01_model_validation/."""
    out = subdirs["validation"]

    # Merge CF clusters with permits
    permit_matched, err = caf.merge_clusters_permits(
        data["all_cf_clusters"], data["WDNR_CAFOs"],
        sum_satellite_counts=True, discrep_analysis=True, only_dairy=True,
    )

    print(err)

    # Permit vs estimate scatter
    pu.plot_permit_v_estimate(
        permit_matched,
        #title="Comparison of AU estimates and permit-reported values",
        include_uncertainty=True,
        color_satellite_summed=False,
        fig_save_path=out / "permit_v_estimate_all_clusters.svg",
    )
    plt.close("all")

    # Discrepancy histograms (raw, pct, abs_pct)
    for metric, fname in [
        ("raw", "permit_estimate_discrepancy_hist_raw"),
        ("pct", "permit_estimate_discrepancy_hist_pct"),
        ("abs_pct", "permit_estimate_discrepancy_hist_abs_pct"),
    ]:
        kwargs = {}
        if metric == "pct":
            kwargs = {"trim_percentiles": (5, 95)}
        elif metric == "abs_pct":
            # abs_pct is always ≥ 0, so extremes are one-sided (upper tail only).
            # (0, 90) excludes the top 10%, keeping 90% of the data.
            kwargs = {"trim_percentiles": (0, 90)}
        cf.plot_permit_estimate_discrepancy_hist(
            permit_matched,
            estimate_col="animal_unit_estimate",
            error_metric=metric,
            save_path=out / f"{fname}.svg",
            title="",
            **kwargs,
        )
        plt.close("all")

    # AU error plot
    cf.permit_CF_AU_error_plot(
        permit_matched,
        fig_save_path=out / "permit_CF_AU_error_plot.svg",
    )
    plt.close("all")

    # EWG comparison
    cf.EWG_cf_count_compare(
        data["cf_clusters_EWG"], data["ewg_afos"],
        save_path=out / "ewg_cf_count_compare.svg",
        estimate_logic="point",
    )
    plt.close("all")

    print(f"  Saved 6 figures to {out}")
    return permit_matched


# ============================================================================
# Section 3a: Cross-regional robustness of validation error rates (reviewer:
# clustering thresholds are empirically defined; cross-regional robustness
# requires further validation)
# ============================================================================

def generate_regional_robustness_analysis(data, permit_matched, subdirs):
    """Test whether WDNR permit recall, AU discrepancy, and EWG precision/
    recall diverge across Wisconsin's 5 WDNR administrative regions under the
    clustering thresholds (500m fuzzy-name distance, 25% building overlap,
    150m proximity) as currently set. → tables/ + 01_model_validation/.

    permit_matched: the dairy-only permit-matched clusters from
        generate_model_validation(), reused so the AU discrepancy numbers
        match the rest of the paper.
    """
    results = rra.run_regional_robustness_analysis(data, permit_matched, subdirs)
    plt.close("all")
    return results


# ============================================================================
# Section 4: Segmentation quality figures
# ============================================================================

def generate_segmentation_quality(data, paths, subdirs,
                                  recalculate_pixel_stats=False):
    """Pixel P/R/IOU, annotation errors, examples. 3 figures → 02_segmentation_quality/."""
    out = subdirs["segmentation"]

    # Pixel-level P/R/IOU curves
    cf.pixel_level_p_r_IOU_curves(
        model_prediction_path=paths["model_prediction_path"],
        vec_thresh=np.array([0.25, 0.4, 0.5, 0.6, 0.75]),
        CF_polygons=data["cf_annotations_exc_EWG"],
        image_bound_map=data["image_bound_map"],
        counties=data["counties"],
        stats_path=paths["analysis_output_path"] / "pixel_level_p_r_IOU_curves.csv",
        recalculate_stats=recalculate_pixel_stats,
        models=["four_band"],
        save_path=out / "pixel_level_p_r_IOU_curves.png",
        suppress_legend=True,
    )
    plt.close("all")

    # Model vs human annotation errors
    cf.model_human_annotation_errors(
        data["four_band_clusters"],
        data["all_cf_clusters"],
        save_path=out / "model_human_annotation_errors.svg",
    )
    plt.close("all")

    # Annotation vs prediction examples
    cf_cluster = data["all_cf_clusters"][
        data["all_cf_clusters"]["jpeg_names"].apply(
            lambda x: "WI_Brown_2_7168_12288.jpeg" in x
        )
    ]
    four_band_cluster = data["four_band_clusters"][
        data["four_band_clusters"]["jpeg_names"].apply(
            lambda x: "WI_Brown_2_7168_12288.jpeg" in x
        )
    ]
    if len(cf_cluster) > 0 and len(four_band_cluster) > 0:
        cf.annotation_prediction_examples(
            cf_cluster, four_band_cluster,
            save_path=out / "annotation_prediction_examples.svg",
        )
        plt.close("all")

    print(f"  Saved figures to {out}")


# ============================================================================
# Section 5: Error analysis figures
# ============================================================================

def generate_error_analysis(data, paths, subdirs, skip_imagery=False):
    """Facility-level P/R curve + case study imagery. → 03_error_analysis/."""
    out = subdirs["error"]

    # Facility-level precision/recall vs threshold
    pr_curve = cf.facility_level_pr_lowerbound_graph(
        data["four_band_clusters"],
        "four_band",
        data["WDNR_CAFOs"],
        data["counties"],
        data["ewg_afos"],
        AFO_lower_bound_threshes=np.arange(0, 1000, 100),
        AFO_lower_bound_logic="point estimate",
        save_path=out / "facility_level_pr_lowerbound.svg",
        suppress_show=True,
    )
    plt.close("all")
    print("\n  Facility-level P/R curve data:")
    print(pr_curve.to_string(index=False))
    pr_curve.to_csv(subdirs["tables"] / "facility_pr_lowerbound.csv", index=False)

    if not skip_imagery:
        # Case study examples via plot_cluster_parcel
        examples = [
            ("small_false_pos_gas_station", "PINK ELEPHANT", 150),
            ("small_false_pos_trucking", "BUI PROPERTIES LLC", 500),
            ("false_neg_missed_area", "DANIEL KAMPS", 500),
            ("true_pos_model_permitted", "TINEDALE FARMS", 500),
            ("large_false_pos_ag_products", "AFP ADVANCED", 500),
            ("large_false_pos_food", "KEMPS LLC", 500),
        ]
        for fname, search_str, zoom in examples:
            subset = data["four_band_clusters"][
                data["four_band_clusters"]["parcel_owner1_names"]
                .fillna("")
                .str.contains(search_str, case=False)
            ]
            if len(subset) == 0:
                print(f"  Warning: no cluster found for '{search_str}', skipping {fname}")
                continue
            try:
                pu.plot_cluster_parcel(
                    subset.iloc[[0]],
                    image_bound_map=data["image_bound_map"],
                    zoom_radius=zoom,
                    include_AU_estimate=True,
                    fig_path=out / f"{fname}.png",
                )
                plt.close("all")
            except Exception as e:
                print(f"  Warning: could not generate {fname}: {e}")

    print(f"  Saved figures to {out}")


# ============================================================================
# Section 6: Image review efficiency stats
# ============================================================================

# Wisconsin has no dedicated Census "Urban Area" (uac) layer in the TIGER
# extract we have on hand (data/WI_urban_areas is a bulk per-county dump with
# no uac file, and the statewide tabblock20 file's UR20/UACE20 urban-flag
# fields are unpopulated in this extract). tl_2020_55_place20.shp
# (incorporated cities/villages) is the closest deterministic, statewide
# proxy for "urban area" and is what the >500m urban-proximity filter uses
# below and in generate_ewg_fp_waterfall().
URBAN_AREA_PROXY_PATH = "WI_urban_areas/tl_2020_55_place20.shp"


def _add_cf_review_flags(four_band_clusters, all_cf_clusters, paths):
    """Add any_images_sent and matched_to_CF_annot to four_band_clusters.

      any_images_sent     – cluster has ≥1 image that was sent to CF for labeling
      matched_to_CF_annot – cluster is within 10m of a human-annotated cluster

    Shared by generate_image_review_stats() and
    generate_cf_unmatched_fp_analysis() so the two analyses can't drift.
    """
    four_band = four_band_clusters.copy()

    all_labeled_images = cf.get_labeled_images(how="csv", data_path=paths["data_path"])
    labeled_set = set(all_labeled_images["Image name"].tolist())
    four_band["any_images_sent"] = four_band["jpeg_names"].apply(
        lambda x: any(item in labeled_set for item in (x if isinstance(x, (list, tuple)) else []))
    )

    # matched_to_CF_annot: sjoin_nearest to all_cf_clusters, max 10m.
    # how='left' so unmatched rows get NaN, then matched_to_CF_annot = ~cf_polygon_indices.isna()
    cf_clusters = all_cf_clusters[["geometry", "polygon_indices"]].rename(
        columns={"polygon_indices": "cf_polygon_indices"}
    ).copy()
    four_band = gpd.sjoin_nearest(
        four_band, cf_clusters, max_distance=10, how="left", distance_col="distance_to_cf"
    )
    # polygon_indices is list-valued (loaded with literal_eval), so
    # drop_duplicates must use a str key.
    four_band["_pstr2"] = four_band["polygon_indices"].apply(str)
    four_band = four_band.sort_values("distance_to_cf").drop_duplicates(
        subset="_pstr2", keep="first"
    ).drop(columns=["_pstr2"])
    four_band["matched_to_CF_annot"] = ~four_band["cf_polygon_indices"].isna()
    four_band = four_band.drop(
        columns=["cf_polygon_indices", "distance_to_cf", "index_right"], errors="ignore"
    )
    return four_band


def generate_image_review_stats(data, paths, subdirs):
    """Compute and save stats on human imagery review reduction. → tables/.

    Replicates the four_band_cfsend construction from all_key_results.ipynb:
      1. any_images_sent  – cluster has ≥1 image that was sent to CF for labeling
      2. matched_to_CF_annot – cluster is within 10m of a human-annotated cluster
      3. Filters: urban area (>500m), parcel owner keywords, non-dairy CAFO (>300m)
      4. need_to_send = not sent AND not already annotated
      5. Count unique images where need_to_send OR any_images_sent
    """
    four_band = _add_cf_review_flags(data["four_band_clusters"], data["all_cf_clusters"], paths)

    # ------------------------------------------------------------------
    # 3. Urban area filter: keep clusters >500m from any urban area
    # ------------------------------------------------------------------
    urban_areas_path = paths["data_path"].parent / URBAN_AREA_PROXY_PATH
    four_band = four_band.drop(columns=["index_right"], errors="ignore")
    try:
        urban_areas = gpd.read_file(urban_areas_path).to_crs(cfg.WI_EPSG)
        four_band = four_band.sjoin_nearest(
            urban_areas[["geometry"]], distance_col="distance_to_urban_area"
        )
        four_band = four_band[four_band["distance_to_urban_area"] > 500]
        four_band = four_band.drop(columns=["index_right"], errors="ignore")
        four_band["_pstr"] = four_band["polygon_indices"].apply(str)
        four_band = four_band.drop_duplicates(subset="_pstr").drop(columns=["_pstr"])
    except Exception as e:
        print(f"  Warning: could not apply urban area filter: {e}")

    # ------------------------------------------------------------------
    # 4. Parcel owner keyword filter
    # ------------------------------------------------------------------
    exclude_keywords = ["STORAG", "CHEMICAL", "ELECTRONIC", "WOOD"]
    four_band = four_band[
        ~four_band["parcel_owner1_names"].apply(
            lambda x: any(kw in str(x) for kw in exclude_keywords)
        )
    ]

    # ------------------------------------------------------------------
    # 5. Non-dairy CAFO filter: exclude clusters within 300m of non-dairy permits
    # ------------------------------------------------------------------
    four_band = four_band.drop(columns=["index_right"], errors="ignore")
    nondairy_cafos = data["WDNR_CAFOs"][data["WDNR_CAFOs"]["AnimalType"] != "Dairy"].to_crs(cfg.WI_EPSG)
    if len(nondairy_cafos) > 0:
        near_nondairy = four_band.sjoin_nearest(
            nondairy_cafos[["geometry"]], max_distance=300, how="inner"
        )
        nondairy_strs = set(near_nondairy["polygon_indices"].apply(str).tolist())
        four_band = four_band[
            ~four_band["polygon_indices"].apply(str).isin(nondairy_strs)
        ]

    # ------------------------------------------------------------------
    # 6. Deduplicate by polygon_indices (mirrors notebook drop_duplicates)
    # ------------------------------------------------------------------
    four_band["polygon_indices_str"] = four_band["polygon_indices"].apply(str)
    four_band = four_band.drop_duplicates(subset="polygon_indices_str").drop(
        columns=["polygon_indices_str"]
    )

    # ------------------------------------------------------------------
    # 7. need_to_send: not yet sent AND not already covered by an annotation
    # ------------------------------------------------------------------
    four_band["need_to_send"] = (
        (four_band["any_images_sent"] == False) &
        (four_band["matched_to_CF_annot"] == False)
    )

    # ------------------------------------------------------------------
    # 8. Count images (mirrors notebook cell 55)
    # ------------------------------------------------------------------
    four_band_send = four_band[
        (four_band["need_to_send"] == True) | (four_band["any_images_sent"] == True)
    ]
    images_to_send = four_band_send["jpeg_names"].explode().nunique()

    total_wi_images = data["image_bound_map"]["filename"].nunique()
    reduction_factor = total_wi_images / images_to_send if images_to_send > 0 else np.nan

    stats = pd.DataFrame({
        "metric": [
            "total_wi_images",
            "images_needing_review",
            "fraction_needing_review",
            "reduction_factor",
        ],
        "value": [
            total_wi_images,
            images_to_send,
            images_to_send / total_wi_images if total_wi_images > 0 else np.nan,
            reduction_factor,
        ],
    })
    stats.to_csv(subdirs["tables"] / "image_review_stats.csv", index=False)

    print(f"  Total WI images: {total_wi_images}")
    print(f"  Images needing review: {images_to_send}")
    print(f"  Reduction factor: {reduction_factor:.1f}x")


def generate_human_annotation_yield_stats(data, paths, subdirs):
    """Quantify the marginal value of human annotation, conditional on a
    cluster having already survived model size-filtering and automated
    post-processing (Reviewer #2: error analysis by pipeline stage).

    Scope matters here: any_images_sent/matched_to_CF_annot cover every
    historical CloudFactory batch sent over the life of the project, under
    whatever size/criteria were in effect at the time — not just the
    current >500 AU decision rule. Computing "fraction sent but not
    annotated" over that whole history overstates how much human review
    weeds out *given everything upstream already filtered*, since it
    includes many small/low-quality historical sends automated filtering
    would exclude today. This function instead restricts to clusters that
    pass the CURRENT decision rule (>500 AU) and the same post-processing
    filters used in generate_ewg_fp_waterfall / generate_image_review_stats
    (urban-area proximity, parcel-owner keyword, non-dairy-CAFO exclusion),
    then asks: of what's actually been sent for annotation within that
    already-filtered population, what fraction came back unconfirmed?

    Writes human_annotation_yield.csv to tables/. Returns the stats
    DataFrame.
    """
    four_band = _add_cf_review_flags(data["four_band_clusters"], data["all_cf_clusters"], paths)

    fb = four_band[four_band["animal_unit_estimate"] > 500].copy()
    n0 = len(fb)

    urban_areas = gpd.read_file(
        paths["data_path"].parent / URBAN_AREA_PROXY_PATH
    ).to_crs(cfg.WI_EPSG)
    fb = fb.sjoin_nearest(urban_areas[["geometry"]], distance_col="dist_to_urban")
    fb = fb[~fb.index.duplicated(keep="first")]
    fb = fb[fb["dist_to_urban"] > 500].copy()
    fb = fb.drop(columns=["index_right", "dist_to_urban"], errors="ignore")

    exclude_keywords = ["STORAG", "CHEMICAL", "ELECTRONIC", "WOOD"]
    fb = fb[
        ~fb["parcel_owner1_names"].apply(lambda x: any(kw in str(x) for kw in exclude_keywords))
    ].copy()

    nondairy_cafos = data["WDNR_CAFOs"][data["WDNR_CAFOs"]["AnimalType"] != "Dairy"].to_crs(cfg.WI_EPSG)
    near_nondairy = fb.sjoin_nearest(nondairy_cafos[["geometry"]], max_distance=300, how="inner")
    nondairy_pstrs = set(near_nondairy["polygon_indices"].apply(str))
    fb = fb[~fb["polygon_indices"].apply(str).isin(nondairy_pstrs)].copy()

    fb["_pstr"] = fb["polygon_indices"].apply(str)
    fb = fb.drop_duplicates(subset="_pstr").drop(columns=["_pstr"])

    def _yield_stats(sub, label):
        sent = sub[sub["any_images_sent"] == True]
        n_sent = len(sent)
        n_confirmed = int((sent["matched_to_CF_annot"] == True).sum())
        n_rejected = int((sent["matched_to_CF_annot"] == False).sum())
        return {
            "scope": label,
            "n_gt500AU_post_processing_passed": len(sub),
            "n_sent_for_annotation": n_sent,
            "n_confirmed_dairy": n_confirmed,
            "n_not_confirmed": n_rejected,
            "pct_not_confirmed_of_sent": 100 * n_rejected / n_sent if n_sent > 0 else np.nan,
        }

    rows = [
        _yield_stats(fb, "statewide"),
        _yield_stats(fb[fb["ewg_region"] == True], "ewg_region"),
        _yield_stats(fb[fb["ewg_region"] == False], "outside_ewg_region"),
    ]
    stats = pd.DataFrame(rows)
    stats.to_csv(subdirs["tables"] / "human_annotation_yield.csv", index=False)

    print(f"\n  Clusters >500 AU passing post-processing filters: {n0} -> {len(fb)}")
    print("\n  Human annotation yield, conditional on prior filtering:")
    print(stats.to_string(index=False))
    print(f"\n  Saved human_annotation_yield.csv to {subdirs['tables']}")
    return stats


# ============================================================================
# Section 6a: False-positive waterfalls (Reviewer #1, Limitation 2)
# ============================================================================
#
# Two separate waterfalls, not one combined pool. The EWG-region set and the
# CF-sent-but-unmatched set differ in a way that matters for what "removing a
# false positive" even means:
#
#   - In the EWG region, EWG's AFO census is treated as a complete ground
#     truth, so "unmatched to an EWG dairy" is a clean starting definition of
#     "candidate false positive" with no pre-filtering baked in. The full
#     minimum-size / urban / keyword / milk-license waterfall is meaningful
#     here because none of those filters have been applied yet.
#   - The CF-sent-but-unmatched set was already screened by size, urban-area,
#     and parcel-keyword criteria *before* being selected for CF review (the
#     same need_to_send logic in generate_image_review_stats). Re-running
#     those same filters on this set would mostly just show they remove
#     ~nothing — an artifact of double-filtering, not a real result. Only the
#     milk-license check adds information here.


def _save_fp_residual_csv(residual_gdf, out_path):
    """Write a false-positive residual to CSV with lat/lon instead of geometry."""
    out = residual_gdf.copy()
    centroids_4326 = out.geometry.centroid.to_crs(4326)
    out["lat"] = centroids_4326.y
    out["lon"] = centroids_4326.x
    out = out[[
        "polygon_indices", "source", "animal_unit_estimate", "cluster_area_m2",
        "parcel_owner1_names", "jpeg_names", "ewg_region", "lat", "lon",
    ]]
    out.to_csv(out_path, index=False)


def generate_ewg_fp_waterfall(data, paths, subdirs):
    """False-positive waterfall for the EWG region (EWG's AFO census treated
    as complete ground truth).

    Candidate definition: ALL four_band_clusters in the EWG region that do
    not sjoin_nearest (400m) to any EWG *dairy* facility (Animal_Typ=='Dairy',
    plus the 'Cattle: Small' legend category that the existing pr-curve code
    in facility_level_pr_lowerbound_graph() also treats as small dairy — a
    match to a non-dairy EWG facility, e.g. hogs, does not rescue a candidate
    here, since the model targets dairy CAFOs specifically). No size
    pre-filter — minimum-size is the first waterfall step below, so its
    contribution is measured rather than baked into the candidate
    definition.

    Waterfall, in the order given in the reviewer response:
      1. minimum-size filter (>500 AU, the paper's chosen operating threshold)
      2. urban-area proximity (>500m from an incorporated place)
      3. parcel-owner keyword exclusion
      4. milk-license rescue (within 500m) — a nearby milk license means the
         detection is probably a real dairy operation EWG's census missed,
         not a model false positive, so it's removed from the FP tally here
         rather than confirmed as a genuine detection.

    Because EWG's census is treated as complete for this region, the full
    final residual — not a sample — is the manual-review target (see
    generate_fp_manual_review_sample with n_sample=None).

    Writes fp_ewg_waterfall.csv and fp_ewg_residual.csv to tables/.
    Returns (waterfall_df, residual_gdf).
    """
    four_band = data["four_band_clusters"]
    ewg_afos = data["ewg_afos"]
    ewg_dairy = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy")
        | ((ewg_afos["Animal_Typ"] == "Cattle") & (ewg_afos["Legend"] == "Cattle: Small"))
    ]

    pool = four_band[four_band["ewg_region"] == True].copy()
    matches = ewg_dairy.sjoin_nearest(pool, max_distance=400)
    matched_idx = set(matches["index_right"].unique())
    pool = pool[~pool.index.isin(matched_idx)].copy()
    n0 = len(pool)

    print(f"\n  EWG-region FP candidate pool (unmatched to an EWG dairy facility): {n0}")

    steps = [{"step": "0_total_candidate_fps", "n_remaining": n0, "n_removed": 0}]
    remaining = pool.copy()

    # ---- Step 1: minimum-size filter (paper's chosen 500 AU operating threshold) ----
    before = len(remaining)
    remaining = remaining[remaining["animal_unit_estimate"] > 500].copy()
    steps.append({
        "step": "1_min_size_gt500AU",
        "n_remaining": len(remaining), "n_removed": before - len(remaining),
    })

    # ---- Step 2: urban-area proximity filter (>500m from incorporated place) ----
    before = len(remaining)
    try:
        urban_areas = gpd.read_file(
            paths["data_path"].parent / URBAN_AREA_PROXY_PATH
        ).to_crs(cfg.WI_EPSG)
        remaining = remaining.sjoin_nearest(
            urban_areas[["geometry"]], distance_col="dist_to_urban"
        )
        remaining = remaining[~remaining.index.duplicated(keep="first")]
        remaining = remaining[remaining["dist_to_urban"] > 500].copy()
        remaining = remaining.drop(columns=["index_right", "dist_to_urban"], errors="ignore")
    except Exception as e:
        print(f"  Warning: could not apply urban area filter: {e}")
    steps.append({
        "step": "2_urban_area_gt500m",
        "n_remaining": len(remaining), "n_removed": before - len(remaining),
    })

    # ---- Step 3: parcel-owner keyword exclusion ----
    exclude_keywords = ["STORAG", "CHEMICAL", "ELECTRONIC", "WOOD"]
    before = len(remaining)
    remaining = remaining[
        ~remaining["parcel_owner1_names"].apply(
            lambda x: any(kw in str(x) for kw in exclude_keywords)
        )
    ].copy()
    steps.append({
        "step": "3_parcel_keyword_exclude",
        "n_remaining": len(remaining), "n_removed": before - len(remaining),
    })

    # ---- Step 4: milk-license rescue (within 500m) ----
    before = len(remaining)
    milk_wi = data["milk_producers"].set_geometry("location").to_crs(cfg.WI_EPSG)
    matched_milk = remaining.sjoin_nearest(milk_wi[["location"]], max_distance=500, how="inner")
    milk_pstrs = set(matched_milk["polygon_indices"].apply(str))
    remaining = remaining[~remaining["polygon_indices"].apply(str).isin(milk_pstrs)].copy()
    steps.append({
        "step": "4_milk_license_rescue_500m",
        "n_remaining": len(remaining), "n_removed": before - len(remaining),
    })

    waterfall = pd.DataFrame(steps)
    waterfall["pct_of_total_removed_this_step"] = 100 * waterfall["n_removed"] / n0
    waterfall["cumulative_pct_resolved"] = 100 * (n0 - waterfall["n_remaining"]) / n0
    waterfall.to_csv(subdirs["tables"] / "fp_ewg_waterfall.csv", index=False)

    print("\n  EWG-region false-positive waterfall:")
    print(waterfall.to_string(index=False))
    print(f"\n  Residual requiring manual review (full census — EWG assumed complete): "
          f"{len(remaining)} ({100 * len(remaining) / n0:.1f}% of {n0})")

    remaining = remaining.copy()
    remaining["source"] = "ewg_unmatched"
    _save_fp_residual_csv(remaining, subdirs["tables"] / "fp_ewg_residual.csv")

    print(f"  Saved fp_ewg_waterfall.csv, fp_ewg_residual.csv to {subdirs['tables']}")
    return waterfall, remaining


def generate_cf_unmatched_fp_analysis(data, paths, subdirs):
    """False-positive analysis for the CF-sent-but-unmatched set: clusters
    with >=1 image sent to CloudFactory for labeling where CF did not draw
    an annotation (any_images_sent & ~matched_to_CF_annot).

    Unlike the EWG-region set, this candidate pool was already screened by
    size/urban-area/parcel-keyword criteria before being selected for CF
    review (the same need_to_send logic in generate_image_review_stats), so
    re-applying those filters here would mostly show they remove ~nothing —
    a double-filtering artifact, not a real result. The only filter that
    adds information at this point is the milk-license rescue: a nearby
    milk license means the detection is probably a real, if EWG-invisible,
    dairy operation rather than a genuine false positive.

    This set is also noisier than the EWG-region one: CF can skip
    annotating an image for reasons unrelated to "not a farm" (partial
    facility at a tile edge, already covered by another image, etc.), and
    there's no complete-census assumption backing it here. So instead of a
    full census, a random sample of the residual is manually reviewed (see
    generate_fp_manual_review_sample).

    Writes fp_cf_unmatched_waterfall.csv and fp_cf_unmatched_residual.csv to
    tables/. Returns (waterfall_df, residual_gdf).
    """
    four_band = _add_cf_review_flags(data["four_band_clusters"], data["all_cf_clusters"], paths)

    pool = four_band[
        (four_band["any_images_sent"] == True) & (four_band["matched_to_CF_annot"] == False)
    ].copy()
    n0 = len(pool)
    print(f"\n  CF-sent-unmatched FP candidate pool: {n0}")

    steps = [{"step": "0_total_candidate_fps", "n_remaining": n0, "n_removed": 0}]
    remaining = pool.copy()

    # ---- Only filter applied: milk-license rescue (within 500m) ----
    before = len(remaining)
    milk_wi = data["milk_producers"].set_geometry("location").to_crs(cfg.WI_EPSG)
    matched_milk = remaining.sjoin_nearest(milk_wi[["location"]], max_distance=500, how="inner")
    milk_pstrs = set(matched_milk["polygon_indices"].apply(str))
    remaining = remaining[~remaining["polygon_indices"].apply(str).isin(milk_pstrs)].copy()
    steps.append({
        "step": "1_milk_license_rescue_500m",
        "n_remaining": len(remaining), "n_removed": before - len(remaining),
    })

    waterfall = pd.DataFrame(steps)
    waterfall["pct_of_total_removed_this_step"] = 100 * waterfall["n_removed"] / n0
    waterfall["cumulative_pct_resolved"] = 100 * (n0 - waterfall["n_remaining"]) / n0
    waterfall.to_csv(subdirs["tables"] / "fp_cf_unmatched_waterfall.csv", index=False)

    print("\n  CF-sent-unmatched false-positive waterfall:")
    print(waterfall.to_string(index=False))
    print(f"\n  Residual for random-sample manual review: {len(remaining)} "
          f"({100 * len(remaining) / n0:.1f}% of {n0})")

    remaining = remaining.copy()
    remaining["source"] = "cf_sent_unmatched"
    _save_fp_residual_csv(remaining, subdirs["tables"] / "fp_cf_unmatched_residual.csv")

    print(f"  Saved fp_cf_unmatched_waterfall.csv, fp_cf_unmatched_residual.csv to {subdirs['tables']}")
    return waterfall, remaining


# ============================================================================
# Section 6b: Manual-review sample for unresolved FP candidates
# ============================================================================

def generate_fp_manual_review_sample(data, paths, subdirs, residual, label, n_sample=None, seed=0):
    """Build the manual-review packet for a false-positive residual (from
    generate_ewg_fp_waterfall() or generate_cf_unmatched_fp_analysis()): one
    plot_cluster_parcel image per candidate (raw NAIP imagery + cluster
    outline) plus a CSV using the same columns as the existing manual-review
    log (Date of check, Identifier, lat-long, Parcel_owner1_names, Dataset,
    Issue/reason for checking, Notes/conclusion) so entries can be pasted
    straight in.

    Never overwrites an existing review CSV — if fp_manual_review_{label}.csv
    already exists, this is a no-op (delete it to force regeneration), so a
    routine pipeline re-run can't clobber completed manual review.

    label: short tag used in output file/folder names (e.g. "ewg" for the
        full-census EWG-region review, "cf_unmatched" for the CF-sent
        random sample).
    n_sample: if given, draw a random subsample of this size (fixed seed)
        instead of reviewing the full residual — use None to review the
        full residual (appropriate when the source is treated as a
        complete census, e.g. the EWG-region waterfall).
    """
    out_dir = subdirs["error"] / f"fp_manual_review_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = subdirs["tables"] / f"fp_manual_review_{label}.csv"

    if out_csv.exists():
        print(f"  {out_csv} already exists — skipping (delete it to force regeneration).")
        return pd.read_csv(out_csv)

    sample = residual.copy()
    if n_sample is not None and n_sample < len(sample):
        sample = sample.sample(n=n_sample, random_state=seed)

    centroids_4326 = sample.geometry.centroid.to_crs(4326)
    sample["lat"] = centroids_4326.y
    sample["lon"] = centroids_4326.x
    sample["lat-long"] = (
        sample["lat"].round(6).astype(str) + ", " + sample["lon"].round(6).astype(str)
    )

    rows = []
    for idx, row in sample.iterrows():
        ident = str(row["polygon_indices"])
        safe_ident = ident.replace("[", "").replace("]", "").replace(", ", "-").replace(",", "-")
        # Large merged clusters can have hundreds of sub-polygon indices,
        # producing a filename longer than the OS limit (255 bytes) -- fall
        # back to a short hash of the full identifier in that case.
        if len(safe_ident) > 150:
            safe_ident = hashlib.md5(ident.encode()).hexdigest()[:16]
        fig_path = out_dir / f"{safe_ident}.png"
        try:
            pu.plot_cluster_parcel(
                sample.loc[[idx]],
                image_bound_map=data["image_bound_map"],
                zoom_radius=400,
                include_AU_estimate=True,
                fig_path=fig_path,
                fig_format="png",
                dpi=120,
            )
            plt.close("all")
        except Exception as e:
            print(f"  Warning: could not plot {ident}: {e}")
            fig_path = None

        rows.append({
            "Date of check": "",
            "Identifier": ident,
            "lat-long": row["lat-long"],
            "Parcel_owner1_names": row["parcel_owner1_names"],
            "Dataset": f"four_band_clusters ({row['source']})",
            "animal_unit_estimate": row["animal_unit_estimate"],
            "plot_image": str(fig_path) if fig_path else "",
            "Issue/reason for checking": "post-processing FP residual — unresolved after automated filters",
            "Notes/conclusion": "",
        })

    sample_out = pd.DataFrame(rows)
    sample_out.to_csv(out_csv, index=False)
    print(f"  Saved {len(sample_out)}-row manual review sample to {out_csv}")
    print(f"  Saved {len(sample_out)} cluster/parcel plots to {out_dir}")
    return sample_out


# ============================================================================
# Section 6c: Manual-review summary statistics (Reviewer #1, Limitation 2)
# ============================================================================
#
# Consumes the human-coded review files, NOT the blank templates generate_
# fp_manual_review_sample() writes. Workflow: (1) generate the blank samples,
# (2) fill in "Notes/conclusion" for each row by hand (imagery in the
# fp_manual_review_{label}/ folders + lat-long for satellite lookup), (3) add
# a "category" column with one of "farm" / "false_positive" / "ambiguous" per
# row reflecting the conclusion in Notes/conclusion, (4) save as
# fp_manual_review_{label}_categorized.csv in tables/, (5) run this function.
# The categorized files checked into tables/ right now reflect a 20-of-42
# (EWG) and 20-of-40 (CF) manually reviewed subset; re-running with more rows
# categorized tightens the confidence intervals below.

def generate_fp_manual_review_summary(subdirs):
    """Summarize human-coded false-positive/farm/ambiguous determinations for
    the EWG and CF-unmatched manual-review samples, with Wilson 95% CIs on
    the false-positive rate, and extrapolate the CF-unmatched rate to its
    full (un-sampled) residual.

    Writes fp_manual_review_summary.csv to tables/ and prints a copy-paste
    summary block. Returns the summary DataFrame, or None if the categorized
    review files aren't present yet (prints instructions in that case).
    """
    from statsmodels.stats.proportion import proportion_confint

    ewg_path = subdirs["tables"] / "fp_manual_review_ewg_categorized.csv"
    cf_path = subdirs["tables"] / "fp_manual_review_cf_unmatched_categorized.csv"
    if not ewg_path.exists() or not cf_path.exists():
        print(
            "  Categorized manual-review files not found "
            f"({ewg_path.name}, {cf_path.name}). Skipping summary — see "
            "generate_fp_manual_review_summary()'s docstring for the "
            "expected workflow/format."
        )
        return None

    ewg = pd.read_csv(ewg_path)
    cf = pd.read_csv(cf_path)

    # CF-unmatched residual size (post milk-license rescue) that the reviewed
    # 20-row (or however many are categorized) sample was drawn from, for
    # extrapolation. Recomputed from the waterfall CSV rather than hardcoded.
    cf_waterfall_path = subdirs["tables"] / "fp_cf_unmatched_waterfall.csv"
    cf_residual_n = None
    if cf_waterfall_path.exists():
        cf_waterfall = pd.read_csv(cf_waterfall_path)
        cf_residual_n = int(cf_waterfall["n_remaining"].iloc[-1])

    rows = []
    for label, df in [("ewg", ewg), ("cf_unmatched", cf)]:
        coded = df[df["category"].isin(["farm", "false_positive", "ambiguous"])]
        n = len(coded)
        n_fp = int((coded["category"] == "false_positive").sum())
        n_farm = int((coded["category"] == "farm").sum())
        n_amb = int((coded["category"] == "ambiguous").sum())
        lo, hi = proportion_confint(n_fp, n, method="wilson") if n > 0 else (np.nan, np.nan)
        row = {
            "sample": label,
            "n_reviewed": n,
            "n_false_positive": n_fp,
            "n_farm": n_farm,
            "n_ambiguous": n_amb,
            "fp_rate": n_fp / n if n > 0 else np.nan,
            "fp_rate_wilson_ci_low": lo,
            "fp_rate_wilson_ci_high": hi,
        }
        if label == "cf_unmatched" and cf_residual_n is not None:
            row["extrapolated_residual_n"] = cf_residual_n
            row["extrapolated_fp_count"] = row["fp_rate"] * cf_residual_n
            row["extrapolated_fp_count_ci_low"] = lo * cf_residual_n
            row["extrapolated_fp_count_ci_high"] = hi * cf_residual_n
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(subdirs["tables"] / "fp_manual_review_summary.csv", index=False)

    print("\n  Manual review summary (human-coded false positive / farm / ambiguous):")
    print(summary.to_string(index=False))
    for row in rows:
        print(
            f"\n  {row['sample']}: {row['n_false_positive']}/{row['n_reviewed']} "
            f"confirmed false positives "
            f"({100 * row['fp_rate']:.0f}%, Wilson 95% CI "
            f"[{100 * row['fp_rate_wilson_ci_low']:.0f}%, "
            f"{100 * row['fp_rate_wilson_ci_high']:.0f}%])"
        )
        if "extrapolated_fp_count" in row:
            print(
                f"    extrapolated to the full {row['extrapolated_residual_n']}-item "
                f"residual: ~{row['extrapolated_fp_count']:.0f} false positives "
                f"[{row['extrapolated_fp_count_ci_low']:.0f}, "
                f"{row['extrapolated_fp_count_ci_high']:.0f}]"
            )

    print(f"\n  Saved fp_manual_review_summary.csv to {subdirs['tables']}")
    return summary


# ============================================================================
# Section 7: Model performance and universe tables
# ============================================================================

def generate_tables(data, subdirs, permit_matched):
    """Model performance table + unpermitted universe table. → tables/.

    permit_matched: the dairy-only permit-matched clusters from generate_model_validation()
    """
    # Model performance
    cf.facility_level_stats_table(
        data["four_band_clusters"],
        data["three_band_clusters"],
        data["poultry_detections"],
        data["inception_detections"],
        data["WDNR_CAFOs"],
        data["counties"],
        data["ewg_afos"],
        AFO_lower_bound=500,
        AFO_lower_bound_logic="point_estimate",
        save_path=subdirs["tables"] / "model_performance_table.csv",
    )

    # Unpermitted universe table: counts of unpermitted vs permitted > 1000 AU
    # across all AU estimate logic columns, with milk producer match breakdown.

    # Match to ANY permit (including non-dairy, no satellite aggregation)
    all_permit_matched, _ = caf.merge_clusters_permits(
        data["all_cf_clusters"], data["WDNR_CAFOs"],
        sum_satellite_counts=False, discrep_analysis=False, only_dairy=False,
    )
    unpermitted_cf = data["all_cf_clusters"][
        ~data["all_cf_clusters"]["polygon_indices"].isin(
            all_permit_matched["polygon_indices"]
        )
    ].copy()

    # Find unpermitted clusters that match a milk producer license within 500m
    milk_producers_wi = data["milk_producers"].to_crs(cfg.WI_EPSG)
    unperm_milk_match = unpermitted_cf.sjoin_nearest(
        milk_producers_wi, max_distance=500,
    )
    # Restore geometry after sjoin (sjoin_nearest creates geometry_left/right)
    if "geometry_left" in unperm_milk_match.columns:
        unperm_milk_match["geometry"] = unperm_milk_match["geometry_left"]
        unperm_milk_match.drop(
            ["geometry_left", "geometry_right"], axis=1, inplace=True, errors="ignore",
        )
        unperm_milk_match = gpd.GeoDataFrame(
            unperm_milk_match, geometry="geometry", crs=cfg.WI_EPSG,
        )
    # Drop duplicate facilities (sjoin_nearest can return multiple rows per left
    # geometry when two milk producers are exactly equidistant). Keep first match.
    unperm_milk_match = unperm_milk_match[
        ~unperm_milk_match.index.duplicated(keep="first")
    ]

    # AU logic columns to compare across
    logic_cols = [
        c for c in [
            "animal_unit_estimate",
            "animal_units_lower", "animal_units_upper",
            "animal_units_0.1_perc", "animal_units_0.25_perc",
            "animal_units_0.5_perc", "animal_units_0.75_perc",
            "animal_units_0.9_perc",
        ]
        if c in data["all_cf_clusters"].columns
    ]

    # Find permitted clusters that also match a milk producer license within 500m
    perm_milk_match = permit_matched.sjoin_nearest(
        milk_producers_wi, max_distance=500,
    )
    if "geometry_left" in perm_milk_match.columns:
        perm_milk_match["geometry"] = perm_milk_match["geometry_left"]
        perm_milk_match.drop(
            ["geometry_left", "geometry_right"], axis=1, inplace=True, errors="ignore",
        )
        perm_milk_match = gpd.GeoDataFrame(
            perm_milk_match, geometry="geometry", crs=cfg.WI_EPSG,
        )
    perm_milk_match = perm_milk_match[~perm_milk_match.index.duplicated(keep="first")]

    unperm = unpermitted_cf[logic_cols].apply(
        lambda x: (x >= 1000).sum(), axis=0,
    )
    perm = permit_matched[logic_cols].apply(
        lambda x: (x >= 1000).sum(), axis=0,
    )
    unperm_milk = unperm_milk_match[logic_cols].apply(
        lambda x: (x >= 1000).sum(), axis=0,
    )
    perm_milk = perm_milk_match[logic_cols].apply(
        lambda x: (x >= 1000).sum(), axis=0,
    )

    # Total permitted dairy CAFOs from the matching process, regardless of AU threshold
    total_permitted_dairy = len(permit_matched)

    perm_unperm_counts = pd.DataFrame({
        "Unpermitted Potential CAFOs": unperm,
        "Unpermitted Potential CAFOs with milk producer license": unperm_milk,
        "Permitted CAFOs meeting this logic": perm,
        "% of permitted dairy CAFOs meeting this logic": 100 * perm / total_permitted_dairy,
        "Permitted CAFOs meeting this logic (with milk license)": perm_milk,
        "% of permitted dairy CAFOs (with milk license)": 100 * perm_milk / total_permitted_dairy,
        "Permit evasion rate (all unpermitted)": unperm / (unperm + total_permitted_dairy),
        "Permit evasion rate (milk licensed only)": unperm_milk / (unperm_milk + total_permitted_dairy),
    })
    perm_unperm_counts = perm_unperm_counts.sort_values(
        by="Unpermitted Potential CAFOs", ascending=False,
    )
    perm_unperm_counts.to_csv(subdirs["tables"] / "unperm_universe_table.csv")

    print(perm_unperm_counts.to_string())
    print(f"  Saved tables to {subdirs['tables']}")


# ============================================================================
# Section 8: Unpermitted analysis figures
# ============================================================================

def generate_unpermitted_analysis(data, subdirs, permit_matched, all_clusters=None):
    """Size distributions, permit rate, WI map. 3 figures → 04_unpermitted_analysis/.

    all_clusters: optional GeoDataFrame returned by generate_risk_assessment() (from
        summary_table()). If provided, it is used directly for plot_permit_rate_by_size
        so the authoritative 4-category 'set' column is used rather than a manually
        constructed one. Should be passed whenever risk assessment has already run.
    """
    out = subdirs["unpermitted"]

    # Identify unpermitted clusters
    all_permit_matched, _ = caf.merge_clusters_permits(
        data["all_cf_clusters"], data["WDNR_CAFOs"],
        sum_satellite_counts=False, discrep_analysis=False, only_dairy=False,
    )
    unpermitted_cf = data["all_cf_clusters"][
        ~data["all_cf_clusters"]["polygon_indices"].isin(
            all_permit_matched["polygon_indices"]
        )
    ].copy()
    unperm_potential = unpermitted_cf[
        unpermitted_cf["animal_unit_estimate"] >= 1000
    ]

    # Size distribution comparison
    cf.plot_unpermitted_vs_permitted_distribution(
        unperm_potential, permit_matched, unpermitted_cf,
        chosen_logic_col="animal_unit_estimate",
        save_path=out / "unpermitted_vs_permitted_size_distribution.svg",
    )

    # AU category permit-rate breakdown (requires all_clusters from summary_table)
    if all_clusters is not None and "permitted" in all_clusters.columns:
        broader_stats = all_clusters[["animal_unit_estimate", "permitted"]].copy()
        broader_stats["animal_unit_category"] = pd.cut(
            broader_stats["animal_unit_estimate"], bins=[500, 1000, 2000, np.inf]
        )
        cat_stats = broader_stats.groupby("animal_unit_category")["permitted"].agg(
            ["mean", "count"]
        )
        print("\n  AU category permit rates (clusters ≥500 AU):")
        print(cat_stats.to_string())

        frac_perm_lt2000 = (
            len(permit_matched[permit_matched["animal_unit_estimate"] < 2000])
            / len(permit_matched)
        )
        frac_unperm_lt2000 = (
            len(unperm_potential[unperm_potential["animal_unit_estimate"] < 2000])
            / len(unperm_potential)
        )
        print(f"\n  Fraction of permitted CAFOs < 2000 AU:        {frac_perm_lt2000:.3f}")
        print(f"  Fraction of unpermitted potential < 2000 AU:  {frac_unperm_lt2000:.3f}")

    # Permit rate by size — use all_clusters from summary_table() when available so
    # the authoritative 4-category 'set' column is used (same as notebook).
    if all_clusters is not None:
        cf.plot_permit_rate_by_size(
            all_clusters,
            save_path=out / "permit_rate_by_size.svg",
        )
    else:
        # Fallback when risk assessment hasn't run yet: construct set column manually.
        # NOTE: this is less accurate than the summary_table() version.
        all_cf_with_set = data["all_cf_clusters"].copy()
        all_cf_with_set["set"] = "Unknown"
        permitted_indices = all_permit_matched["polygon_indices"]
        all_cf_with_set.loc[
            all_cf_with_set["polygon_indices"].isin(permitted_indices), "set"
        ] = "Permitted dairy CAFOs"
        cf.plot_permit_rate_by_size(
            all_cf_with_set,
            save_path=out / "permit_rate_by_size.svg",
        )

    # WI CAFO map
    cf.WI_CAFO_map(
        data["all_cf_clusters"], data["WDNR_CAFOs"], data["counties"],
        CAFO_thresh_logic="point",
        save_path=out / "WI_CAFO_map.svg",
    )
    plt.close("all")

    print(f"  Saved 3 figures to {out}")
    return unpermitted_cf, unperm_potential


# ============================================================================
# Section 8a: Formal permitted vs. unpermitted statistical tests (Reviewer)
# ============================================================================

def generate_perm_unperm_statistical_tests(all_clusters, subdirs):
    """Formal hypothesis tests, effect sizes, and CIs for the permitted vs.
    unpermitted-potential comparisons that are stated descriptively in the text
    (Section discussing Figure fig:perm_unperm_size / Table
    tab:permitted_unpermitted_physical).

    Addresses reviewer comment: "The discussion relies on descriptive
    comparisons. Perform formal tests to determine whether differences between
    permitted and unpermitted CAFOs are statistically significant; report
    confidence intervals, hypothesis tests, or effect sizes."

    Produces, all operating on the same `all_clusters['set']` partition that
    feeds the physical-characteristics table so the numbers are consistent:

      1. Continuous comparisons (footprint area, AU, #parcels, #buildings)
         between "Permitted dairy CAFOs" and "Unpermitted potential CAFOs":
         Welch t-test, Mann-Whitney U, Cohen's d, Cliff's delta, a 95% CI on
         the difference in means, and a bootstrap 95% CI on the ratio of means
         (backs the "N% smaller footprint" statement).
      2. Permit rate by AU size bin with Wilson 95% CIs, a Cochran-Armitage
         trend test and a Spearman rank correlation (backs "permit status is
         strongly correlated with size"), plus a Fisher exact test contrasting
         the lowest and highest size bins (backs "12% ... to 91%").
      3. Descriptive top-300-by-size reframing (backs "64 unpermitted farms
         would enter the top-size group").

    Writes two CSVs to tables/ and prints a copy-paste-ready summary block.
    """
    import scipy.stats as ss
    from statsmodels.stats.proportion import proportion_confint

    out_tables = subdirs["tables"]
    PERM, UNP = "Permitted dairy CAFOs", "Unpermitted potential CAFOs"

    df = all_clusters.copy()
    perm = df[df["set"] == PERM]
    unp = df[df["set"] == UNP]

    def _cohens_d(a, b):
        na, nb = len(a), len(b)
        sp = np.sqrt(
            ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
            / (na + nb - 2)
        )
        return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

    def _cliffs_delta(U, n1, n2):
        # Cliff's delta = 2*U/(n1*n2) - 1 (equivalently the rank-biserial
        # correlation), computed from the Mann-Whitney U statistic.
        return 2 * U / (n1 * n2) - 1

    def _compare(name, a, b, seed=0, n_boot=10000):
        """a = permitted values, b = unpermitted-potential values."""
        a = a.dropna().to_numpy()
        b = b.dropna().to_numpy()
        na, nb = len(a), len(b)
        ma, mb = a.mean(), b.mean()

        t = ss.ttest_ind(a, b, equal_var=False)
        U = ss.mannwhitneyu(a, b, alternative="two-sided")

        # Welch CI on the difference in means (permitted - unpermitted)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / na + vb / nb)
        dfw = se ** 4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        tcrit = ss.t.ppf(0.975, dfw)
        diff = ma - mb
        diff_ci = (diff - tcrit * se, diff + tcrit * se)

        # Bootstrap CI on ratio of means (unpermitted / permitted)
        rng = np.random.default_rng(seed)
        ratios = np.array([
            rng.choice(b, nb).mean() / rng.choice(a, na).mean()
            for _ in range(n_boot)
        ])
        r_lo, r_hi = np.percentile(ratios, [2.5, 97.5])

        return {
            "variable": name,
            "n_permitted": na,
            "n_unpermitted": nb,
            "mean_permitted": ma,
            "mean_unpermitted": mb,
            "median_permitted": np.median(a),
            "median_unpermitted": np.median(b),
            "mean_diff": diff,
            "mean_diff_ci_low": diff_ci[0],
            "mean_diff_ci_high": diff_ci[1],
            "ratio_unperm_over_perm": mb / ma,
            "ratio_ci_low": r_lo,
            "ratio_ci_high": r_hi,
            "pct_smaller": (1 - mb / ma) * 100,
            "welch_t": t.statistic,
            "welch_p": t.pvalue,
            "mwu_U": U.statistic,
            "mwu_p": U.pvalue,
            "cohens_d": _cohens_d(a, b),
            "cliffs_delta": _cliffs_delta(U.statistic, na, nb),
        }

    # ── 1. Continuous comparisons ────────────────────────────────────────────
    comparisons = [
        ("cluster_area_m2", "Building footprint (m^2)"),
        ("animal_unit_estimate", "Animal unit estimate (AU)"),
        ("n_parcels", "Number of land parcels"),
        ("n_buildings", "Number of buildings"),
    ]
    rows = [
        _compare(label, perm[col], unp[col])
        for col, label in comparisons
        if col in df.columns
    ]
    cont = pd.DataFrame(rows)
    cont.to_csv(out_tables / "perm_unperm_statistical_tests.csv", index=False)

    # ── 2. Permit rate by size bin ───────────────────────────────────────────
    allf = df.copy()
    allf["is_perm"] = (allf["set"] == PERM).astype(int)
    bins = [1000, 1500, 2000, 3000, np.inf]
    labels = ["1000-1500", "1500-2000", "2000-3000", "3000+"]
    pr_rows = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sub = allf[
            (allf["animal_unit_estimate"] >= lo)
            & (allf["animal_unit_estimate"] < hi)
        ]
        k, n = int(sub["is_perm"].sum()), len(sub)
        lo_ci, hi_ci = proportion_confint(k, n, method="wilson")
        pr_rows.append({
            "au_bin": lab, "n_farms": n, "n_permitted": k,
            "permit_rate_pct": 100 * k / n,
            "wilson_ci_low_pct": 100 * lo_ci,
            "wilson_ci_high_pct": 100 * hi_ci,
        })
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(out_tables / "permit_rate_by_size_ci.csv", index=False)

    # Cochran-Armitage trend test across ordered size bins (scores 0..k-1)
    scores = np.arange(len(pr_rows))
    n_i = pr["n_farms"].to_numpy()
    x_i = pr["n_permitted"].to_numpy()
    N, R = n_i.sum(), x_i.sum()
    pbar = R / N
    T = np.sum(scores * (x_i - n_i * pbar))
    var = pbar * (1 - pbar) * (
        np.sum(n_i * scores ** 2) - (np.sum(n_i * scores)) ** 2 / N
    )
    z_ca = T / np.sqrt(var)
    p_ca = 2 * (1 - ss.norm.cdf(abs(z_ca)))

    # Fisher exact: lowest vs highest size bin
    lowbin = allf[
        (allf["animal_unit_estimate"] >= 1000)
        & (allf["animal_unit_estimate"] < 1500)
    ]
    highbin = allf[allf["animal_unit_estimate"] >= 3000]
    table = [
        [int(lowbin["is_perm"].sum()), len(lowbin) - int(lowbin["is_perm"].sum())],
        [int(highbin["is_perm"].sum()), len(highbin) - int(highbin["is_perm"].sum())],
    ]
    or_fisher, p_fisher = ss.fisher_exact(table)

    # Spearman rank correlation of size vs permit status (farms >= 1000 AU)
    big = allf[allf["animal_unit_estimate"] >= 1000]
    rho, p_rho = ss.spearmanr(big["animal_unit_estimate"], big["is_perm"])

    # ── 3. Top-300-by-size reframing ─────────────────────────────────────────
    n_perm_total = len(perm)
    top = df.sort_values("animal_unit_estimate", ascending=False).head(n_perm_total)
    top_composition = top["set"].value_counts()
    n_unperm_in_top = int(top_composition.get(UNP, 0))

    # % of permitted in the 1000-2000 AU band
    perm_1000_2000 = perm[
        (perm["animal_unit_estimate"] >= 1000)
        & (perm["animal_unit_estimate"] < 2000)
    ]
    pct_perm_1000_2000 = 100 * len(perm_1000_2000) / n_perm_total

    # ── Print copy-paste summary ─────────────────────────────────────────────
    def _p(p):
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("FORMAL TESTS: permitted vs. unpermitted-potential CAFOs")
    lines.append("(copy-paste-ready summary; full numbers in "
                 "tables/perm_unperm_statistical_tests.csv)")
    lines.append("=" * 78)

    for r in rows:
        lines.append(f"\n{r['variable']}:")
        lines.append(
            f"  permitted   mean={r['mean_permitted']:,.1f} "
            f"(median {r['median_permitted']:,.1f}), n={r['n_permitted']}"
        )
        lines.append(
            f"  unpermitted mean={r['mean_unpermitted']:,.1f} "
            f"(median {r['median_unpermitted']:,.1f}), n={r['n_unpermitted']}"
        )
        lines.append(
            f"  diff in means = {r['mean_diff']:,.1f} "
            f"[95% CI {r['mean_diff_ci_low']:,.1f}, {r['mean_diff_ci_high']:,.1f}]"
        )
        lines.append(
            f"  unpermitted is {r['pct_smaller']:.0f}% smaller "
            f"(ratio {r['ratio_unperm_over_perm']:.3f} "
            f"[95% CI {r['ratio_ci_low']:.3f}, {r['ratio_ci_high']:.3f}])"
        )
        lines.append(
            f"  Welch t={r['welch_t']:.2f} ({_p(r['welch_p'])}); "
            f"Mann-Whitney {_p(r['mwu_p'])}; "
            f"Cohen's d={r['cohens_d']:.2f}, Cliff's delta={r['cliffs_delta']:.2f}"
        )

    lines.append("\nPermit rate by size (Wilson 95% CIs):")
    for r in pr_rows:
        lines.append(
            f"  {r['au_bin']} AU: {r['permit_rate_pct']:.1f}% "
            f"[{r['wilson_ci_low_pct']:.1f}, {r['wilson_ci_high_pct']:.1f}] "
            f"(n={r['n_farms']})"
        )
    lines.append(
        f"  Cochran-Armitage trend test: z={z_ca:.1f}, {_p(p_ca)}"
    )
    lines.append(
        f"  Fisher exact (1000-1500 vs 3000+): OR={or_fisher:.3f}, {_p(p_fisher)}"
    )
    lines.append(
        f"  Spearman rho(size, permitted | AU>=1000) = {rho:.2f}, "
        f"{_p(p_rho)} (n={len(big)})"
    )

    lines.append(
        f"\nTop {n_perm_total} farms by estimated size: "
        f"{int(top_composition.get(PERM, 0))} permitted, "
        f"{n_unperm_in_top} unpermitted potential."
    )
    lines.append(
        f"Permitted CAFOs in 1000-2000 AU band: {len(perm_1000_2000)} "
        f"({pct_perm_1000_2000:.0f}% of {n_perm_total} permitted)."
    )
    lines.append("=" * 78 + "\n")

    summary_text = "\n".join(lines)
    print(summary_text)
    (out_tables / "perm_unperm_tests_summary.txt").write_text(summary_text)

    print(f"  Saved perm_unperm_statistical_tests.csv, permit_rate_by_size_ci.csv, "
          f"perm_unperm_tests_summary.txt to {out_tables}")
    return cont, pr


# ============================================================================
# Section 8b: Spatial clustering statistics (permitted vs. unpermitted potential)
# ============================================================================

def generate_spatial_clustering_analysis(data, all_clusters, subdirs, k=8, n_permutations=999, seed=0):
    """Formal spatial-clustering tests for permitted vs. unpermitted potential CAFOs.

    Replaces the visual "no strong clustering" claim with join-count/Moran's I,
    Getis-Ord Gi*, a K-function difference test, and a nearest-neighbor cross-statistic,
    all with permutation/Monte Carlo inference. → 07_spatial_clustering/.
    """
    out = subdirs["spatial"]
    results = sps.run_spatial_clustering_analysis(
        all_clusters, out_dir=out, k=k, n_permutations=n_permutations, seed=seed,
        county_data=data["counties"],
    )
    plt.close("all")
    return results


# ============================================================================
# Section 8c: Descriptive permit-status regression (Reviewer #2)
# ============================================================================

def generate_permit_status_regression(data, all_clusters, paths, subdirs):
    """Descriptive logistic regression of permit status on size + confounders.

    Addresses Reviewer #2: isolates the factors associated with holding a WPDES
    permit while controlling for size, region (WDNR administrative region and
    county dairy density), operational scale, and location. Optionally adds
    Census/ACS county socioeconomic controls when a CENSUS_API_KEY is set (or a
    cached ACS file exists). → tables/ + 04_unpermitted_analysis/.
    """
    census_cache = paths["data_path"] / "census_acs_county.csv"
    res = psr.run_permit_status_regression(
        all_clusters,
        data["counties"],
        save_table_path=subdirs["tables"] / "permit_status_regression.csv",
        save_fig_path=subdirs["unpermitted"] / "permit_status_or_forest.svg",
        census_cache_path=census_cache,
    )
    plt.close("all")
    return res


# ============================================================================
# Section 9: Risk assessment (summary table, risk indices, figures)
# ============================================================================

WEIGHTS_DICT = {
    "water_distance": 0.134,
    "impaired_water_distance": 0.115,
    "swqma_300ft_dummy": 0.058,
    "swqma_1000ft_dummy": 0.038,
    "gw_0": 0.115,
    "gw_20": 0.096,
    "bedrock_lt5ft_dummy": 0.115,
    "silurian_0_2_dummy": 0.058,
    "silurian_2_5_dummy": 0.038,
    "mean_slope": 0.115,
    "slope_greater_12_dummy": 0.077,
    "cafo_w_restrict_distance": 0.019,
    "cafo_r_restrict_distance": 0.019,
}

INVERT_VARIABLES = [
    "water_distance",
    "impaired_water_distance",
    "swqma_300ft_distance",
    "swqma_1000ft_distance",
    "bedrock_lt5ft_distance",
    "silurian_0_2_distance",
    "silurian_2_5_distance",
    "cafo_w_restrict_distance",
    "cafo_r_restrict_distance",
    "slope_greater_12_distance",
]


INTERNAL_PII_CLUSTERS_PATH_NAME = "all_clusters_internal_PII.parquet"


def generate_risk_assessment(data, paths, subdirs, skip_imagery=False,
                             precomputed_path=None):
    """Summary table, risk indices, risk figures. → 05_risk_assessment/ + tables/.

    precomputed_path: if provided (and the file exists), load all_clusters directly
        from that GeoParquet (e.g. the publication dataset) instead of loading
        snapmaps and running the expensive summary_table() + distance calculations.
        All downstream risk-index figures are still produced; the summary table CSV
        is skipped in this mode.

    When the full (non-precomputed) path runs, all_clusters — which still carries
    parcel_owner1_names/2_names from the underlying cluster data — is also cached to
    tables/all_clusters_internal_PII.parquet. This is a SEPARATE file from the public
    publication dataset (which never includes parcel owner names) and exists only so
    generate_online_validation_sample() can look up owner names on a later
    --use-precomputed (fast) run without redoing this expensive step. Never publish or
    share this file — it contains PII (parcel owner names).
    """
    out_risk = subdirs["risk"]
    out_tables = subdirs["tables"]

    if precomputed_path is not None and Path(precomputed_path).exists():
        # ── fast path: skip snapmaps + summary_table ─────────────────────────
        print(f"  Using pre-computed cluster data: {precomputed_path}")
        print("  (snapmaps loading and summary_table() skipped)")
        all_clusters = gpd.read_parquet(precomputed_path)
        if "set" not in all_clusters.columns and "type" in all_clusters.columns:
            # Older publication-dataset exports used 'type' before it was renamed to 'set'.
            all_clusters = all_clusters.rename(columns={"type": "set"})
    else:
        # ── full path: load snapmaps + run summary_table ──────────────────────
        snapmaps = data.get("snapmaps")
        if snapmaps is None:
            print("  Loading snapmaps...")
            snapmaps = process_snapmaps.load_snapmaps(
                feather=True,
                feather_dir=paths["data_path"] / "snapmaps_feather",
                raw_gdb=paths["data_path"] / "NM_590_CAFO_STATEWIDE.gdb",
                crs=cfg.WI_EPSG,
                simplify_geometries=True, tolerance=10,
            )

        # Diagnostic snapshot — helps detect if earlier pipeline steps mutated data
        _cf = data["all_cf_clusters"]
        print(f"\n  [diag] all_cf_clusters: {len(_cf)} rows, CRS={_cf.crs}")
        print(f"  [diag] columns: {sorted(_cf.columns.tolist())}")
        _geom0 = _cf.geometry.iloc[0]
        print(f"  [diag] first geom centroid: x={_geom0.centroid.x:.2f}, y={_geom0.centroid.y:.2f}")
        print(f"  [diag] animal_unit_estimate range: "
              f"{_cf['animal_unit_estimate'].min():.0f} – {_cf['animal_unit_estimate'].max():.0f}")

        results_point, all_clusters = cf.summary_table(
            data["all_cf_clusters"],
            data["WDNR_CAFOs"],
            data["four_band_clusters"],
            data["counties"],
            data["milk_producers"],
            paths["data_path"] / "water_data" / "DEM_30m" / "WI_slope.tif",
            data["all_waters"],
            data["impaired_waters"],
            data["water_table_depth"],
            snapmaps_layers=snapmaps,
            AFO_lower_bound=500,
            just_under_CAFO_thresh=900,
            CAFO_thresh_logic="point",
            save_path=out_tables / "summary_table_point_estimate.csv",
            size_plot_au_save_path=out_risk / "size_au_unperm_vs_perm.svg",
        )
        plt.close("all")
        del snapmaps

        # Cache the parcel-owner-containing clusters (see docstring) — CONTAINS PII.
        pii_cache_path = out_tables / INTERNAL_PII_CLUSTERS_PATH_NAME
        try:
            cache_df = all_clusters.copy()
            # pd.cut-derived columns (e.g. 'size_category' from sample_and_calc_au)
            # are IntervalDtype, which pyarrow can't cast to parquet — stringify
            # any interval/categorical columns before writing.
            for col in cache_df.columns:
                if col == cache_df.geometry.name:
                    continue
                dtype = cache_df[col].dtype
                if isinstance(dtype, pd.CategoricalDtype) or "interval" in str(dtype).lower():
                    cache_df[col] = cache_df[col].astype(str)
            cache_df.to_parquet(pii_cache_path)
            print(
                f"  Cached all_clusters with parcel owner names (INTERNAL — contains "
                f"PII, do not publish/share) → {pii_cache_path}"
            )
        except Exception as e:
            print(f"  Warning: could not cache internal PII clusters: {e}")

    # ── common to both paths ──────────────────────────────────────────────────
    all_clusters["permitted"] = all_clusters["set"] == "Permitted dairy CAFOs"

    # Risk indices — guard so precomputed columns aren't needlessly recomputed
    if "hand_built_risk_index" not in all_clusters.columns:
        all_clusters = cf.add_hand_built_risk_index(
            all_clusters, WEIGHTS_DICT, INVERT_VARIABLES,
        )
    if "pca_risk_index" not in all_clusters.columns:
        all_clusters = cf.add_pca_risk_index(
            all_clusters, list(WEIGHTS_DICT.keys()),
            invert_variables=INVERT_VARIABLES,
        )

    # Add AU category for plotting
    all_clusters["animal_unit_category"] = pd.cut(
        all_clusters["animal_unit_estimate"],
        bins=[500, 800, 1000, 1250, 1500, 2000, 5000, np.inf],
    )

    # Risk index summary statistics table
    risk_var_stats = all_clusters[list(WEIGHTS_DICT.keys())].agg(
        ["mean", "std", "min", "max"]
    )
    risk_var_stats.to_csv(out_tables / "risk_index_summary_stats.csv")
    print(f"  Saved risk_index_summary_stats.csv")

    # Risk index by AU category figures
    cf.plot_risk_index_by_au_category(
        all_clusters, "hand_built_risk_index",
        save_path=out_risk / "hand_built_index_by_au_cat.svg",
    )
    cf.plot_risk_index_by_au_category(
        all_clusters, "pca_risk_index",
        save_path=out_risk / "pca_index_by_au_cat.svg",
    )

    # Multivariate regression: does permit status remain associated with risk
    # once size (and other facility/location factors) are controlled for
    # simultaneously, rather than just compared within AU bins? Reviewer comment
    # on the bin plots above. Also tests robustness to alternative risk-index designs.
    cf.run_permit_risk_regression(
        all_clusters, WEIGHTS_DICT, INVERT_VARIABLES,
        risk_col="hand_built_risk_index",
        size_col="animal_unit_estimate",
        n_robustness_draws=500,
        save_table_path=out_tables / "risk_regression_summary.csv",
        save_fig_path=out_risk / "risk_regression_weight_robustness.svg",
    )
    # Same regression, but controlling for cluster footprint area (a directly
    # measured facility size) instead of the AU estimate, which carries Monte
    # Carlo estimation uncertainty — checks the finding isn't an artifact of
    # noise in the AU estimation model.
    cf.run_permit_risk_regression(
        all_clusters, WEIGHTS_DICT, INVERT_VARIABLES,
        risk_col="hand_built_risk_index",
        size_col="cluster_area_m2",
        n_robustness_draws=500,
        save_table_path=out_tables / "risk_regression_summary_area.csv",
        save_fig_path=out_risk / "risk_regression_weight_robustness_area.svg",
    )

    # Sensitivity analysis: does the permitted vs unpermitted finding hold under
    # random weight perturbation?
    cf.plot_risk_sensitivity_bands(
        all_clusters, WEIGHTS_DICT, INVERT_VARIABLES,
        n_draws=500,
        save_path=out_risk / "hand_built_index_weight_sensitivity.svg",
    )
          
    # Top 300 breakdown
    cf.plot_top_300_breakdown(
        all_clusters,
        save_path=out_risk / "top_300_breakdown_combined_index.svg",
        save_table_path=out_tables / "top_300_breakdown.csv",
    )

    # Risk comparison case studies — explicit cases from all_key_results.ipynb
    if not skip_imagery:
        try:
            _generate_risk_case_studies(all_clusters, data, out_risk)
        except Exception as e:
            print(f"  Warning: could not generate risk case studies: {e}")

    print(f"  Saved figures to {out_risk}")
    return all_clusters


# ============================================================================
# Section 10: Publication dataset
# ============================================================================

def generate_publication_dataset(all_clusters, milk_producers, subdirs):
    """Create and export the publication dataset as GeoParquet + GeoJSON."""
    out = subdirs["pub_dataset"]

    pub_data = all_clusters.copy()

    # Merge milk producer info for matched facilities
    # milk_producers has active geometry 'location' (coalesced in load_data());
    # select that as the geometry and drop the original 'geometry' column.
    milk_cols = ["BusinessName", "FarmAddress", "TypeofMilk", "County"]
    milk_subset = gpd.GeoDataFrame(
        milk_producers[milk_cols].copy(),
        geometry=milk_producers.geometry,  # uses the active geometry ('location')
        crs=milk_producers.crs,
    )
    milk_subset = milk_subset.rename(
        columns={c: f"milk_{c}" for c in milk_cols}
    )

    # sjoin_nearest to find closest milk producer within 500m
    pub_with_milk = pub_data.sjoin_nearest(
        milk_subset.to_crs(cfg.WI_EPSG), max_distance=500, how="left",
    )
    # Drop duplicates from multiple milk matches (keep closest)
    pub_with_milk = pub_with_milk[
        ~pub_with_milk.index.duplicated(keep="first")
    ]

    # Select columns for export
    columns_to_keep = [
        # Geometry
        "geometry",
        # Parcel matching
        "parcel_owner1_names", "parcel_owner2_names",
        # AU estimates
        "animal_unit_estimate", "animal_units_lower", "animal_units_upper",
        # Permit status
        "set",
        # Permit info
        "Facility ID (FIN)", "Number ofAnimalUnits", "allowable_animal_units",
        "AnimalType", "has_satellite_sites",
        # Milk license
        "matched_milk", "milk_BusinessName", "milk_FarmAddress",
        "milk_TypeofMilk", "milk_County",
        # Water risk
        "water_distance", "impaired_water_distance", "closest_water_impaired",
        "hydro_intermit_distance", "hydro_perennial_distance",
        # Groundwater
        "gw_0", "gw_20", "gw_50",
        # Geology
        "bedrock_lt5ft_distance", "bedrock_lt5ft_dummy",
        "silurian_0_2_distance", "silurian_0_2_dummy",
        "silurian_2_5_distance", "silurian_2_5_dummy",
        "shallow_silurian_distance", "shallow_silurian_dummy",
        # Slope
        "mean_slope", "slope_greater_12_distance", "slope_greater_12_dummy",
        # Regulatory
        "swqma_300ft_distance", "swqma_300ft_dummy",
        "swqma_1000ft_distance", "swqma_1000ft_dummy",
        "cafo_w_restrict_distance", "cafo_r_restrict_distance",
        # Risk indices
        "hand_built_risk_index", "pca_risk_index",
        # Metadata
        "n_buildings", "n_parcels", "cluster_area_m2",
    ]
    available_cols = [c for c in columns_to_keep if c in pub_with_milk.columns]
    pub_export = gpd.GeoDataFrame(
        pub_with_milk[available_cols], geometry="geometry",
    )

    # Handle non-serializable types (tuples/lists in polygon_indices, parcel names)
    for col in pub_export.columns:
        if col == "geometry":
            continue
        sample = pub_export[col].dropna()
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, tuple)):
            pub_export[col] = pub_export[col].apply(
                lambda x: json.dumps(list(x)) if isinstance(x, (list, tuple)) else x
            )

    # Export GeoParquet in WI State Plane
    pub_wi = pub_export.copy()
    if pub_wi.crs is None or pub_wi.crs.to_epsg() != cfg.WI_EPSG:
        pub_wi = pub_wi.to_crs(epsg=cfg.WI_EPSG)
    pub_wi.to_parquet(out / "wi_cafo_facilities.parquet")

    # Export GeoJSON in WGS84
    pub_wgs84 = pub_export.to_crs(epsg=4326)
    pub_wgs84.to_file(out / "wi_cafo_facilities.geojson", driver="GeoJSON")

    print(f"  Publication dataset: {len(pub_export)} facilities exported")
    print(f"  Columns: {len(available_cols)}")
    print(f"  Saved to {out}")
    return pub_export


# ============================================================================
# Section 11: Risk case-study helper
# ============================================================================

def _centroid_latlon(row):
    """Return (lat, lon) of the cluster geometry centroid in WGS84."""
    centroid = (
        gpd.GeoSeries([row.geometry], crs=cfg.WI_EPSG)
        .to_crs("EPSG:4326")
        .iloc[0]
        .centroid
    )
    return centroid.y, centroid.x  # lat, lon


def _generate_risk_case_studies(all_clusters, data, out_risk):
    """Print data + save plot for four fixed risk-comparison facilities.

    Cases:
      1. High-risk unpermitted  — Gobel Dairy LLC
      2. Low-risk permitted     — Zinke Dairy Farms LLC
      3. Closest-to-impaired-water unpermitted — Brickner-Meikle Family Farms LLC
      4. Low-risk permitted     — Jay Stauffacher
    """
    import plotting_utils as pu

    cols_to_print = [
        "parcel_owner1_names", "animal_unit_estimate",
        "animal_units_lower", "animal_units_upper",
        "cluster_area_m2", "hand_built_risk_index",
    ] + list(WEIGHTS_DICT.keys())
    cols_to_print = [c for c in cols_to_print if c in all_clusters.columns]

    def _find(search_str):
        """Return first row whose parcel_owner1_names contains search_str."""
        mask = all_clusters["parcel_owner1_names"].apply(
            lambda x: search_str.upper() in str(x).upper()
        )
        matches = all_clusters[mask]
        if len(matches) == 0:
            raise ValueError(f"No cluster found matching '{search_str}'")
        return matches.iloc[0]

    cases = [
        ("1", "High-risk unpermitted",               "GOBEL DAIRY LLC",              "risk_case1_gobel_high_risk_unperm.png",         False),
        ("2", "Low-risk permitted",                  "ZINKE DAIRY FARMS LLC",         "risk_case2_zinke_low_risk_perm.png",             True),
        ("3", "Closest-to-impaired-water unpermitted","BRICKNER-MEIKLE FAMILY FARMS", "risk_case3_brickner_closest_impaired_unperm.png",False),
        ("4", "Low-risk permitted",                  "JAY STAUFFACHER",               "risk_case4_stauffacher_low_risk_perm.png",        True),
    ]

    for case_num, label, search_str, fname, is_permitted in cases:
        try:
            row = _find(search_str)
            print(f"\n  [Risk case {case_num}] {label} ({search_str}):")
            print(row[cols_to_print].to_string())
            lat, lon = _centroid_latlon(row)
            print(f"  Google Maps: {lat:.6f}, {lon:.6f}")

            cluster_to_plot = all_clusters[
                all_clusters["polygon_indices"].apply(lambda x: x == row["polygon_indices"])
            ]
            # all_clusters' active geometry is 'centroid' (points), swapped in by
            # analyze_water_pollution_stats — plot_cluster_parcel draws .boundary/.plot()
            # on whatever's active, so without this reset it silently plots empty point
            # boundaries instead of the actual cluster polygon outline/fill.
            cluster_to_plot = cluster_to_plot.set_geometry("geometry")
            plot_kwargs = dict(
                image_bound_map=data["image_bound_map"],
                parcel_data=data["parcels"],
                annotate_parcels=False,
                zoom_radius=500,
                show_scale_bar=True,
                display=False,
                fig_path=out_risk / fname,
                dpi=150,
            )
            if not is_permitted:
                plot_kwargs["permit_data"] = data["WDNR_CAFOs"]
            else:
                plot_kwargs["show_cluster_outline"] = True
                plot_kwargs["cluster_fill"] = True
            pu.plot_cluster_parcel(cluster_to_plot, **plot_kwargs)
            plt.close("all")
        except Exception as e:
            print(f"  [Risk case {case_num}] Skipped: {e}")


# ============================================================================
# Section 12: Utilities
# ============================================================================

class _Tee:
    """Duplicate writes to both the real stdout and an open log file."""
    def __init__(self, logfile):
        self._log = logfile
        self._out = sys.__stdout__

    def write(self, data):
        self._out.write(data)
        self._log.write(data)

    def flush(self):
        self._out.flush()
        self._log.flush()

    def isatty(self):
        return False



# ============================================================================
# Section 11b: Milk license match statistics
# ============================================================================

def generate_milk_license_match_stats(data, subdirs):
    """Milk license match statistics for clusters outside EWG region.

    Reports, for four_band_clusters_exc_EWG and cf_clusters_exc_EWG (after
    removing permit-matched clusters):
      - Count matched, total count, % matched at >500 and >1000 AU point estimates
      - Sum of animal_unit_estimate for matched vs. total
      - Same stats broken down by AU size category

    Also reports match-distance statistics across all regions (both cluster sets)
    and produces a CDF figure of match distances to support sensitivity analysis
    around the 500 m radius threshold.

    Uses polygon_indices (not DataFrame index) as the stable cluster identifier
    throughout, to avoid false matches from duplicate indices after concat ops.
    """
    milk_prods = data["milk_producers"].to_crs(cfg.WI_EPSG)
    au_bins   = [500, 1000, 2000, np.inf]
    au_labels = ["500–1000", "1000–2000", "2000+"]

    def _do_match(clusters, milk_prods, max_distance=500):
        """Return clusters GDF with milk_matched boolean and match_dist_m columns.

        Deduplication is done on polygon_indices (converted to string), so
        duplicate DataFrame indices (e.g. after pd.concat) cannot cause false
        matches.
        """
        matched = clusters.sjoin_nearest(
            milk_prods, max_distance=max_distance, distance_col="match_dist_m",
        )
        # When left/right geometries have the same column name sjoin_nearest
        # renames them; restore original geometry so subsequent ops work.
        if "geometry_left" in matched.columns:
            matched = matched.rename(columns={"geometry_left": "geometry"}).drop(
                columns=["geometry_right"], errors="ignore"
            )
            matched = gpd.GeoDataFrame(matched, geometry="geometry", crs=clusters.crs)

        # Keep one row per cluster (equidistant ties → keep first match).
        matched["_pstr"] = matched["polygon_indices"].apply(str)
        matched = matched.drop_duplicates(subset="_pstr")

        # Build pstr → distance mapping before dropping the helper column.
        dist_map = matched.set_index("_pstr")["match_dist_m"]
        matched_pstrs = set(matched["_pstr"])
        matched = matched.drop(columns=["_pstr"])

        # Flag matched clusters in the full set via polygon_indices string.
        result = clusters.copy()
        result["_pstr"] = result["polygon_indices"].apply(str)
        result["milk_matched"]  = result["_pstr"].isin(matched_pstrs)
        result["match_dist_m"]  = result["_pstr"].map(dist_map)
        result = result.drop(columns=["_pstr"])
        return result

    def _print_match_stats(clusters_with_flags, label):
        print(f"\n  ── {label} ──")

        for thresh in [500, 1000]:
            sub = clusters_with_flags[
                clusters_with_flags["animal_unit_estimate"] >= thresh
            ]
            n_total   = len(sub)
            n_matched = int(sub["milk_matched"].sum())
            pct       = 100 * n_matched / n_total if n_total else float("nan")
            print(f"\n    >{thresh} AU (point estimate):")
            print(f"      Total clusters:   {n_total:>5}")
            print(f"      Milk-matched:     {n_matched:>5}  ({pct:.1f}%)")

        # By AU size category (≥500 AU only)
        sub500 = clusters_with_flags[
            clusters_with_flags["animal_unit_estimate"] >= 500
        ].copy()
        sub500["au_cat"] = pd.cut(
            sub500["animal_unit_estimate"],
            bins=au_bins, labels=au_labels, right=False,
        )
        cat = (
            sub500.groupby("au_cat", observed=True)
            .apply(lambda g: pd.Series({
                "total":      len(g),
                "matched":    int(g["milk_matched"].sum()),
                "au_total":   g["animal_unit_estimate"].sum(),
                "au_matched": g.loc[g["milk_matched"], "animal_unit_estimate"].sum(),
            }))
        )
        cat["pct_matched"] = 100 * cat["matched"] / cat["total"]
        print("\n    By AU category (≥500 AU):")
        print(cat.to_string())

    def _drop_permit_matched(clusters, permits, match_distance=400):
        """Return clusters that have no WDNR permit within match_distance meters."""
        wdnr = permits[["geometry"]].to_crs(cfg.WI_EPSG)
        joined = clusters.sjoin_nearest(
            wdnr, max_distance=match_distance, how="left", distance_col="_pd",
        )
        joined["_pstr"] = joined["polygon_indices"].apply(str)
        permit_matched_pstrs = set(joined.loc[joined["_pd"].notna(), "_pstr"])
        result = clusters.copy()
        result["_pstr"] = result["polygon_indices"].apply(str)
        out = result[~result["_pstr"].isin(permit_matched_pstrs)].drop(columns=["_pstr"])
        n_dropped = len(clusters) - len(out)
        print(f"    (dropped {n_dropped} permit-matched clusters)")
        return out

    wdnr_permits = data["WDNR_CAFOs"]

    # ── four-band clusters (exc. EWG, exc. permit-matched) ───────────────────
    fb_exc_raw = data["four_band_clusters_exc_EWG"]
    fb_exc_unperm = _drop_permit_matched(fb_exc_raw, wdnr_permits)
    fb_exc = _do_match(fb_exc_unperm, milk_prods)
    _print_match_stats(fb_exc, "Four-band model clusters (outside EWG, unpermitted)")

    # ── CF-annotated clusters (exc. EWG, exc. permit-matched) ────────────────
    cf_exc_raw = data["cf_clusters_exc_EWG"]
    cf_exc_unperm = _drop_permit_matched(cf_exc_raw, wdnr_permits)
    cf_exc = _do_match(cf_exc_unperm, milk_prods)
    _print_match_stats(cf_exc, "CF-annotated clusters (outside EWG, unpermitted)")

    # ── Match-distance sensitivity (all regions) ─────────────────────────────
    print("\n  ── Match-distance sensitivity (all regions, 500 m threshold) ──")
    dist_series = {}
    for clusters_all, label in [
        (data["four_band_clusters"], "Four-band (all regions)"),
        (data["all_cf_clusters"],    "CF-annotated (all regions)"),
    ]:
        matched_all = _do_match(clusters_all, milk_prods)
        dists = matched_all.loc[
            matched_all["milk_matched"], "match_dist_m"
        ].dropna()
        dist_series[label] = dists
        print(f"\n    {label}  (n matched = {len(dists)})")
        print(f"      mean:   {dists.mean():.1f} m")
        print(f"      median: {dists.median():.1f} m")
        print(f"      p25/p75:{dists.quantile(0.25):.1f} / {dists.quantile(0.75):.1f} m")
        print(f"      max:    {dists.max():.1f} m")
        dist_bins = pd.cut(
            dists, bins=[0, 50, 100, 200, 500],
            labels=["0–50 m", "50–100 m", "100–200 m", "200–500 m"],
        )
        print("      Distance distribution:")
        for bucket, cnt in dist_bins.value_counts().sort_index().items():
            print(f"        {bucket}: {cnt}")

    # CDF figure: cumulative fraction of matches vs. distance threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    thresholds = np.linspace(0, 1000, 500)
    for label, dists in dist_series.items():
        n_total = len(dists)
        cumulative = np.array([(dists <= t).sum() / n_total for t in thresholds])
        ax.plot(thresholds, cumulative, label=label)
    ax.axvline(500, color="gray", linestyle="--", linewidth=1, label="500 m threshold")
    ax.set_xlabel("Distance threshold (m)")
    ax.set_ylabel("Cumulative fraction of matches")
    ax.set_title("Milk license match sensitivity to distance threshold")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = subdirs["tables"] / "milk_match_distance_sensitivity.svg"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"\n  Saved match-distance sensitivity plot → {save_path}")


# ============================================================================
# Section 11c: 2024 permit update match (out-of-sample confirmation)
# ============================================================================

PERMIT_2024_DIR_NAME = "wdnr_cafos_2024_aug"
PERMIT_2024_MATCH_DISTANCE = 400  # metres — matches caf.merge_clusters_permits elsewhere


def _load_permits_2024(paths):
    """Load and clean the updated (Aug 2024) WDNR CAFO Main/Satellite shapefiles.

    Mirrors the cleaning in notebooks/2024_permit_update_analysis.ipynb:
      - Main: keep only PERMIT_STA == 'Current'.
      - Satellite: drop rows with the shapefile float-overflow sentinel
        coordinates used for null geometries.
      - Combine into one GeoDataFrame with a unique CAFO_index and a
        SATELLITE_ column (null for main sites), matching data["WDNR_CAFOs"]'s
        schema closely enough to reuse the same matching logic.
    """
    permit_dir = paths["data_path"] / PERMIT_2024_DIR_NAME
    new_main = gpd.read_file(permit_dir / "CAFO Main.shp")
    new_sat = gpd.read_file(permit_dir / "CAFO Satellite.shp")

    main_clean = new_main[new_main["PERMIT_STA"] == "Current"].copy()
    main_clean["SATELLITE_"] = np.nan
    main_clean = main_clean.rename(
        columns={"FIN": "Facility ID (FIN)", "PERMITTEE_": "PermitteeName"}
    )

    COORD_OVERFLOW = 1e20  # shapefile float overflow sentinel for null geometries
    sat_bounds = new_sat.geometry.bounds
    sat_clean = new_sat[sat_bounds["minx"].abs() <= COORD_OVERFLOW].copy()
    sat_clean = sat_clean.rename(columns={"FIN": "Facility ID (FIN)"})

    main_clean["CAFO_index"] = range(len(main_clean))
    sat_clean["CAFO_index"] = range(len(main_clean), len(main_clean) + len(sat_clean))

    permits_2024 = pd.concat([main_clean, sat_clean], ignore_index=True)
    permits_2024 = gpd.GeoDataFrame(permits_2024, geometry="geometry", crs=cfg.WI_EPSG)
    return permits_2024


def generate_permit_2024_update_match(data, paths, all_clusters, subdirs):
    """Re-match unpermitted potential CAFOs against an updated (Aug 2024) WDNR
    permit shapefile, issued after the paper's original ~2022 permit snapshot.

    Reproduces notebooks/2024_permit_update_analysis.ipynb inside the main
    pipeline so the match is regenerated from the same in-memory all_clusters
    used throughout the paper (rather than the static published geojson).

    For each unpermitted potential CAFO (all_clusters['set'] ==
    'Unpermitted potential CAFOs'), matches to the nearest 2024 permit within
    400m, preferring a main-site match over a satellite-site match (same
    logic as caf.merge_clusters_permits). Classifies each match as
    'genuinely new' (Facility ID not present in the original 2022 permit
    data) vs. one that should have matched already in the original analysis.

    Writes permit_2024_update_matches.csv to tables/ with the auto-computed
    match columns plus blank columns for the manual permit-file lookups
    reported in the paper (approval-time AU, stated expansion AU, approval
    date, earliest portal document date, link, notes) — same schema as the
    "2024 permit matched unpermitted detections" Google Sheet tab, so a
    completed sheet can be dropped back in at this path. Never overwrites an
    existing CSV (delete it to force a fresh match), so pulling the
    filled-in sheet back and re-running is safe.
    """
    out_csv = subdirs["tables"] / "permit_2024_update_matches.csv"
    if out_csv.exists():
        print(f"  {out_csv} already exists — skipping (delete it to force a fresh match).")
        return pd.read_csv(out_csv)

    permits_2024 = _load_permits_2024(paths)
    permits_main_2024 = permits_2024[permits_2024["SATELLITE_"].isna()].copy()
    permits_sat_2024 = permits_2024[permits_2024["SATELLITE_"].notna()].copy()

    old_permits = data["WDNR_CAFOs"]
    old_main_fins = set(
        old_permits.loc[old_permits["SATELLITE_"].isna(), "Facility ID (FIN)"]
        .dropna().astype(int)
    )

    # Select only the columns needed for matching. all_clusters already carries
    # a 'Facility ID (FIN)' column from the original 2022 permit match (NaN for
    # unpermitted rows) — sjoin_nearest would otherwise silently rename both
    # sides' 'Facility ID (FIN)' to _left/_right on collision with the 2024
    # permit data's own column of the same name.
    unpermitted = all_clusters.loc[
        all_clusters["set"] == "Unpermitted potential CAFOs",
        ["polygon_indices", "animal_unit_estimate", "animal_units_lower",
         "animal_units_upper", "geometry"],
    ].copy()
    # .loc[] keeps all_clusters' active-geometry pointer (upstream code swaps it to
    # 'centroid' via set_geometry), which isn't among the selected columns above —
    # reset it explicitly or sjoin_nearest below fails to resolve .geometry/.crs.
    unpermitted = unpermitted.set_geometry("geometry")
    print(f"  Unpermitted potential CAFOs to re-match: {len(unpermitted)}")

    matched_main = unpermitted.sjoin_nearest(
        permits_main_2024[["Facility ID (FIN)", "PermitteeName", "CAFO_index", "geometry"]],
        max_distance=PERMIT_2024_MATCH_DISTANCE, distance_col="match_distance", how="left",
    ).drop(columns=["index_right"], errors="ignore")
    matched_main["_match_priority"] = 0

    matched_sat = unpermitted.sjoin_nearest(
        permits_sat_2024[["Facility ID (FIN)", "SATELLITE_", "CAFO_index", "geometry"]],
        max_distance=PERMIT_2024_MATCH_DISTANCE, distance_col="match_distance", how="left",
    ).drop(columns=["index_right"], errors="ignore")
    matched_sat["_match_priority"] = 1

    combined = pd.concat([matched_main, matched_sat])
    combined["_pstr"] = combined["polygon_indices"].apply(str)
    combined = combined.sort_values(["_match_priority", "match_distance"])
    combined = combined.drop_duplicates(subset="_pstr", keep="first").drop(columns=["_pstr"])

    newly_matched = combined[combined["CAFO_index"].notna()].copy()
    still_unmatched = combined[combined["CAFO_index"].isna()].copy()

    newly_matched["fin_int"] = newly_matched["Facility ID (FIN)"].astype(float).astype("Int64")
    newly_matched["permit_in_2022"] = newly_matched["fin_int"].apply(
        lambda f: int(f) in old_main_fins if pd.notna(f) else False
    )
    newly_matched["genuinely_new_permit"] = ~newly_matched["permit_in_2022"]

    print(f"  Now matching a 2024 permit (<={PERMIT_2024_MATCH_DISTANCE}m): {len(newly_matched)}")
    print(
        f"    Genuinely new permit (issued after 2022):        "
        f"{int(newly_matched['genuinely_new_permit'].sum())}"
    )
    print(
        f"    FIN existed in 2022 (missed match originally):   "
        f"{int(newly_matched['permit_in_2022'].sum())}"
    )
    print(f"  Still unmatched in 2024:                            {len(still_unmatched)}")

    out = newly_matched.sort_values("match_distance")[[
        "polygon_indices", "animal_unit_estimate", "animal_units_lower", "animal_units_upper",
        "Facility ID (FIN)", "PermitteeName", "match_distance", "permit_in_2022",
        "genuinely_new_permit",
    ]].copy()
    out["polygon_indices"] = out["polygon_indices"].apply(str)

    # Blank columns for manual permit-file lookups (matches the Google Sheet
    # schema used for the manuscript's out-of-sample confirmation appendix).
    for col in [
        "Projected expansion (projected) at time of approval",
        "Current found online at application approval time",
        "Date created of approved initial permit",
        "Au at initial app time",
        "Date created for earliest doc on portal",
        "Date of imagery",
        "Link",
        "notes",
    ]:
        out[col] = ""

    out.to_csv(out_csv, index=False)
    print(f"  Saved {len(out)}-row 2024 permit match table to {out_csv}")
    return out


# ============================================================================
# Section 11d: Online validation sample (unpermitted potential CAFOs)
# ============================================================================

VALIDATION_SAMPLE_SEED = 42
VALIDATION_SAMPLE_N_PER_GROUP = 40


def generate_online_validation_sample(data, all_clusters, subdirs,
                                       n_per_group=VALIDATION_SAMPLE_N_PER_GROUP,
                                       seed=VALIDATION_SAMPLE_SEED):
    """Draw a reproducible stratified random sample of unpermitted potential
    CAFOs for manual online validation, and write a fill-in-the-blanks CSV.

    Stratifies by milk-license match status (all_clusters['matched_milk'],
    the same 500m sjoin_nearest match used throughout the paper) and draws
    n_per_group facilities from each stratum of the >=1000 AU unpermitted
    universe (all_clusters['set'] == 'Unpermitted potential CAFOs'), using a
    fixed random_state so the draw is identical on every pipeline run.

    Writes online_validation_sample.csv to tables/ with:
      - identifier (polygon_indices, stable join key back to all_clusters)
      - milk_license_match_group ('with_milk_license_match' / 'without_...')
      - lat, lon, lat-long (paste directly into Google Maps/Earth)
      - parcel_owner1_names, parcel_owner2_names
      - milk_BusinessName, milk_FarmAddress, milk_TypeofMilk, milk_County,
        milk_match_distance_m (blank when milk_license_match_group is
        'without_milk_license_match')
      - animal_unit_estimate, animal_units_lower, animal_units_upper
      - three blank columns for manual completion after pulling this into a
        Google Sheet: 'Confirmed active dairy facility', 'Any information
        about size available online - sources', 'Online information about
        herd size - estimate'

    Never overwrites an existing CSV (delete it to force a fresh draw) — a
    routine pipeline re-run can't clobber a partially/fully filled-in sheet
    pulled back from Google Sheets. Pair with analyze_online_validation_sample()
    to read the completed sheet back in for synthesis.

    If all_clusters is missing parcel_owner1_names (e.g. running with
    --use-precomputed off the public, PII-free publication dataset), this
    transparently swaps in tables/all_clusters_internal_PII.parquet when that
    cache exists (written by a prior full run of generate_risk_assessment()),
    so owner names don't require re-running the expensive summary_table() step.
    """
    out_csv = subdirs["tables"] / "online_validation_sample.csv"
    if out_csv.exists():
        print(f"  {out_csv} already exists — skipping (delete it to force a fresh draw).")
        return pd.read_csv(out_csv)

    if "parcel_owner1_names" not in all_clusters.columns:
        pii_cache_path = subdirs["tables"] / INTERNAL_PII_CLUSTERS_PATH_NAME
        if pii_cache_path.exists():
            print(
                f"  all_clusters has no parcel_owner1_names (likely --use-precomputed) — "
                f"loading cached internal clusters from {pii_cache_path} instead."
            )
            all_clusters = gpd.read_parquet(pii_cache_path)
        else:
            print(
                "  Warning: all_clusters has no parcel_owner1_names and no "
                f"{pii_cache_path} cache exists — sample will omit parcel owner names. "
                "Run the pipeline once without --use-precomputed to build the cache."
            )

    if "matched_milk" not in all_clusters.columns:
        raise ValueError(
            "all_clusters is missing 'matched_milk' — run generate_risk_assessment() first."
        )

    unp = all_clusters[all_clusters["set"] == "Unpermitted potential CAFOs"].copy()
    with_milk = unp[unp["matched_milk"] == True]
    without_milk = unp[unp["matched_milk"] == False]

    def _draw(pool, group_label):
        n = min(n_per_group, len(pool))
        if n < n_per_group:
            print(
                f"  Warning: only {len(pool)} candidates available for '{group_label}' "
                f"(requested {n_per_group}); using all of them."
            )
        drawn = pool.sample(n=n, random_state=seed).copy()
        drawn["milk_license_match_group"] = group_label
        return drawn

    drawn_with = _draw(with_milk, "with_milk_license_match")
    drawn_without = _draw(without_milk, "without_milk_license_match")
    sample = pd.concat([drawn_with, drawn_without]).copy()

    # Milk-license match details for the sampled rows only (same 500m
    # threshold used to build 'matched_milk' in summary_table()).
    milk_wi = data["milk_producers"].to_crs(cfg.WI_EPSG)
    milk_cols = ["BusinessName", "FarmAddress", "TypeofMilk", "County"]
    milk_subset = milk_wi[milk_cols + [milk_wi.geometry.name]].rename(
        columns={c: f"milk_{c}" for c in milk_cols}
    )
    sample = sample.sjoin_nearest(
        milk_subset, max_distance=500, how="left", distance_col="milk_match_distance_m",
    )
    sample = sample[~sample.index.duplicated(keep="first")]
    sample = sample.drop(columns=["index_right"], errors="ignore")

    # Lat/lon centroid in WGS84, pasteable straight into Google Maps/Earth.
    centroids_4326 = sample.geometry.centroid.to_crs(4326)
    sample["lat"] = centroids_4326.y
    sample["lon"] = centroids_4326.x
    sample["lat-long"] = (
        sample["lat"].round(6).astype(str) + ", " + sample["lon"].round(6).astype(str)
    )
    sample["identifier"] = sample["polygon_indices"].apply(str)

    out_cols = [
        "identifier", "milk_license_match_group",
        "lat", "lon", "lat-long",
        "parcel_owner1_names", "parcel_owner2_names",
        "milk_BusinessName", "milk_FarmAddress", "milk_TypeofMilk", "milk_County",
        "milk_match_distance_m",
        "animal_unit_estimate", "animal_units_lower", "animal_units_upper",
    ]
    out_cols = [c for c in out_cols if c in sample.columns]
    out = pd.DataFrame(sample[out_cols])
    out["Confirmed active dairy facility"] = ""
    out["Any information about size available online - sources"] = ""
    out["Online information about herd size - estimate"] = ""

    # Shuffle row order (fixed seed) so the two strata are interleaved rather
    # than block-ordered, then reset the index for a clean CSV.
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    out.to_csv(out_csv, index=False)
    print(
        f"  Drew {len(drawn_with)} with milk-license match + {len(drawn_without)} without "
        f"({len(out)} total) → {out_csv}"
    )
    return out


def analyze_online_validation_sample(subdirs):
    """Read the completed online-validation CSV (see
    generate_online_validation_sample()) back in and report confirmation
    rates by milk-license-match group.

    Prints a reminder and returns without writing anything if the sample
    hasn't been drawn yet, or hasn't been filled in yet. Expects 'Confirmed
    active dairy facility' to contain Yes/No-style text; blank/unrecognized
    values are excluded from the rate calculation (and counted separately)
    rather than treated as either answer, so partial completion of the
    Google Sheet doesn't skew the reported rate.
    """
    in_csv = subdirs["tables"] / "online_validation_sample.csv"
    if not in_csv.exists():
        print(f"  {in_csv} not found — run generate_online_validation_sample() first.")
        return None

    df = pd.read_csv(in_csv)
    confirmed_col = "Confirmed active dairy facility"
    resp = df[confirmed_col].astype(str).str.strip().str.lower()
    is_yes = resp.isin(["yes", "y", "true", "1"])
    is_no = resp.isin(["no", "n", "false", "0"])
    answered = is_yes | is_no
    n_answered = int(answered.sum())

    if n_answered == 0:
        print(
            f"  {in_csv} has no completed rows yet — fill in '{confirmed_col}' "
            f"(and the other columns) in the pulled-back Google Sheet, then re-run."
        )
        return df

    df["_confirmed_active"] = np.where(is_yes, True, np.where(is_no, False, np.nan))
    summary = df[answered].groupby("milk_license_match_group")["_confirmed_active"].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "pct_confirmed_active", "count": "n_answered"})
    summary["pct_confirmed_active"] *= 100

    overall = df.loc[answered, "_confirmed_active"].mean() * 100
    print(f"\n  Online validation sample: {n_answered}/{len(df)} rows answered.")
    print(summary.to_string())
    print(f"\n  Overall confirmed-active rate: {overall:.1f}%")

    out_path = subdirs["tables"] / "online_validation_summary.csv"
    summary.to_csv(out_path)
    print(f"  Saved {out_path}")
    return df


# ============================================================================
# Section 12: Main orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper results for WI-CAFO study",
    )
    parser.add_argument(
        "--skip-pixel-stats", action="store_true", default=True,
        help="Use cached pixel P/R/IOU stats (default: True)",
    )
    parser.add_argument(
        "--recalculate-pixel-stats", action="store_true",
        help="Force recalculation of pixel P/R/IOU stats",
    )
    parser.add_argument(
        "--skip-snapmaps", action="store_true",
        help="Skip loading snapmaps GDB (faster for testing)",
    )
    parser.add_argument(
        "--skip-imagery", action="store_true",
        help="Skip figures requiring NAIP tile downloads from GCS",
    )
    parser.add_argument(
        "--use-precomputed", action="store_true",
        help=(
            "Skip snapmaps loading and summary_table() by loading pre-computed "
            "cluster data from paper_results/publication_dataset/wi_cafo_facilities.parquet. "
            "Requires that the publication dataset has already been generated."
        ),
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip space-parameter sensitivity analysis (OAT sweeps + welfare scenarios).",
    )
    parser.add_argument(
        "--skip-spatial-stats", action="store_true",
        help="Skip spatial clustering statistics (join-count/Moran's I, Getis-Ord Gi*, "
             "K-function difference, NN cross-statistic).",
    )
    parser.add_argument(
        "--spatial-permutations", type=int, default=999,
        help="Number of Monte Carlo permutations for spatial clustering tests (default: 999).",
    )
    parser.add_argument(
        "--sensitivity-workers", type=int, default=4,
        help="Parallel threads for sensitivity analysis (default: 4).",
    )
    args = parser.parse_args()

    recalc_pixel = args.recalculate_pixel_stats and not args.skip_pixel_stats

    # Suppress interactive display
    matplotlib.use("Agg")
    plt.ioff()
    sns.set_theme(style="whitegrid")
    apply_paper_style()

    start = time.time()

    # Setup
    paths, subdirs = setup_paths()

    # Open log file — tee all stdout to it for the rest of the run
    log_path = Path("paper_results") / "paper_results.log"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(log_file)
    print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_path.resolve()}\n")

    try:
        _run_pipeline(args, paths, subdirs, recalc_pixel)
        elapsed = time.time() - start
        print(f"\nAll paper results generated in {elapsed / 60:.1f} minutes.")
        print(f"Outputs saved to: paper_results/")
        print(f"Log saved to:     {log_path.resolve()}")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()

    print(f"Run complete. Log: {log_path.resolve()}")


def _run_pipeline(args, paths, subdirs, recalc_pixel):
    """Inner pipeline — called by main() with stdout already tee'd to log."""
    use_precomputed = getattr(args, "use_precomputed", False)

    # When using precomputed publication dataset, snapmaps are never needed.
    read_snapmaps = (not args.skip_snapmaps) and (not use_precomputed)
    data = load_all_data(paths, read_snapmaps=read_snapmaps)

    # Resolve precomputed path (default: publication dataset parquet)
    precomputed_path = None
    if use_precomputed:
        _p = Path("paper_results") / "publication_dataset" / "wi_cafo_facilities.parquet"
        if _p.exists():
            precomputed_path = _p
        else:
            print(
                "  Warning: --use-precomputed set but publication dataset not found "
                f"at {_p}. Falling back to full pipeline (snapmaps will be loaded)."
            )
            data = load_all_data(paths, read_snapmaps=True)

    # Baseline snapshot — compare to the one printed just before summary_table()
    _cf0 = data["all_cf_clusters"]
    print(f"[baseline] all_cf_clusters: {len(_cf0)} rows, CRS={_cf0.crs}")
    _g0 = _cf0.geometry.iloc[0]
    print(f"[baseline] first geom centroid: x={_g0.centroid.x:.2f}, y={_g0.centroid.y:.2f}")
    print(f"[baseline] animal_unit_estimate range: "
          f"{_cf0['animal_unit_estimate'].min():.0f} – {_cf0['animal_unit_estimate'].max():.0f}")

    # 1. Model validation
    print("\n=== 1/8: Model Validation ===")
    permit_matched = generate_model_validation(data, subdirs)

    # 1a. Cross-regional robustness of validation error rates (reviewer:
    # clustering thresholds are empirically defined; test whether permit
    # recall / AU discrepancy / EWG precision diverge across WDNR regions).
    print("\n=== 1a: Cross-Regional Robustness ===")
    generate_regional_robustness_analysis(data, permit_matched, subdirs)

    # 2. Segmentation quality
    print("\n=== 2/8: Segmentation Quality ===")
    generate_segmentation_quality(
        data, paths, subdirs, recalculate_pixel_stats=recalc_pixel,
    )

    # 3. Error analysis
    print("\n=== 3/8: Error Analysis ===")
    generate_error_analysis(
        data, paths, subdirs, skip_imagery=args.skip_imagery,
    )

    # 4. Image review stats
    print("\n=== 4/8: Image Review Stats ===")
    generate_image_review_stats(data, paths, subdirs)

    # 4z. Human-annotation yield conditional on prior filtering (Reviewer #2)
    print("\n=== 4z: Human Annotation Yield ===")
    generate_human_annotation_yield_stats(data, paths, subdirs)

    # 4a. EWG-region false-positive waterfall (Reviewer #1, Limitation 2) —
    # EWG's census is treated as complete, so the full residual is reviewed.
    print("\n=== 4a: EWG-Region FP Waterfall ===")
    _, ewg_fp_residual = generate_ewg_fp_waterfall(data, paths, subdirs)

    # 4b. CF-sent-but-unmatched false positives — only the milk-license
    # rescue applies (size/urban/keyword were already applied before
    # sending to CF), so a random sample is manually reviewed instead.
    print("\n=== 4b: CF-Sent-Unmatched FP Analysis ===")
    _, cf_fp_residual = generate_cf_unmatched_fp_analysis(data, paths, subdirs)

    if not args.skip_imagery:
        print("\n=== 4c: FP Manual Review Samples ===")
        generate_fp_manual_review_sample(
            data, paths, subdirs, ewg_fp_residual, label="ewg", n_sample=None,
        )
        generate_fp_manual_review_sample(
            data, paths, subdirs, cf_fp_residual, label="cf_unmatched", n_sample=40,
        )

    # 4d. Summary stats from human-coded review (no-op until *_categorized.csv
    # files exist — see generate_fp_manual_review_summary()'s docstring).
    print("\n=== 4d: FP Manual Review Summary ===")
    generate_fp_manual_review_summary(subdirs)

    # 5. Model performance + universe tables
    print("\n=== 5/8: Performance & Universe Tables ===")
    generate_tables(data, subdirs, permit_matched)

    # 6. Risk assessment (heaviest section — loads snapmaps, runs summary_table).
    # Must run before unpermitted analysis so the authoritative 4-category 'set'
    # column from summary_table() can be passed to plot_permit_rate_by_size.
    print("\n=== 6/8: Risk Assessment ===")
    all_clusters = generate_risk_assessment(
        data, paths, subdirs, skip_imagery=args.skip_imagery,
        precomputed_path=precomputed_path,
    )

    # 7. Unpermitted analysis — receives all_clusters so permit_rate_by_size uses
    # the same 'set' column as the notebook (from summary_table()).
    print("\n=== 7/13: Unpermitted Analysis ===")
    generate_unpermitted_analysis(data, subdirs, permit_matched, all_clusters=all_clusters)

    # 7a. Formal tests backing the descriptive permitted-vs-unpermitted claims
    # in the size-distribution discussion (reviewer: report CIs / hypothesis
    # tests / effect sizes rather than descriptive comparisons).
    print("\n=== 7a/13: Permitted vs. Unpermitted Statistical Tests ===")
    generate_perm_unperm_statistical_tests(all_clusters, subdirs)

    # 7b. Descriptive permit-status regression (Reviewer #2): what correlates
    # with holding a permit, net of size, region, and operational factors.
    print("\n=== 7b/13: Permit-Status Regression ===")
    generate_permit_status_regression(data, all_clusters, paths, subdirs)

    # 7c. Online validation sample: reproducible random draw of unpermitted
    # potential CAFOs (stratified by milk-license match) for manual
    # confirmation via Google Maps/online search. Never overwrites an
    # existing draw, so pulling the filled-in sheet back and re-running is
    # safe. analyze_online_validation_sample() reports confirmation rates
    # once the sheet has been filled in.
    print("\n=== 7c/13: Online Validation Sample ===")
    generate_online_validation_sample(data, all_clusters, subdirs)
    analyze_online_validation_sample(subdirs)

    # 8. Spatial clustering statistics (join-count/Moran's I, Getis-Ord Gi*,
    # K-function difference, NN cross-statistic) for permitted vs. unpermitted potential.
    if not getattr(args, "skip_spatial_stats", False):
        print("\n=== 8/13: Spatial Clustering Statistics ===")
        generate_spatial_clustering_analysis(
            data, all_clusters, subdirs,
            n_permutations=getattr(args, "spatial_permutations", 999),
        )
    else:
        print("\n=== 8/13: Spatial Clustering Statistics (skipped) ===")

    # 9. Publication dataset
    print("\n=== 9/13: Publication Dataset ===")
    generate_publication_dataset(all_clusters, data["milk_producers"], subdirs)

    # 10. Milk license match statistics
    print("\n=== 10/13: Milk License Match Statistics ===")
    generate_milk_license_match_stats(data, subdirs)

    # 10a. Re-match against an updated (Aug 2024) WDNR permit shapefile, issued
    # after the paper's original ~2022 permit snapshot — out-of-sample
    # confirmation that some unpermitted potential CAFOs are real, if
    # previously unrecorded, farms (Section 5.1 of the manuscript).
    print("\n=== 10a/13: 2024 Permit Update Match ===")
    generate_permit_2024_update_match(data, paths, all_clusters, subdirs)

    # 11. Space-parameter sensitivity (OAT sweeps + welfare scenarios)
    if not getattr(args, "skip_sensitivity", False):
        print("\n=== 11/13: Space-Parameter Sensitivity ===")
        import sensitivity_space_params as ssp
        ssp.generate_space_param_sensitivity(
            data, subdirs,
            n_workers=getattr(args, "sensitivity_workers", 4),
        )
    else:
        print("\n=== 11/13: Space-Parameter Sensitivity (skipped) ===")


if __name__ == "__main__":
    main()
