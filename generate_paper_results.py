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

def generate_image_review_stats(data, paths, subdirs):
    """Compute and save stats on human imagery review reduction. → tables/.

    Replicates the four_band_cfsend construction from all_key_results.ipynb:
      1. any_images_sent  – cluster has ≥1 image that was sent to CF for labeling
      2. matched_to_CF_annot – cluster is within 10m of a human-annotated cluster
      3. Filters: urban area (>500m), parcel owner keywords, non-dairy CAFO (>300m)
      4. need_to_send = not sent AND not already annotated
      5. Count unique images where need_to_send OR any_images_sent
    """
    four_band = data["four_band_clusters"].copy()

    # ------------------------------------------------------------------
    # 1. any_images_sent: check jpeg_names against CF labeled image list
    # ------------------------------------------------------------------
    all_labeled_images = cf.get_labeled_images(how="json", data_path=paths["data_path"])
    labeled_set = set(all_labeled_images["Image name"].tolist())
    four_band["any_images_sent"] = four_band["jpeg_names"].apply(
        lambda x: any(item in labeled_set for item in (x if isinstance(x, (list, tuple)) else []))
    )

    # ------------------------------------------------------------------
    # 2. matched_to_CF_annot: sjoin_nearest to all_cf_clusters, max 10m
    #    Mirrors notebook cell 5: how='left' so unmatched rows get NaN,
    #    then matched_to_CF_annot = ~cf_polygon_indices.isna()
    # ------------------------------------------------------------------
    cf_clusters = data["all_cf_clusters"][["geometry", "polygon_indices"]].rename(
        columns={"polygon_indices": "cf_polygon_indices"}
    ).copy()
    four_band = gpd.sjoin_nearest(
        four_band, cf_clusters, max_distance=10, how="left", distance_col="distance_to_cf"
    )
    # polygon_indices is list-valued (loaded with literal_eval), so drop_duplicates
    # must use a str key — identical to step 6 below.
    four_band["_pstr2"] = four_band["polygon_indices"].apply(str)
    four_band = four_band.sort_values("distance_to_cf").drop_duplicates(
        subset="_pstr2", keep="first"
    ).drop(columns=["_pstr2"])
    four_band["matched_to_CF_annot"] = ~four_band["cf_polygon_indices"].isna()
    four_band = four_band.drop(
        columns=["cf_polygon_indices", "distance_to_cf", "index_right"], errors="ignore"
    )

    # ------------------------------------------------------------------
    # 3. Urban area filter: keep clusters >500m from any urban area
    # ------------------------------------------------------------------
    urban_areas_path = paths["data_path"] / "WI_urban_areas"
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


def generate_risk_assessment(data, paths, subdirs, skip_imagery=False,
                             precomputed_path=None):
    """Summary table, risk indices, risk figures. → 05_risk_assessment/ + tables/.

    precomputed_path: if provided (and the file exists), load all_clusters directly
        from that GeoParquet (e.g. the publication dataset) instead of loading
        snapmaps and running the expensive summary_table() + distance calculations.
        All downstream risk-index figures are still produced; the summary table CSV
        is skipped in this mode.
    """
    out_risk = subdirs["risk"]
    out_tables = subdirs["tables"]

    if precomputed_path is not None and Path(precomputed_path).exists():
        # ── fast path: skip snapmaps + summary_table ─────────────────────────
        print(f"  Using pre-computed cluster data: {precomputed_path}")
        print("  (snapmaps loading and summary_table() skipped)")
        all_clusters = gpd.read_parquet(precomputed_path)
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
    print("\n=== 7/8: Unpermitted Analysis ===")
    generate_unpermitted_analysis(data, subdirs, permit_matched, all_clusters=all_clusters)

    # 8. Publication dataset
    print("\n=== 8/8: Publication Dataset ===")
    generate_publication_dataset(all_clusters, data["milk_producers"], subdirs)

    # 9. Milk license match statistics
    print("\n=== 9/9: Milk License Match Statistics ===")
    generate_milk_license_match_stats(data, subdirs)


if __name__ == "__main__":
    main()
