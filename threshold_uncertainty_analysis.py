#!/usr/bin/env python3
"""
Uncertainty-propagation and threshold-classification analysis for WI-CAFO study.

Responds to Reviewer #2's two linked comments:
  1. Comprehensive sensitivity analysis showing how AU-estimation uncertainty
     propagates to the estimated number of unpermitted CAFOs (not just the
     three-point lower/point/upper table already in the manuscript).
  2. Classification accuracy near the 1,000 AU permitting threshold: using
     permitted CAFOs (for which we have WDNR-reported ground-truth AU), how
     does our estimate's uncertainty affect classification right at the
     threshold, and at an alternative threshold with more nearby facilities?

Usage:
    python threshold_uncertainty_analysis.py

Outputs -> paper_results/08_uncertainty_propagation/:
  percentile_sweep_results.csv       unpermitted count vs. percentile of the AU
                                      simulation used (1st-99th, plus 2.5/97.5)
  percentile_sweep.svg               plot of the above
  threshold_classification_facility_level.csv   per-permitted-CAFO detail
  threshold_classification_summary.csv          accuracy by distance-to-threshold band
  threshold_classification.svg       P(estimate >= threshold) vs. true AU
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import config.config_params as cfg
import cluster_analysis_functions as caf
import estimate_animal_units as est_au


# Fine percentile grid for the simulation-interval sweep (1st-99th, step 1).
# Built once so the exact same float values are used both when requesting the
# percentiles from sample_and_calc_au and when reading back the resulting
# "animal_units_{p}_perc" columns.
FINE_PERCENTILES = [round(i / 100, 2) for i in range(1, 100)]

# The manuscript's existing three-point rule, for cross-validation against
# this recomputation (same seed -> should reproduce 127 / 290 / 568 exactly).
REFERENCE_POINTS = {"lower (2.5%)": 0.025, "point (mean)": None, "upper (97.5%)": 0.975}

# Threshold near which most permitted CAFOs cluster (1500-2500 AU band), used
# as a higher-density alternative to the regulatory 1,000 AU cutoff so the
# classification-accuracy analysis has more nearby ground-truth facilities.
ALT_THRESHOLD = 2000

DISTANCE_BINS = [0, 100, 250, 500, np.inf]
DISTANCE_LABELS = ["0-100", "100-250", "250-500", "500+"]


def _milk_matched_unpermitted(all_cf, WDNR_CAFOs, milk_producers):
    """Unpermitted clusters matched to a milk-producer license (same logic used
    throughout the pipeline, e.g. generate_paper_results.py / sensitivity_space_params.py)."""
    all_perm, _ = caf.merge_clusters_permits(
        all_cf, WDNR_CAFOs,
        sum_satellite_counts=False, discrep_analysis=False, only_dairy=False,
    )
    unpermitted = all_cf[
        ~all_cf["polygon_indices"].isin(all_perm["polygon_indices"])
    ].copy()

    milk_wi = milk_producers.to_crs(cfg.WI_EPSG)
    unperm_milk = unpermitted.sjoin_nearest(milk_wi, max_distance=500)
    if "geometry_left" in unperm_milk.columns:
        unperm_milk["geometry"] = unperm_milk["geometry_left"]
        unperm_milk.drop(
            ["geometry_left", "geometry_right"], axis=1, inplace=True, errors="ignore"
        )
        unperm_milk = gpd.GeoDataFrame(unperm_milk, geometry="geometry", crs=cfg.WI_EPSG)
    unperm_milk = unperm_milk[~unperm_milk.index.duplicated(keep="first")]
    return unperm_milk


# ============================================================================
# 1. Percentile sweep: unpermitted count as a function of the simulation
#    percentile used, generalizing the lower/point/upper table to a curve.
# ============================================================================

def run_percentile_sweep(data, out_dir):
    print("\n[1/2] Percentile sweep of the AU simulation interval...")
    all_cf = est_au.sample_and_calc_au(
        data["all_cf_clusters"].copy(),
        include_area_uncertainty=False,
        optional_other_perc_threshes=FINE_PERCENTILES,
    )
    unperm_milk = _milk_matched_unpermitted(all_cf, data["WDNR_CAFOs"], data["milk_producers"])

    rows = []
    for p in FINE_PERCENTILES:
        col = f"animal_units_{p}_perc"
        count = int((unperm_milk[col] >= 1000).sum())
        rows.append({"percentile": p * 100, "unpermitted_count": count})

    # Cross-validate against the manuscript's existing three-point rule.
    for label, p in REFERENCE_POINTS.items():
        if p is None:
            count = int((unperm_milk["animal_unit_estimate"] >= 1000).sum())
            pct = 50.0  # mean, plotted near the median for reference only
        else:
            col = "animal_units_lower" if p == 0.025 else "animal_units_upper"
            count = int((unperm_milk[col] >= 1000).sum())
            pct = p * 100
        rows.append({"percentile": pct, "unpermitted_count": count, "reference_rule": label})
        print(f"    {label}: {count} unpermitted potential CAFOs (milk-matched)")

    df = pd.DataFrame(rows).sort_values("percentile").reset_index(drop=True)
    df.to_csv(out_dir / "percentile_sweep_results.csv", index=False)

    # Plot
    fine = df[df["reference_rule"].isna()] if "reference_rule" in df.columns else df
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fine["percentile"], fine["unpermitted_count"], color="steelblue", lw=2)
    for label, p in REFERENCE_POINTS.items():
        pct = 50.0 if p is None else p * 100
        y = df.loc[df["percentile"] == pct, "unpermitted_count"]
        if len(y):
            ax.scatter([pct], [y.values[-1]], color="firebrick", zorder=5)
            ax.annotate(label, (pct, y.values[-1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Percentile of per-facility AU simulation distribution used")
    ax.set_ylabel("Estimated unpermitted potential CAFOs\n(milk-license matched)")
    ax.set_title("Unpermitted-CAFO count across the full simulation interval")
    fig.tight_layout()
    fig.savefig(out_dir / "percentile_sweep.svg")
    plt.close(fig)

    print(f"  Saved percentile_sweep_results.csv + percentile_sweep.svg ({len(df)} rows)")
    return df


# ============================================================================
# 2. Threshold classification accuracy, using permitted CAFOs' WDNR-reported
#    (ground-truth) animal units as the true label.
# ============================================================================

def run_threshold_classification(data, out_dir, thresholds=(1000, ALT_THRESHOLD)):
    print("\n[2/2] Threshold classification accuracy (permitted CAFOs, ground truth)...")
    all_cf = est_au.sample_and_calc_au(
        data["all_cf_clusters"].copy(), include_area_uncertainty=False,
    )
    permit_matched, _ = caf.merge_clusters_permits(
        all_cf, data["WDNR_CAFOs"],
        sum_satellite_counts=True, discrep_analysis=True, only_dairy=True,
    )
    permit_matched = permit_matched.dropna(
        subset=["allowable_animal_units", "animal_unit_estimate", "animal_units_upper"]
    ).copy()

    # Normal approximation from (mean, SE), consistent with how satellite-site
    # uncertainty is already combined into animal_units_lower/upper (mean +/- 1.96*SE).
    permit_matched["se"] = (
        permit_matched["animal_units_upper"] - permit_matched["animal_unit_estimate"]
    ) / 1.96

    facility_rows = permit_matched[
        ["Facility ID (FIN)", "allowable_animal_units", "animal_unit_estimate", "se"]
    ].copy()

    summary_rows = []
    for threshold in thresholds:
        z = (permit_matched["animal_unit_estimate"] - threshold) / permit_matched["se"]
        p_ge = norm.cdf(z)
        true_label = permit_matched["allowable_animal_units"] >= threshold
        point_class = permit_matched["animal_unit_estimate"] >= threshold
        correct = point_class == true_label

        facility_rows[f"p_estimate_ge_{threshold}"] = p_ge
        facility_rows[f"true_label_ge_{threshold}"] = true_label
        facility_rows[f"point_correct_{threshold}"] = correct

        dist = (permit_matched["allowable_animal_units"] - threshold).abs()
        band = pd.cut(dist, bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False)

        overall_acc = correct.mean()
        overall_n = len(permit_matched)
        print(f"    Threshold={threshold}: n={overall_n}, overall accuracy={overall_acc:.3f}")
        summary_rows.append({
            "threshold": threshold, "distance_band": "all", "n": overall_n,
            "accuracy": round(overall_acc, 3),
            "mean_p_correct_direction": round(
                np.where(true_label, p_ge, 1 - p_ge).mean(), 3
            ),
        })

        for label in DISTANCE_LABELS:
            mask = band == label
            n = int(mask.sum())
            if n == 0:
                continue
            acc = correct[mask].mean()
            p_correct_dir = np.where(true_label[mask], p_ge[mask], 1 - p_ge[mask]).mean()
            summary_rows.append({
                "threshold": threshold, "distance_band": label, "n": n,
                "accuracy": round(acc, 3),
                "mean_p_correct_direction": round(p_correct_dir, 3),
            })
            print(f"      |true AU - {threshold}| in [{label}): n={n}, "
                  f"accuracy={acc:.3f}, mean P(correct direction)={p_correct_dir:.3f}")

    facility_rows.to_csv(out_dir / "threshold_classification_facility_level.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "threshold_classification_summary.csv", index=False)

    # Plot: P(estimate >= threshold) vs. true (WDNR-reported) AU, one panel per threshold.
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5))
    if len(thresholds) == 1:
        axes = [axes]
    for ax, threshold in zip(axes, thresholds):
        z = (permit_matched["animal_unit_estimate"] - threshold) / permit_matched["se"]
        p_ge = norm.cdf(z)
        colors = np.where(permit_matched["allowable_animal_units"] >= threshold,
                           "steelblue", "firebrick")
        ax.scatter(permit_matched["allowable_animal_units"], p_ge, c=colors, s=18, alpha=0.7)
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("WDNR-reported (true) animal units")
        ax.set_ylabel("P(our estimate $\\geq$ threshold)")
        ax.set_title(f"Threshold = {threshold} AU (n={len(permit_matched)})")
        xlim = (max(0, threshold - 1500), threshold + 1500)
        ax.set_xlim(xlim)
    fig.suptitle("Classification confidence near the permitting threshold\n"
                 "(permitted CAFOs only, using WDNR-reported ground-truth AU)")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_classification.svg")
    plt.close(fig)

    print(f"  Saved threshold_classification_{{facility_level,summary}}.csv + .svg")
    return facility_rows, summary_df


# ============================================================================
# Main entry point
# ============================================================================

def generate_uncertainty_propagation_analysis(data, subdirs):
    out = subdirs.get("uncertainty_propagation", Path("paper_results") / "08_uncertainty_propagation")
    os.makedirs(out, exist_ok=True)

    percentile_df = run_percentile_sweep(data, out)
    facility_df, summary_df = run_threshold_classification(data, out)

    print(f"\nSaved all outputs to {out}")
    return {"percentile_sweep": percentile_df, "classification_facility": facility_df,
            "classification_summary": summary_df}


def main():
    matplotlib.use("Agg")
    plt.ioff()

    import generate_paper_results as gpr
    paths, subdirs = gpr.setup_paths()
    subdirs["uncertainty_propagation"] = Path("paper_results") / "08_uncertainty_propagation"
    os.makedirs(subdirs["uncertainty_propagation"], exist_ok=True)

    print("Loading data (no snapmaps)...")
    data = gpr.load_all_data(paths, read_snapmaps=False)

    return generate_uncertainty_propagation_analysis(data, subdirs)


if __name__ == "__main__":
    main()
