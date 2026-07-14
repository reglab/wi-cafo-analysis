#!/usr/bin/env python3
"""
Space-parameter sensitivity analysis for WI-CAFO study.

Implements:
  1. One-at-a-time (OAT) mean/SD sweeps — local sensitivity / importance ranking
  2. Joint welfare scenario analysis — coherent low-/high-welfare parameter sets

Usage (standalone):
    python sensitivity_space_params.py
    python sensitivity_space_params.py --n-workers 8

Integration with generate_paper_results.py:
    import sensitivity_space_params as ssp
    ssp.generate_space_param_sensitivity(data, subdirs, n_workers=4)

Outputs → paper_results/06_space_param_sensitivity/:
  oat_sweep_results.csv
  oat_<param_key>.svg            (one per OAT sweep)
  oat_importance_ranking.svg
  welfare_scenario_results.csv
  welfare_scenario_params.csv
  welfare_scenario_comparison.svg
"""

import argparse
import copy
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config.config_params as cfg
import cluster_analysis_functions as caf
import estimate_animal_units as est_au
import create_figures as cf


# ── thread safety ────────────────────────────────────────────────────────────
# Shapely 2.0 releases the GIL for geometry ops → threads genuinely parallelize
# geopandas/numpy work. matplotlib is not thread-safe, so we serialize its
# calls with a module-level lock. _CSV_LOCK guards checkpoint file appends.
_MATPLOTLIB_LOCK = threading.Lock()
_CSV_LOCK = threading.Lock()


# ============================================================================
# Constants
# ============================================================================

# Bounds widened vs. original (milking cows 7–15 → 6–18; heifers_800_1200 12→13)
# so OAT sweeps through the extremes never pile mass at the truncation boundary.
BASELINE_TRUNCNORM_PARAMS = {
    "calves":           {"min_val": 3,   "max_val": 7,  "mean": 4,   "std": 0.4},
    "milking_cows":     {"min_val": 6,   "max_val": 18, "mean": 12,  "std": 1.2},
    "heifers_800_1200": {"min_val": 6.5, "max_val": 13, "mean": 9.5, "std": 0.95},
    "heifers_400_800":  {"min_val": 4,   "max_val": 8,  "mean": 6,   "std": 0.6},
    "beef_cattle":      {"min_val": 4,   "max_val": 8,  "mean": 5,   "std": 0.5},
}

# OAT sweeps — param_key format: "<param_type>_<animal_type>", split on first "_"
OAT_SWEEPS = [
    ("mean_milking_cows",     [6.5, 8, 10, 12, 14, 16, 18]),   # baseline 12
    ("mean_heifers_800_1200", [5.5, 7, 9.5, 11, 13]),          # baseline 9.5
    ("mean_heifers_400_800",  [3.5, 5, 6, 7, 9]),              # baseline 6
    ("mean_calves",           [2.5, 3.5, 4, 5, 6]),            # baseline 4 (expect ~flat)
    ("std_milking_cows",      [0.6, 1.2, 1.8, 2.4]),           # baseline 1.2
]

# Welfare scenarios — coherent shifts of mean, min, AND max across all stages.
# Milking-cow anchors from plan: low (min 6, μ 7.5, max 9), high (min 13, μ 16, max 20).
# Other stages scaled proportionally: low factor 7.5/12 ≈ 0.625, high factor 16/12 ≈ 1.333.
WELFARE_SCENARIOS = {
    "baseline": BASELINE_TRUNCNORM_PARAMS,
    "low_welfare": {
        # Crowded/overstocked housing → smaller si → more farms exceed threshold
        "calves":           {"min_val": 2,   "max_val": 4,  "mean": 2.5,  "std": 0.25},
        "milking_cows":     {"min_val": 6,   "max_val": 9,  "mean": 7.5,  "std": 0.75},
        "heifers_800_1200": {"min_val": 4,   "max_val": 8,  "mean": 6.0,  "std": 0.60},
        "heifers_400_800":  {"min_val": 2.5, "max_val": 5,  "mean": 3.75, "std": 0.38},
        "beef_cattle":      {"min_val": 2.5, "max_val": 5,  "mean": 3.1,  "std": 0.31},
    },
    "high_welfare": {
        # Spacious/bedded-pack housing → larger si → fewer farms exceed threshold
        # This is the conservative stress-test for false positives.
        "calves":           {"min_val": 4,   "max_val": 9,  "mean": 5.3,  "std": 0.53},
        "milking_cows":     {"min_val": 13,  "max_val": 20, "mean": 16,   "std": 1.5},
        "heifers_800_1200": {"min_val": 9,   "max_val": 17, "mean": 12.7, "std": 1.27},
        "heifers_400_800":  {"min_val": 5.3, "max_val": 11, "mean": 8.0,  "std": 0.80},
        "beef_cattle":      {"min_val": 5.3, "max_val": 11, "mean": 6.7,  "std": 0.67},
    },
}

KEY_METRICS = [
    "permit_recall",
    "ewg_precision",
    "unpermitted_potential_CAFOs_milk_license_estimate",
    "median_abs_discrep_pct",
    "n_within_uncertainty_bounds",
]

METRIC_LABELS = {
    "permit_recall":                                     "Permit recall",
    "ewg_precision":                                     "EWG precision",
    "unpermitted_potential_CAFOs_milk_license_estimate": "Unpermitted ≥1000 AU\n(milk-licensed)",
    "median_abs_discrep_pct":                            "Median |discrepancy| %",
    "n_within_uncertainty_bounds":                       "Fraction within\nuncertainty interval",
}


# ============================================================================
# Checkpoint helpers
# ============================================================================

def _append_row(row, csv_path, lock):
    """Thread-safe single-row append to a CSV checkpoint file."""
    with lock:
        df_row = pd.DataFrame([row])
        write_header = not csv_path.exists()
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)


# ============================================================================
# Core computation
# ============================================================================

def _run_full_params(scenario_name, truncnorm_params, data_dict):
    """Compute all key metrics for one complete set of truncnorm params.

    Parameters
    ----------
    scenario_name : str
        Label for this row (e.g. "baseline", "mean_milking_cows", "high_welfare").
    truncnorm_params : dict
        Full truncnorm params dict (deep-copied internally).
    data_dict : dict
        Keys: all_cf_clusters, four_band_clusters, WDNR_CAFOs, counties,
              ewg_afos, milk_producers.

    Returns
    -------
    dict of scalar metrics.
    """
    params = copy.deepcopy(truncnorm_params)

    # Fresh copies — never mutate original data
    all_cf = est_au.sample_and_calc_au(
        data_dict["all_cf_clusters"].copy(),
        include_area_uncertainty=False,
        truncnorm_params=params,
    )
    four_band = est_au.sample_and_calc_au(
        data_dict["four_band_clusters"].copy(),
        include_area_uncertainty=True,
        truncnorm_params=params,
    )

    # Permit match (dairy, satellite-summed) → discrepancy metrics
    permit_matched, err = caf.merge_clusters_permits(
        all_cf, data_dict["WDNR_CAFOs"],
        sum_satellite_counts=True, discrep_analysis=True, only_dairy=True,
    )

    # Facility-level P/R at AFO threshold = 500 AU
    # Serialized through a lock because matplotlib is not thread-safe
    with _MATPLOTLIB_LOCK:
        recall_df = cf.facility_level_pr_lowerbound_graph(
            four_band, "four_band",
            data_dict["WDNR_CAFOs"], data_dict["counties"], data_dict["ewg_afos"],
            AFO_lower_bound_threshes=np.arange(0, 1000, 50),
            AFO_lower_bound_logic="point estimate",
            save_path=None,
            suppress_show=True,
        )
        plt.close("all")

    recall_500 = recall_df[recall_df["AFO_lower_bound_thresh"] == 500]
    if len(recall_500) == 0:
        raise ValueError(f"[{scenario_name}] No AFO_lower_bound_thresh=500 row in recall_df")

    # Unpermitted clusters (any permit type, no satellite aggregation)
    all_perm, _ = caf.merge_clusters_permits(
        all_cf, data_dict["WDNR_CAFOs"],
        sum_satellite_counts=False, discrep_analysis=False, only_dairy=False,
    )
    unpermitted = all_cf[
        ~all_cf["polygon_indices"].isin(all_perm["polygon_indices"])
    ].copy()

    # Milk-producer match — mirrors generate_tables() in generate_paper_results.py
    milk_wi = data_dict["milk_producers"].to_crs(cfg.WI_EPSG)
    unperm_milk = unpermitted.sjoin_nearest(milk_wi, max_distance=500)
    if "geometry_left" in unperm_milk.columns:
        unperm_milk["geometry"] = unperm_milk["geometry_left"]
        unperm_milk.drop(
            ["geometry_left", "geometry_right"], axis=1, inplace=True, errors="ignore"
        )
        unperm_milk = gpd.GeoDataFrame(unperm_milk, geometry="geometry", crs=cfg.WI_EPSG)
    unperm_milk = unperm_milk[~unperm_milk.index.duplicated(keep="first")]

    def _count_ge_1000(df, col):
        return int((df[col] >= 1000).sum()) if col in df.columns else np.nan

    return {
        "scenario":                    scenario_name,
        "milking_cow_mean":            params["milking_cows"]["mean"],
        "milking_cow_min":             params["milking_cows"]["min_val"],
        "milking_cow_max":             params["milking_cows"]["max_val"],
        "median_abs_discrep":          err["median_abs_discrep"].values[0],
        "median_abs_discrep_pct":      err["median_abs_discrep_pct"].values[0],
        "n_within_uncertainty_bounds": err["n_within_uncertainty_bounds"].values[0],
        "ewg_large_recall":            recall_500["ewg_large_recall"].values[0],
        "permit_recall":               recall_500["permit_recall"].values[0],
        "ewg_precision":               recall_500["ewg_precision"].values[0],
        "unpermitted_potential_CAFOs_milk_license_estimate": _count_ge_1000(unperm_milk, "animal_unit_estimate"),
        "unpermitted_potential_CAFOs_milk_license_lower":    _count_ge_1000(unperm_milk, "animal_units_lower"),
        "unpermitted_potential_CAFOs_milk_license_upper":    _count_ge_1000(unperm_milk, "animal_units_upper"),
        "permitted_CAFOs_meeting_logic_estimate":            _count_ge_1000(permit_matched, "animal_unit_estimate"),
        "permitted_CAFOs_meeting_logic_lower":               _count_ge_1000(permit_matched, "animal_units_lower"),
        "permitted_CAFOs_meeting_logic_upper":               _count_ge_1000(permit_matched, "animal_units_upper"),
    }


def _run_one_param_value(param_to_change, value, truncnorm_params, data_dict):
    """OAT iteration: change one param key, leave all others at baseline.

    param_to_change: "<param_type>_<animal_type>", e.g. "mean_milking_cows"
    """
    param_type, animal_type = param_to_change.split("_", 1)
    params = copy.deepcopy(truncnorm_params)
    params[animal_type][param_type] = value
    row = _run_full_params(param_to_change, params, data_dict)
    row["param_changed"] = param_to_change
    row["value"] = value
    return row


# ============================================================================
# Welfare-based selection (differential mixing) analysis
# ============================================================================

def _unpermitted_au_estimates(truncnorm_params, data_dict):
    """Per-farm point-estimate AU for milk-license-matched unpermitted farms.

    The unpermitted set and the milk-license match are both spatial and do not
    depend on space params, so only the per-farm AU (and thus the ≥1000 crossing)
    changes across parameter sets. Returns a pd.Series indexed by the facility's
    native cluster index.
    """
    params = copy.deepcopy(truncnorm_params)
    all_cf = est_au.sample_and_calc_au(
        data_dict["all_cf_clusters"].copy(),
        include_area_uncertainty=False,
        truncnorm_params=params,
    )
    all_perm, _ = caf.merge_clusters_permits(
        all_cf, data_dict["WDNR_CAFOs"],
        sum_satellite_counts=False, discrep_analysis=False, only_dairy=False,
    )
    unpermitted = all_cf[
        ~all_cf["polygon_indices"].isin(all_perm["polygon_indices"])
    ].copy()
    milk_wi = data_dict["milk_producers"].to_crs(cfg.WI_EPSG)
    unperm_milk = unpermitted.sjoin_nearest(milk_wi, max_distance=500)
    if "geometry_left" in unperm_milk.columns:
        unperm_milk["geometry"] = unperm_milk["geometry_left"]
        unperm_milk.drop(
            ["geometry_left", "geometry_right"], axis=1, inplace=True, errors="ignore"
        )
        unperm_milk = gpd.GeoDataFrame(unperm_milk, geometry="geometry", crs=cfg.WI_EPSG)
    unperm_milk = unperm_milk[~unperm_milk.index.duplicated(keep="first")]
    return unperm_milk["animal_unit_estimate"]


def _run_welfare_mixing(baseline_params, high_welfare_params, data_dict, out_dir,
                        phi_grid=None):
    """Differential welfare-selection sensitivity.

    Holds permitted farms at baseline and shifts a fraction phi of *unpermitted*
    farms to high-welfare (spacious) housing. Because the unpermitted count
    depends only on the unpermitted set, the expected count under independent
    per-farm Bernoulli(phi) assignment is exact:
        E[count(phi)] = sum_f [(1-phi)*1{AU_base_f >= 1000} + phi*1{AU_high_f >= 1000}]
    Only two AU passes are needed (baseline + high-welfare).

    Reports only the estimated number of unpermitted potential CAFOs vs. phi.
    """
    if phi_grid is None:
        phi_grid = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0

    print("  Welfare mixing: computing unpermitted AU under baseline params...")
    au_base = _unpermitted_au_estimates(baseline_params, data_dict)
    print("  Welfare mixing: computing unpermitted AU under high-welfare params...")
    au_high = _unpermitted_au_estimates(high_welfare_params, data_dict)

    common = au_base.index.intersection(au_high.index)
    base_ge = (au_base.loc[common] >= 1000).astype(int)
    high_ge = (au_high.loc[common] >= 1000).astype(int)

    rows = []
    for phi in phi_grid:
        expected = float(((1 - phi) * base_ge + phi * high_ge).sum())
        rows.append({
            "phi_unpermitted_high_welfare": phi,
            "expected_unpermitted_cafos": round(expected, 1),
        })
        print(f"    phi={phi}: expected unpermitted CAFOs = {expected:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "welfare_mixing_results.csv", index=False)
    print(f"  Saved welfare_mixing_results.csv "
          f"(phi=0 → {rows[0]['expected_unpermitted_cafos']}, "
          f"phi=1 → {rows[-1]['expected_unpermitted_cafos']})")
    return df


# ============================================================================
# Parallel orchestration
# ============================================================================

def _run_oat_parallel(oat_sweeps, baseline_params, data_dict, n_workers, out_dir):
    """Fan-out all OAT iterations across threads. Returns sorted DataFrame.

    Checkpoints each row to oat_sweep_results.csv as it completes so a
    partial run can be resumed without re-running finished iterations.
    """
    checkpoint = out_dir / "oat_sweep_results.csv"

    # Resume: skip iterations already written to the checkpoint
    done = set()
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        done = set(zip(existing["param_changed"], existing["value"].astype(float)))
        if done:
            print(f"  Resuming OAT: {len(done)} iterations already cached, skipping.")

    n_total = sum(len(values) for _, values in oat_sweeps)
    tasks = [
        (param_key, value, baseline_params)
        for param_key, values in oat_sweeps
        for value in values
        if (param_key, float(value)) not in done
    ]
    n_run = len(tasks)
    n_params = len(oat_sweeps)
    print(f"  OAT: {n_run}/{n_total} iterations to run, {n_params} parameters "
          f"({n_workers} threads)")

    completed = len(done)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_one_param_value, pk, v, bp, data_dict): (pk, v)
            for pk, v, bp in tasks
        }
        for future in as_completed(futures):
            pk, v = futures[future]
            try:
                row = future.result()
                _append_row(row, checkpoint, _CSV_LOCK)
                completed += 1
                print(
                    f"    [{completed}/{n_total}] {pk}={v}: "
                    f"permit_recall={row['permit_recall']:.2f}, "
                    f"unperm_milk={row['unpermitted_potential_CAFOs_milk_license_estimate']}"
                )
            except Exception as e:
                print(f"    ERROR [{completed+1}/{n_total}] {pk}={v}: {e}")

    df = pd.read_csv(checkpoint)
    return df.sort_values(["param_changed", "value"]).reset_index(drop=True)


def _run_scenarios_parallel(welfare_scenarios, data_dict, n_workers, out_dir):
    """Fan-out welfare scenarios across threads. Returns DataFrame in scenario order.

    Checkpoints each row to welfare_scenario_results.csv as it completes.
    """
    checkpoint = out_dir / "welfare_scenario_results.csv"
    scenario_order = list(welfare_scenarios.keys())
    n_total = len(scenario_order)

    # Resume: skip scenarios already written to the checkpoint
    done = set()
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        done = set(existing["scenario"].tolist())
        if done:
            print(f"  Resuming welfare scenarios: {sorted(done)} already cached, skipping.")

    remaining = {name: params for name, params in welfare_scenarios.items() if name not in done}
    n_run = len(remaining)
    print(f"  Welfare scenarios: {scenario_order} — {n_run}/{n_total} to run "
          f"({min(n_workers, max(n_run, 1))} threads)")

    completed = len(done)
    with ThreadPoolExecutor(max_workers=min(n_workers, max(n_run, 1))) as pool:
        futures = {
            pool.submit(_run_full_params, name, params, data_dict): name
            for name, params in remaining.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                row = future.result()
                _append_row(row, checkpoint, _CSV_LOCK)
                completed += 1
                print(
                    f"    [{completed}/{n_total}] {name}: "
                    f"permit_recall={row['permit_recall']:.2f}, "
                    f"unperm_milk={row['unpermitted_potential_CAFOs_milk_license_estimate']}"
                )
            except Exception as e:
                print(f"    ERROR [{completed+1}/{n_total}] {name}: {e}")

    df = pd.read_csv(checkpoint)
    order_map = {s: i for i, s in enumerate(scenario_order)}
    df["_order"] = df["scenario"].map(order_map)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ============================================================================
# Figures
# ============================================================================

def _plot_oat_results(oat_df, out_dir):
    """Per-sweep line plots + combined importance ranking bar chart."""
    sweep_params = oat_df["param_changed"].unique()

    for param_key in sweep_params:
        sweep = oat_df[oat_df["param_changed"] == param_key].sort_values("value")
        n_metrics = len(KEY_METRICS)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))

        param_type, animal_type = param_key.split("_", 1)
        baseline_val = BASELINE_TRUNCNORM_PARAMS.get(animal_type, {}).get(param_type)

        for ax, metric in zip(axes, KEY_METRICS):
            ax.plot(sweep["value"], sweep[metric], marker="o", color="steelblue", lw=1.5)
            if baseline_val is not None:
                ax.axvline(
                    baseline_val, color="gray", linestyle="--", linewidth=1, label="baseline"
                )
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
            ax.set_xlabel(param_key.replace("_", " "), fontsize=8)

        axes[0].legend(fontsize=8)
        fig.suptitle(f"OAT: {param_key.replace('_', ' ')}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"oat_{param_key}.svg")
        plt.close(fig)

    # Importance ranking: % range of each metric across each sweep
    importance_rows = []
    for param_key in sweep_params:
        sweep = oat_df[oat_df["param_changed"] == param_key]
        param_type, animal_type = param_key.split("_", 1)
        baseline_val = BASELINE_TRUNCNORM_PARAMS.get(animal_type, {}).get(param_type)
        for metric in KEY_METRICS:
            rng = sweep[metric].max() - sweep[metric].min()
            if baseline_val is not None:
                # row closest to the baseline parameter value
                baseline_row = sweep.loc[(sweep["value"] - baseline_val).abs().idxmin()]
                denom = abs(baseline_row[metric])
            else:
                denom = sweep[metric].mean()
            importance_rows.append({
                "param": param_key,
                "metric": METRIC_LABELS.get(metric, metric),
                "pct_range": 100 * rng / denom if denom != 0 else np.nan,
            })

    imp_df = pd.DataFrame(importance_rows)
    pivot = imp_df.pivot(index="param", columns="metric", values="pct_range")
    # Enforce KEY_METRICS column order (using labels)
    ordered_labels = [METRIC_LABELS.get(m, m) for m in KEY_METRICS]
    pivot = pivot[[c for c in ordered_labels if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_ylabel("% range across sweep (relative to baseline value)")
    ax.set_title("OAT importance ranking")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [p.replace("_", " ") for p in pivot.index], rotation=25, ha="right", fontsize=9
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "oat_importance_ranking.svg")
    plt.close(fig)

    # Save importance table alongside
    imp_df.to_csv(out_dir / "oat_importance_ranking.csv", index=False)


def _plot_welfare_scenarios(welfare_df, out_dir):
    """Bar chart comparing baseline / low-welfare / high-welfare on key metrics."""
    scenario_order = [s for s in ["low_welfare", "baseline", "high_welfare"]
                      if s in welfare_df["scenario"].values]
    wdf = welfare_df.set_index("scenario").reindex(scenario_order)

    n_metrics = len(KEY_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5))
    colors = {
        "baseline":     "steelblue",
        "low_welfare":  "tomato",
        "high_welfare": "seagreen",
    }

    for ax, metric in zip(axes, KEY_METRICS):
        vals = wdf[metric]
        bars = ax.bar(
            range(len(vals)),
            vals,
            color=[colors.get(s, "gray") for s in vals.index],
            edgecolor="white",
        )
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(
            [s.replace("_", "\n") for s in vals.index], fontsize=8
        )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                label = f"{v:.2f}" if abs(v) < 10 else f"{int(round(v))}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    label,
                    ha="center", va="bottom", fontsize=8,
                )

    fig.suptitle(
        "Key metrics: low-welfare / baseline / high-welfare scenarios\n"
        "(high-welfare = spacious housing → conservative false-positive stress test)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "welfare_scenario_comparison.svg")
    plt.close(fig)


# ============================================================================
# Main entry point (callable from generate_paper_results)
# ============================================================================

def generate_space_param_sensitivity(
    data,
    subdirs,
    baseline_params=None,
    oat_sweeps=None,
    welfare_scenarios=None,
    n_workers=4,
):
    """Run space-parameter sensitivity analysis and save all outputs.

    Parameters
    ----------
    data : dict
        Standard data dict from load_all_data() in generate_paper_results.py.
    subdirs : dict
        Must include key "sensitivity" pointing to the output directory.
    baseline_params : dict, optional
        Baseline truncnorm params. Default: BASELINE_TRUNCNORM_PARAMS
        (widened milking-cow bounds vs. the original 7–15).
    oat_sweeps : list of (str, list) tuples, optional
        Default: OAT_SWEEPS.
    welfare_scenarios : dict, optional
        Default: WELFARE_SCENARIOS.
    n_workers : int
        Threads for parallel execution (default: 4).

    Returns
    -------
    dict with keys "oat" (DataFrame) and "welfare" (DataFrame).
    """
    out = subdirs["sensitivity"]
    os.makedirs(out, exist_ok=True)

    if baseline_params is None:
        baseline_params = BASELINE_TRUNCNORM_PARAMS
    if oat_sweeps is None:
        oat_sweeps = OAT_SWEEPS
    if welfare_scenarios is None:
        welfare_scenarios = WELFARE_SCENARIOS

    data_dict = {
        "all_cf_clusters":    data["all_cf_clusters"],
        "four_band_clusters": data["four_band_clusters"],
        "WDNR_CAFOs":         data["WDNR_CAFOs"],
        "counties":           data["counties"],
        "ewg_afos":           data["ewg_afos"],
        "milk_producers":     data["milk_producers"],
    }

    # ── 1. OAT sweeps ──────────────────────────────────────────────────────
    print("\n  [1/3] OAT sweeps...")
    oat_df = _run_oat_parallel(oat_sweeps, baseline_params, data_dict, n_workers, out)
    _plot_oat_results(oat_df, out)
    print(f"  Saved oat_sweep_results.csv + figures ({len(oat_df)} rows)")

    # ── 2. Welfare scenarios ────────────────────────────────────────────────
    print("\n  [2/3] Welfare scenarios...")
    welfare_df = _run_scenarios_parallel(welfare_scenarios, data_dict, n_workers, out)
    _plot_welfare_scenarios(welfare_df, out)

    # Params table — documents exactly what was run
    scenario_params = [
        {"scenario": name, "animal_type": animal, **d}
        for name, params in welfare_scenarios.items()
        for animal, d in params.items()
    ]
    pd.DataFrame(scenario_params).to_csv(out / "welfare_scenario_params.csv", index=False)

    print(f"\n  Welfare results:")
    display_cols = ["scenario"] + KEY_METRICS
    available = [c for c in display_cols if c in welfare_df.columns]
    print(welfare_df[available].to_string(index=False))

    # ── 3. Welfare-based selection (differential mixing) ─────────────────────
    print("\n  [3/3] Welfare-selection mixing...")
    high_welfare = welfare_scenarios.get("high_welfare")
    if high_welfare is None:
        print("    (skipped — no 'high_welfare' scenario defined)")
        mixing_df = None
    else:
        mixing_df = _run_welfare_mixing(baseline_params, high_welfare, data_dict, out)

    print(f"\n  Saved all outputs to {out}")

    return {"oat": oat_df, "welfare": welfare_df, "mixing": mixing_df}


# ============================================================================
# Standalone CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Space-parameter sensitivity analysis for WI-CAFO study",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Parallel threads (default: 4; Shapely 2.0 releases GIL for spatial ops)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Smoke-test mode: run 1 OAT iteration (mean_milking_cows=12) and "
            "baseline scenario only. Fast end-to-end check before the full grid."
        ),
    )
    parser.add_argument(
        "--mixing-only", action="store_true",
        help=(
            "Run only the welfare-selection mixing analysis (two AU passes: "
            "baseline + high-welfare). Fast; outputs welfare_mixing_results.csv."
        ),
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    plt.ioff()
    sns.set_theme(style="whitegrid")

    start = time.time()
    print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import generate_paper_results as gpr
    paths, subdirs = gpr.setup_paths()
    subdirs["sensitivity"] = Path("paper_results") / "06_space_param_sensitivity"
    os.makedirs(subdirs["sensitivity"], exist_ok=True)

    print("Loading data (no snapmaps)...")
    data = gpr.load_all_data(paths, read_snapmaps=False)

    if args.mixing_only:
        print("\n[MIXING-ONLY] Welfare-selection differential mixing analysis.")
        data_dict = {
            "all_cf_clusters":    data["all_cf_clusters"],
            "four_band_clusters": data["four_band_clusters"],
            "WDNR_CAFOs":         data["WDNR_CAFOs"],
            "counties":           data["counties"],
            "ewg_afos":           data["ewg_afos"],
            "milk_producers":     data["milk_producers"],
        }
        results = _run_welfare_mixing(
            BASELINE_TRUNCNORM_PARAMS,
            WELFARE_SCENARIOS["high_welfare"],
            data_dict,
            subdirs["sensitivity"],
        )
    elif args.test:
        print("\n[TEST MODE] Running 1 OAT value + baseline scenario only.")
        test_oat = [("mean_milking_cows", [12])]
        test_scenarios = {"baseline": BASELINE_TRUNCNORM_PARAMS}
        results = generate_space_param_sensitivity(
            data, subdirs,
            oat_sweeps=test_oat,
            welfare_scenarios=test_scenarios,
            n_workers=1,
        )
    else:
        results = generate_space_param_sensitivity(
            data, subdirs, n_workers=args.n_workers,
        )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Outputs: paper_results/06_space_param_sensitivity/")
    return results


if __name__ == "__main__":
    main()
