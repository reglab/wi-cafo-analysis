#!/usr/bin/env python3
"""Cross-regional robustness of the model validation stats (Reviewer comment
on the clustering appendix): "Thresholds including 500-meter distance, fuzzy
owner-name matching, and 25% building overlap are empirically defined;
cross-regional robustness requires further validation."

These three thresholds (cluster.py: same_name_distance_threshold,
fuzzy_threshold, overlap_factor) are what turn raw model polygon detections
into farm-level clusters, and their downstream effect shows up entirely in
the model's error rates against ground truth (WDNR permit recall, AU
discrepancy, EWG precision/recall). This module tests whether those error
rates diverge across Wisconsin's five WDNR administrative regions (the same
``dnr_region`` partition used in permit_status_regression.py) under the
thresholds as currently set. If they don't diverge, that is evidence the
thresholds generalize geographically rather than being overfit to wherever
they were tuned.

Two ground-truth sources have different regional coverage:
  - WDNR permits exist statewide (all 5 regions) -> permit recall and AU
    discrepancy are tested across all 5 regions, well powered (N~300 permits).
  - EWG's independent AFO census only covers 9 counties, which fall in 3 of
    the 5 regions (Northeast, South Central, West Central) -> EWG
    precision/recall by region is reported as a secondary, lower-power check
    restricted to those 3 regions.

Usage (standalone):
    python regional_robustness_analysis.py

Integration with generate_paper_results.py:
    import regional_robustness_analysis as rra
    rra.run_regional_robustness_analysis(data, permit_matched, subdirs)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

import config.config_params as cfg

# WDNR permit records store region as a short code; counties['DNR_REGION']
# (used throughout the rest of the pipeline, e.g. permit_status_regression.py)
# spells the same five regions out in full. This is the single mapping between
# the two representations.
DNR_REGION_CODE_MAP = {
    "NE": "Northeast Region",
    "WC": "West Central Region",
    "SC": "South Central Region",
    "SE": "Southeast Region",
    "NO": "Northern Region",
}

# The 3 of 5 DNR regions that overlap cfg.EWG_COUNTIES (EWG's independent AFO
# census only covers 9 counties, none in Northern or Southeast WI).
EWG_COVERED_REGIONS = ["Northeast Region", "South Central Region", "West Central Region"]

AU_RECALL_THRESH = 500      # paper's chosen operating AU threshold (point estimate)
MATCH_DISTANCE = 400        # metres; matches caf.merge_clusters_permits / facility_level_pr_lowerbound_graph


def _wilson_ci_pct(k, n):
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, method="wilson")
    return 100 * lo, 100 * hi


def _attach_dnr_region(gdf, counties):
    """Spatially attach the WDNR administrative region (5 units) to a
    GeoDataFrame via containment, falling back to nearest county for the
    handful of centroids that fall just outside a polygon boundary.

    Mirrors the county-join pattern in permit_status_regression.build_regression_frame.
    """
    counties_wi = counties.to_crs(cfg.WI_EPSG)[["DNR_REGION", "geometry"]]
    gdf_wi = gdf.to_crs(cfg.WI_EPSG)
    joined = gpd.sjoin(
        gdf_wi, counties_wi, how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")
    if joined["DNR_REGION"].isna().any():
        miss = joined["DNR_REGION"].isna()
        near = gpd.sjoin_nearest(
            gdf_wi[miss], counties_wi, how="left"
        ).drop(columns=["index_right"], errors="ignore")
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, "DNR_REGION"] = near["DNR_REGION"]
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined


def _match_flag(left, right, max_distance=MATCH_DISTANCE):
    """Boolean array (aligned to `left`'s index): does each `left` row have a
    `right` row within max_distance? Dedupes equidistant-tie duplicates.
    """
    m = left.sjoin_nearest(
        right[["geometry"]], max_distance=max_distance, how="left", distance_col="_d"
    )
    m = m[~m.index.duplicated(keep="first")]
    return m.loc[left.index, "_d"].notna().to_numpy()


# ============================================================================
# 1. WDNR permit recall by region (statewide, N~300)
# ============================================================================

def analyze_permit_recall_by_region(four_band_clusters, wdnr_permits,
                                     au_thresh=AU_RECALL_THRESH,
                                     match_distance=MATCH_DISTANCE):
    """Per-region permit recall + chi-square test + logistic LR test.

    Returns (region_table, tests_dict, permits_frame).
    """
    permits = wdnr_permits[
        (wdnr_permits["SATELLITE_"].isna()) & (wdnr_permits["AnimalType"] == "Dairy")
    ].copy()
    permits["dnr_region"] = permits["DNRRegion"].map(DNR_REGION_CODE_MAP)
    n_dropped = int(permits["dnr_region"].isna().sum())
    permits = permits.dropna(subset=["dnr_region"]).copy()

    detections = four_band_clusters[
        four_band_clusters["animal_unit_estimate"] > au_thresh
    ]
    permits["detected"] = _match_flag(permits, detections, match_distance).astype(int)

    region_rows = []
    for region, sub in permits.groupby("dnr_region"):
        k, n = int(sub["detected"].sum()), len(sub)
        lo, hi = _wilson_ci_pct(k, n)
        region_rows.append({
            "dnr_region": region, "n_permits": n, "n_detected": k,
            "recall_pct": 100 * k / n,
            "wilson_ci_low_pct": lo, "wilson_ci_high_pct": hi,
        })
    region_table = pd.DataFrame(region_rows).sort_values("dnr_region").reset_index(drop=True)

    ct = pd.crosstab(permits["dnr_region"], permits["detected"])
    chi2, chi2_p, chi2_dof, chi2_expected = ss.chi2_contingency(ct)
    min_expected = float(chi2_expected.min())

    # Logistic LR test: does adding region improve fit over an intercept-only model?
    m0 = smf.logit("detected ~ 1", data=permits).fit(disp=False)
    m_region = smf.logit("detected ~ C(dnr_region)", data=permits).fit(disp=False)
    lr_stat = 2 * (m_region.llf - m0.llf)
    lr_df = m_region.df_model - m0.df_model
    lr_p = ss.chi2.sf(lr_stat, df=lr_df)

    # Same LR test net of permit size (log AU), to rule out region acting only
    # as a proxy for "regions differ in typical farm size".
    permits["log_permit_au"] = np.log(
        permits["Number ofAnimalUnits"].replace(0, np.nan)
    )
    sub_size = permits.dropna(subset=["log_permit_au"])
    m0_size = smf.logit("detected ~ log_permit_au", data=sub_size).fit(disp=False)
    m_region_size = smf.logit(
        "detected ~ log_permit_au + C(dnr_region)", data=sub_size
    ).fit(disp=False)
    lr_stat_size = 2 * (m_region_size.llf - m0_size.llf)
    lr_df_size = m_region_size.df_model - m0_size.df_model
    lr_p_size = ss.chi2.sf(lr_stat_size, df=lr_df_size)

    tests = {
        "n_permits_used": len(permits),
        "n_permits_dropped_no_region": n_dropped,
        "chi2_stat": chi2, "chi2_dof": chi2_dof, "chi2_p": chi2_p,
        "chi2_min_expected_count": min_expected,
        "logit_lr_stat": lr_stat, "logit_lr_df": lr_df, "logit_lr_p": lr_p,
        "logit_lr_stat_size_controlled": lr_stat_size,
        "logit_lr_df_size_controlled": lr_df_size,
        "logit_lr_p_size_controlled": lr_p_size,
        "n_permits_size_controlled": len(sub_size),
    }
    return region_table, tests, permits


# ============================================================================
# 2. AU estimate discrepancy by region (statewide, N~300)
# ============================================================================

def analyze_au_discrepancy_by_region(permit_matched, counties):
    """Per-region AU discrepancy (permit vs. model estimate) + Kruskal-Wallis
    + OLS F-test.

    Returns (region_table, tests_dict, discrepancy_frame).
    """
    df = permit_matched.copy()
    if "DNRRegion" in df.columns and df["DNRRegion"].notna().any():
        df["dnr_region"] = df["DNRRegion"].map(DNR_REGION_CODE_MAP)
    else:
        df = _attach_dnr_region(df, counties)
        df["dnr_region"] = df["DNR_REGION"]

    df = df.dropna(subset=["dnr_region", "Number ofAnimalUnits", "discrep"]).copy()
    df = df[df["Number ofAnimalUnits"] > 0]
    df["abs_pct_discrep"] = 100 * (df["discrep"] / df["Number ofAnimalUnits"]).abs()

    region_rows = []
    groups = []
    for region, sub in df.groupby("dnr_region"):
        vals = sub["abs_pct_discrep"]
        region_rows.append({
            "dnr_region": region, "n": len(sub),
            "median_abs_pct_discrep": vals.median(),
            "mean_abs_pct_discrep": vals.mean(),
            "q25_abs_pct_discrep": vals.quantile(0.25),
            "q75_abs_pct_discrep": vals.quantile(0.75),
        })
        groups.append(vals.to_numpy())
    region_table = pd.DataFrame(region_rows).sort_values("dnr_region").reset_index(drop=True)

    kw_stat, kw_p = ss.kruskal(*groups)
    ols = smf.ols("abs_pct_discrep ~ C(dnr_region)", data=df).fit()

    tests = {
        "n_used": len(df),
        "kruskal_stat": kw_stat, "kruskal_dof": len(groups) - 1, "kruskal_p": kw_p,
        "ols_f_stat": ols.fvalue, "ols_f_p": ols.f_pvalue,
    }
    return region_table, tests, df


# ============================================================================
# 3. EWG precision/recall by region (secondary; 3 EWG-covered regions only)
# ============================================================================

def analyze_ewg_precision_recall_by_region(four_band_clusters, ewg_afos, counties,
                                            au_thresh=AU_RECALL_THRESH,
                                            match_distance=MATCH_DISTANCE):
    """Per-region EWG precision/recall, restricted to the 3 DNR regions that
    overlap EWG's 9-county ground-truth census. Secondary/lower-power check.

    Returns (region_table, tests_dict).
    """
    # "Large dairy" only, matching cf.facility_level_pr_lowerbound_graph's
    # ewg_large_dairy_afos — the paper's own headline recall metric at the
    # chosen 500 AU threshold. Using "any dairy size" here instead would
    # understate recall by counting small EWG dairy farms the model was never
    # meant to catch above the operating threshold.
    ewg_dairy = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Large")
    ].copy()

    detections = four_band_clusters[
        (four_band_clusters["ewg_region"] == True)
        & (four_band_clusters["animal_unit_estimate"] > au_thresh)
    ].copy()
    detections = _attach_dnr_region(detections, counties)
    detections = detections[detections["DNR_REGION"].isin(EWG_COVERED_REGIONS)]

    ewg_all = _attach_dnr_region(ewg_afos, counties)
    ewg_dairy = _attach_dnr_region(ewg_dairy, counties)
    ewg_dairy = ewg_dairy[ewg_dairy["DNR_REGION"].isin(EWG_COVERED_REGIONS)]

    detections["matched"] = _match_flag(detections, ewg_all, match_distance).astype(int)
    ewg_dairy["matched"] = _match_flag(ewg_dairy, detections, match_distance).astype(int)

    region_rows = []
    for region in EWG_COVERED_REGIONS:
        det_r = detections[detections["DNR_REGION"] == region]
        dairy_r = ewg_dairy[ewg_dairy["DNR_REGION"] == region]
        k_p, n_p = int(det_r["matched"].sum()), len(det_r)
        k_r, n_r = int(dairy_r["matched"].sum()), len(dairy_r)
        p_lo, p_hi = _wilson_ci_pct(k_p, n_p)
        r_lo, r_hi = _wilson_ci_pct(k_r, n_r)
        region_rows.append({
            "dnr_region": region,
            "n_detections": n_p, "n_matched_to_ewg": k_p,
            "precision_pct": 100 * k_p / n_p if n_p else np.nan,
            "precision_ci_low_pct": p_lo, "precision_ci_high_pct": p_hi,
            "n_ewg_large_dairy_afos": n_r, "n_matched_to_detection": k_r,
            "recall_pct": 100 * k_r / n_r if n_r else np.nan,
            "recall_ci_low_pct": r_lo, "recall_ci_high_pct": r_hi,
        })
    region_table = pd.DataFrame(region_rows)

    ct_p = pd.crosstab(detections["DNR_REGION"], detections["matched"])
    ct_r = pd.crosstab(ewg_dairy["DNR_REGION"], ewg_dairy["matched"])
    chi2_p_stat, chi2_p_p, chi2_p_dof, exp_p = ss.chi2_contingency(ct_p)
    chi2_r_stat, chi2_r_p, chi2_r_dof, exp_r = ss.chi2_contingency(ct_r)

    tests = {
        "precision_chi2_stat": chi2_p_stat, "precision_chi2_dof": chi2_p_dof,
        "precision_chi2_p": chi2_p_p, "precision_chi2_min_expected": float(exp_p.min()),
        "recall_chi2_stat": chi2_r_stat, "recall_chi2_dof": chi2_r_dof,
        "recall_chi2_p": chi2_r_p, "recall_chi2_min_expected": float(exp_r.min()),
        "note": "Secondary/lower-power check: EWG ground truth covers only "
                f"{len(EWG_COVERED_REGIONS)} of 5 DNR regions.",
    }
    return region_table, tests


# ============================================================================
# Figure
# ============================================================================

def _plot_regional_robustness(recall_table, discrep_table, ewg_table, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    order = recall_table.sort_values("recall_pct")["dnr_region"]
    rt = recall_table.set_index("dnr_region").loc[order]
    yerr = np.array([
        rt["recall_pct"] - rt["wilson_ci_low_pct"],
        rt["wilson_ci_high_pct"] - rt["recall_pct"],
    ])
    ax.barh(range(len(rt)), rt["recall_pct"], xerr=yerr, color="steelblue", capsize=3)
    ax.set_yticks(range(len(rt)))
    ax.set_yticklabels([r.replace(" Region", "") for r in rt.index], fontsize=9)
    ax.set_xlabel("WDNR permit recall (%)")
    ax.set_title("Permit recall by region\n(N in parens)", fontsize=10)
    for i, (region, row) in enumerate(rt.iterrows()):
        ax.text(row["recall_pct"] + 2, i, f"(n={int(row['n_permits'])})", va="center", fontsize=7.5)
    ax.set_xlim(0, 110)

    ax = axes[1]
    order2 = discrep_table.sort_values("median_abs_pct_discrep")["dnr_region"]
    dt = discrep_table.set_index("dnr_region").loc[order2]
    ax.barh(range(len(dt)), dt["median_abs_pct_discrep"], color="tomato")
    ax.set_yticks(range(len(dt)))
    ax.set_yticklabels([r.replace(" Region", "") for r in dt.index], fontsize=9)
    ax.set_xlabel("Median |AU discrepancy| (%)")
    ax.set_title("AU discrepancy by region\n(N in parens)", fontsize=10)
    for i, (region, row) in enumerate(dt.iterrows()):
        ax.text(row["median_abs_pct_discrep"] + 1, i, f"(n={int(row['n'])})", va="center", fontsize=7.5)

    ax = axes[2]
    et = ewg_table.set_index("dnr_region")
    x = np.arange(len(et))
    width = 0.35
    ax.bar(x - width / 2, et["precision_pct"], width, label="Precision", color="seagreen")
    ax.bar(x + width / 2, et["recall_pct"], width, label="Recall (large dairy)", color="goldenrod")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" Region", "") for r in et.index], fontsize=8, rotation=15)
    ax.set_ylabel("%")
    ax.set_title("EWG precision/recall by region\n(secondary; 3/5 regions have EWG coverage)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ============================================================================
# Orchestration
# ============================================================================

def run_regional_robustness_analysis(data, permit_matched, subdirs, save_fig=True):
    """Run all three regional-divergence checks and save tables + a summary.

    Parameters
    ----------
    data : dict
        Standard data dict from load_all_data() (needs four_band_clusters,
        WDNR_CAFOs, ewg_afos, counties).
    permit_matched : GeoDataFrame
        Output of generate_model_validation() / caf.merge_clusters_permits(
        sum_satellite_counts=True, discrep_analysis=True, only_dairy=True) —
        reused here so the AU discrepancy numbers match the rest of the paper.
    subdirs : dict
        Needs "tables" and "validation" keys.

    Returns dict with the three region tables + test dicts.
    """
    out_tables = subdirs["tables"]
    out_fig = subdirs["validation"]

    print("\n  [1/3] WDNR permit recall by region...")
    recall_table, recall_tests, _ = analyze_permit_recall_by_region(
        data["four_band_clusters"], data["WDNR_CAFOs"],
    )
    recall_table.to_csv(out_tables / "regional_robustness_permit_recall.csv", index=False)
    print(recall_table.to_string(index=False))

    print("\n  [2/3] AU estimate discrepancy by region...")
    discrep_table, discrep_tests, _ = analyze_au_discrepancy_by_region(
        permit_matched, data["counties"],
    )
    discrep_table.to_csv(out_tables / "regional_robustness_au_discrepancy.csv", index=False)
    print(discrep_table.to_string(index=False))

    print("\n  [3/3] EWG precision/recall by region (secondary, 3/5 regions)...")
    ewg_table, ewg_tests = analyze_ewg_precision_recall_by_region(
        data["four_band_clusters"], data["ewg_afos"], data["counties"],
    )
    ewg_table.to_csv(out_tables / "regional_robustness_ewg_precision_recall.csv", index=False)
    print(ewg_table.to_string(index=False))

    tests_df = pd.DataFrame([
        {"check": "permit_recall_chi2_homogeneity", **{
            k: v for k, v in recall_tests.items()
            if k in ("chi2_stat", "chi2_dof", "chi2_p", "chi2_min_expected_count")
        }},
        {"check": "permit_recall_logit_lr_region", **{
            k: v for k, v in recall_tests.items()
            if k in ("logit_lr_stat", "logit_lr_df", "logit_lr_p")
        }},
        {"check": "permit_recall_logit_lr_region_size_controlled", **{
            "logit_lr_stat": recall_tests["logit_lr_stat_size_controlled"],
            "logit_lr_df": recall_tests["logit_lr_df_size_controlled"],
            "logit_lr_p": recall_tests["logit_lr_p_size_controlled"],
        }},
        {"check": "au_discrepancy_kruskal_wallis", **{
            "stat": discrep_tests["kruskal_stat"], "dof": discrep_tests["kruskal_dof"],
            "p": discrep_tests["kruskal_p"],
        }},
        {"check": "au_discrepancy_ols_f_test", **{
            "stat": discrep_tests["ols_f_stat"], "p": discrep_tests["ols_f_p"],
        }},
        {"check": "ewg_precision_chi2_homogeneity", **{
            "stat": ewg_tests["precision_chi2_stat"], "dof": ewg_tests["precision_chi2_dof"],
            "p": ewg_tests["precision_chi2_p"],
            "min_expected": ewg_tests["precision_chi2_min_expected"],
        }},
        {"check": "ewg_recall_chi2_homogeneity", **{
            "stat": ewg_tests["recall_chi2_stat"], "dof": ewg_tests["recall_chi2_dof"],
            "p": ewg_tests["recall_chi2_p"],
            "min_expected": ewg_tests["recall_chi2_min_expected"],
        }},
    ])
    tests_df.to_csv(out_tables / "regional_robustness_formal_tests.csv", index=False)

    if save_fig:
        _plot_regional_robustness(
            recall_table, discrep_table, ewg_table,
            out_fig / "regional_robustness.svg",
        )

    def _p(p):
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"

    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("CROSS-REGIONAL ROBUSTNESS: does model error diverge across WDNR regions?")
    lines.append("(copy-paste-ready summary; full numbers in tables/regional_robustness_*.csv)")
    lines.append("=" * 78)

    lines.append(f"\nWDNR permit recall by region (N={recall_tests['n_permits_used']} dairy "
                 f"permits statewide, {recall_tests['n_permits_dropped_no_region']} dropped "
                 f"for missing region):")
    for _, r in recall_table.iterrows():
        lines.append(
            f"  {r['dnr_region']:<22} recall={r['recall_pct']:5.1f}% "
            f"[{r['wilson_ci_low_pct']:.1f}, {r['wilson_ci_high_pct']:.1f}] "
            f"(n={int(r['n_permits'])})"
        )
    lines.append(
        f"  Chi-square test of homogeneity across regions: "
        f"chi2={recall_tests['chi2_stat']:.2f}, dof={recall_tests['chi2_dof']}, "
        f"{_p(recall_tests['chi2_p'])} "
        f"(min expected cell count={recall_tests['chi2_min_expected_count']:.1f})"
    )
    lines.append(
        f"  Logistic LR test (region term): "
        f"LR={recall_tests['logit_lr_stat']:.2f}, df={recall_tests['logit_lr_df']}, "
        f"{_p(recall_tests['logit_lr_p'])}"
    )
    lines.append(
        f"  Same LR test, controlling for permit size (log AU): "
        f"LR={recall_tests['logit_lr_stat_size_controlled']:.2f}, "
        f"df={recall_tests['logit_lr_df_size_controlled']}, "
        f"{_p(recall_tests['logit_lr_p_size_controlled'])} "
        f"(n={recall_tests['n_permits_size_controlled']})"
    )

    lines.append(f"\nAU estimate discrepancy by region (N={discrep_tests['n_used']} matched "
                 "permits statewide):")
    for _, r in discrep_table.iterrows():
        lines.append(
            f"  {r['dnr_region']:<22} median |discrep|={r['median_abs_pct_discrep']:5.1f}% "
            f"(IQR {r['q25_abs_pct_discrep']:.1f}-{r['q75_abs_pct_discrep']:.1f}) "
            f"(n={int(r['n'])})"
        )
    lines.append(
        f"  Kruskal-Wallis test across regions: "
        f"H={discrep_tests['kruskal_stat']:.2f}, dof={discrep_tests['kruskal_dof']}, "
        f"{_p(discrep_tests['kruskal_p'])}"
    )
    lines.append(
        f"  OLS F-test (region dummies): "
        f"F={discrep_tests['ols_f_stat']:.2f}, {_p(discrep_tests['ols_f_p'])}"
    )

    lines.append(
        "\nEWG precision/recall by region (SECONDARY — EWG ground truth covers "
        f"only {len(EWG_COVERED_REGIONS)}/5 DNR regions, lower power):"
    )
    for _, r in ewg_table.iterrows():
        lines.append(
            f"  {r['dnr_region']:<22} "
            f"precision={r['precision_pct']:5.1f}% (n={int(r['n_detections'])})  "
            f"recall={r['recall_pct']:5.1f}% (n={int(r['n_ewg_large_dairy_afos'])})"
        )
    lines.append(
        f"  Precision chi-square: chi2={ewg_tests['precision_chi2_stat']:.2f}, "
        f"dof={ewg_tests['precision_chi2_dof']}, {_p(ewg_tests['precision_chi2_p'])} "
        f"(min expected={ewg_tests['precision_chi2_min_expected']:.1f})"
    )
    lines.append(
        f"  Recall chi-square: chi2={ewg_tests['recall_chi2_stat']:.2f}, "
        f"dof={ewg_tests['recall_chi2_dof']}, {_p(ewg_tests['recall_chi2_p'])} "
        f"(min expected={ewg_tests['recall_chi2_min_expected']:.1f})"
    )
    lines.append("=" * 78 + "\n")

    summary_text = "\n".join(lines)
    print(summary_text)
    (out_tables / "regional_robustness_summary.txt").write_text(summary_text)
    print(f"  Saved regional_robustness_*.csv + summary.txt to {out_tables}")
    if save_fig:
        print(f"  Saved regional_robustness.svg to {out_fig}")

    return {
        "permit_recall": (recall_table, recall_tests),
        "au_discrepancy": (discrep_table, discrep_tests),
        "ewg_precision_recall": (ewg_table, ewg_tests),
    }


# ============================================================================
# Standalone CLI
# ============================================================================

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()

    import generate_paper_results as gpr
    import cluster_analysis_functions as caf

    paths, subdirs = gpr.setup_paths()
    print("Loading data (no snapmaps)...")
    data = gpr.load_all_data(paths, read_snapmaps=False)

    permit_matched, _ = caf.merge_clusters_permits(
        data["all_cf_clusters"], data["WDNR_CAFOs"],
        sum_satellite_counts=True, discrep_analysis=True, only_dairy=True,
    )

    run_regional_robustness_analysis(data, permit_matched, subdirs)


if __name__ == "__main__":
    main()
