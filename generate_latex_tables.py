"""
Generates the 20 auto-updating LaTeX table snippets used by
revised_manuscript/main_reproducible.tex (each pulled in via \\input{}).

Consolidates the table-generator functions drafted by five parallel review
passes over paper_results/ this session. Every function reads its source
CSV(s)/JSON fresh and writes a complete `\\begin{table}...\\end{table}` (or
`\\begin{longtable}`) block to manuscript_inputs/<name>.tex (flat — no
subfolders, per the journal's submission requirement that all LaTeX inputs
sit at the project root), reproducing the manuscript's existing
caption/label/column structure with freshly-computed cell values.

Called from generate_paper_results.py's _run_pipeline() as the final step, so
every full pipeline run regenerates all 20 tables from whatever is currently
in paper_results/. manuscript_inputs/ is meant to be copy-pasted wholesale
into the Overleaf project root alongside main_reproducible.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config.config_params as cfg

GENERATED_TABLES_DIR = Path("manuscript_inputs")


def _out_dir(subdirs):
    GENERATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_TABLES_DIR


def _fmt_int(x):
    try:
        if pd.isna(x):
            return "N/A"
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)) or not str(x).strip():
        return "N/A"
    dt = pd.to_datetime(x)
    return dt.strftime("%b %Y")


# ============================================================================
# Batch 1: main-body unpermitted-estimates table, AU/space params, welfare
# sensitivity, threshold classification
# ============================================================================

def generate_table_unpermitted_estimates(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "unperm_universe_table.csv", index_col=0)
    rows = [
        ("Upper bound $\\geq$ 1000 AU", "animal_units_upper"),
        ("Point estimate $\\geq$ 1000 AU", "animal_unit_estimate"),
        ("Lower bound $\\geq$ 1000 AU", "animal_units_lower"),
    ]
    body_lines = []
    for label, row_key in rows:
        row = df.loc[row_key]
        n_unperm = int(round(row["Unpermitted Potential CAFOs"]))
        n_milk = int(round(row["Unpermitted Potential CAFOs with milk producer license"]))
        pct_permitted = row["% of permitted dairy CAFOs meeting this logic"]
        body_lines.append(f"         {label} & {n_unperm} & {n_milk} & {pct_permitted:.0f}\\% \\\\")
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\begin{{tabular}}{{|p{{0.35\\linewidth}}|p{{0.15\\linewidth}}|p{{0.15\\linewidth}}|p{{0.15\\linewidth}}|}}
    \\hline
       {{Decision rule}} & {{Estimated unpermitted CAFOs}} &
       {{Unpermitted CAFOs matching to milk license}} &
       {{Percentage of permitted CAFOs meeting rule}} \\\\
       \\hline
{body}
        \\hline
    \\end{{tabular}}
    \\caption{{Total number of unpermitted detections of different sizes, as
    determined by point estimates, lower uncertainty bounds, and upper
    uncertainty bounds. The third column does not enforce the milk license match
    condition on permitted CAFOs, which further reduces the numbers by 2 to 4
    percentage points.}}
    \\label{{unpermitted_estimates}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "unpermitted_estimates.tex").write_text(tex)


def generate_table_livestock_data(paths, subdirs):
    def fmt(x):
        f = float(x)
        return str(int(f)) if f == int(f) else f"{f:g}"

    stage_order = [
        ("calves", "Calves", cfg.CALVES_AU_FACTOR),
        ("milking_cows", "Milking Cows", cfg.MILKING_AND_DRY_COWS_AU_FACTOR),
        ("heifers_800_1200", "Heifers (800--1200 lbs)", cfg.HEIFERS_800_1200_AU_FACTOR),
        ("heifers_400_800", "Heifers (400--800 lbs)", cfg.HEIFERS_400_800_AU_FACTOR),
        ("beef_cattle", "Beef Cattle", cfg.STEERS_AU_FACTOR),
    ]
    body_lines = []
    for i, (key, label, u_i) in enumerate(stage_order):
        p = cfg.truncnorm_params[key]
        line = (
            f" {label}  & {fmt(u_i)} & {fmt(p['min_val'])} & {fmt(p['max_val'])} & "
            f"{fmt(p['mean'])} & {fmt(p['std'])} \\\\"
        )
        if i == len(stage_order) - 1:
            line += " [1ex]"
        body_lines.append(line)
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{||c c c c c c||}}
 \\hline
 Animal stage & $u_i$ & $\\text{{min}}_i$ ($\\text{{m}}^2$) & $\\text{{max}}_i$ ($\\text{{m}}^2$) & $\\mu_i$ ($\\text{{m}}^2$) & $\\sigma_i$ ($\\text{{m}}^2$) \\\\ [0.5ex]
 \\hline\\hline
{body}
 \\hline
 \\end{{tabular}}
\\caption{{Animal Unit and Space Distribution Parameters}}
\\label{{table:livestock_data}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "livestock_data.tex").write_text(tex)


def generate_table_welfare_sensitivity(paths, subdirs):
    csv_path = subdirs["sensitivity"] / "welfare_scenario_results.csv"
    df = pd.read_csv(csv_path).set_index("scenario")
    scenario_order = [
        ("low_welfare", "Low-welfare (crowded)"),
        ("baseline", "Baseline"),
        ("high_welfare", "High-welfare (spacious)"),
    ]
    estimate_col = "unpermitted_potential_CAFOs_milk_license_estimate"
    lower_col = "unpermitted_potential_CAFOs_milk_license_lower"
    upper_col = "unpermitted_potential_CAFOs_milk_license_upper"
    has_upper = upper_col in df.columns
    if not has_upper:
        print(
            "WARNING [generate_table_welfare_sensitivity]: "
            f"'{upper_col}' missing from {csv_path} (known stale-cache issue). "
            "Writing 'N/A' for Upper (97.5%) instead of fabricating a number."
        )

    def fmt_mu(x):
        f = float(x)
        return str(int(f)) if f == int(f) else f"{f:g}"

    rows = []
    for key, label in scenario_order:
        row = df.loc[key]
        mu = fmt_mu(row["milking_cow_mean"])
        lower = int(round(row[lower_col]))
        point = int(round(row[estimate_col]))
        upper = str(int(round(row[upper_col]))) if has_upper else "N/A"
        recall = f"{row['permit_recall']:.2f}"
        pad = " " * max(1, 24 - len(label))
        rows.append(f"    {label}{pad}& {mu} & {lower} & {point} & {upper} & {recall} \\\\")
    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\begin{{tabular}}{{|p{{0.28\\linewidth}}|c|c|c|c|c|}}
    \\hline
    Scenario & Milking-cow $\\mu$ ($\\text{{m}}^2$) &
    Lower (2.5\\%) & Point (mean) & Upper (97.5\\%) & Permit recall \\\\
    \\hline
{body}
    \\hline
    \\end{{tabular}}
    \\caption{{Unpermitted potential CAFOs (milk-license matched) under coherent
    low-, baseline-, and high-welfare space-per-animal assumptions, with the
    full 95\\% simulation interval (point-estimate rule). Permit recall is the
    share of currently permitted dairy CAFOs the rule recovers.}}
    \\label{{tab:welfare_sensitivity}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "welfare_sensitivity.tex").write_text(tex)


def generate_table_threshold_classification(paths, subdirs):
    uncertainty_dir = subdirs["uncertainty_propagation"]
    df = pd.read_csv(uncertainty_dir / "threshold_classification_summary.csv")
    band_order = [
        ("0-100", "0--100 AU"), ("100-250", "100--250 AU"),
        ("250-500", "250--500 AU"), ("500+", "500+ AU"),
    ]

    def fmt_cell(threshold, band_key):
        row = df[(df["threshold"] == threshold) & (df["distance_band"] == band_key)]
        if row.empty:
            return "N/A"
        r = row.iloc[0]
        return f"{int(r['n'])}, {round(r['accuracy'] * 100)}\\%"

    def overall_acc(threshold):
        row = df[(df["threshold"] == threshold) & (df["distance_band"] == "all")]
        return round(row.iloc[0]["accuracy"] * 100) if not row.empty else "N/A"

    row_1000 = " & ".join(fmt_cell(1000, k) for k, _ in band_order)
    row_2000 = " & ".join(fmt_cell(2000, k) for k, _ in band_order)
    acc_1000, acc_2000 = overall_acc(1000), overall_acc(2000)
    header_cols = " & ".join(label for _, label in band_order)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\begin{{tabular}}{{|p{{0.3\\linewidth}}|c|c|c|c|}}
    \\hline
    Distance from threshold & {header_cols} \\\\
    \\hline
    Threshold = 1{{,}}000 AU (n, accuracy) & {row_1000} \\\\
    Threshold = 2{{,}}000 AU (n, accuracy) & {row_2000} \\\\
    \\hline
    \\end{{tabular}}
    \\caption{{Classification accuracy (point estimate vs.\\ WDNR-reported ground
    truth) among the 300 permitted CAFOs, by distance from the classification
    threshold. Overall accuracy is {acc_1000}\\% at 1{{,}}000~AU and {acc_2000}\\% at 2{{,}}000~AU;
    the difference reflects how many permitted CAFOs happen to sit near each
    threshold, not a difference in estimator quality.}}
    \\label{{tab:threshold_classification}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "threshold_classification.tex").write_text(tex)


# ============================================================================
# Batch 2: 2024 permit matches, regional robustness (AU discrepancy, permit
# recall, EWG precision/recall)
# ============================================================================

def generate_table_permit_2024_matches(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "permit_2024_update_matches.csv")
    df["PermitteeName"] = df["PermitteeName"].astype(str).str.replace("’", "'", regex=False)
    body_lines = []
    for _, row in df.iterrows():
        est = _fmt_int(row["animal_unit_estimate"])
        lo = _fmt_int(row["animal_units_lower"])
        hi = _fmt_int(row["animal_units_upper"])
        approval_au = _fmt_int(row.get("Current found online at application approval time"))
        expansion_au = _fmt_int(row.get("Projected expansion (projected) at time of approval"))
        approval_date = _fmt_date(row.get("Date created of approved initial permit"))
        body_lines.append(
            f"    {row['PermitteeName']} & {est} ({lo}--{hi}) & {approval_au} & "
            f"{expansion_au} & {approval_date} \\\\"
        )
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\small
    \\begin{{tabular}}{{|p{{0.24\\linewidth}}|p{{0.22\\linewidth}}|p{{0.16\\linewidth}}|p{{0.16\\linewidth}}|p{{0.14\\linewidth}}|}}
    \\hline
    Facility & AU est.\\ (95\\% interval) & Approval-time AU & Stated expansion AU & Approval date \\\\
    \\hline
{body}
    \\hline
    \\end{{tabular}}
    \\caption{{Facilities newly matched to an updated (August 2024) WDNR permit
    shapefile, with approval-time and stated-expansion AU from the permit file
    where available.}}
    \\label{{table:permit_2024_matches}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "permit_2024_matches.tex").write_text(tex)


def generate_table_regional_au_discrepancy(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "regional_robustness_au_discrepancy.csv")
    formal = pd.read_csv(subdirs["tables"] / "regional_robustness_formal_tests.csv").set_index("check")
    df = df.copy()
    df["region_label"] = df["dnr_region"].str.replace(" Region", "", regex=False)
    df = df.sort_values("median_abs_pct_discrep")
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(
            f"{row['region_label']} & {int(row['n'])} & {row['median_abs_pct_discrep']:.1f}\\% & "
            f"{row['q25_abs_pct_discrep']:.1f}--{row['q75_abs_pct_discrep']:.1f}\\% \\\\"
        )
    body = "\n".join(body_lines)
    n_total = int(df["n"].sum())
    kw = formal.loc["au_discrepancy_kruskal_wallis"]
    ols = formal.loc["au_discrepancy_ols_f_test"]
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
WDNR Region & N & Median abs.\\ \\% discrepancy & IQR \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Median absolute percentage discrepancy between AU point estimates and
permit-reported AU, by WDNR administrative region (N={n_total}). Kruskal-Wallis
test: H={kw['stat']:.2f}, {int(kw['dof'])} d.f., p={kw['p']:.2f}.
OLS F-test for a region effect (controlling for permit size): p={ols['p']:.2f}.}}
\\label{{tab:regional_au_discrep}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "regional_au_discrepancy.tex").write_text(tex)


def generate_table_regional_permit_recall(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "regional_robustness_permit_recall.csv")
    formal = pd.read_csv(subdirs["tables"] / "regional_robustness_formal_tests.csv").set_index("check")
    df = df.copy()
    df["region_label"] = df["dnr_region"].str.replace(" Region", "", regex=False)
    df = df.sort_values("recall_pct")
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(
            f"{row['region_label']} & {int(row['n_permits'])} & {row['recall_pct']:.1f} & "
            f"[{row['wilson_ci_low_pct']:.1f}, {row['wilson_ci_high_pct']:.1f}] \\\\"
        )
    body = "\n".join(body_lines)
    n_total = int(df["n_permits"].sum())
    chi2 = formal.loc["permit_recall_chi2_homogeneity"]
    lr1 = formal.loc["permit_recall_logit_lr_region"]
    lr2 = formal.loc["permit_recall_logit_lr_region_size_controlled"]
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
WDNR Region & N permits & Recall (\\%) & 95\\% CI \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Recall of permitted dairy CAFOs by WDNR administrative region (N={n_total} of
300 dairy permits statewide; {300 - n_total} dropped for missing region code). Chi-square test
of homogeneity: $\\chi^2={chi2['chi2_stat']:.2f}$, {int(chi2['chi2_dof'])} d.f., p={chi2['chi2_p']:.2f}.
Logistic likelihood-ratio test for a region effect: p={lr1['logit_lr_p']:.2f} (p={lr2['logit_lr_p']:.2f} controlling for permit size).}}
\\label{{tab:regional_permit_recall}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "regional_permit_recall.tex").write_text(tex)


def generate_table_regional_ewg_precision_recall(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "regional_robustness_ewg_precision_recall.csv")
    formal = pd.read_csv(subdirs["tables"] / "regional_robustness_formal_tests.csv").set_index("check")
    df = df.copy()
    df["region_label"] = df["dnr_region"].str.replace(" Region", "", regex=False)
    df = df.sort_values("precision_pct")
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append(
            f"{row['region_label']} & {int(row['n_detections'])} & {row['precision_pct']:.1f} & "
            f"{int(row['n_ewg_large_dairy_afos'])} & {row['recall_pct']:.1f} \\\\"
        )
    body = "\n".join(body_lines)
    prec_chi2 = formal.loc["ewg_precision_chi2_homogeneity"]
    rec_chi2 = formal.loc["ewg_recall_chi2_homogeneity"]
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrrr}}
\\toprule
WDNR Region & N detections & Precision (\\%) & N large-dairy AFOs & Recall (\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Precision and recall against the EWG independent census, by WDNR
administrative region, restricted to the three regions overlapping EWG's
nine-county study area. Chi-square test of homogeneity: precision $\\chi^2={prec_chi2['stat']:.2f}$,
{int(prec_chi2['dof'])} d.f., p={prec_chi2['p']:.3f}; recall $\\chi^2={rec_chi2['stat']:.2f}$,
{int(rec_chi2['dof'])} d.f., p={rec_chi2['p']:.2f}.}}
\\label{{tab:regional_ewg_pr}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "regional_ewg_precision_recall.tex").write_text(tex)


# ============================================================================
# Batch 3: FP waterfalls, manual review summary, model performance
# ============================================================================

def generate_table_fp_ewg_waterfall(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "fp_ewg_waterfall.csv").set_index("step")
    label_map = {
        "0_total_candidate_fps": "Total candidates (unmatched to EWG dairy)",
        "1_min_size_gt500AU": "Minimum-size ($>$500 AU)",
        "2_urban_area_gt500m": "Urban-area proximity ($>$500m)",
        "3_parcel_keyword_exclude": "Parcel-owner keyword exclusion",
        "4_milk_license_rescue_500m": "Milk-license match (rescued)",
    }
    body_lines = []
    for i, (step, label) in enumerate(label_map.items()):
        row = df.loc[step]
        n_remaining = f"{int(row['n_remaining']):,}"
        n_removed = "---" if i == 0 else f"{int(row['n_removed']):,}"
        cum_pct = "---" if i == 0 else f"{row['cumulative_pct_resolved']:.1f}\\%"
        body_lines.append(f"{label} & {n_remaining} & {n_removed} & {cum_pct} \\\\")
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
Filtering step & Remaining & Removed & Cumulative \\% resolved \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{False-positive resolution waterfall, EWG study region. ``Rescued''
candidates are excluded from the false-positive tally because a nearby
licensed dairy suggests the detection is a real farm missing from EWG's
census, not a model error.}}
\\label{{t:fp_ewg_waterfall}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "fp_ewg_waterfall.tex").write_text(tex)


def generate_table_fp_cf_waterfall(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "fp_cf_unmatched_waterfall.csv").set_index("step")
    label_map = {
        "0_total_candidate_fps": "Total candidates (sent for review, unannotated)",
        "1_milk_license_rescue_500m": "Milk-license match (rescued)",
    }
    body_lines = []
    for i, (step, label) in enumerate(label_map.items()):
        row = df.loc[step]
        n_remaining = f"{int(row['n_remaining']):,}"
        n_removed = "---" if i == 0 else f"{int(row['n_removed']):,}"
        cum_pct = "---" if i == 0 else f"{row['cumulative_pct_resolved']:.1f}\\%"
        body_lines.append(f"{label} & {n_remaining} & {n_removed} & {cum_pct} \\\\")
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
Filtering step & Remaining & Removed & Cumulative \\% resolved \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{False-positive resolution waterfall, CloudFactory-reviewed
detections. Minimum-size, urban-proximity, and parcel-keyword filters are
not reapplied here, as this candidate set was already screened by the same
criteria prior to selection for human review.}}
\\label{{t:fp_cf_waterfall}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "fp_cf_waterfall.tex").write_text(tex)


def generate_table_fp_manual_review_summary(paths, subdirs):
    summary = pd.read_csv(subdirs["tables"] / "fp_manual_review_summary.csv").set_index("sample")
    ewg_cat = pd.read_csv(subdirs["tables"] / "fp_manual_review_ewg_categorized.csv")
    cf_cat = pd.read_csv(subdirs["tables"] / "fp_manual_review_cf_unmatched_categorized.csv")
    n_ewg_total, n_cf_total = len(ewg_cat), len(cf_cat)

    def row_for(sample, total):
        r = summary.loc[sample]
        n_rev = int(r["n_reviewed"])
        fp_pct = round(r["fp_rate"] * 100)
        ci_lo = round(r["fp_rate_wilson_ci_low"] * 100)
        ci_hi = round(r["fp_rate_wilson_ci_high"] * 100)
        return (
            f"{n_rev} of {total}", int(r["n_farm"]), int(r["n_false_positive"]),
            int(r["n_ambiguous"]), f"{fp_pct}\\% [{ci_lo}\\%, {ci_hi}\\%]"
        )

    ewg_row = row_for("ewg", n_ewg_total)
    cf_row = row_for("cf_unmatched", n_cf_total)
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Sample & Reviewed & Farm & False positive & Ambiguous & FP rate (95\\% CI) \\\\
\\midrule
EWG-region residual & {ewg_row[0]} & {ewg_row[1]} & {ewg_row[2]} & {ewg_row[3]} & {ewg_row[4]} \\\\
CF-reviewed residual & {cf_row[0]}\\textsuperscript{{$\\dagger$}} & {cf_row[1]} & {cf_row[2]} & {cf_row[3]} & {cf_row[4]} \\\\
\\bottomrule
\\end{{tabular}}
\\par\\textsuperscript{{$\\dagger$}}\\textit{{Random sample of the {int(summary.loc['cf_unmatched', 'extrapolated_residual_n']) if 'extrapolated_residual_n' in summary.columns else n_cf_total}-item residual.}}
\\caption{{Manual review outcomes for the unresolved false-positive
residuals. Confidence intervals are Wilson score intervals on the
false-positive rate.}}
\\label{{t:fp_manual_review_summary}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "fp_manual_review_summary.tex").write_text(tex)


def generate_table_model_performance(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "model_performance_table.csv")
    idx_col = df.columns[0]
    df = df.set_index(idx_col)
    row_map = [
        ("Four band", "Our model", ""),
        ("Inceptionv3", "\\citet{Handan-Nader:2019aa} (classification)", ""),
        ("Poultry", "\\citet{Robinson2022} (segmentation)", ""),
    ]
    body_lines = []
    for key, label, _ in row_map:
        r = df.loc[key]
        n = int(r["Number of predictions in the EWG study region"])
        recall = r["Recall of permitted dairy CAFOs outside EWG study region"]
        ewg_large_recall = r["Recall of large dairy farms inside the EWG study region"]
        ewg_precision = r["Precision over all farms inside the EWG study region"]
        body_lines.append(
            f"{label}\n & {recall:.2f} & {ewg_large_recall:.2f}\\textsuperscript{{$\\dagger$}} & "
            f"{ewg_precision:.2f}\\textsuperscript{{$\\dagger$}} (n={n:,}) \\\\[0.7cm]"
        )
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
 \\begin{{center}}
 \\begin{{tabularx}}{{0.95\\textwidth}}{{ | >{{\\RaggedRight}}X | >{{\\RaggedRight}}X | >{{\\RaggedRight}}X | >{{\\RaggedRight}}X |}}
 \\hline
 & Recall of permitted dairy CAFOs &
   Recall of large dairy farms in EWG study region &
   Precision over all dairy farms in EWG study region \\\\[1.8cm]
\\hline
{body}
\\hline
\\end{{tabularx}}
\\par \\textsuperscript{{$\\dagger$}}\\textit{{EWG study region used for training and
validation of three-band and four-band models.}}
 \\caption{{Facility-level recall and precision, where n represents the number of
 facilities, at our chosen threshold of keeping all detections above 500
 estimated AU. Recall of permitted dairy CAFOs statewide excludes satellite
 facilities. ``All dairy farms in EWG study region'' includes all facilities
 labeled by \\citet{{ewgWI}}, and ``large dairy farms in EWG study region'' includes
 the subset of these facilities which EWG labeled as large dairy operations.}}
 \\label{{t: model_metrics}}
 \\end{{center}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "model_performance.tex").write_text(tex)


# ============================================================================
# Batches 4+5: physical comparison, permit-status regression, spatial
# clustering summary + county breakdown, risk-param comparison, risk
# regressions (AU + area), online-validation summary
# ============================================================================

def generate_table_physical_comparison(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "summary_table_point_estimate.csv", index_col=0)
    cols = ["Permitted dairy CAFOs", "Unpermitted potential CAFOs", "AFOs between 500 and 1000 AU"]
    n = {c: int(df.loc["n_clusters", c]) for c in cols}

    def cell(row, c, se_row=None, decimals=0, comma=True):
        v = df.loc[row, c]
        fmt = f"{{:,.{decimals}f}}" if comma else f"{{:.{decimals}f}}"
        s = fmt.format(v)
        if se_row is not None:
            se = df.loc[se_row, c]
            se_s = fmt.format(se)
            return s, f"({se_s})"
        return s

    headers = " & ".join(
        f"{{\\makecell[c]{{{c.split()[0]} \\\\ {' '.join(c.split()[1:-1]) if len(c.split())>2 else c.split()[1]} \\\\ (n={n[c]:,})}}}}"
        for c in cols
    )
    rows_spec = [
        ("Mean estimated facility size ($m^2$)", "cluster_area_m2", "SE_area", 0),
        ("Mean estimated animal units", "animal_unit_estimate", "SE_animal_units", 0),
        ("Mean number of buildings", "n_buildings", "SE_n_buildings", 2),
        ("Mean number of parcels", "n_parcels", "SE_n_parcels", 2),
    ]
    body_lines = []
    for label, row, se_row, decimals in rows_spec:
        vals, ses = [], []
        for c in cols:
            v, se = cell(row, c, se_row, decimals)
            vals.append(v)
            ses.append(se)
        body_lines.append(f"      {label}  & {{{vals[0]}}} & {{{vals[1]}}} & {{{vals[2]}}} \\\\")
        body_lines.append(f"      & {{{ses[0]}}} & {{{ses[1]}}} & {{{ses[2]}}} \\\\[0.15cm]")
    milk = [f"{df.loc['matched_milk', c] * 100:.0f}" for c in cols]
    body_lines.append(f"      \\% with milk license & {milk[0]} & {milk[1]} & {milk[2]} \\\\[0.15cm]")
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
    \\begin{{center}}
    \\begin{{tabular}}{{|l|S[table-format=2.2]|S[table-format=2.2]|S[table-format=2.2]|}}
        \\hline
       & {headers} \\\\
       \\hline
{body}
        \\hline
    \\end{{tabular}}
    \\caption{{Comparison between facility clusters in different groups. Numbers
    in parentheses are standard errors.}}
    \\label{{tab:permitted_unpermitted_physical}}
    \\end{{center}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "physical_comparison.tex").write_text(tex)


def generate_table_permit_status_regression(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "permit_status_regression.csv")

    def stars(p):
        if p < 0.001:
            return "$^{***}$"
        if p < 0.01:
            return "$^{**}$"
        if p < 0.05:
            return "$^{*}$"
        return ""

    def fmt_or(v):
        if v >= 100:
            return f"{v:.0f}"
        if v >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    def or_cell(model, term):
        row = df[(df["model"] == model) & (df["term"] == term)]
        if row.empty:
            return "", ""
        r = row.iloc[0]
        star = stars(r["p_value"])
        return f"{fmt_or(r['odds_ratio'])}{star}", f"[{fmt_or(r['or_ci_low'])}, {fmt_or(r['or_ci_high'])}]"

    models = [
        "1. Size only", "2. + operational scale",
        "3. + region fixed effects", "4. + location / visibility",
    ]
    term_rows = [
        ("Size: log(animal units)", "log_au"),
        ("No.\\ of buildings", "n_buildings"),
        ("Holds state milk license", "milk_license"),
        ("log(distance to surface water)", "log_water_dist"),
        ("Shallow carbonate (karst) bedrock", "karst"),
        ("Shallow groundwater ($\\leq$0 ft)", "shallow_gw"),
        ("County CAFO density (per 1,000 km2)", "county_cafo_density"),
    ]
    body_lines = []
    for label, term in term_rows:
        cells = [or_cell(m, term) for m in models]
        if all(c[0] == "" for c in cells):
            continue
        or_line = " & ".join(c[0] if c[0] else "" for c in cells)
        ci_line = " & ".join(c[1] if c[1] else "" for c in cells)
        body_lines.append(f"        {label}          & {or_line} \\\\")
        body_lines.append(f"                                    & {ci_line} \\\\[0.1cm]")
    body = "\n".join(body_lines)
    n_688 = int(df[df["model"] == "1. Size only"]["n"].iloc[0])
    n_500 = int(df[df["model"].str.startswith("[>=500")]["n"].iloc[0]) if df["model"].str.startswith("[>=500").any() else None
    m5 = df[df["model"] == "5. + census (income, pop density)"]
    income_row = m5[m5["term"] == "log_median_income"]
    popdens_row = m5[m5["term"] == "county_pop_density"]
    census_sentence = ""
    if not income_row.empty and not popdens_row.empty:
        ir, pr = income_row.iloc[0], popdens_row.iloc[0]
        census_sentence = (
            f" A robustness specification adding county-level Census controls "
            f"(median household income, population density) leaves the core "
            f"coefficients essentially unchanged (income OR {fmt_or(ir['odds_ratio'])}, "
            f"p={ir['p_value']:.2f}; population density OR {fmt_or(pr['odds_ratio'])}, "
            f"p={pr['p_value']:.2f}; neither significant)."
        )
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\renewcommand{{\\arraystretch}}{{1.05}}
    \\begin{{tabular}}{{lcccc}}
        \\hline
         & (1) & (2) & (3) & (4) \\\\
         & Size only & \\makecell{{+ Operational}} & \\makecell{{+ Region FE}} & \\makecell{{+ Location}} \\\\
        \\hline
{body}
        \\hline
    \\end{{tabular}}
    \\caption{{Descriptive logistic regression of permit status on facility size
    and a set of observable confounders (odds ratios, 95\\% CI in brackets;
    $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$). N={n_688} in Models 1--4.{census_sentence}}}
    \\label{{tab:permit_status_regression}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "permit_status_regression.tex").write_text(tex)


def generate_table_spatial_clustering_summary(paths, subdirs):
    import json
    stats = json.loads((subdirs["spatial"] / "spatial_clustering_stats.json").read_text())
    jc = stats["join_count_moran"]["join_counts"]
    mi = stats["join_count_moran"]["moran_i"]
    go = stats["getis_ord_gi_star"]
    kf = stats["k_function_difference"]
    nn = stats["nn_cross_statistic"]
    tex = f"""\\begin{{table}}[!ht]
\\centering
\\begin{{tabular}}{{|p{{0.32\\linewidth}}|p{{0.22\\linewidth}}|p{{0.14\\linewidth}}|p{{0.24\\linewidth}}|}}
\\hline
\\textbf{{Test}} & \\textbf{{Statistic}} & \\textbf{{$p$-value}} & \\textbf{{Interpretation}} \\\\
\\hline
Moran's $I$ (permit status) & $I = {mi['I']:.3f}$ & {mi['p_sim']:.3f} & Weak, significant positive spatial autocorrelation \\\\
\\hline
Join count (mixed pairs) & {jc['BW_mixed']:,.0f} mixed pairs observed & {jc['p_sim_bw_fewer_than_expected']:.3f} & Significantly fewer permitted--unpermitted neighbor pairs than chance \\\\
\\hline
Getis-Ord $G_i^*$ & {go['pct_significant_hot_spots']:.1f}\\% hot spots, {go['pct_significant_cold_spots']:.1f}\\% cold spots & --- & Localized, significant pockets of each status \\\\
\\hline
$K$-function difference & $D(r) < 0$ for $r \\geq 8$km & {kf['mad_global_p_sim']:.3f} (global) & Permitted CAFOs more clustered among themselves at regional scale \\\\
\\hline
NN cross-statistic & {nn['observed_mean_nn_dist_m']:,.0f}m vs.\\ {nn['null_mean_nn_dist_m']:,.0f}m expected & {nn['p_sim_two_sided']:.3f} & No local-scale avoidance or attraction \\\\
\\hline
\\end{{tabular}}
\\caption{{Summary of formal spatial-clustering tests comparing permitted and
unpermitted potential CAFO locations. All tests use {stats['n_permutations']} Monte Carlo permutations.}}
\\label{{tab:spatial_stats_summary}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "spatial_clustering_summary.tex").write_text(tex)


def generate_table_spatial_county_breakdown(paths, subdirs, top_n=6):
    df = pd.read_csv(subdirs["spatial"] / "county_breakdown.csv")
    cold = df.sort_values("n_cold_perm", ascending=False).head(top_n)
    hot = df.sort_values("n_hot_unperm", ascending=False).head(top_n)
    body_lines = []
    for _, row in cold.iterrows():
        body_lines.append(
            f"{row['COUNTY_NAM']:<12} & {int(row['n_total'])} & {row['pct_permitted']:.0f}\\% & "
            f"Permitted-heavy cold spot (n={int(row['n_cold_perm'])}) \\\\"
        )
    body_lines.append("\\hline")
    for _, row in hot.iterrows():
        body_lines.append(
            f"{row['COUNTY_NAM']:<12} & {int(row['n_total'])} & {row['pct_permitted']:.0f}\\% & "
            f"Unpermitted-heavy hot spot (n={int(row['n_hot_unperm'])}) \\\\"
        )
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\begin{{tabular}}{{|l|c|c|c|}}
    \\hline
    \\textbf{{County}} & \\textbf{{Facilities}} & \\textbf{{\\% permitted}} & \\textbf{{Significant cluster}} \\\\
    \\hline
    {body}
    \\hline
    \\end{{tabular}}
    \\caption{{Counties with the largest concentrations of significant Getis-Ord
    hot/cold spots of permit status.}}
    \\label{{tab:spatial_county_breakdown}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "spatial_county_breakdown.tex").write_text(tex)


def generate_table_risk_param_comparison(paths, subdirs):
    import geopandas as gpd
    import scipy.stats as ss
    pii_path = subdirs["tables"] / "all_clusters_internal_PII.parquet"
    df = gpd.read_parquet(pii_path)
    PERM, UNP = "Permitted dairy CAFOs", "Unpermitted potential CAFOs"
    perm, unp = df[df["set"] == PERM], df[df["set"] == UNP]
    n_perm, n_unp = len(perm), len(unp)

    def welch_p(col):
        a = perm[col].dropna().astype(float).to_numpy()
        b = unp[col].dropna().astype(float).to_numpy()
        return ss.ttest_ind(a, b, equal_var=False).pvalue

    def means(col):
        return perm[col].mean(), unp[col].mean()

    rows = [
        ("Shallow soils", [
            ("Proportion with bedrock $<$5ft", "bedrock_lt5ft_dummy", "prop"),
            ("Proportion with shallow Silurian bedrock", "silurian_2_5_dummy", "prop"),
        ]),
        ("Distance to waterways", [
            ("Distance to intermittent streams", "hydro_intermit_distance", "dist"),
            ("Distance to perennial streams", "hydro_perennial_distance", "dist"),
            ("Distance to any water body", "water_distance", "dist"),
            ("Distance to NPS impaired water body", "impaired_water_distance", "dist"),
        ]),
        ("Risky zones for CAFOs", [
            ("Proportion with slope $>$12\\%", "slope_greater_12_dummy", "prop"),
            ("In Surface Water QMA (ponds/lakes)", "swqma_1000ft_dummy", "prop"),
            ("Distance to spreading restriction (water table)", "cafo_w_restrict_distance", "dist"),
            ("Distance to spreading restriction (bedrock depth)", "cafo_r_restrict_distance", "dist"),
        ]),
    ]

    def fmt_val(v, kind):
        return f"{v:.2f}" if kind == "prop" else f"{v:.1f}"

    body_lines = []
    for i, (category, metrics) in enumerate(rows):
        body_lines.append(f"\\multirow{{{len(metrics)}}}{{*}}{{{category}}}")
        for label, col, kind in metrics:
            mp, mu = means(col)
            p = welch_p(col)
            body_lines.append(f"& {label} & {fmt_val(mp, kind)} & {fmt_val(mu, kind)} & {p:.2f} \\\\")
        if i < len(rows) - 1:
            body_lines += ["", "\\midrule", ""]
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[htbp]
\\small
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{ll lll}}
\\toprule
\\textbf{{Category}} & \\textbf{{Metric}} &
\\textbf{{Perm (n={n_perm})}} &
\\textbf{{Unperm (n={n_unp})}} &
\\textbf{{p-value}} \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}
}}
\\caption{{Comparison between permitted dairy CAFOs and unpermitted potential dairy
CAFOs. Distances are in meters; proportions refer to the share of facilities in
the group. NPS = Nonpoint-source (e.g., pollutants from agricultural waste,
runoff, manure, etc.). Surface Water QMA's are designated Quality Management
Areas that are monitored.}}
\\label{{tab:permitted_unpermitted_comp}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "risk_param_comparison.tex").write_text(tex)


_MODEL_LABELS = [
    ("1. Size-only", "1. Size only (permit + log AU/area)"),
    ("2. Full controls", "2. Full controls (+ buildings, parcels, milk match)"),
    ("3a. Robustness: PCA risk index", "3a. Robustness: PCA-based index"),
    ("3b. Robustness: equal-weighted index", "3b. Robustness: equal-weighted index"),
]


def _regression_table_tex(df, label, caption, size_word):
    body_lines = []
    for prefix, disp_label in _MODEL_LABELS:
        row = df[df["model"].str.startswith(prefix)]
        if row.empty:
            continue
        r = row.iloc[0]
        disp_label = disp_label.replace("permit + log AU/area", f"permit + log {size_word}")
        body_lines.append(
            f"{disp_label} & {r['permitted_coef']:.4f} & {r['robust_se']:.4f} & "
            f"{r['p_value']:.3f} & {r['r_squared']:.4f} & {int(r['n'])} \\\\"
        )
    body = "\n".join(body_lines)
    return f"""\\begin{{table}}[htbp]
\\small
\\centering
\\sisetup{{
  round-mode = places,
  round-precision = 4,
}}
\\begin{{tabular}}{{l S[table-format=-1.4] S[table-format=1.4] S[table-format=1.3] S[table-format=1.4] S[table-format=4.0]}}
\\toprule
{{\\textbf{{Model}}}} & {{\\boldmath$\\beta$ \\textbf{{(permit)}}}} & {{\\textbf{{Robust SE}}}} & {{\\textbf{{p-value}}}} & {{\\boldmath$R^2$}} & {{\\textbf{{n}}}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def generate_table_risk_regression_au(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "risk_regression_summary.csv")
    draws = df[df["model"].str.startswith("3c.")].iloc[0]
    caption = (
        "Regression of environmental risk index on permit status. Model 1 "
        "controls only for facility size (log estimated AU). Model 2 adds facility "
        "structural complexity (buildings, parcels) and milk-license match status. "
        "Models 3a--3b repeat Model 2's specification substituting alternative risk "
        "index constructions (PCA-based; equal-weighted) for the dependent variable. "
        "Standard errors are heteroskedasticity-robust (HC1). A further robustness "
        "check re-estimating Model 2 under 500 random re-weightings of the hand-built "
        "index (Dirichlet draws) yields a mean coefficient of "
        f"{draws['permitted_coef']:.3f} (SD {draws['robust_se']:.3f}), with "
        f"{draws['pct_draws_same_sign']*100:.0f}\\% of draws sharing the sign of "
        "the published-weight estimate and "
        f"{draws['pct_draws_significant_p05']*100:.1f}\\% significant at $p<0.05$."
    )
    tex = _regression_table_tex(df, "tab:risk-regression", caption, "AU")
    (_out_dir(subdirs) / "risk_regression_au.tex").write_text(tex)


def generate_table_risk_regression_area(paths, subdirs):
    df = pd.read_csv(subdirs["tables"] / "risk_regression_summary_area.csv")
    draws = df[df["model"].str.startswith("3c.")].iloc[0]
    caption = (
        "Regression of environmental risk index on permit status, using "
        "directly measured cluster footprint area (log square meters) in place of "
        "estimated AU as the size control (cf.\\ Table~\\ref{tab:risk-regression}). "
        "Specifications and standard errors follow the same construction as "
        "Table~\\ref{tab:risk-regression}. A further robustness check re-estimating "
        "Model 2 under 500 random re-weightings of the hand-built index yields a mean "
        f"coefficient of {draws['permitted_coef']:.3f} (SD {draws['robust_se']:.3f}), with "
        f"{draws['pct_draws_same_sign']*100:.1f}\\% of draws sharing the sign of the "
        f"published-weight estimate and {draws['pct_draws_significant_p05']*100:.1f}\\% "
        "significant at $p<0.05$ --- both figures essentially unchanged from the AU-based specification."
    )
    tex = _regression_table_tex(df, "tab:risk-regression-area", caption, "footprint area")
    (_out_dir(subdirs) / "risk_regression_area.tex").write_text(tex)


def generate_table_online_validation(paths, subdirs):
    summary = pd.read_csv(subdirs["tables"] / "online_validation_summary.csv").set_index("milk_license_match_group")
    matched, unmatched = summary.loc["with_milk_license_match"], summary.loc["without_milk_license_match"]
    n_matched, n_unmatched = int(matched.sum()), int(unmatched.sum())

    def cell(count, total):
        count = int(count)
        if count == 0:
            return "0"
        return f"{count} ({round(count / total * 100)}\\%)"

    row_specs = [
        ("Confirmed active dairy", "confirmed_dairy"),
        ("Calf-rearing operation (not a milking herd)", "calf_rearing"),
        ("Unclear, but appears to be a cattle operation", "unclear_cattle"),
        ("Confirmed non-dairy (false positive)", "confirmed_non_dairy"),
    ]
    body_lines = [f"    Facilities reviewed & {n_matched} & {n_unmatched} \\\\"]
    for label, col in row_specs:
        body_lines.append(f"    {label} & {cell(matched[col], n_matched)} & {cell(unmatched[col], n_unmatched)} \\\\")
    body = "\n".join(body_lines)
    tex = f"""\\begin{{table}}[!ht]
    \\centering
    \\begin{{tabular}}{{|p{{0.45\\linewidth}}|p{{0.2\\linewidth}}|p{{0.2\\linewidth}}|}}
    \\hline
    & Milk-license matched & Not matched \\\\
    \\hline
{body}
    \\hline
    \\end{{tabular}}
    \\caption{{Manual review outcomes for 25 randomly sampled unpermitted
    potential CAFOs, stratified by milk-license match status. The two
    confirmed false positives were a hog-genetics facility and a
    storage/food-distribution site.}}
    \\label{{table:manual_validation_summary}}
\\end{{table}}
"""
    (_out_dir(subdirs) / "online_validation_summary.tex").write_text(tex)


ALL_TABLE_GENERATORS = [
    generate_table_unpermitted_estimates,
    generate_table_livestock_data,
    generate_table_welfare_sensitivity,
    generate_table_threshold_classification,
    generate_table_permit_2024_matches,
    generate_table_regional_au_discrepancy,
    generate_table_regional_permit_recall,
    generate_table_regional_ewg_precision_recall,
    generate_table_fp_ewg_waterfall,
    generate_table_fp_cf_waterfall,
    generate_table_fp_manual_review_summary,
    generate_table_model_performance,
    generate_table_physical_comparison,
    generate_table_permit_status_regression,
    generate_table_spatial_clustering_summary,
    generate_table_spatial_county_breakdown,
    generate_table_risk_param_comparison,
    generate_table_risk_regression_au,
    generate_table_risk_regression_area,
    generate_table_online_validation,
]


def generate_all_tables(paths, subdirs):
    """Regenerate all 20 auto-updating manuscript table .tex snippets.

    Each generator is independent and reads its own source CSV(s)/JSON fresh
    from paper_results/ — a failure in one (e.g. a missing upstream file)
    shouldn't prevent the others from regenerating.
    """
    print(f"\n  Regenerating {len(ALL_TABLE_GENERATORS)} manuscript tables -> "
          f"{GENERATED_TABLES_DIR}/")
    n_ok, n_failed = 0, 0
    for func in ALL_TABLE_GENERATORS:
        try:
            func(paths, subdirs)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            print(f"    WARNING: {func.__name__} failed: {e!r}")
    print(f"  {n_ok} tables regenerated, {n_failed} failed.")
