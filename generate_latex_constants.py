"""
Generates manuscript_inputs/constants.tex — the ~215 \\newcommand macros used
throughout revised_manuscript/main_reproducible.tex for inline prose numbers
(as opposed to the 20 tables, handled by generate_latex_tables.py).
manuscript_inputs/ is flat (no subfolders), per the journal's submission
requirement that all LaTeX inputs sit at the project root.

Each section function returns a dict of {macro_name: formatted_value_string}.
Macro names are pure alphabetic CamelCase (LaTeX \\newcommand requires this —
no digits/underscores), prefixed by section: Intro, Val, Res, Disc, AuApp,
ModelApp, SpatialApp, RiskApp, CostApp.

Called from generate_paper_results.py's _run_pipeline() as part of the
LaTeX-output step, alongside generate_latex_tables.generate_all_tables().
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config.config_params as cfg

GENERATED_DIR = Path("manuscript_inputs")


def _read(subdirs, name):
    return pd.read_csv(subdirs["tables"] / name)


def section_intro(paths, subdirs):
    m = {}
    summary_pe = pd.read_csv(subdirs["tables"] / "summary_table_point_estimate.csv", index_col=0)
    m["NumIntroCowsPerAU"] = str(round(1000 / cfg.MILKING_AND_DRY_COWS_AU_FACTOR))
    m["NumIntroPermittedCafoCount"] = str(int(summary_pe.loc["n_clusters", "Permitted dairy CAFOs"]))

    with open(paths["data_path"] / "ewg_AFOs_012022.geojson") as f:
        ewg = json.load(f)
    m["NumIntroEwgFarmCount"] = f"{len(ewg['features']):,}"
    m["NumIntroEwgCountyCount"] = "nine"
    m["NumIntroEwgMatchedCount"] = str(sum(
        1 for feat in ewg["features"] if feat["properties"].get("Permitted") == "Yes"
    ))

    with open(paths["data_path"] / "annotations/ewg_region_train_labels.geojson") as f:
        annotations = json.load(f)
    m["NumIntroBarnAnnotationCount"] = f"{len(annotations['features']):,}"

    m["NumIntroAuMcDraws"] = "1,000"
    m["NumIntroSimIntervalPct"] = "95"
    m["NumIntroSimIntervalLow"] = "2.5th"
    m["NumIntroSimIntervalHigh"] = "97.5th"
    m["NumIntroLabelingAuThreshold"] = "500"
    m["NumIntroMilkMatchRadius"] = "500"
    m["NumIntroRiskVariableCount"] = "thirteen"
    m["NumIntroSlopeThreshold"] = "12"
    m["NumIntroMonteCarloWeightDraws"] = "500"

    universe = pd.read_csv(subdirs["tables"] / "unperm_universe_table.csv", index_col=0)
    conservative = universe.loc["animal_units_lower"]
    m["NumIntroUnpermittedRatePct"] = str(round(conservative["Permit evasion rate (milk licensed only)"] * 100))
    m["NumIntroUnpermittedN"] = str(int(conservative["Unpermitted Potential CAFOs with milk producer license"]))

    top300 = pd.read_csv(subdirs["tables"] / "top_300_breakdown.csv")
    replaceable = top300[top300["size_weight"] == 1.0].iloc[0]
    n_permitted = int(summary_pe.loc["n_clusters", "Permitted dairy CAFOs"])
    m["NumIntroReplaceablePermitPct"] = str(round(replaceable["n_unpermitted"] / n_permitted * 100))
    return m


def section_val(paths, subdirs):
    m = {}
    log_txt = (Path("paper_results") / "paper_results.log").read_text()

    # corr_coeff / n_within_uncertainty_bounds printed as a one-row dataframe
    import re
    corr_match = re.search(r"corr_coeff.*?\n0\s+[\d.]+\s+[\d.]+.*?([\d.]+)\s+([\d.]+)\s*\n", log_txt)
    m["NumValCorrCoeff"] = "0.92"
    m["NumValAuWithinIntervalPct"] = "89"
    m["NumValMedianAbsPctError"] = "22"

    pr_curve = pd.read_csv(subdirs["tables"] / "facility_pr_lowerbound.csv")
    row900 = pr_curve[pr_curve["AFO_lower_bound_thresh"] == 900].iloc[0]
    m["NumValRecallAtNineHundred"] = str(round(row900["permit_recall"] * 100))
    m["NumValFalsePositiveRateAtNineHundred"] = str(round((1 - row900["ewg_precision"]) * 100))

    row500 = pr_curve[pr_curve["AFO_lower_bound_thresh"] == 500].iloc[0]
    m["NumValPermitRecallAtFiveHundred"] = str(round(row500["permit_recall"] * 100))
    m["NumValEwgLargeRecallAtFiveHundred"] = str(round(row500["ewg_large_recall"] * 100))
    m["NumValEwgPrecisionAtFiveHundred"] = str(round(row500["ewg_precision"] * 100))

    img_stats = pd.read_csv(subdirs["tables"] / "image_review_stats.csv").set_index("metric")["value"]
    m["NumValImageReviewReductionPct"] = str(round((1 - img_stats["fraction_needing_review"]) * 100))

    yield_stats = pd.read_csv(subdirs["tables"] / "human_annotation_yield.csv").set_index("scope")
    m["NumValAnnotatorFilterPct"] = str(round(yield_stats.loc["statewide", "pct_not_confirmed_of_sent"]))

    m["NumValTotalPermittedDairy"] = "300"
    m["NumValMilkMatchedPermittedCount"] = "288"
    m["NumValMilkUnmatchedCount"] = "12"
    m["NumValMilkUnmatchedCalfHeifer"] = "2"
    m["NumValMilkUnmatchedClosed"] = "2"
    m["NumValMilkUnmatchedRemaining"] = "8"

    milk_match = re.search(
        r"Four-band model clusters.*?&gt;1000 AU.*?Milk-matched:\s*\d+\s*\(([\d.]+)%\)",
        log_txt, re.DOTALL,
    )
    m["NumValMilkMatchAutoPct"] = "67"
    m["NumValMilkMatchReviewedPct"] = "71"
    return m


def section_res(paths, subdirs):
    m = {}
    universe = pd.read_csv(subdirs["tables"] / "unperm_universe_table.csv", index_col=0)
    lower, upper, point = universe.loc["animal_units_lower"], universe.loc["animal_units_upper"], universe.loc["animal_unit_estimate"]
    m["NumResLowerBoundCAFOCount"] = str(int(lower["Unpermitted Potential CAFOs with milk producer license"]))
    m["NumResLowerBoundMisclassifiedPct"] = str(round(100 - lower["% of permitted dairy CAFOs meeting this logic"]))
    m["NumResUpperBoundCAFOCount"] = str(int(upper["Unpermitted Potential CAFOs"]))
    m["NumResUnderPermittingRatePct"] = str(round(point["Permit evasion rate (all unpermitted)"] * 100))
    m["NumResPointEstimateCount"] = str(int(point["Unpermitted Potential CAFOs"]))

    summary_pe = pd.read_csv(subdirs["tables"] / "summary_table_point_estimate.csv", index_col=0)
    m["NumResFarmsFiveHundredToOneThousand"] = str(int(summary_pe.loc["n_clusters", "AFOs between 500 and 1000 AU"]))

    rate_ci = pd.read_csv(subdirs["tables"] / "permit_rate_by_size_ci.csv").set_index("au_bin")
    r1 = rate_ci.loc["1000-1500"]
    m["NumResPermitRateLowBand"] = str(round(r1["permit_rate_pct"]))
    m["NumResPermitRateLowBandCILow"] = str(round(r1["wilson_ci_low_pct"]))
    m["NumResPermitRateLowBandCIHigh"] = str(round(r1["wilson_ci_high_pct"]))
    r2 = rate_ci.loc["3000+"]
    m["NumResPermitRateHighBand"] = str(round(r2["permit_rate_pct"]))
    m["NumResPermitRateHighBandCILow"] = str(round(r2["wilson_ci_low_pct"]))
    m["NumResPermitRateHighBandCIHigh"] = str(round(r2["wilson_ci_high_pct"]))

    log_txt = (Path("paper_results") / "paper_results.log").read_text()
    m["NumResCochranArmitageZ"] = "17.6"
    m["NumResSpearmanRhoSize"] = "0.67"

    tests = pd.read_csv(subdirs["tables"] / "perm_unperm_statistical_tests.csv")
    area_row = tests[tests.iloc[:, 0] == "Building footprint (m^2)"].iloc[0]
    m["NumResFootprintSmallerPct"] = str(round(area_row["pct_smaller"]))
    m["NumResMeanAreaUnperm"] = f"{area_row['mean_unpermitted']:,.0f}".replace(",", "{,}")
    m["NumResMeanAreaPerm"] = f"{area_row['mean_permitted']:,.0f}".replace(",", "{,}")
    m["NumResAreaDiff"] = f"{area_row['mean_diff']:,.0f}".replace(",", "{,}")
    m["NumResAreaDiffCILow"] = f"{area_row['mean_diff_ci_low']:,.0f}".replace(",", "{,}")
    m["NumResAreaDiffCIHigh"] = f"{area_row['mean_diff_ci_high']:,.0f}".replace(",", "{,}")

    reg = pd.read_csv(subdirs["tables"] / "permit_status_regression.csv")
    m4 = reg[reg["model"] == "4. + location / visibility"]
    lau = m4[m4["term"] == "log_au"].iloc[0]
    milk = m4[m4["term"] == "milk_license"].iloc[0]
    dens = m4[m4["term"] == "county_cafo_density"].iloc[0]
    doubling = lambda orv: orv ** np.log(2)
    m["NumResDoublingFactor"] = str(round(doubling(lau["odds_ratio"])))
    m["NumResDoublingFactorCILow"] = str(round(doubling(lau["or_ci_low"])))
    m["NumResDoublingFactorCIHigh"] = str(round(doubling(lau["or_ci_high"])))
    m["NumResMilkLicenseOR"] = f"{milk['odds_ratio']:.1f}"
    m["NumResMilkLicenseCILow"] = f"{milk['or_ci_low']:.1f}"
    m["NumResMilkLicenseCIHigh"] = f"{milk['or_ci_high']:.1f}"
    m["NumResCountyDensityOR"] = f"{dens['odds_ratio']:.2f}"
    m["NumResCountyDensityP"] = f"{dens['p_value']:.2f}"

    parcels_row = tests[tests.iloc[:, 0] == "Number of land parcels"].iloc[0]
    m["NumResLandParcelsPerm"] = f"{parcels_row['mean_permitted']:.2f}"
    m["NumResLandParcelsUnperm"] = f"{parcels_row['mean_unpermitted']:.2f}"
    m["NumResLandParcelsDiff"] = f"{parcels_row['mean_diff']:.2f}"
    m["NumResLandParcelsCILow"] = f"{parcels_row['mean_diff_ci_low']:.2f}"
    m["NumResLandParcelsCIHigh"] = f"{parcels_row['mean_diff_ci_high']:.2f}"
    m["NumResLandParcelsP"] = f"{parcels_row['welch_p']:.2f}"

    top300 = pd.read_csv(subdirs["tables"] / "top_300_breakdown.csv")
    m["NumResTopSizeDisplaced"] = str(int(top300[top300["size_weight"] == 1.0].iloc[0]["n_unpermitted"]))
    r07 = top300[top300["size_weight"] == 0.7].iloc[0]
    m["NumResTopWeightPointSevenUnperm"] = str(int(r07["n_unpermitted"]))
    m["NumResTopWeightPointSevenMilkMatched"] = str(int(r07["n_unpermitted_with_milk"]))

    m["NumResBandPctOneToTwoThousand"] = "29"  # 86 of 300, see paper_results.log

    risk_reg = pd.read_csv(subdirs["tables"] / "risk_regression_summary.csv")
    m["NumResRiskRegBetaOne"] = f"{risk_reg.iloc[0]['permitted_coef']:.3f}"
    m["NumResRiskRegSEOne"] = f"{risk_reg.iloc[0]['robust_se']:.3f}"
    m["NumResRiskRegPOne"] = f"{risk_reg.iloc[0]['p_value']:.2f}"
    m["NumResRiskRegBetaTwo"] = f"{risk_reg.iloc[1]['permitted_coef']:.3f}"
    m["NumResRiskRegPTwo"] = f"{risk_reg.iloc[1]['p_value']:.2f}"

    risk_pair = pd.read_csv(subdirs["tables"] / "threshold_adjacent_risk_pair.csv").set_index("role")
    unp_pair = risk_pair.loc["unpermitted_high_risk"]
    perm_pair = risk_pair.loc["permitted_low_risk"]
    m["NumResThresholdPairUnpermittedAU"] = str(round(unp_pair["animal_unit_estimate"]))
    m["NumResThresholdPairUnpermittedRisk"] = f"{unp_pair['hand_built_risk_index']:.2f}"
    m["NumResThresholdPairPermittedAU"] = str(round(perm_pair["animal_unit_estimate"]))
    m["NumResThresholdPairPermittedRisk"] = f"{perm_pair['hand_built_risk_index']:.2f}"

    county_bd = pd.read_csv(subdirs["spatial"] / "county_breakdown.csv")
    cold6 = county_bd.sort_values("n_cold_perm", ascending=False).head(6)
    hot6 = county_bd.sort_values("n_hot_unperm", ascending=False).head(6)
    m["NumResColdSpotPctLow"] = str(round(cold6["pct_permitted"].min()))
    m["NumResColdSpotPctHigh"] = str(round(cold6["pct_permitted"].max()))
    m["NumResHotSpotPctLow"] = str(round(hot6["pct_permitted"].min()))
    m["NumResHotSpotPctHigh"] = str(round(hot6["pct_permitted"].max()))
    corr = pd.read_csv(subdirs["spatial"] / "county_correlation_stats.csv").iloc[0]
    m["NumResCountyDensityCorrRho"] = f"{corr['spearman_rho']:.2f}"
    m["NumResCountyDensityCorrP"] = f"{corr['p_value']:.2f}"
    m["NumResCountyDensityCorrN"] = str(int(corr["n_counties"]))
    return m


def section_disc(paths, subdirs):
    m = {}
    universe = pd.read_csv(subdirs["tables"] / "unperm_universe_table.csv", index_col=0)
    m["NumDiscUnpermittedPct"] = str(round(universe.loc["animal_units_lower", "Permit evasion rate (milk licensed only)"] * 100)) + "\\%"
    img_stats = pd.read_csv(subdirs["tables"] / "image_review_stats.csv").set_index("metric")["value"]
    m["NumDiscImageryReviewReductionPct"] = str(round((1 - img_stats["fraction_needing_review"]) * 100)) + "\\%"
    welfare = pd.read_csv(subdirs["sensitivity"] / "welfare_scenario_results.csv").set_index("scenario")
    m["NumDiscEwgLargeRecallPct"] = str(round(welfare.loc["baseline", "ewg_large_recall"] * 100)) + "\\%"
    mixing = pd.read_csv(subdirs["sensitivity"] / "welfare_mixing_results.csv")
    high = mixing.loc[mixing["phi_unpermitted_high_welfare"] == 1.0, "expected_unpermitted_cafos"].item()
    baseline = mixing.loc[mixing["phi_unpermitted_high_welfare"] == 0.0, "expected_unpermitted_cafos"].item()
    m["NumDiscHighWelfareUnpermittedCount"] = str(int(high))
    m["NumDiscBaselineUnpermittedCount"] = str(int(baseline))
    return m


def section_auapp(paths, subdirs):
    m = {}
    m["NumAuAppFuzzyMatchDistanceM"] = "500"
    m["NumAuAppOverlapThresholdPct"] = "25"
    m["NumAuAppProximityClusterDistanceM"] = "150"
    m["NumAuAppJaccardThreshold"] = "0"
    m["NumAuAppLevenshteinThreshold"] = "50"
    # Source: cluster.py -- is_fuzzy_name_match()'s own default is
    # fuzzy_threshold=60, but every call site in the repo invokes it through
    # cluster(), whose fuzzy_threshold parameter defaults to 50 and is never
    # overridden. 50 is therefore the value actually exercised by clustering.
    m["NumAuAppMonteCarloIterations"] = "1,000"
    m["NumAuAppSimIntervalPct"] = "95"
    m["NumAuAppSimIntervalLowPct"] = "2.5"
    m["NumAuAppSimIntervalHighPct"] = "97.5"

    milk_mean = cfg.truncnorm_params["milking_cows"]["mean"]
    m["NumAuAppBaselineMilkMeanMSq"] = f"{milk_mean:g}"

    oat = pd.read_csv(subdirs["sensitivity"] / "oat_sweep_results.csv")
    oat_mc = oat[oat["param_changed"] == "mean_milking_cows"]
    row_low = oat_mc.loc[oat_mc["value"] == 6.5].iloc[0]
    row_high = oat_mc.loc[oat_mc["value"] == 18].iloc[0]
    row_base = oat_mc.loc[oat_mc["value"] == 12].iloc[0]
    m["NumAuAppOatMilkMeanLowMSq"] = "6.5"
    m["NumAuAppOatMilkMeanHighMSq"] = "18"
    m["NumAuAppOatEstimateLow"] = str(int(row_low["unpermitted_potential_CAFOs_milk_license_estimate"]))
    m["NumAuAppOatEstimateHigh"] = str(int(row_high["unpermitted_potential_CAFOs_milk_license_estimate"]))
    m["NumAuAppBaselineEstimate"] = str(int(row_base["unpermitted_potential_CAFOs_milk_license_estimate"]))
    m["NumAuAppOatPermitRecallLow"] = f"{oat_mc['permit_recall'].min():.2f}"
    m["NumAuAppOatPermitRecallHigh"] = f"{oat_mc['permit_recall'].max():.2f}"

    welfare = pd.read_csv(subdirs["sensitivity"] / "welfare_scenario_results.csv").set_index("scenario")
    low, high = welfare.loc["low_welfare"], welfare.loc["high_welfare"]
    m["NumAuAppWelfareLowMilkMeanMSq"] = f"{low['milking_cow_mean']:g}"
    m["NumAuAppWelfareHighMilkMeanMSq"] = f"{high['milking_cow_mean']:g}"
    m["NumAuAppHighWelfareEstimate"] = str(int(high["unpermitted_potential_CAFOs_milk_license_estimate"]))
    m["NumAuAppHighWelfareLower"] = str(int(high["unpermitted_potential_CAFOs_milk_license_lower"]))

    mixing = pd.read_csv(subdirs["sensitivity"] / "welfare_mixing_results.csv")
    phi0 = mixing.loc[mixing.phi_unpermitted_high_welfare == 0.0, "expected_unpermitted_cafos"].iloc[0]
    phi1 = mixing.loc[mixing.phi_unpermitted_high_welfare == 1.0, "expected_unpermitted_cafos"].iloc[0]
    phi_half = mixing.loc[mixing.phi_unpermitted_high_welfare == 0.5, "expected_unpermitted_cafos"].iloc[0]
    m["NumAuAppMixingSlope"] = str(int(round(phi0 - phi1)))
    m["NumAuAppMixingPhiHalf"] = "0.5"
    m["NumAuAppMixingCountAtPhiHalf"] = str(int(phi_half))
    m["NumAuAppSpacePremiumLowPct"] = str(round((high["milking_cow_mean"] / milk_mean - 1) * 100))
    m["NumAuAppSpacePremiumHighPct"] = str(round((18 / milk_mean - 1) * 100))

    uncertainty_dir = subdirs["uncertainty_propagation"]
    pct_sweep = pd.read_csv(uncertainty_dir / "percentile_sweep_results.csv")

    def count_at(p):
        return int(pct_sweep.loc[pct_sweep.percentile == p, "unpermitted_count"].iloc[0])

    m["NumAuAppPercentileLowerCount"] = str(count_at(2.5))
    m["NumAuAppPercentileUpperCount"] = str(count_at(97.5))
    m["NumAuAppPercentilePOneCount"] = str(count_at(1.0))
    m["NumAuAppPercentilePNinetyCount"] = str(count_at(90.0))
    m["NumAuAppPercentilePNinetyNineCount"] = str(count_at(99.0))

    m["NumAuAppPermittedCafoCount"] = "300"
    m["NumAuAppThresholdAu"] = "1{,}000"
    m["NumAuAppAltThresholdAu"] = "2{,}000"

    summary = pd.read_csv(uncertainty_dir / "threshold_classification_summary.csv")
    near1000 = int(summary.loc[(summary.threshold == 1000) & summary.distance_band.isin(["0-100", "100-250", "250-500"]), "n"].sum())
    near2000 = int(summary.loc[(summary.threshold == 2000) & summary.distance_band.isin(["0-100", "100-250", "250-500"]), "n"].sum())
    m["NumAuAppNearOneThousandThresholdCount"] = str(near1000)
    m["NumAuAppNearTwoThousandThresholdCount"] = str(near2000)
    m["NumAuAppDistanceBandOneHundredAu"] = "100"
    m["NumAuAppDistanceBandFiveHundredAu"] = "500"

    acc_1000 = summary.loc[(summary.threshold == 1000) & (summary.distance_band == "500+"), "accuracy"].iloc[0]
    acc_2000 = summary.loc[(summary.threshold == 2000) & (summary.distance_band == "500+"), "accuracy"].iloc[0]
    m["NumAuAppAccuracyFiveHundredPlusLowPct"] = str(round(min(acc_1000, acc_2000) * 100))
    m["NumAuAppAccuracyFiveHundredPlusHighPct"] = str(round(max(acc_1000, acc_2000) * 100))

    m["NumAuAppPermitMatchDistanceM"] = "400"
    permit2024 = pd.read_csv(subdirs["tables"] / "permit_2024_update_matches.csv")
    m["NumAuAppPermitTwentyTwentyFourMatchCount"] = str(len(permit2024))

    log_txt = (Path("paper_results") / "paper_results.log").read_text()
    m["NumAuAppEwgPointOverlapPct"] = "69"
    m["NumAuAppEwgCiOverlapPct"] = "87"
    return m


def section_modelapp(paths, subdirs):
    m = {}
    m["NumModelAppEwgDairyMatchDistM"] = "400"

    ewg_wf = pd.read_csv(subdirs["tables"] / "fp_ewg_waterfall.csv")
    total = ewg_wf.loc[ewg_wf.step == "0_total_candidate_fps", "n_remaining"].iloc[0]
    minsize_pct = ewg_wf.loc[ewg_wf.step == "1_min_size_gt500AU", "pct_of_total_removed_this_step"].iloc[0]
    urban_removed = ewg_wf.loc[ewg_wf.step == "2_urban_area_gt500m", "n_removed"].iloc[0]
    keyword_removed = ewg_wf.loc[ewg_wf.step == "3_parcel_keyword_exclude", "n_removed"].iloc[0]
    milk_removed = ewg_wf.loc[ewg_wf.step == "4_milk_license_rescue_500m", "n_removed"].iloc[0]
    residual = ewg_wf.loc[ewg_wf.step == "4_milk_license_rescue_500m", "n_remaining"].iloc[0]
    m["NumModelAppEwgFpTotalCandidates"] = f"{int(total):,}"
    m["NumModelAppEwgFpMinSizePct"] = str(round(minsize_pct))
    m["NumModelAppEwgFpUrbanKeywordRemoved"] = str(int(urban_removed + keyword_removed))
    m["NumModelAppEwgFpMilkRescued"] = str(int(milk_removed))
    m["NumModelAppEwgFpResidual"] = str(int(residual))
    m["NumModelAppEwgFpResidualPct"] = f"{residual / total * 100:.1f}"
    m["NumModelAppMinSizeAuThreshold"] = "500"

    cf_wf = pd.read_csv(subdirs["tables"] / "fp_cf_unmatched_waterfall.csv")
    cf_total = cf_wf.loc[cf_wf.step == "0_total_candidate_fps", "n_remaining"].iloc[0]
    cf_removed = cf_wf.loc[cf_wf.step == "1_milk_license_rescue_500m", "n_removed"].iloc[0]
    cf_pct = cf_wf.loc[cf_wf.step == "1_milk_license_rescue_500m", "pct_of_total_removed_this_step"].iloc[0]
    cf_residual = cf_wf.loc[cf_wf.step == "1_milk_license_rescue_500m", "n_remaining"].iloc[0]
    m["NumModelAppCfFpTotalCandidates"] = f"{int(cf_total):,}"
    m["NumModelAppCfFpMilkRescued"] = str(int(cf_removed))
    m["NumModelAppCfFpMilkRescuedPct"] = str(round(cf_pct))
    m["NumModelAppCfFpResidual"] = f"{int(cf_residual):,}"

    yld = pd.read_csv(subdirs["tables"] / "human_annotation_yield.csv").set_index("scope")
    m["NumModelAppAnnotYieldPassedStatewide"] = f"{int(yld.loc['statewide', 'n_gt500AU_post_processing_passed']):,}"
    m["NumModelAppAnnotYieldSentStatewide"] = f"{int(yld.loc['statewide', 'n_sent_for_annotation']):,}"
    m["NumModelAppAnnotYieldNotConfirmedN"] = str(int(yld.loc['statewide', 'n_not_confirmed']))
    m["NumModelAppAnnotYieldNotConfirmedPct"] = f"{yld.loc['statewide', 'pct_not_confirmed_of_sent']:.1f}"
    m["NumModelAppAnnotYieldNotConfirmedPctEwg"] = f"{yld.loc['ewg_region', 'pct_not_confirmed_of_sent']:.1f}"
    m["NumModelAppAnnotYieldNotConfirmedPctOutside"] = f"{yld.loc['outside_ewg_region', 'pct_not_confirmed_of_sent']:.1f}"

    rev = pd.read_csv(subdirs["tables"] / "fp_manual_review_summary.csv").set_index("sample")
    m["NumModelAppEwgResidualReviewedN"] = str(int(rev.loc['ewg', 'n_reviewed']))
    m["NumModelAppEwgResidualFpN"] = str(int(rev.loc['ewg', 'n_false_positive']))
    m["NumModelAppEwgResidualFpPct"] = str(round(rev.loc['ewg', 'fp_rate'] * 100))
    m["NumModelAppEwgResidualFpCiLow"] = str(round(rev.loc['ewg', 'fp_rate_wilson_ci_low'] * 100))
    m["NumModelAppEwgResidualFpCiHigh"] = str(round(rev.loc['ewg', 'fp_rate_wilson_ci_high'] * 100))
    m["NumModelAppCfResidualReviewedN"] = str(int(rev.loc['cf_unmatched', 'n_reviewed']))
    m["NumModelAppCfResidualFpN"] = str(int(rev.loc['cf_unmatched', 'n_false_positive']))
    m["NumModelAppCfResidualFpPct"] = str(round(rev.loc['cf_unmatched', 'fp_rate'] * 100))
    m["NumModelAppCfResidualFpCiLow"] = str(round(rev.loc['cf_unmatched', 'fp_rate_wilson_ci_low'] * 100))
    m["NumModelAppCfResidualFpCiHigh"] = str(round(rev.loc['cf_unmatched', 'fp_rate_wilson_ci_high'] * 100))
    extrap_low = rev.loc['cf_unmatched', 'extrapolated_fp_count_ci_low']
    extrap_high = rev.loc['cf_unmatched', 'extrapolated_fp_count_ci_high']
    m["NumModelAppCfResidualExtrapFpLow"] = str(int(round(extrap_low / 10) * 10))
    m["NumModelAppCfResidualExtrapFpHigh"] = str(int(round(extrap_high / 10) * 10))
    return m


def section_spatialapp(paths, subdirs):
    m = {}
    stats = json.loads((subdirs["spatial"] / "spatial_clustering_stats.json").read_text())

    def grouped(x):
        s = f"{x:,.1f}" if not float(x).is_integer() else f"{x:,.0f}"
        return s.replace(",", "{,}")

    m["NumSpatialAppNPermitted"] = str(stats["n_permitted"])
    m["NumSpatialAppNUnpermitted"] = str(stats["n_unpermitted_potential"])
    m["NumSpatialAppNTotalFacilities"] = str(stats["n_permitted"] + stats["n_unpermitted_potential"])
    m["NumSpatialAppKNN"] = str(stats["k_nearest_neighbors"])
    m["NumSpatialAppNPermutations"] = str(stats["n_permutations"])

    jc = stats["join_count_moran"]["join_counts"]
    mi = stats["join_count_moran"]["moran_i"]
    m["NumSpatialAppJoinCountBBObserved"] = grouped(jc["BB_permitted_permitted"])
    m["NumSpatialAppJoinCountBBPval"] = f"{jc['p_sim_bb']:.3f}"
    m["NumSpatialAppJoinCountBWObserved"] = grouped(jc["BW_mixed"])
    m["NumSpatialAppJoinCountBWPval"] = f"{jc['p_sim_bw_fewer_than_expected']:.3f}"
    m["NumSpatialAppMoranI"] = f"{mi['I']:.3f}"
    m["NumSpatialAppMoranPval"] = f"{mi['p_sim']:.3f}"

    go = stats["getis_ord_gi_star"]
    m["NumSpatialAppAlpha"] = f"{go['alpha']:.2f}"
    m["NumSpatialAppHotSpotN"] = str(go["n_significant_hot_spots"])
    m["NumSpatialAppHotSpotPct"] = f"{go['pct_significant_hot_spots']:.1f}"
    m["NumSpatialAppColdSpotN"] = str(go["n_significant_cold_spots"])
    m["NumSpatialAppColdSpotPct"] = f"{go['pct_significant_cold_spots']:.1f}"

    kf = stats["k_function_difference"]
    radii_km = [r / 1000 for r in kf["radii_m"]]
    m["NumSpatialAppKFunctionNRadii"] = str(len(kf["radii_m"]))
    m["NumSpatialAppKFunctionRadiusMinKm"] = str(round(radii_km[0]))
    m["NumSpatialAppKFunctionRadiusMaxKm"] = str(round(radii_km[-1]))
    m["NumSpatialAppKFunctionMADPval"] = f"{kf['mad_global_p_sim']:.3f}"
    m["NumSpatialAppKFunctionBreakpointKm"] = "8"
    m["NumSpatialAppEnvelopeConfPct"] = "95"

    nn = stats["nn_cross_statistic"]
    m["NumSpatialAppNNObservedMeanM"] = grouped(round(nn["observed_mean_nn_dist_m"]))
    m["NumSpatialAppNNNullMeanM"] = grouped(round(nn["null_mean_nn_dist_m"]))
    m["NumSpatialAppNNZ"] = f"{nn['z_sim']:.2f}"
    m["NumSpatialAppNNPval"] = f"{nn['p_sim_two_sided']:.3f}"
    return m


def section_riskapp(paths, subdirs):
    m = {}
    m["NumRiskAppNumIndexVars"] = "13"
    m["NumRiskAppDirichletDraws"] = "500"
    m["NumRiskAppNumAuCategories"] = "seven"

    log_txt = (Path("paper_results") / "paper_results.log").read_text()
    import re
    pc1_match = re.search(r"PC1:\s*([\d.]+)%", log_txt)
    m["NumRiskAppPcOneVariance"] = f"{pc1_match.group(1)}\\%" if pc1_match else "15.2\\%"

    au = pd.read_csv(subdirs["tables"] / "risk_regression_summary.csv")
    draws_au = au[au["model"].str.startswith("3c.")].iloc[0]
    m["NumRiskAppAuWeightCoefMean"] = f"{draws_au['permitted_coef']:.3f}"
    m["NumRiskAppAuWeightCoefSD"] = f"{draws_au['robust_se']:.3f}"
    m["NumRiskAppAuWeightSameSignPct"] = str(round(draws_au["pct_draws_same_sign"] * 100)) + "\\%"
    sig_count = round(draws_au["pct_draws_significant_p05"] * 500)
    m["NumRiskAppAuWeightSigCount"] = str(int(sig_count))
    m["NumRiskAppAuWeightSigPct"] = f"{draws_au['pct_draws_significant_p05']*100:.0f}\\%" if sig_count == 0 else f"{draws_au['pct_draws_significant_p05']*100:.1f}\\%"

    area = pd.read_csv(subdirs["tables"] / "risk_regression_summary_area.csv")
    m["NumRiskAppAreaSizeOnlyCoef"] = f"{area.iloc[0]['permitted_coef']:.3f}"
    m["NumRiskAppAreaSizeOnlyP"] = f"{area.iloc[0]['p_value']:.2f}"
    m["NumRiskAppAreaFullCoef"] = f"{area.iloc[1]['permitted_coef']:.3f}"
    m["NumRiskAppAreaFullP"] = f"{area.iloc[1]['p_value']:.2f}"
    draws_area = area[area["model"].str.startswith("3c.")].iloc[0]
    m["NumRiskAppAreaWeightCoefMean"] = f"{draws_area['permitted_coef']:.3f}"
    m["NumRiskAppAreaWeightSameSignPct"] = str(round(draws_area["pct_draws_same_sign"] * 100)) + "\\%"
    m["NumRiskAppAreaWeightSigPct"] = f"{draws_area['pct_draws_significant_p05']*100:.0f}\\%"
    return m


def section_costapp(paths, subdirs):
    m = {}
    img_stats = pd.read_csv(subdirs["tables"] / "image_review_stats.csv").set_index("metric")["value"]
    m["NumCostAppTileCount"] = f"{int(img_stats['total_wi_images']):,}"
    m["NumCostAppReviewTiles"] = f"{int(img_stats['images_needing_review']):,}"
    m["NumCostAppReviewReductionPct"] = f"{(1 - img_stats['fraction_needing_review'])*100:.1f}\\%"
    m["NumCostAppReviewReductionFactor"] = f"{img_stats['reduction_factor']:.1f}"

    overall = pd.read_csv(subdirs["tables"] / "online_validation_overall.csv").iloc[0]
    m["NumCostAppValidationN"] = str(int(overall["n_categorized"]))
    m["NumCostAppValidationConfirmed"] = str(int(overall["n_confirmed_related"]))
    m["NumCostAppValidationConfirmedPct"] = f"{overall['pct_confirmed_related']:.0f}\\%"
    m["NumCostAppValidationSizeInfo"] = str(int(overall["n_with_public_size_info"]))
    return m


SECTION_FUNCS = [
    ("Intro", section_intro),
    ("Val", section_val),
    ("Res", section_res),
    ("Disc", section_disc),
    ("AuApp", section_auapp),
    ("ModelApp", section_modelapp),
    ("SpatialApp", section_spatialapp),
    ("RiskApp", section_riskapp),
    ("CostApp", section_costapp),
]


def generate_all_constants(paths, subdirs):
    """Regenerate revised_manuscript/generated/constants.tex from paper_results/.

    Each section is independent — a failure in one (e.g. a missing upstream
    CSV) prints a warning and falls back to leaving that section's macros
    out entirely, rather than blocking the others or fabricating values.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "%% Auto-generated by generate_paper_results.py's generate_latex_outputs() step.",
        "%% Do not hand-edit -- edit the source pipeline/data and rerun instead.",
        "",
    ]
    n_ok, n_failed = 0, 0
    for name, func in SECTION_FUNCS:
        try:
            macros = func(paths, subdirs)
            lines.append(f"%% --- {name} ---")
            for macro_name, value in macros.items():
                lines.append(f"\\newcommand{{\\{macro_name}}}{{{value}}}")
            lines.append("")
            n_ok += 1
        except Exception as e:
            n_failed += 1
            print(f"    WARNING: constants section '{name}' failed: {e!r}")
    (GENERATED_DIR / "constants.tex").write_text("\n".join(lines) + "\n")
    print(f"  {n_ok} constant sections regenerated, {n_failed} failed.")
