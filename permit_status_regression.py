#!/usr/bin/env python3
"""Descriptive (associational) regression of permit status on facility size,
region, and operational/locational characteristics.

Addresses Reviewer #2: the raw "larger farms are more likely to be permitted"
relationship may reflect visibility, inspection frequency, reporting
requirements, economic capacity, or operational characteristics rather than
size per se. This module fits logistic regressions that isolate the factors
associated with holding a WPDES CAFO permit *while controlling for size and
other confounders*, and reports odds ratios with (county-clustered) robust SEs.

The analysis is explicitly descriptive: it quantifies which observable
characteristics co-vary with permit status, not the causal effect of any one of
them. It is built to run on the publication dataset
(``paper_results/publication_dataset/wi_cafo_facilities.parquet``) plus the
county-boundaries shapefile, both of which are already loaded by the main
pipeline.

Regional / socioeconomic controls
---------------------------------
Two region controls are derived fully offline from the WDNR county-boundaries
shapefile via a spatial join:

  * ``dnr_region``   – WDNR administrative region (5 units). This is the
    regulator office responsible for CAFO inspection and enforcement, so it is
    a direct proxy for the "inspection frequency / regulator capacity" channel.
  * ``county_cafo_density`` – detected facilities (>=500 AU) per 1,000 km2 of
    county land area. A proxy for local dairy intensity / how visually
    "expected" a large barn complex is in that landscape (the visibility
    channel).

An optional Census/ACS county merge (median household income = economic
capacity; population density = rurality/visibility) is supported through
``load_census_county_covariates`` but is skipped gracefully when neither a
cached file nor a ``CENSUS_API_KEY`` is available, so the pipeline never breaks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.formula.api as smf

import config.config_params as cfg


# ---------------------------------------------------------------------------
# Optional Census/ACS county covariates (economic capacity / rurality)
# ---------------------------------------------------------------------------

def load_census_county_covariates(cache_path=None, year=2021):
    """Return a DataFrame of WI county ACS covariates keyed by 5-digit FIPS.

    Columns: ``county_fips``, ``median_hh_income``, ``county_population``.

    Resolution order:
      1. A cached CSV at ``cache_path`` (committed alongside the data), if present.
      2. The Census ACS5 API, if a ``CENSUS_API_KEY`` environment variable is set
         (the public API now requires a key). The result is written to
         ``cache_path`` for reproducibility.

    Returns ``None`` (and prints a note) if neither source is available, so the
    caller can proceed without socioeconomic controls.
    """
    import os

    if cache_path is not None and Path(cache_path).exists():
        df = pd.read_csv(cache_path, dtype={"county_fips": str})
        df["county_fips"] = df["county_fips"].str.zfill(5)
        return df

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print(
            "  [census] no cached ACS file and no CENSUS_API_KEY set — "
            "proceeding without socioeconomic controls."
        )
        return None

    try:
        import requests

        url = (
            f"https://api.census.gov/data/{year}/acs/acs5"
            "?get=NAME,B19013_001E,B01003_001E&for=county:*&in=state:55"
            f"&key={api_key}"
        )
        rows = requests.get(url, timeout=30).json()
        cols = rows[0]
        rec = pd.DataFrame(rows[1:], columns=cols)
        out = pd.DataFrame({
            "county_fips": ("55" + rec["county"].astype(str).str.zfill(3)),
            "median_hh_income": pd.to_numeric(rec["B19013_001E"], errors="coerce"),
            "county_population": pd.to_numeric(rec["B01003_001E"], errors="coerce"),
        })
        if cache_path is not None:
            out.to_csv(cache_path, index=False)
        return out
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  [census] ACS fetch failed ({e!r}); proceeding without it.")
        return None


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_regression_frame(all_clusters, counties, census_cache_path=None):
    """Attach region controls and construct model variables.

    Parameters
    ----------
    all_clusters : GeoDataFrame
        Detected facilities with (at minimum) ``animal_unit_estimate``,
        ``cluster_area_m2``, ``n_buildings``, ``matched_milk``, a permit-status
        column (``set`` or ``type``), water/karst/groundwater fields, and a
        geometry in WI State Plane (EPSG:3071).
    counties : GeoDataFrame
        WDNR county boundaries (with ``COUNTY_NAM``, ``COUNTY_FIP``,
        ``DNR_REGION``).

    Returns
    -------
    GeoDataFrame with added columns used by the regressions.
    """
    df = all_clusters.copy()

    # Harmonize the permit-status label column (publication export uses 'type';
    # the in-memory summary_table output uses 'set').
    status_col = "set" if "set" in df.columns else "type"
    df["permitted"] = (df[status_col] == "Permitted dairy CAFOs").astype(int)

    # ── spatial join to counties (region controls) ────────────────────────────
    counties_wi = counties.to_crs(cfg.WI_EPSG)
    keep = [c for c in ["COUNTY_NAM", "COUNTY_FIP", "DNR_REGION", "geometry"]
            if c in counties_wi.columns]
    df = df.to_crs(cfg.WI_EPSG)
    joined = gpd.sjoin(
        df, counties_wi[keep], how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")
    # A handful of centroids can fall just outside a polygon boundary; fill via
    # nearest county.
    if joined["COUNTY_NAM"].isna().any():
        miss = joined["COUNTY_NAM"].isna()
        near = gpd.sjoin_nearest(
            df[miss], counties_wi[keep], how="left"
        ).drop(columns=["index_right"], errors="ignore")
        # sjoin_nearest can emit >1 row per facility on equidistant ties.
        near = near[~near.index.duplicated(keep="first")]
        for c in ["COUNTY_NAM", "COUNTY_FIP", "DNR_REGION"]:
            joined.loc[near.index, c] = near[c]
    # sjoin can duplicate rows on boundary ties; keep one per original facility.
    joined = joined[~joined.index.duplicated(keep="first")]
    df = joined
    df["dnr_region"] = df["DNR_REGION"].fillna("Unknown")
    df["county_name"] = df["COUNTY_NAM"]
    # COUNTY_FIP in the WDNR shapefile is the bare 3-digit county code (e.g. "59"),
    # while Census county FIPS are state(55)+county(3-digit). Build the full 5-digit
    # GEOID so the ACS merge keys line up.
    df["county_fips"] = "55" + df["COUNTY_FIP"].astype(str).str.zfill(3)

    # ── county detected-CAFO density (per 1,000 km2) ──────────────────────────
    # Uses the full detected universe (>=500 AU, i.e. every row here) as the
    # numerator so it measures local dairy intensity, not just permitted counts.
    county_area_km2 = (
        counties_wi.assign(area_km2=counties_wi.geometry.area / 1e6)
        .set_index("COUNTY_NAM")["area_km2"]
    )
    county_counts = df.groupby("county_name").size()
    dens = (county_counts / county_area_km2.reindex(county_counts.index) * 1000.0)
    df["county_cafo_density"] = df["county_name"].map(dens)

    # ── model variables ───────────────────────────────────────────────────────
    df["log_au"] = np.log(df["animal_unit_estimate"])
    df["log_area"] = np.log(df["cluster_area_m2"])
    df["milk_license"] = df["matched_milk"].astype(int)
    df["log_water_dist"] = np.log(df["water_distance"] + 1)
    # Karst / shallow-carbonate bedrock vulnerability (NE WI dolomite): use the
    # broadest available shallow-silurian dummy, falling back to silurian_0_2.
    for karst_src in ["shallow_silurian_dummy", "silurian_0_2_dummy"]:
        if karst_src in df.columns:
            df["karst"] = df[karst_src].astype(int)
            break
    df["shallow_gw"] = df["gw_0"].astype(int) if "gw_0" in df.columns else 0

    # ── optional census merge ─────────────────────────────────────────────────
    census = load_census_county_covariates(cache_path=census_cache_path)
    if census is not None:
        df = df.merge(census, on="county_fips", how="left")
        # population density (people / km2) as a rurality/visibility proxy
        df["county_pop_density"] = (
            df["county_population"] / county_area_km2.reindex(
                df["county_name"]
            ).values
        )
        df["log_median_income"] = np.log(df["median_hh_income"])
        df["_has_census"] = True
    else:
        df["_has_census"] = False

    return df


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

_MODEL_SPECS = [
    ("1. Size only", "permitted ~ log_au"),
    ("2. + operational scale", "permitted ~ log_au + n_buildings + milk_license"),
    ("3. + region fixed effects",
     "permitted ~ log_au + n_buildings + milk_license + C(dnr_region)"),
    ("4. + location / visibility",
     "permitted ~ log_au + n_buildings + milk_license + C(dnr_region)"
     " + log_water_dist + karst + shallow_gw + county_cafo_density"),
]

# Pretty labels for reporting (odds-ratio table / forest plot)
_TERM_LABELS = {
    "log_au": "Size: log(animal units)",
    "n_buildings": "No. of buildings",
    "milk_license": "Holds state milk license",
    "log_water_dist": "log(distance to surface water)",
    "karst": "Shallow carbonate (karst) bedrock",
    "shallow_gw": "Shallow groundwater (<=0 ft)",
    "county_cafo_density": "County CAFO density (per 1,000 km2)",
    "log_median_income": "log(county median HH income)",
    "county_pop_density": "County population density",
}


def _fit_logit(df, formula, cluster_col="county_name"):
    """Fit a logistic regression with county-clustered robust SEs.

    Falls back to standard MLE SEs if clustering fails (e.g. too few clusters).
    """
    terms = [t.strip() for t in formula.split("~")[1].replace("+", " ").split()]
    base_cols = [c for c in terms
                 if c in df.columns] + ["permitted", cluster_col]
    # Pull the C(...) source columns too
    for t in terms:
        if t.startswith("C(") and t.endswith(")"):
            inner = t[2:-1]
            if inner in df.columns:
                base_cols.append(inner)
    sub = df[[c for c in dict.fromkeys(base_cols) if c in df.columns]].dropna()
    try:
        model = smf.logit(formula, data=sub).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": sub[cluster_col]},
        )
    except Exception:
        model = smf.logit(formula, data=sub).fit(disp=False)
    return model, len(sub)


def _tidy_odds_ratios(model, model_name, n):
    """Return a tidy DataFrame of odds ratios + 95% CI for one fitted model."""
    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    out = pd.DataFrame({
        "model": model_name,
        "term": model.params.index,
        "coef": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "or_ci_low": np.exp(ci["ci_low"].values),
        "or_ci_high": np.exp(ci["ci_high"].values),
        "p_value": model.pvalues.values,
        "n": n,
        "pseudo_r2": model.prsquared,
    })
    return out


def run_permit_status_regression(
    all_clusters,
    counties,
    save_table_path=None,
    save_fig_path=None,
    au_threshold=1000,
    census_cache_path=None,
    verbose=True,
):
    """Fit and report the descriptive permit-status logistic regressions.

    Primary universe: facilities whose point-estimate animal units meet the
    CAFO permitting threshold (``au_threshold``, default 1,000 AU) — i.e. the
    facilities that *appear* to require coverage. Outcome = 1 if the facility
    holds a WPDES permit, 0 if it is an unpermitted potential CAFO. Size (log
    AU) still varies widely within this universe, so its coefficient is
    identified net of the threshold.

    Also fits the same size-only and full specifications on the full detected
    universe (>=500 AU) as a descriptive robustness check on the size gradient.

    Returns a dict with the fitted models, the tidy odds-ratio table, and the
    analysis frame.
    """
    frame = build_regression_frame(
        all_clusters, counties, census_cache_path=census_cache_path
    )

    primary = frame[frame["animal_unit_estimate"] >= au_threshold].copy()

    if verbose:
        print(f"\n  Descriptive permit-status regression")
        print(f"  Primary universe: facilities >= {au_threshold} AU "
              f"(point estimate), N = {len(primary)}")
        print(f"    permitted:   {int(primary['permitted'].sum())}")
        print(f"    unpermitted: {int((primary['permitted'] == 0).sum())}")
        print(f"  DNR region distribution (primary universe):")
        print(primary["dnr_region"].value_counts().to_string())

    models = {}
    tidy_rows = []
    for name, formula in _MODEL_SPECS:
        m, n = _fit_logit(primary, formula)
        models[name] = m
        tidy_rows.append(_tidy_odds_ratios(m, name, n))
        if verbose:
            print(f"\n  ── {name}  (N={n}, McFadden R2={m.prsquared:.3f}) ──")
            print(f"     OR log(AU) = {np.exp(m.params['log_au']):.2f} "
                  f"(p={m.pvalues['log_au']:.1e})")

    # Optional census-augmented full model. Guard against a failed/empty ACS
    # merge (e.g. FIPS mismatch) so the pipeline never breaks on it.
    if (frame["_has_census"].iloc[0]
            and "log_median_income" in primary.columns
            and primary["log_median_income"].notna().sum() > 50):
        cformula = (_MODEL_SPECS[-1][1]
                    + " + log_median_income + county_pop_density")
        mc, nc = _fit_logit(primary, cformula)
        models["5. + census (income, pop density)"] = mc
        tidy_rows.append(_tidy_odds_ratios(
            mc, "5. + census (income, pop density)", nc))
        if verbose:
            print(f"\n  ── 5. + census  (N={nc}, McFadden R2={mc.prsquared:.3f}) ──")
            print(f"     OR log(median HH income) = "
                  f"{np.exp(mc.params['log_median_income']):.3f} "
                  f"(p={mc.pvalues['log_median_income']:.2g})")
    elif frame["_has_census"].iloc[0] and verbose:
        print("  [census] merge produced insufficient coverage; skipping Model 5.")

    # Full-universe (>=500 AU) descriptive robustness on the size gradient
    for name, formula in [_MODEL_SPECS[0], _MODEL_SPECS[-1]]:
        m, n = _fit_logit(frame, formula)
        rob_name = f"[>=500 AU universe] {name}"
        models[rob_name] = m
        tidy_rows.append(_tidy_odds_ratios(m, rob_name, n))

    tidy = pd.concat(tidy_rows, ignore_index=True)
    tidy["term_label"] = tidy["term"].map(_TERM_LABELS).fillna(tidy["term"])

    if save_table_path is not None:
        tidy.to_csv(save_table_path, index=False)
        if verbose:
            print(f"\n  Saved odds-ratio table -> {save_table_path}")

    if save_fig_path is not None:
        _plot_or_forest(models["4. + location / visibility"], save_fig_path)
        if verbose:
            print(f"  Saved OR forest plot -> {save_fig_path}")

    return {"models": models, "tidy": tidy, "frame": frame, "primary": primary}


def _plot_or_forest(model, save_path):
    """Forest plot of odds ratios (with 95% CI) for the full model."""
    import matplotlib.pyplot as plt

    params = model.params.drop("Intercept", errors="ignore")
    # Drop the region fixed-effect dummies from the plot for legibility
    params = params[~params.index.str.startswith("C(dnr_region)")]
    ci = model.conf_int().loc[params.index]

    labels = [_TERM_LABELS.get(t, t) for t in params.index]
    or_vals = np.exp(params.values)
    lo = np.exp(ci[0].values)
    hi = np.exp(ci[1].values)

    order = np.argsort(or_vals)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.2))
    ax.errorbar(
        or_vals[order], y,
        xerr=[or_vals[order] - lo[order], hi[order] - or_vals[order]],
        fmt="o", color="#2b6cb0", ecolor="#2b6cb0", capsize=3, lw=1.2,
    )
    ax.axvline(1.0, color="0.4", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio for holding a permit (95% CI, log scale)")
    ax.set_title("Correlates of permit status among facilities >=1,000 AU")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    # Standalone quick-test against the publication dataset.
    import yaml

    with open("config/config.yml") as f:
        configs = yaml.safe_load(f)
    data_path = Path(configs["data_path"])

    pub = gpd.read_parquet(
        "paper_results/publication_dataset/wi_cafo_facilities.parquet"
    )
    counties = gpd.read_file(
        data_path / "geospatial/county_boundaries/County_Boundaries_24K.shp"
    )
    out_tables = Path("paper_results/tables")
    out_fig = Path("paper_results/04_unpermitted_analysis")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    res = run_permit_status_regression(
        pub, counties,
        save_table_path=out_tables / "permit_status_regression.csv",
        save_fig_path=out_fig / "permit_status_or_forest.svg",
        census_cache_path=data_path / "census_acs_county.csv",
    )
    print("\n=== Tidy odds-ratio table (primary models) ===")
    show = res["tidy"][~res["tidy"]["term"].str.startswith("C(")]
    show = show[~show["model"].str.startswith("[>=500")]
    print(show[["model", "term_label", "odds_ratio",
                "or_ci_low", "or_ci_high", "p_value"]].to_string(index=False))
