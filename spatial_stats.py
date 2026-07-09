"""Spatial-statistics tests for clustering/segregation between permitted and unpermitted
potential CAFO locations.

Replaces the visual "no strong clustering" claim with formal, permutation-based tests:
  1. Join-count statistic + Moran's I on the binary permit indicator, over a KNN spatial
     weights matrix -- tests whether permitted and unpermitted facilities are spatially
     segregated (cluster into like-labeled neighborhoods) vs. spatially interspersed.
  2. Getis-Ord Gi* local hot-spot analysis on the local unpermitted share -- locates
     statistically significant clusters ("hot spots") of non-permitting.
  3. A cross-type point-pattern test: the difference between the permitted and
     unpermitted univariate Ripley's K functions, and a nearest-neighbor cross-statistic
     (mean distance from each unpermitted facility to its nearest permitted facility) --
     both tested against a random-labeling permutation null (locations fixed, labels
     reshuffled), to test whether unpermitted locations cluster relative to the
     permitted set.

All inference is Monte Carlo / permutation based; no test relies on a parametric
asymptotic null.
"""
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree


def _centroids_xy(gdf):
    c = gdf.geometry.centroid
    return np.column_stack([c.x.values, c.y.values])


# ---------------------------------------------------------------------------
# 1. Join counts + Moran's I on the binary permit indicator
# ---------------------------------------------------------------------------

def join_count_moran(gdf, label_col="permitted", k=8, permutations=999, seed=None):
    """Join-count statistic and Moran's I for a binary label over a KNN weights matrix.

    Args:
        gdf: GeoDataFrame with a binary/boolean label_col and point or polygon geometry
            (polygons are reduced to centroids).
        label_col: name of the binary column (1 = permitted, 0 = unpermitted).
        k: number of nearest neighbors for the spatial weights matrix.
        permutations: number of Monte Carlo permutations for inference.
        seed: optional RNG seed for reproducibility.

    Returns:
        dict of join-count and Moran's I statistics with permutation-based p-values.
    """
    from libpysal.weights import KNN
    from esda.join_counts import Join_Counts
    from esda.moran import Moran

    coords = _centroids_xy(gdf)
    w = KNN(coords, k=k)
    w.transform = "b"

    y = gdf[label_col].astype(int).values

    jc = Join_Counts(y, w, permutations=permutations)
    mi = Moran(y, w, permutations=permutations, two_tailed=True)

    # esda's p_sim_bw is an upper-tail test (probability BW is *larger* than
    # expected); segregation implies BW is *smaller* than expected, so also
    # report the corresponding lower-tail pseudo p-value directly.
    p_sim_bw_lower = (np.sum(jc.sim_bw <= jc.bw) + 1) / (permutations + 1)

    return {
        "k": k,
        "permutations": permutations,
        "n": int(len(y)),
        "n_permitted": int(y.sum()),
        "n_unpermitted": int((1 - y).sum()),
        "join_counts": {
            "BB_permitted_permitted": float(jc.bb),
            "WW_unpermitted_unpermitted": float(jc.ww),
            "BW_mixed": float(jc.bw),
            "p_sim_bb": float(jc.p_sim_bb),
            "p_sim_bw": float(jc.p_sim_bw),
            "p_sim_bw_fewer_than_expected": float(p_sim_bw_lower),
            "p_sim_positive_autocorr": float(jc.p_sim_autocorr_pos),
            "p_sim_negative_autocorr": float(jc.p_sim_autocorr_neg),
        },
        "moran_i": {
            "I": float(mi.I),
            "expected_I": float(mi.EI),
            "z_sim": float(mi.z_sim),
            "p_sim": float(mi.p_sim),
        },
    }


# ---------------------------------------------------------------------------
# 2. Getis-Ord Gi* local hot-spot analysis
# ---------------------------------------------------------------------------

def getis_ord_hotspots(gdf, value_col="unpermitted_share", k=8, permutations=999,
                        alpha=0.05, seed=None):
    """Local Getis-Ord Gi* hot-spot analysis.

    Args:
        gdf: GeoDataFrame with value_col (continuous or 0/1 indicator) and geometry.
        value_col: column to test for local clustering of high (hot) / low (cold) values.
        k: number of nearest neighbors for the spatial weights matrix.
        permutations: number of conditional permutations for inference.
        alpha: significance threshold for counting hot/cold spots.
        seed: optional RNG seed.

    Returns:
        dict summary plus the input gdf with 'gi_star_z' and 'gi_star_p_sim' columns
        attached under key 'gdf'.
    """
    from libpysal.weights import KNN
    from esda.getisord import G_Local

    coords = _centroids_xy(gdf)
    w = KNN(coords, k=k)
    w.transform = "b"

    y = gdf[value_col].astype(float).values
    gl = G_Local(y, w, star=True, permutations=permutations, seed=seed)

    out = gdf.copy()
    out["gi_star_z"] = gl.Zs
    out["gi_star_p_sim"] = gl.p_sim

    n_hot = int(((gl.Zs > 0) & (gl.p_sim < alpha)).sum())
    n_cold = int(((gl.Zs < 0) & (gl.p_sim < alpha)).sum())

    return {
        "k": k,
        "permutations": permutations,
        "alpha": alpha,
        "value_col": value_col,
        "n": int(len(y)),
        "n_significant_hot_spots": n_hot,
        "n_significant_cold_spots": n_cold,
        "pct_significant_hot_spots": 100 * n_hot / len(y),
        "pct_significant_cold_spots": 100 * n_cold / len(y),
        "gdf": out,
    }


# ---------------------------------------------------------------------------
# 3. Cross-type point-pattern tests: K-function difference + NN cross-statistic
# ---------------------------------------------------------------------------

def _ripley_k(coords, radii, area):
    """Univariate Ripley's K, no edge correction: K(r) = area/n^2 * sum_i sum_{j!=i} 1(d_ij<=r)."""
    n = len(coords)
    tree = cKDTree(coords)
    counts = tree.count_neighbors(tree, radii)  # includes self-pairs (n) at every radius
    ordered_pairs = np.asarray(counts, dtype=float) - n
    return area * ordered_pairs / (n * n)


def _hull_area(coords):
    from scipy.spatial import ConvexHull
    return float(ConvexHull(coords).volume)  # 'volume' is the 2D area for 2D points


def k_function_difference_test(gdf, label_col="permitted", radii=None,
                                n_permutations=999, seed=None):
    """Difference between the unpermitted and permitted univariate K-functions,
    D(r) = K_unpermitted(r) - K_permitted(r), tested against a random-labeling
    permutation null (locations fixed, labels reshuffled preserving group sizes).

    A positive D(r) means unpermitted facilities are more clustered among
    themselves than permitted facilities are, at spatial scale r.

    Returns a dict with the observed D(r), the pointwise 95% permutation envelope,
    pointwise p-values, and a single global p-value from a maximum-absolute-deviation
    (MAD) test that avoids the multiple-comparisons problem of testing every radius.
    """
    coords = _centroids_xy(gdf)
    y = gdf[label_col].astype(int).values
    n = len(y)
    n_permitted = int(y.sum())
    n_unpermitted = n - n_permitted

    area = _hull_area(coords)

    if radii is None:
        nn_dist = cKDTree(coords).query(coords, k=2)[0][:, 1]
        r_min = max(1000.0, np.percentile(nn_dist, 25))
        r_max = 10 * np.percentile(nn_dist, 95)
        radii = np.geomspace(r_min, r_max, 20)
    radii = np.asarray(radii, dtype=float)

    def _d_of(labels):
        perm_mask = labels.astype(bool)
        k_perm = _ripley_k(coords[perm_mask], radii, area)
        k_unperm = _ripley_k(coords[~perm_mask], radii, area)
        return k_unperm - k_perm

    d_obs = _d_of(y)

    rng = np.random.default_rng(seed)
    sims = np.empty((n_permutations, len(radii)))
    for i in range(n_permutations):
        perm_labels = np.zeros(n, dtype=int)
        perm_labels[rng.choice(n, n_permitted, replace=False)] = 1
        sims[i] = _d_of(perm_labels)

    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)
    pointwise_p = (1 + np.sum(np.abs(sims) >= np.abs(d_obs), axis=0)) / (n_permutations + 1)

    mad_obs = np.max(np.abs(d_obs))
    mad_sims = np.max(np.abs(sims), axis=1)
    global_p = (1 + np.sum(mad_sims >= mad_obs)) / (n_permutations + 1)

    return {
        "label_col": label_col,
        "n_permutations": n_permutations,
        "n_permitted": n_permitted,
        "n_unpermitted": n_unpermitted,
        "hull_area_m2": area,
        "radii_m": radii.tolist(),
        "d_observed": d_obs.tolist(),
        "envelope_lo_2.5pct": lo.tolist(),
        "envelope_hi_97.5pct": hi.tolist(),
        "pointwise_p_sim": pointwise_p.tolist(),
        "mad_global_p_sim": float(global_p),
    }


def nn_cross_test(gdf, label_col="permitted", n_permutations=999, seed=None):
    """Nearest-neighbor cross-statistic: mean distance from each unpermitted facility
    to its nearest permitted facility, tested against a random-labeling permutation
    null (locations fixed, labels reshuffled preserving group sizes).

    A mean distance significantly *smaller* than the permutation null indicates
    unpermitted facilities cluster near permitted ones; significantly *larger*
    indicates unpermitted facilities avoid permitted locations.
    """
    coords = _centroids_xy(gdf)
    y = gdf[label_col].astype(int).values
    n = len(y)
    n_permitted = int(y.sum())

    def _mean_nn_dist(labels):
        perm_mask = labels.astype(bool)
        permitted_coords = coords[perm_mask]
        unpermitted_coords = coords[~perm_mask]
        tree = cKDTree(permitted_coords)
        d, _ = tree.query(unpermitted_coords, k=1)
        return float(d.mean())

    obs = _mean_nn_dist(y)

    rng = np.random.default_rng(seed)
    sims = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_labels = np.zeros(n, dtype=int)
        perm_labels[rng.choice(n, n_permitted, replace=False)] = 1
        sims[i] = _mean_nn_dist(perm_labels)

    p_sim_two_sided = (1 + np.sum(np.abs(sims - sims.mean()) >= np.abs(obs - sims.mean()))) / (n_permutations + 1)

    return {
        "label_col": label_col,
        "n_permutations": n_permutations,
        "observed_mean_nn_dist_m": obs,
        "null_mean_nn_dist_m": float(sims.mean()),
        "null_std_nn_dist_m": float(sims.std()),
        "z_sim": float((obs - sims.mean()) / sims.std()),
        "p_sim_two_sided": float(p_sim_two_sided),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_spatial_clustering_analysis(all_clusters, out_dir=None, k=8, n_permutations=999,
                                     radii=None, seed=0, set_col="set",
                                     permitted_label="Permitted dairy CAFOs",
                                     unpermitted_label="Unpermitted potential CAFOs",
                                     county_data=None):
    """Run all four spatial-clustering tests on permitted vs. unpermitted-potential CAFOs.

    Args:
        all_clusters: GeoDataFrame with a categorical column (set_col) distinguishing
            permitted dairy CAFOs, unpermitted potential CAFOs, and (optionally) other
            categories that are dropped for this analysis.
        out_dir: if provided, write spatial_clustering_stats.json and figures here.
        k: number of nearest neighbors for the KNN spatial weights matrix.
        n_permutations: number of Monte Carlo permutations for all tests.
        radii: optional array of distances (m) for the K-function test; auto-chosen if None.
        seed: RNG seed for reproducibility.
        set_col, permitted_label, unpermitted_label: identify the two groups to compare.
        county_data: optional GeoDataFrame of WI county boundaries, used as a basemap
            for the Getis-Ord Gi* hot-spot map figure.

    Returns:
        dict of results from all four tests, plus the filtered GeoDataFrame used.
    """
    sub = all_clusters[
        all_clusters[set_col].isin([permitted_label, unpermitted_label])
    ].copy()
    sub["permitted"] = (sub[set_col] == permitted_label).astype(int)
    sub["unpermitted_share"] = 1 - sub["permitted"]

    print(f"  Spatial clustering analysis: n={len(sub)} "
          f"({sub['permitted'].sum()} permitted, {(1 - sub['permitted']).sum()} unpermitted potential)")

    print("  Running join-count / Moran's I test...")
    jc_moran = join_count_moran(sub, label_col="permitted", k=k,
                                 permutations=n_permutations, seed=seed)

    print("  Running Getis-Ord Gi* hot-spot analysis...")
    gi_star = getis_ord_hotspots(sub, value_col="unpermitted_share", k=k,
                                  permutations=n_permutations, seed=seed)
    gi_star_gdf = gi_star.pop("gdf")

    print(f"  Running K-function difference test ({n_permutations} permutations)...")
    k_diff = k_function_difference_test(sub, label_col="permitted", radii=radii,
                                         n_permutations=n_permutations, seed=seed)

    print(f"  Running nearest-neighbor cross-statistic test ({n_permutations} permutations)...")
    nn_cross = nn_cross_test(sub, label_col="permitted",
                              n_permutations=n_permutations, seed=seed)

    results = {
        "n_permitted": int(sub["permitted"].sum()),
        "n_unpermitted_potential": int((1 - sub["permitted"]).sum()),
        "k_nearest_neighbors": k,
        "n_permutations": n_permutations,
        "join_count_moran": jc_moran,
        "getis_ord_gi_star": gi_star,
        "k_function_difference": k_diff,
        "nn_cross_statistic": nn_cross,
    }

    if out_dir is not None:
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "spatial_clustering_stats.json", "w") as f:
            json.dump(results, f, indent=2)

        gi_star_gdf.to_file(out_dir / "gi_star_hotspots.geojson", driver="GeoJSON")

        _plot_k_function_difference(k_diff, out_dir / "k_function_difference.svg")
        _plot_gi_star_map(gi_star_gdf, county_data, out_dir / "gi_star_hotspot_map.svg")

        print(f"  Saved spatial_clustering_stats.json, gi_star_hotspots.geojson, "
              f"k_function_difference.svg, gi_star_hotspot_map.svg to {out_dir}")

    _print_summary(results)

    results["gdf"] = sub
    return results


def _plot_k_function_difference(k_diff, save_path):
    """D(r) = K_unpermitted(r) - K_permitted(r) with its 95% permutation envelope."""
    import matplotlib.pyplot as plt
    import config.config_params as cfg

    r = np.asarray(k_diff["radii_m"]) / 1000
    d_obs = np.asarray(k_diff["d_observed"])
    lo = np.asarray(k_diff["envelope_lo_2.5pct"])
    hi = np.asarray(k_diff["envelope_hi_97.5pct"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(r, lo, hi, color="gray", alpha=0.3,
                     label="95\\% permutation envelope\n(random-labeling null)")
    ax.plot(r, d_obs, color="black", linewidth=cfg.FIG_LINEWIDTH, label="Observed $D(r)$")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance $r$ (km)")
    ax.set_ylabel(r"$D(r) = K_{\mathrm{unpermitted}}(r) - K_{\mathrm{permitted}}(r)$")
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)
    plt.close(fig)


def _plot_gi_star_map(gi_star_gdf, county_data, save_path, alpha=0.05):
    """Map of significant Getis-Ord Gi* hot spots (unpermitted-heavy) and cold spots
    (permitted-heavy), colored to match the permitted/unpermitted palette used
    throughout the paper.
    """
    import matplotlib.pyplot as plt
    import config.config_params as cfg

    fig, ax = plt.subplots(figsize=(7, 7))

    if county_data is not None:
        county_data.to_crs(cfg.WI_EPSG).plot(
            ax=ax, edgecolor="black", linewidth=1, color="lightgray", alpha=0.2,
        )

    c = gi_star_gdf.geometry.centroid
    sig = gi_star_gdf["gi_star_p_sim"] < alpha
    hot = sig & (gi_star_gdf["gi_star_z"] > 0)
    cold = sig & (gi_star_gdf["gi_star_z"] < 0)
    nonsig = ~sig

    ax.scatter(c[nonsig].x, c[nonsig].y, c="lightgray", s=6,
               label=f"Not significant (n={nonsig.sum()})")
    ax.scatter(c[hot].x, c[hot].y, c=cfg.COLOR_UNPERMITTED, s=14,
               label=f"Unpermitted-heavy hot spot (n={hot.sum()})")
    ax.scatter(c[cold].x, c[cold].y, c=cfg.COLOR_PERMITTED, s=14,
               label=f"Permitted-heavy cold spot (n={cold.sum()})")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)
    plt.close(fig)


def _print_summary(results):
    jc = results["join_count_moran"]
    gi = results["getis_ord_gi_star"]
    kd = results["k_function_difference"]
    nn = results["nn_cross_statistic"]

    print("\n  === Spatial clustering summary ===")
    print(f"  Moran's I on permit indicator: I={jc['moran_i']['I']:.4f}, "
          f"p_sim={jc['moran_i']['p_sim']:.4f}")
    print(f"  Join counts: BB={jc['join_counts']['BB_permitted_permitted']:.1f} "
          f"(p={jc['join_counts']['p_sim_bb']:.3f}), "
          f"WW={jc['join_counts']['WW_unpermitted_unpermitted']:.1f}, "
          f"BW={jc['join_counts']['BW_mixed']:.1f} (p={jc['join_counts']['p_sim_bw']:.3f}), "
          f"positive-autocorrelation p={jc['join_counts']['p_sim_positive_autocorr']:.3f}")
    print(f"  Getis-Ord Gi*: {gi['n_significant_hot_spots']} significant hot spots "
          f"({gi['pct_significant_hot_spots']:.1f}%), "
          f"{gi['n_significant_cold_spots']} significant cold spots "
          f"({gi['pct_significant_cold_spots']:.1f}%) at alpha={gi['alpha']}")
    print(f"  K-function difference: global MAD test p_sim={kd['mad_global_p_sim']:.4f}")
    print(f"  NN cross-statistic: observed mean dist={nn['observed_mean_nn_dist_m']:.0f}m, "
          f"null mean={nn['null_mean_nn_dist_m']:.0f}m, "
          f"z={nn['z_sim']:.2f}, p_sim={nn['p_sim_two_sided']:.4f}")
