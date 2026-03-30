# This function contains functions for analyzing model or human label clusters.

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys
from copy import deepcopy
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import cluster
import rasterio
import affine
from rasterio.mask import mask
import fiona
import math
import numpy as np
from shapely.ops import unary_union
from pyproj import CRS


# our files
sys.path.append("../")
import estimate_animal_units as est_au
import time


## ---- Helper functions ---- ##
def remove_index_l_r(df):
    """Helper function to remove index_left and index_right columns from a dataframe
    Args:
        df: dataframe
    Returns: dataframe with index_left and index_right columns removed
    """
    if "index_left" in df.columns:
        df = df.drop("index_left", axis=1)
    if "index_right" in df.columns:
        df = df.drop("index_right", axis=1)

    return df


## ---- Main functions ---- ##
def match_cluster_names(test_cluster, parcels, match_type):
    """This function looks through all pairs of owner name within a cluster and provides the number of
      related connected components (adding a column to the cluster dataframe)

    Args: test cluster, parcels, match matching type
    Returns: test_cluster with a new column
    """
    # for cluster i, find the polygons, and then find the owners:
    matching_names = []
    for cluster_i in test_cluster.index.values:
        parcels_for_cluster_i = parcels.iloc[
            test_cluster.iloc[cluster_i]["parcel_indices"]
        ]

        n = len(parcels_for_cluster_i["OWNERNME1"])

        # combine owner names into one list (with structure:  parcel1,owner1, parcel1,owner2, parcel2,owner1, parcel2,owner2, etc)
        list_of_names = []
        for x in range(n):
            list_of_names.append(parcels_for_cluster_i["OWNERNME1"].iloc[x])
            list_of_names.append(parcels_for_cluster_i["OWNERNME2"].iloc[x])

        # fill a graph with diagonal and off-diagonals as 1 to represent the owner1-owner2 match (even if their names are unrelated)
        if n == 2:
            name_graph = np.ones((2 * n, 2 * n))
        else:
            name_graph = np.eye(n * 2)
            for i in range((2 * n) - 1):
                if i % 2 == 0:
                    name_graph[i + 1][i] = 1
                    name_graph[i][i + 1] = 1

            list_of_names, parcels_for_cluster_i["OWNERNME1"], parcels_for_cluster_i[
                "OWNERNME2"
            ]

            for ni in range(2 * n):
                for nj in range(2 * n):
                    if name_graph[ni][nj] == 1:
                        continue
                    else:
                        if match_type == "fuzzy":
                            match = cluster.is_fuzzy_name_match(
                                list_of_names[ni], list_of_names[nj]
                            )
                        if match_type == "same":
                            match = list_of_names[ni] == list_of_names[nj]

                        if match:
                            name_graph[ni][nj] = 1

        graph = csr_matrix(name_graph)
        n_components, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )
        if n_components > 1:
            matching_names.append(False)
        else:
            matching_names.append(True)
    if match_type == "fuzzy":
        test_cluster["fuzzy_related_names"] = matching_names
    if match_type == "same":
        test_cluster["same_names"] = matching_names

    return test_cluster


def compare_discrep_in_novel_clusters(test_matched_clusters, base_matched_clusters):
    """Compare the AU estimates between test and base clustering techniques
    Args:
        - test_matched_clusters: dataframe with polygon indices and parcel indices to test against the base
        - base_matched_clusters dataframe with polygon indices and parcel indices

    Returns:
        - diff_au_discrep: dataframe with change in discrep between the two clustering methods
        - diff_matched_c: dataframe with permit information for differently grouped clusters au estimate
    """
    diff_matched_c = get_cluster_diff(test_matched_clusters, base_matched_clusters)
    diff_matched_c["polygon_indices"].astype(str)
    test_matched_clusters["polygon_indices"].astype(str)
    base_matched_clusters["polygon_indices"].astype(str)

    diff_matched_c["parcel_owner1_names"] = None
    diff_matched_c["parcel_owner2_names"] = None
    diff_matched_c["Number ofAnimalUnits"] = None
    diff_matched_c["animal_unit_estimate"] = None
    diff_matched_c["left_discrep"] = None
    diff_matched_c["right_discrep"] = None
    diff_matched_c["diff_ID"] = None
    for i, row in tqdm(diff_matched_c.iterrows()):
        if not pd.isna(row["left_index"]):
            test_c = test_matched_clusters.loc[[int(row["left_index"])]]
            # assert (str(test_c['polygon_indices']) == str(row['polygon_indices'])), f"check that your left and right data frames are correct {test_c['polygon_indices']}, {row['polygon_indices']}"
            diff_matched_c.at[i, "parcel_owner1_names"] = test_c[
                "parcel_owner1_names"
            ].values
            diff_matched_c.at[i, "parcel_owner2_names"] = test_c[
                "parcel_owner2_names"
            ].values
            diff_matched_c.at[i, "Number ofAnimalUnits"] = test_c[
                "Number ofAnimalUnits"
            ].values
            diff_matched_c.at[i, "animal_unit_estimate"] = test_c[
                "animal_unit_estimate"
            ].values
            diff_matched_c.at[i, "left_discrep"] = test_c["discrep"].values
            diff_matched_c.at[i, "diff_ID"] = int(np.floor(i / 2))

        if not pd.isna(row["right_index"]):
            base_c = base_matched_clusters.loc[[int(row["right_index"])]]
            diff_matched_c.at[i, "parcel_owner1_names"] = base_c[
                "parcel_owner1_names"
            ].values
            diff_matched_c.at[i, "parcel_owner2_names"] = base_c[
                "parcel_owner2_names"
            ].values
            diff_matched_c.at[i, "Number ofAnimalUnits"] = base_c[
                "Number ofAnimalUnits"
            ].values
            diff_matched_c.at[i, "animal_unit_estimate"] = base_c[
                "animal_unit_estimate"
            ].values
            diff_matched_c.at[i, "right_discrep"] = base_c["discrep"].values
            diff_matched_c.at[i, "diff_ID"] = int(np.floor(i / 2))

    return diff_matched_c


def merge_clusters_permits(
    clusters: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame,
    match_distance: float = 400,
    drop_multi_matches: bool = True,
    sum_satellite_counts: bool = True,
    discrep_analysis: bool = False,
    only_dairy: bool = True,
    **kwargs,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Matches clusters to WDNR CAFO permit data, and calculates discrepancy between our area-based
        animal unit estimate and the permit's animal unit estimate.
    Args:
        - clusters: cluster dataframe, output of cluster.cluster
        - permits: WDNR CAFO permit data, including main and satellite sites
        - match_distance: max distance to match a cluster with a CAFO permit, in ft
        - sum_satellite_counts: whether or not to collect animal unit estimates from
            satellite sites into a single estimate for comparison with WDNR
    Returns:
        - matched_clusters: dataframe of clusters matched to permit locations, with discrepancy in
            animal unit estimate
        - discrep_stats: dataframe of discrep statistics
    """
    # Drop non-dairy permits
    if only_dairy:
        permits = permits[permits['AnimalType'] == 'Dairy']
    
    # Merge clusters with main facilities first
    matched_main_facility_clusters = clusters.sjoin_nearest(
        permits[pd.isna(permits["SATELLITE_"])],
        max_distance=match_distance,
        distance_col="match_distance",
    ).drop("index_right", axis=1)
    # Merge clusters with satellite facilities second
    matched_sat_facility_clusters = clusters.sjoin_nearest(
        permits[~pd.isna(permits["SATELLITE_"])],
        max_distance=match_distance,
        distance_col="match_distance",
    ).drop("index_right", axis=1)
    # Combine the two sets, keeping only the main facility match if a cluster is matched to both.
    # Sort so that main-facility rows (from matched_main_facility_clusters) always appear first,
    # then by match_distance to break any remaining ties deterministically.
    matched_main_facility_clusters["_match_priority"] = 0
    matched_sat_facility_clusters["_match_priority"] = 1
    matched_clusters = pd.concat(
        [matched_main_facility_clusters, matched_sat_facility_clusters]
    )
    matched_clusters = matched_clusters.sort_values(
        ["_match_priority", "match_distance"]
    ).drop(columns=["_match_priority"])
    matched_clusters.drop_duplicates("polygon_indices", inplace=True)

    if drop_multi_matches:
        # Where more than one cluster matches to a permit, take the closer cluster.
        matched_clusters.sort_values(by=["CAFO_index", "match_distance"], inplace=True)
        matched_clusters.drop_duplicates(
            subset="CAFO_index", keep="first", inplace=True
        )

    if "animal_unit_estimate" not in matched_clusters.columns:
        matched_clusters = est_au.sample_and_calc_au(matched_clusters, **kwargs)

    if sum_satellite_counts:
        # Keep only main facilities, adjusting our animal unit estimate to reflect the sum of
        # estimates at clusters matched to satellite facilities
        if "animal_units_lower" in matched_clusters.columns:
            # If the animal unit estimates have lower and upper bounds, combine the uncertainties.
            # NOTE: assumes that animal unit estimation uncertainty is independent among satellite sites of the same CAFO.
            matched_clusters["animal_unit_variance"] = (
                (
                    matched_clusters["animal_units_upper"]
                    - matched_clusters["animal_unit_estimate"]
                )
                / 1.96
            ) ** 2

            # Assuming matched_clusters is a GeoDataFrame
            matched_clusters_summed = matched_clusters.groupby(
                ["Facility ID (FIN)"]
            ).agg(
                animal_unit_estimate=pd.NamedAgg(
                    column="animal_unit_estimate", aggfunc="sum"
                ),
                animal_unit_variance=pd.NamedAgg(
                    column="animal_unit_variance", aggfunc="sum"
                ),
                allowable_animal_units=pd.NamedAgg(
                    column="Number ofAnimalUnits", aggfunc="sum"
                ),
                geometry=pd.NamedAgg(
                    column="geometry", aggfunc=unary_union
                ),  # Combine geometries
                has_satellite_sites=pd.NamedAgg(
                    column="SATELLITE_", aggfunc=lambda x: any(~pd.isna(x))
                ),
            )
            matched_clusters_summed["animal_units_lower"] = (
                matched_clusters_summed.apply(
                    lambda x: x.animal_unit_estimate
                    - np.sqrt(x.animal_unit_variance) * 1.96,
                    axis=1,
                )
            )

            matched_clusters_summed["animal_units_upper"] = (
                matched_clusters_summed.apply(
                    lambda x: x.animal_unit_estimate
                    + np.sqrt(x.animal_unit_variance) * 1.96,
                    axis=1,
                )
            )

            matched_clusters_summed.drop("animal_unit_variance", axis=1, inplace=True)
            matched_clusters.drop(
                ["animal_units_lower", "animal_units_upper"], axis=1, inplace=True
            )

        else:
            matched_clusters_summed = matched_clusters.groupby(
                ["Facility ID (FIN)"]
            ).agg(
                geometry=pd.NamedAgg(
                    column="geometry", aggfunc=unary_union
                ),  # Combine geometries
                animal_unit_estimate=pd.NamedAgg(
                    column="animal_unit_estimate", aggfunc="sum"
                ),
                allowable_animal_units=pd.NamedAgg(
                    column="Number ofAnimalUnits", aggfunc="sum"
                ),
                has_satellite_sites=pd.NamedAgg(
                    column="SATELLITE_", aggfunc=lambda x: any(~pd.isna(x))
                ),
            )

        matched_clusters = matched_clusters.drop("animal_unit_estimate", axis=1).merge(
            matched_clusters_summed, on="Facility ID (FIN)"
        )
        matched_clusters = matched_clusters[pd.isna(matched_clusters["SATELLITE_"])]
        # reset the geometry, after the merge
        matched_clusters["geometry"] = matched_clusters["geometry_y"]
        matched_clusters.drop(
            ["geometry_x", "geometry_y"], axis=1, inplace=True, errors="ignore"
        )
        matched_clusters = gpd.GeoDataFrame(matched_clusters, geometry="geometry")
        matched_clusters["original_cluster_area_m2"] = matched_clusters.cluster_area_m2
        matched_clusters["cluster_area_m2"] = matched_clusters.geometry.area
    if discrep_analysis:
        # Drop cases where WDNR estimate of number of animal units is missing
        #print(f"""Dropping {sum(pd.isna(matched_clusters["Number ofAnimalUnits"]))} cases where WDNR estimate of number of animal units is missing""")
        
        matched_clusters = matched_clusters[
            ~pd.isna(matched_clusters["Number ofAnimalUnits"])
        ]

        # Discrepancy analysis
        matched_clusters["discrep"] = (
            matched_clusters["animal_unit_estimate"]
            - matched_clusters["Number ofAnimalUnits"]
        )
        matched_clusters.replace([np.inf, -np.inf], np.nan, inplace=True)

        # saving a few discrep stats
        mean_discrep = matched_clusters["discrep"].mean().round(2)
        mean_abs_discrep = np.abs(matched_clusters["discrep"]).mean().round(2)
        median_abs_discrep = np.abs(matched_clusters["discrep"]).median().round(2)
        median_abs_discrep_pct = (
            np.abs(
                matched_clusters["discrep"] / matched_clusters["Number ofAnimalUnits"]
            )
            .median()
            .round(2)
        )
        r2 = r2_score(
            matched_clusters["Number ofAnimalUnits"],
            matched_clusters["animal_unit_estimate"],
        ).round(2)

        corr_coeff = np.corrcoef(
            matched_clusters["Number ofAnimalUnits"],
            matched_clusters["animal_unit_estimate"],
        )[0, 1].round(2)

        n_within_uncertainty_bounds = np.NaN
        if "animal_units_lower" in matched_clusters.columns:
            n_within_uncertainty_bounds = (
                matched_clusters[
                    (
                        matched_clusters["Number ofAnimalUnits"]
                        > matched_clusters["animal_units_lower"]
                    )
                    & (
                        matched_clusters["Number ofAnimalUnits"]
                        < matched_clusters["animal_units_upper"]
                    )
                    & (matched_clusters["Number ofAnimalUnits"] > 0)
                ].shape[0]
                / matched_clusters[
                    (matched_clusters["Number ofAnimalUnits"] > 0)
                ].shape[0]
            )
        discrep_data = {
            "mean_discrep": [mean_discrep],
            "mean_abs_discrep": [mean_abs_discrep],
            "median_abs_discrep": [median_abs_discrep],
            "median_abs_discrep_pct": [median_abs_discrep_pct],
            "r2": [r2],
            "corr_coeff": [corr_coeff],
            "n_within_uncertainty_bounds": [n_within_uncertainty_bounds],
        }
        discrep_stats = pd.DataFrame(discrep_data)
    else:
        discrep_stats = "-"

    return matched_clusters, discrep_stats


# def merge_clusters_permits(
#     clusters: gpd.GeoDataFrame,
#     permits: gpd.GeoDataFrame,
#     match_distance: float = 400,
#     drop_multi_matches: bool = True,
#     sum_satellite_counts: bool = True,
#     discrep_analysis: bool = False,
#     **kwargs,
# ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
#     """Matches clusters to WDNR CAFO permit data, and calculates discrepancy between our area-based
#         animal unit estimate and the permit's animal unit estimate.
#     Args:
#         - clusters: cluster dataframe, output of cluster.cluster
#         - permits: WDNR CAFO permit data, including main and satellite sites
#         - match_distance: max distance to match a cluster with a CAFO permit, in ft
#         - sum_satellite_counts: whether or not to collect animal unit estimates from
#             satellite sites into a single estimate for comparison with WDNR
#     Returns:
#         - matched_clusters: dataframe of clusters matched to permit locations, with discrepancy in
#             animal unit estimate
#         - discrep_stats: dataframe of discrep statistics
#     """
#     # Merge clusters with main facilities first
#     matched_main_facility_clusters = clusters.sjoin_nearest(
#         permits[pd.isna(permits["SATELLITE_"])],
#         max_distance=match_distance,
#         distance_col="match_distance",
#     ).drop("index_right", axis=1)
#     # Merge clusters with satellite facilities second
#     matched_sat_facility_clusters = clusters.sjoin_nearest(
#         permits[~pd.isna(permits["SATELLITE_"])],
#         max_distance=match_distance,
#         distance_col="match_distance",
#     ).drop("index_right", axis=1)
#     # Combine the two sets, keeping only the main facility match if a cluster is matched to both
#     matched_clusters = pd.concat(
#         [matched_main_facility_clusters, matched_sat_facility_clusters]
#     )
#     matched_clusters.drop_duplicates("polygon_indices", inplace=True)

#     if drop_multi_matches:
#         # Where more than one cluster matches to a permit, take the closer cluster.
#         matched_clusters.sort_values(by=["CAFO_index", "match_distance"], inplace=True)
#         matched_clusters.drop_duplicates(
#             subset="CAFO_index", keep="first", inplace=True
#         )

#     if "animal_unit_estimate" not in matched_clusters.columns:
#         matched_clusters = est_au.sample_and_calc_au_units(matched_clusters, **kwargs)

#     if sum_satellite_counts:
#         # Keep only main facilities, adjusting our animal unit estimate to reflect the sum of
#         # estimates at clusters matched to satellite facilities
#         if "animal_units_lower" in matched_clusters.columns:
#             # If the animal unit estimates have lower and upper bounds, combine the uncertainties.
#             # NOTE: assumes that animal unit estimation uncertainty is independent among satellite sites of the same CAFO.
#             matched_clusters["animal_unit_variance"] = (
#                 (
#                     matched_clusters["animal_units_upper"]
#                     - matched_clusters["animal_unit_estimate"]
#                 )
#                 / 1.96
#             ) ** 2
#             matched_clusters_summed = matched_clusters.groupby(
#                 ["Facility ID (FIN)"]
#             ).agg(
#                 animal_unit_estimate=pd.NamedAgg(
#                     column="animal_unit_estimate", aggfunc="sum"
#                 ),
#                 animal_unit_variance=pd.NamedAgg(
#                     column="animal_unit_variance", aggfunc="sum"
#                 ),
#             )
#             matched_clusters_summed["animal_units_lower"] = (
#                 matched_clusters_summed.apply(
#                     lambda x: x.animal_unit_estimate
#                     - np.sqrt(x.animal_unit_variance) * 1.96,
#                     axis=1,
#                 )
#             )
#             matched_clusters_summed["animal_units_upper"] = (
#                 matched_clusters_summed.apply(
#                     lambda x: x.animal_unit_estimate
#                     + np.sqrt(x.animal_unit_variance) * 1.96,
#                     axis=1,
#                 )
#             )
#             matched_clusters_summed.drop("animal_unit_variance", axis=1, inplace=True)
#             matched_clusters.drop(
#                 ["animal_units_lower", "animal_units_upper"], axis=1, inplace=True
#             )
#         else:
#             matched_clusters_summed = matched_clusters.groupby(
#                 ["Facility ID (FIN)"]
#             ).agg(
#                 animal_unit_estimate=pd.NamedAgg(
#                     column="animal_unit_estimate", aggfunc="sum"
#                 )
#             )
#         matched_clusters = matched_clusters.drop("animal_unit_estimate", axis=1).merge(
#             matched_clusters_summed, on="Facility ID (FIN)"
#         )
#         matched_clusters = matched_clusters[pd.isna(matched_clusters["SATELLITE_"])]
#     if discrep_analysis:
#         # Discrepancy analysis
#         matched_clusters["discrep"] = (
#             matched_clusters["animal_unit_estimate"]
#             - matched_clusters["Number ofAnimalUnits"]
#         )
#         matched_clusters.replace([np.inf, -np.inf], np.nan, inplace=True)

#         # saving a few discrep stats
#         mean_discrep = matched_clusters["discrep"].mean().round(2)
#         mean_abs_discrep = np.abs(matched_clusters["discrep"]).mean().round(2)
#         median_abs_discrep = np.abs(matched_clusters["discrep"]).median().round(2)
#         median_abs_discrep_pct = (
#             np.abs(
#                 matched_clusters["discrep"] / matched_clusters["Number ofAnimalUnits"]
#             )
#             .median()
#             .round(2)
#         )
#         r2 = r2_score(
#             matched_clusters["Number ofAnimalUnits"],
#             matched_clusters["animal_unit_estimate"],
#         ).round(2)
#         n_within_uncertainty_bounds = np.NaN
#         if "animal_units_lower" in matched_clusters.columns:
#             n_within_uncertainty_bounds = (
#                 matched_clusters[
#                     (
#                         matched_clusters["Number ofAnimalUnits"]
#                         > matched_clusters["animal_units_lower"]
#                     )
#                     & (
#                         matched_clusters["Number ofAnimalUnits"]
#                         < matched_clusters["animal_units_upper"]
#                     )
#                     & (matched_clusters["Number ofAnimalUnits"] > 0)
#                 ].shape[0]
#                 / matched_clusters[
#                     (matched_clusters["Number ofAnimalUnits"] > 0)
#                 ].shape[0]
#             )
#         discrep_data = {
#             "mean_discrep": [mean_discrep],
#             "mean_abs_discrep": [mean_abs_discrep],
#             "median_abs_discrep": [median_abs_discrep],
#             "median_abs_discrep_pct": [median_abs_discrep_pct],
#             "r2": [r2],
#             "n_within_uncertainty_bounds": [n_within_uncertainty_bounds],
#         }
#         discrep_stats = pd.DataFrame(discrep_data)
#     else:
#         discrep_stats = "-"

#     return matched_clusters, discrep_stats


def stratified_sample(dairies, base_value, width):
    """This function returns subsets of the sample that have WDNR animal unit estimation values within +/- the width of the base value"""
    # select for cafos within a certain size (near CAFO cut-off)
    near_x_dairies = dairies[
        dairies["Number ofAnimalUnits"].between(base_value - width, base_value + width)
    ]

    return near_x_dairies


def get_cluster_diff(
    clusters_a: gpd.GeoDataFrame, clusters_b: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Find clusters which are not in common between two clustered datasets. Essentially an anti-join.
    Args:
        clusters_a: first set of clusters (output of cluster.cluster)
        clusters_b: second set of clusters (output of cluster.cluster)
    Returns:
        DataFrame of clusters.
    """
    # Copy dataframes to avoid modification
    clusters_left = deepcopy(clusters_a)
    clusters_right = deepcopy(clusters_b)
    # Set up indices
    clusters_left["left_index"] = range(0, clusters_left.shape[0], 1)
    clusters_right["right_index"] = range(0, clusters_right.shape[0], 1)

    # Make sure polygon_indices are sorted and mergable
    clusters_left["polygon_indices"].apply(lambda x: x.sort())
    clusters_right["polygon_indices"].apply(lambda x: x.sort())
    clusters_left["polygon_indices"] = clusters_left["polygon_indices"].astype(str)
    clusters_right["polygon_indices"] = clusters_right["polygon_indices"].astype(str)

    # Perform outer merge
    outer_merge = clusters_left.merge(
        clusters_right, on=["polygon_indices"], how="outer"
    )
    # Calculate anti-join
    diff = outer_merge[
        (pd.isna(outer_merge["left_index"])) | (pd.isna(outer_merge["right_index"]))
    ]

    # Coalesce results
    result = pd.concat(
        [
            diff[["polygon_indices", "parcel_indices_x", "left_index"]]
            .dropna()
            .reset_index(drop=True),
            diff[["polygon_indices", "parcel_indices_y", "right_index"]]
            .dropna()
            .reset_index(drop=True),
        ]
    )
    result["parcel_indices"] = result["parcel_indices_x"].combine_first(
        result["parcel_indices_y"]
    )
    result.sort_values(by="polygon_indices", inplace=True)
    result = result[
        ["polygon_indices", "parcel_indices", "left_index", "right_index"]
    ].reset_index(drop=True)

    return result


# ------------------ #
# TODO: This function is more of a util, so should live in a different script
def calculate_slope_from_dem(dem_data_path: str, save_path: str, nodata_value: float = -9999.0):
    """
    Calculate slope (in degrees) from a digital elevation model without external geomorphology libs.
    Args:
        - dem_data_path: path to digital elevation model data
        - save_path: path to save slope data
        - nodata_value: value to write where DEM has no data
    """
    with rasterio.open(dem_data_path) as dem_data:
        dem = dem_data.read(1, masked=True).astype("float32")
        # Fill masked cells with NaN so gradient math propagates nodata regions
        dem_array = dem.filled(np.nan)

        transform = dem_data.transform
        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        # Compute dz/dx and dz/dy using pixel spacing; ignore warnings from NaNs
        with np.errstate(invalid="ignore"):
            gy, gx = np.gradient(dem_array, pixel_size_y, pixel_size_x, edge_order=1)
            slope_radians = np.arctan(np.sqrt(gx**2 + gy**2))
            slope_degrees = np.degrees(slope_radians)

        slope_degrees[np.isnan(dem_array)] = nodata_value

        metadata = dem_data.meta.copy()
        metadata.update({
            "dtype": "float32",
            "nodata": nodata_value,
            "driver": "GTiff",
        })

        output_path = save_path / "WI_slope.tif"
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(slope_degrees.astype("float32"), 1)


def analyze_water_pollution_stats(
    cluster_data: gpd.GeoDataFrame,
    slope_data_path: str,
    all_waters: gpd.GeoDataFrame,
    impaired_waters: gpd.GeoDataFrame,
    water_table_depth: gpd.GeoDataFrame,
    snapmaps_layers: dict,
    slope_distance_buffer: float = 500,
    slope_metadata: dict = {
        "driver": "GTiff",
        "dtype": np.dtype("float32"),
        "nodata": -9999.0,
        "width": 16282,
        "height": 17310,
        "count": 1,
        "crs": 3071,
        "transform": affine.Affine(
            30.0, 0.0, 289830.6823445987, 0.0, -30.0, 739413.4560018647
        ),
    },
):
    """
    This function analyzes water pollution risk statistics for a given cluster dataset.
    Args:
        - cluster_data: cluster dataset
        - slope_data_path: path to raster slope data (output of calculate_slope_from_dem)
        - all_waters: all open waters data
        - impaired_waters: impaired waters data
        - water_table_depth: water table depth data
        - snapmaps_layers: dictionary of SNAPMaps layers, with each value being a gpd.GeoDataFrame. 
            Includes layers for: silurian bedrock shallow soils, nmp restricted spreading areas, etc.
        - slope_distance_buffer: distance for buffer around cluster geometry to cover at least 1 pixel of the slope raster
        - slope_metadata: attributes of slope raster
    Returns:
        Table of water pollution risk statistics
    """
    #Quick CRS check
    expected_crs = CRS.from_epsg(3071)
    if cluster_data.crs != expected_crs:
        print(f"Cluster data CRS is {cluster_data.crs}, expected {expected_crs}. Reprojecting...")
        cluster_data = cluster_data.to_crs(3071)
    for layer in snapmaps_layers.values():
        if layer.crs != expected_crs:
            print(f"{layer} CRS is {layer.crs}, expected {expected_crs}. Reprojecting...")
            layer = layer.to_crs(3071)

    # Calculate distance to water bodies
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            all_waters[["geometry"]].to_crs(3071), distance_col="water_distance"
        )
    )
    cluster_data = cluster_data.sort_values("water_distance").drop_duplicates("polygon_indices", keep="first")

    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            impaired_waters[["geometry"]].to_crs(3071),
            distance_col="impaired_water_distance",
        )
    )
    cluster_data = cluster_data.sort_values("impaired_water_distance").drop_duplicates("polygon_indices", keep="first")
    cluster_data["closest_water_impaired"] = (
        cluster_data["impaired_water_distance"] < cluster_data["water_distance"]
    )

    # Buffer cluster geometry in order to cover at least 1 pixel of the slope raster
    cluster_data["buffered_geometry"] = cluster_data["geometry"].buffer(
        slope_distance_buffer
    )
    mean_slopes = []

    # Iterate over each row (multipolygon) in the GeoDataFrame
    for index, row in tqdm(cluster_data.iterrows()):
        # Mask the raster using the geometry of the current multipolygon
        with rasterio.open(slope_data_path, "r", **slope_metadata) as slope:
            masked_data, mask_transform = mask(
                slope, [row["buffered_geometry"]], crop=True
            )

        # Calculate the mean of the masked data
        mean_slope = np.mean(masked_data[masked_data != slope.nodata])

        # Append the mean value to the list
        mean_slopes.append(mean_slope)

    # Add a new column to the GeoDataFrame with the mean slope values
    cluster_data["mean_slope"] = mean_slopes

    # Reset geometry to cluster centroid
    cluster_data["centroid"] = cluster_data.centroid
    cluster_data.set_geometry("centroid", inplace=True)

    # Calculate water table depth
    cluster_data = remove_index_l_r(
        cluster_data.sjoin(water_table_depth[["geometry", "WTGW_VALUE"]], how="left")
    )
    # Recode
    cluster_data["gw_50"] = cluster_data["WTGW_VALUE"] == 10
    cluster_data["gw_20"] = cluster_data["WTGW_VALUE"] == 5
    cluster_data["gw_0"] = cluster_data["WTGW_VALUE"] == 1
    cluster_data.drop("WTGW_VALUE", axis=1, inplace=True)

    # SNAPmaps data
    # Change the primary key to hashable type, so we can deduplicate the spatial joins as they happen
    # Note: the spatial joins could return multiple records if multiple polygons are equidistant, 
    # so we need to deduplicate by keeping only the minimum distance for each cluster
    cluster_data['polygon_indices'] = cluster_data['polygon_indices'].apply(tuple)

    # Soil depth less than 5ft depth to bedrock
    start_time = time.time()
    print(f"Shape of cluster_data before sjoin_nearest with NM_590_DEPTH_BEDROCK_LT5FT: {cluster_data.shape}")
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['NM_590_DEPTH_BEDROCK_LT5FT'][["geometry"]],
            distance_col="bedrock_lt5ft_distance"
        )
    )
    print(f"Time for sjoin_nearest with NM_590_DEPTH_BEDROCK_LT5FT: {time.time() - start_time} seconds")


    # Deduplicate in case multiple records returned - keep only the minimum distance for each cluster
    # Each cluster has the key 'polygon_indices'
    cluster_data = cluster_data.sort_values('bedrock_lt5ft_distance').drop_duplicates('polygon_indices', keep='first')



    # Create a dummy var for if the facility is in one of these areas
    cluster_data['bedrock_lt5ft_dummy'] = cluster_data['bedrock_lt5ft_distance'] == 0

    # Thickness of soil over silurian bedrocks
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['SILURIAN_2_5'][["geometry"]],
            distance_col="silurian_2_5_distance",
        )
    )
    print(f"Time for sjoin_nearest with SILURIAN_2_5: {time.time() - start_time} seconds")

    # Drop duplicate matche sof equal distance
    cluster_data = cluster_data.sort_values('silurian_2_5_distance').drop_duplicates('polygon_indices', keep='first')

    cluster_data['silurian_2_5_dummy'] = cluster_data['silurian_2_5_distance'] == 0

    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['SILURIAN_0_2'][["geometry"]],
            distance_col="silurian_0_2_distance",
        )
    )
    print(f"Time for sjoin_nearest with SILURIAN_0_2: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('silurian_0_2_distance').drop_duplicates('polygon_indices', keep='first')

    cluster_data['silurian_0_2_dummy'] = cluster_data['silurian_0_2_distance'] == 0

    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['NM_590_SHALLOW_SILURIAN'][["geometry"]],
            distance_col="shallow_silurian_distance",
        )
    )
    print(f"Time for sjoin_nearest with NM_590_SHALLOW_SILURIAN: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('shallow_silurian_distance').drop_duplicates('polygon_indices', keep='first')
    cluster_data['shallow_silurian_dummy'] = cluster_data['shallow_silurian_distance'] == 0

    # Distance to Surface Water Quality Management Areas (SWQMA)
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['NM_590_SWQMA_300FT'][["geometry"]],
            distance_col="swqma_300ft_distance",
        )
    )
    print(f"Time for sjoin_nearest with NM_590_SWQMA_300FT: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('swqma_300ft_distance').drop_duplicates('polygon_indices', keep='first')

    cluster_data['swqma_300ft_dummy'] = cluster_data['swqma_300ft_distance'] == 0

    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['SWQMA1000FT'][["geometry"]],
            distance_col="swqma_1000ft_distance",
        )
    )
    print(f"Time for sjoin_nearest with SWQMA1000FT: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('swqma_1000ft_distance').drop_duplicates('polygon_indices', keep='first')

    cluster_data['swqma_1000ft_dummy'] = cluster_data['swqma_1000ft_distance'] == 0

    # Distance to cafo w soil spreading restriction areas
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['CAFO_W'][["geometry"]],
            distance_col="cafo_w_restrict_distance",
        )
    )
    print(f"Time for sjoin_nearest with CAFO_W: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('cafo_w_restrict_distance').drop_duplicates('polygon_indices', keep='first')

    # Distance to cafo r soil spreading restriction areas
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['CAFO_R'][["geometry"]],
            distance_col="cafo_r_restrict_distance",
        )
    )
    print(f"Time for sjoin_nearest with CAFO_R: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('cafo_r_restrict_distance').drop_duplicates('polygon_indices', keep='first')

    # Slope > 12 areas
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['SLOPES_GREATER_12'][["geometry"]],
            distance_col="slope_greater_12_distance",
        )
    )
    print(f"Time for sjoin_nearest with SLOPES_GREATER_12: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('slope_greater_12_distance').drop_duplicates('polygon_indices', keep='first')

    cluster_data['slope_greater_12_dummy'] = cluster_data['slope_greater_12_distance'] == 0

    # Water layers
    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['HYDRO_INTERMIT'][["geometry"]],
            distance_col="hydro_intermit_distance",
        )
    )
    print(f"Time for sjoin_nearest with HYDRO_INTERMIT: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('hydro_intermit_distance').drop_duplicates('polygon_indices', keep='first')

    start_time = time.time()
    cluster_data = remove_index_l_r(
        cluster_data.sjoin_nearest(
            snapmaps_layers['HYDRO_PERENNIAL'][["geometry"]],
            distance_col="hydro_perennial_distance",
        )
    )
    print(f"Time for sjoin_nearest with HYDRO_PERENNIAL: {time.time() - start_time} seconds")

    cluster_data = cluster_data.sort_values('hydro_perennial_distance').drop_duplicates('polygon_indices', keep='first')

    # Commengint out water because this is the biggest shapefile that sinks a lot of time 
    # start_time = time.time()
    # cluster_data = remove_index_l_r(
    #     cluster_data.sjoin_nearest(
    #         snapmaps_layers['WATER'][["geometry"]],
    #         distance_col="snapmaps_water_distance",
    #     )
    # )
    # print(f"Time for sjoin_nearest with WATER: {time.time() - start_time} seconds")

    # cluster_data = cluster_data.loc[cluster_data.groupby('polygon_indices')['snapmaps_water_distance'].idxmin()]

    return cluster_data
