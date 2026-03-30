# This script contains functions to load and analyze segmentation model outputs,
# primarily in terms of performance relative to human labels.

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
import cluster as cluster
import itertools
from pathlib import Path
import config.config_params as cfg


def match_CF_three_four_band(
    cf_clusters_exc_EWG, three_band_clusters, four_band_clusters
):
    """match the CF clusters to the three and four band clusters
    Args:
        cf_clusters_exc_EWG: CF clusters outside of the EWG study region
        three_band_clusters: three band clusters
        four_band_clusters: four band clusters
    Returns:
        CF_three_four_band_match: a dataframe with the CF cluster index, three band cluster index, and four band cluster index
    """

    cf_clusters_exc_EWG["CF_cluster_index"] = cf_clusters_exc_EWG.index
    three_band_clusters["three_band_cluster_index"] = three_band_clusters.index
    four_band_clusters["four_band_cluster_index"] = four_band_clusters.index

    CF_four_band_match = cf_clusters_exc_EWG.sjoin(
        four_band_clusters, how="inner", op="intersects"
    )

    CF_four_band_match.drop(columns=["index_right"], inplace=True)

    CF_three_four_band_match = CF_four_band_match.sjoin(
        three_band_clusters, how="inner", op="intersects"
    )

    CF_three_four_band_match = CF_three_four_band_match[
        ["CF_cluster_index", "three_band_cluster_index", "four_band_cluster_index"]
    ]

    return CF_three_four_band_match


def label_by_ewg_region(dataset: gpd.GeoDataFrame,
                     county_data: gpd.GeoDataFrame,
                     predicate: str = "intersects"):
    """Adds a column specifying whether a polygon is in the EWG region.
    Args:
        dataset: polygon or cluster dataset
        county_data: counties dataset
        predicate: how to join the datasets, default "intersects"
    """
    if 'train_counties' not in county_data.columns:
        county_data['train_counties'] = county_data['COUNTY_NAM'].isin(cfg.EWG_COUNTIES)
    dataset_ewg = dataset.sjoin(county_data[county_data['train_counties']], predicate=predicate)

    # Add column for whether geometry is in the EWG region:
    # cluster dataset
    if 'polygon_indices' in dataset.columns:
        dataset['ewg_region'] = dataset["polygon_indices"].isin(dataset_ewg['polygon_indices'])
    # polygon dataset
    elif 'polygon_index' in dataset.columns:
        dataset['ewg_region'] = dataset["polygon_index"].isin(dataset_ewg['polygon_index'])
    else:
        raise ValueError("polygons must have either 'polygon_index' or 'polygon_indices' column")

    return dataset


# -----------CLUSTER LEVEL ANALYSIS FUNCTIONS----------------- #
def match_predictions_to_labels(three_band_clusters=None, four_band_clusters=None, CF_clusters=None, how='inner'):
    """
    This function matches the model predictions to the CF labels. It does this by finding the nearest CF label to each model prediction.
    Args:
        three_band_clusters: dataframe with model predictions from the three band model
        four_band_clusters: dataframe with model predictions from the four band model
        CF_clusters: dataframe with CF labels
        how: how to join the dataframes, default 'inner', could be 'left', 'right', 'outer'
    Returns:
        corr_test_three_band: dataframe with the model predictions from the three band model and the nearest CF label
    """
    CF_clusters["CF_cluster_area"] = CF_clusters.geometry.area.values

    if three_band_clusters is not None:
        three_band_clusters["model_cluster_area"] = three_band_clusters.geometry.area.values
        corr_test_three_band = gpd.sjoin_nearest(three_band_clusters,
        CF_clusters, max_distance=10, how=how)
    
    if four_band_clusters is not None:
        four_band_clusters["model_cluster_area"] = four_band_clusters.geometry.area.values
        corr_test_four_band = gpd.sjoin_nearest(four_band_clusters,
                                            CF_clusters, max_distance=10, how=how)

   
    if three_band_clusters is not None:
        if four_band_clusters is not None:
            return corr_test_three_band, corr_test_four_band
        else:
            return corr_test_three_band
    elif four_band_clusters is not None:
        return corr_test_four_band
    else:
        raise ValueError("Must provide at least one of three_band_clusters or four_band_clusters")

def find_clusters_in_region_viewed_by_CF(CF_annotations, predictions):
    """Find the model predicted clusters that was viewed by CF. Here, we use the jpeg names of the CF labels
    (from CF excluding the EWG region).

    Args:
        CF_annoations: dataframe with the CF polygons (with jpeg name(s) as a column)
        predictions: dataframe with the image bounds (with jpeg name(s) as a column)
    Returns:
        predictions_in_cf_labeled_region: dataframe with the region that was viewed by CF
    """
    if "jpeg_names" in CF_annotations.columns:
        images_sent_to_CF = np.unique(
            list(itertools.chain.from_iterable(CF_annotations["jpeg_names"].values))
        )
        predictions_in_cf_labeled_region = predictions.loc[
            predictions.apply(
                lambda row: any(
                    jpeg_name in images_sent_to_CF for jpeg_name in row["jpeg_names"]
                ),
                axis=1,
            )
        ]

    return predictions_in_cf_labeled_region


def get_unit_recalls(labels: gpd.GeoDataFrame, preds: gpd.GeoDataFrame):
    """Given prediction and label geometry, compute per-unit recalls.

    Args:
        labels: label polygons
        preds: prediction polygons

    Returns:
        unit recalls
    """
    labels_join_preds = labels.sjoin(preds, how="inner", predicate="intersects")

    overlap = labels.geometry.loc[labels_join_preds.index].intersection(
        preds.geometry.loc[labels_join_preds.index_right], align=False
    )

    return overlap.area.groupby(level=0).sum().divide(labels.area, fill_value=0)


def get_unit_precisions(labels: gpd.GeoDataFrame, preds: gpd.GeoDataFrame):
    """Given prediction and label geometry, compute per-unit precisions.

    Args:
        labels: label polygons
        preds: prediction polygons

    Returns:
        unit precisions
    """
    labels_join_preds = labels.sjoin(preds, how="inner", predicate="intersects")

    overlap = preds.geometry.loc[labels_join_preds.index_right].intersection(
        labels.geometry.loc[labels_join_preds.index], align=False
    )

    return overlap.area.groupby(level=0).sum().divide(preds.area, fill_value=0)


def compute_cluster_unit_precision_recall(
    cf_clusters_exc_EWG: gpd.GeoDataFrame,
    three_band_clusters_exc_EWG: gpd.GeoDataFrame,
    four_band_clusters_exc_EWG: gpd.GeoDataFrame,
):
    """Compute unit precision and recall for three and four band models
    Args:
        cf_clusters_exc_EWG (GeoDataFrame): CF clusters outside of EWG study region
        three_band_clusters_exc_EWG (GeoDataFrame): Three band model clusters outside of EWG study region
        four_band_clusters_exc_EWG (GeoDataFrame): Four band model clusters outside of EWG study region
    Returns:
        three_band_precision (array): Unit precision for three band model
        three_band_recall (array): Unit recall for three band model
        four_band_precision (array): Unit precision for four band model
        four_band_recall (array): Unit recall for four band model
    """
    # find the subset of clusters that are within NAIP tiles viewed by CF
    four_band_clusters_in_CF_test = find_clusters_in_region_viewed_by_CF(
        cf_clusters_exc_EWG, four_band_clusters_exc_EWG
    )
    three_band_clusters_in_CF_test = find_clusters_in_region_viewed_by_CF(
        cf_clusters_exc_EWG, three_band_clusters_exc_EWG
    )
    # find unit precisions and recalls for three and four band models
    four_band_precision = get_unit_precisions(
        cf_clusters_exc_EWG, four_band_clusters_in_CF_test
    )

    four_band_recall = get_unit_recalls(
        cf_clusters_exc_EWG, four_band_clusters_in_CF_test
    )

    three_band_precision = get_unit_precisions(
        cf_clusters_exc_EWG, three_band_clusters_in_CF_test
    )

    three_band_recall = get_unit_recalls(
        cf_clusters_exc_EWG, three_band_clusters_in_CF_test
    )
    return (
        three_band_precision,
        three_band_recall,
        four_band_precision,
        four_band_recall,
    )
# -----------POLYGON LEVEL ANALYSIS FUNCTIONS----------------- #
def load_polygons_from_batches(vec_thresh, model_prediction_path, county_data, model):
    """Load the model predictions from the batches, combine all predictions with same threshold, and store as a dataframe"""
    all_predictions_all_thresh = gpd.GeoDataFrame()
    # load all thresholds and all batches and store in all_predictions_all_thresh
    for thresh in tqdm(vec_thresh):
        thresh_predictions = cluster.load_polygons(model_prediction_path / f"{model}_all_predictions-{thresh}.geojson",
                                                    county_data=county_data)
        thresh_predictions["thresh"] = thresh
        thresh_predictions = thresh_predictions[['thresh', 'geometry']]
        all_predictions_all_thresh = pd.concat([all_predictions_all_thresh, thresh_predictions])

    return all_predictions_all_thresh


def find_region_viewed_by_CF(CF_polygons, image_bound_map):
    """Find the region that was viewed by CF using the jpeg names of the CF labels (from CF excluding the EWG region). This is used for the predicted polygons
    Args:
        CF_polygons: dataframe with the CF polygons
        image_bound_map: dataframe with the image bounds
    Returns:
        labeled_region: dataframe with the region that was viewed by CF"""
    # use the file names of the jpegs from the CF polygons to get the image bounds
    images_sent_to_CF = set([jpeg_name for jpeg_names in CF_polygons['jpeg_names'] for jpeg_name in jpeg_names])
    labeled_region = image_bound_map[
        image_bound_map["filename"].isin(list(images_sent_to_CF))
    ]
    return labeled_region


def subset_model_pred(model_predictions, labeled_region):
    """filter the model predictions to just the predictions in NAIP tiles that were sent to CF
    Args:
        model_predictions: dataframe with model predictions
        labeled_region: dataframe with the NAIP tiles that were sent to CF
    Returns:
        prediction_subset: dataframe with the model predictions that overlap with the NAIP tiles that were sent to CF

    """

    # Spatial dissolve the labeled_region
    dissolved_region = labeled_region.dissolve().explode(index_parts=True)[["geometry"]]

    # Check which model_predictions overlap/intersect with the dissolved region
    prediction_subset = gpd.overlay(
        model_predictions, dissolved_region, how="intersection"
    )

    return prediction_subset


def pixel_level_IOU_p_r(polygon_pred, CF_polygons, labeled_region, CRS):
    """Calculate the IOU, precision and recall for a model and CF polygons. Our approach here is to view all predictions
      and CF labels as multipolygons and then find the intersection between the two.
    Args:
        polygon_pred: dataframe with model predictions
        CF_polygons: dataframe with CF polygons
        labeled_region: list of counties that were viewed by CF
        CRS: coordinate reference system
    Returns:
        precision: precision (float)
        recall: recall (float)
        IOU: IOU (float)
    """
    # subset model predictions
    polygon_pred_subset = subset_model_pred(polygon_pred, labeled_region)

    # dissolve the overlapping/duplicated geometries
    polygon_pred_subset_dissolved = polygon_pred_subset.dissolve().explode(
        index_parts=True
    )[["geometry"]]
    CF_polygons_dissolved = CF_polygons.dissolve().explode(index_parts=True)[
        ["geometry"]
    ]

    # combine all prediction and labels into single multipolygon objects
    merged_pred_polygons = polygon_pred_subset_dissolved["geometry"].unary_union
    merged_CF = CF_polygons_dissolved["geometry"].unary_union

    # Create GeoDataFrames from the multipolygon objects
    merged_pred_polygons_gdf = gpd.GeoDataFrame(
        geometry=[merged_pred_polygons], crs=CRS
    )
    merged_CF_gdf = gpd.GeoDataFrame(geometry=[merged_CF], crs=CRS)

    # calculate the intersection between the model predictions and the CF polygons
    union = merged_CF_gdf.overlay(merged_pred_polygons_gdf, how="intersection")
    union = union.dissolve().explode(index_parts=True)[
        ["geometry"]
    ]  # this step might not be necessary

    # now we can calculate the IOU, precision and recall for the model predictions
    precision = union.area.sum() / merged_pred_polygons_gdf.geometry.area
    recall = union.area.sum() / merged_CF_gdf.geometry.area
    IOU = union.area.sum() / (
        merged_CF_gdf.geometry.area
        + merged_pred_polygons_gdf.geometry.area
        - union.area.sum()
    )
    return precision.values[0], recall.values[0], IOU.values[0]


# Create function to calculate building-level recall -- WE ARE NOT CURRENTLY USING THIS FUNCTION
def calc_barn_recall(cf_polygons, model_predictions):
    """Calculate Recall from model predictions.
    Inputs:
    cf_polygons: Ground truth polygons
    model_predictions: CV model predictions

    Returns:
    model_cf_overlaps: dataframe with percent recall
    """
    # make sure that area is a column in the prediction dataframe:
    if "area" not in model_predictions.keys():
        model_predictions["area"] = model_predictions.area
    # Deduplicate CF polygons (known issue)
    cf_polygons_clean = cf_polygons.dissolve().explode(index_parts=True)[["geometry"]]
    cf_polygons_clean["human_area"] = cf_polygons_clean.area
    cf_polygons_clean = cf_polygons_clean.sjoin(
        cf_polygons[["geometry", "polygon_index", "jpeg_name"]]
    )
    cf_polygons_clean.drop_duplicates("human_area", inplace=True)
    cf_polygons_clean.drop("index_right", axis=1, inplace=True)

    # Sum area of all model predictions overlapping with the same CF polygon
    model_cf_overlaps = (
        cf_polygons_clean.sjoin(model_predictions, how="left")
        .groupby(["polygon_index_left", "human_area"])
        .agg(
            model_area=pd.NamedAgg(column="area", aggfunc="sum"),
            model_polys=pd.NamedAgg(column="polygon_index_right", aggfunc="unique"),
        )
    )
    model_cf_overlaps.reset_index(inplace=True)
    model_cf_overlaps["model_polys"] = model_cf_overlaps["model_polys"].apply(
        lambda x: str(x)
    )

    # Retain CF polygons with no model prediction
    model_misses = model_cf_overlaps[model_cf_overlaps["model_area"] == 0]

    # Sum area of all CF polygons overlapping with the same model prediction
    model_cf_overlaps = (
        model_cf_overlaps[model_cf_overlaps["model_area"] != 0]
        .groupby(["model_area", "model_polys"])
        .agg(human_area=pd.NamedAgg(column="human_area", aggfunc="sum"))
    )
    model_cf_overlaps.reset_index(inplace=True)

    model_cf_overlaps = pd.concat(
        [model_cf_overlaps, model_misses.drop("polygon_index_left", axis=1)]
    )

    # Calculate pct recall
    model_cf_overlaps["pct"] = (
        model_cf_overlaps["model_area"] / model_cf_overlaps["human_area"]
    )
    model_cf_overlaps["pct"] = model_cf_overlaps["pct"].apply(lambda x: min(x, 1))
    return model_cf_overlaps


def calculate_prediction_metrics(
    statewide_clusters: gpd.GeoDataFrame(),
    ewg_afos: gpd.GeoDataFrame(),
    WDNR_CAFOs: gpd.GeoDataFrame(),
    counties: gpd.GeoDataFrame(),
    match_distance: float = 400,
):
    """
    This function calculates our various precision and recall metrics for a given set of clusters.
    Inputs:
        statewide_clusters: dataframe of clustered model predictions, statewide
        ewg_afos: dataframe of ewg AFO/CAFO locations in the EWG study region
        WDNR_CAFOs: dataframe of all WDNR-permitted CAFOs statewide
        counties: dataframe of all WI counties
        match_distance: distance threshold for matching clusters to EWG/WDNR locations in meters,
            default 400
    Returns:
        Dataframe of model metrics and sample sizes.
    """
    # Prep data
    counties["train_counties"] = counties["COUNTY_NAM"].isin(cfg.EWG_COUNTIES)

    WDNR_CAFOs = WDNR_CAFOs.sjoin(
        counties[["geometry", "COUNTY_NAM", "train_counties"]]
    ).drop(columns=["index_right"])

    statewide_clusters = statewide_clusters.sjoin(
        counties[["geometry", "COUNTY_NAM", "train_counties"]]
    ).drop(columns=["index_right"])

    # Calculate recall of permitted dairy CAFOs outside the EWG study region
    CAFO_set = WDNR_CAFOs[
        (~WDNR_CAFOs["train_counties"])
        & pd.isna(WDNR_CAFOs["SATELLITE_"])
        & (WDNR_CAFOs["AnimalType"] == "Dairy")
    ]
    tru_pos = CAFO_set.sjoin_nearest(
        statewide_clusters[~statewide_clusters["train_counties"]],
        max_distance=match_distance,
    ).drop_duplicates("CAFO_index")

    WDNR_recall = (
        1
        - CAFO_set[~CAFO_set["CAFO_index"].isin(tru_pos["CAFO_index"])].shape[0]
        / CAFO_set.drop_duplicates("CAFO_index").shape[0]
    )

    # Calculate precision over all EWG AFOs/CAFOs in the EWG study region
    matched_ewg = statewide_clusters[
        statewide_clusters["train_counties"]
    ].sjoin_nearest(ewg_afos, max_distance=match_distance)
    # Drop duplicates (i.e., more than one cluster matched to a single EWG location)
    matched_ewg = matched_ewg.drop_duplicates("EWGID")
    ewg_precision = (
        matched_ewg.shape[0]
        / statewide_clusters[statewide_clusters["train_counties"]].shape[0]
    )

    # Calculate recall of large EWG AFOs/CAFOs in the EWG study region
    ewg_large_dairy = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Large")
    ]
    matched_ewg_large_dairy = ewg_large_dairy.sjoin_nearest(
        statewide_clusters, max_distance=match_distance
    ).drop_duplicates("EWGID")
    ewg_large_dairy_recall = (
        matched_ewg_large_dairy.shape[0]
        / ewg_large_dairy.drop_duplicates("EWGID").shape[0]
    )

    # Compile results
    results = pd.DataFrame(
        data={
            "metric": [
                "Recall of permitted dairy CAFOs (outside EWG study region)",
                "Recall of large EWG dairy AFOs/CAFOs (EWG study region)",
                "Precision over all EWG AFOs/CAFOs (EWG study region)",
            ],
            "value": [
                round(WDNR_recall, 2),
                round(ewg_large_dairy_recall, 2),
                round(ewg_precision, 2),
            ],
            "n": [
                statewide_clusters[~statewide_clusters["train_counties"]].shape[0],
                statewide_clusters[statewide_clusters["train_counties"]].shape[0],
                statewide_clusters[statewide_clusters["train_counties"]].shape[0],
            ],
        }
    )

    return results
