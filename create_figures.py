# This script contains functions to create all figures in the paper draft.

import geopandas as gpd
import pandas as pd
import yaml
from pathlib import Path
import sys
import os
import seaborn as sns
import shapely
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from labellines import labelLines

import pandas as pd
import numpy as np
import scipy.stats as ss
import fiona
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# our files
sys.path.append("..")
import config.config_params as cfg

sys.path.append("../")
import naip_utils
import plotting_utils
import rasterio.plot
import rasterio.merge
import estimate_animal_units as est_au
import cluster
import process_snapmaps
import cluster_analysis_functions as caf
import analyze_model_outputs as amo

# Set default fonts
plt.rcParams["font.family"] = cfg.FIG_FONT_FAMILY
plt.rcParams[f"font.{cfg.FIG_FONT_FAMILY}"] = cfg.FIG_FONT


def ensure_export_format(save_path, fig_format=None):
    """Ensure save_path has the correct file extension based on export format.
    
    Args:
        save_path: Path object or string to save the figure
        fig_format: Export format ('svg' or 'png'). If None, uses cfg.FIG_EXPORT_FORMAT
        
    Returns:
        Path with correct extension
    """
    if fig_format is None:
        fig_format = cfg.FIG_EXPORT_FORMAT
    
    save_path = Path(save_path)
    # Replace any existing extension with the desired format
    return save_path.with_suffix(f'.{fig_format}')


def cluster_level_precision_recall_hist_single_threshold(
    cf_clusters_exc_EWG: gpd.GeoDataFrame,
    three_band_clusters_exc_EWG: gpd.GeoDataFrame,
    four_band_clusters_exc_EWG: gpd.GeoDataFrame,
    fig_save_path: str = None,
):
    """Create a histogram of the cluster level precision and recall for a single threshold.
    Args:
        cf_clusters_exc_EWG: geopandas dataframe of CF clusters
        three_band_clusters_exc_EWG: geopandas dataframe of three band model predictions
        four_band_clusters_exc_EWG: geopandas dataframe of four band model predictions
        fig_save_path: path to save the figure
    Returns:
        None
    """
    # compute the unit recall and precision for clusters that are within the test region viewed by CF
    (
        three_band_precision,
        three_band_recall,
        four_band_precision,
        four_band_recall,
    ) = amo.compute_cluster_unit_precision_recall(
        cf_clusters_exc_EWG, three_band_clusters_exc_EWG, four_band_clusters_exc_EWG
    )
    # plot the histogram
    plotting_utils.plot_cluster_level_precision_recall_histograms(
        four_band_precision,
        four_band_recall,
        three_band_precision,
        three_band_recall,
        fig_save_path,
    )
    return None

def model_human_annotation_errors(
        model_clusters: gpd.GeoDataFrame,
        CF_clusters: gpd.GeoDataFrame,
        save_path=None
):
    """
    Plots a histogram  of errors between model predictions and human annotations.
    model_clusters: geopandas dataframe of model predictions
        CF_clusters: geopandas dataframe of CF clusters
        save_path: path to save the figure
    """

    # match the predictions to the CF clusters
    corr_test = amo.match_predictions_to_labels(
        four_band_clusters=model_clusters, CF_clusters=CF_clusters, how='inner'
    )

    corr_test["discrep"] = corr_test["model_cluster_area"] - corr_test["CF_cluster_area"]
    # Plot histogram of discrepancies
    fig = sns.histplot(
        corr_test,
        x="discrep",
        kde=False
    )
    fig.set_title("Model area prediction errors")
    fig.set_xlabel("Model predicted area - human labeled area (m²)")
    fig.set_ylabel("Number of facilities")

    # Truncate x-axis to focus on the middle 95% of data (example limits)
    plt.xlim(-5000, 5000)

    # Print some statistics, like median absolute discrepancy 
    print("Median discrepancy:", corr_test["discrep"].median())
    print("Median absolute discrepancy:", corr_test["discrep"].abs().median())
    print("Mean discrepancy:", corr_test["discrep"].mean())
    print("Mean absolute discrepancy:", corr_test["discrep"].abs().mean())
    
    # Percentage of cases where discrepancy > 0
    print(f"Percentage of cases with discrepancy > 0: {corr_test['discrep'].gt(0).mean() * 100:.2f}%")

    # Print percentage of cases where absolute discrepancy > 1500m^2
    print(f"Percentage of cases with absolute discrepancy > 1500m²: {corr_test['discrep'].abs().gt(1500).mean() * 100:.2f}%")

    # Save plot
    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)


def plot_discrepancy_by_size_category(
    model_clusters: gpd.GeoDataFrame,
    CF_clusters: gpd.GeoDataFrame,
    save_dir=None
):
    """
    Creates error bar plots of raw and absolute discrepancies between model and human annotations,
    grouped by size category.

    Args:
        model_clusters: GeoDataFrame of model predictions with 'model_cluster_area'
        CF_clusters: GeoDataFrame of human annotations with 'CF_cluster_area' and 'size_category'
        save_dir: Directory to save plots (optional)
    """
    # Match predictions to CF clusters
    corr_test = amo.match_predictions_to_labels(
        four_band_clusters=model_clusters, CF_clusters=CF_clusters, how='inner'
    )

    # Calculate discrepancies
    corr_test["discrep"] = corr_test["model_cluster_area"] - corr_test["CF_cluster_area"]
    corr_test["abs_discrep"] = corr_test["discrep"].abs()

    if 'size_category' not in corr_test.columns:
        corr_test["size_category"] = pd.cut(corr_test["CF_cluster_area"], bins=[0, 50, 500, 1000, 2500, 5000,  10000, 500000])


    # Convert size category intervals to strings for labeling
    corr_test["size_category_str"] = corr_test["size_category"].astype(str)

   # No conversion to string for grouping or sorting
    corr_test = corr_test[corr_test["size_category"] != pd.Interval(left=0, right=500)]  # Drop smallest if needed

    # Group using the Interval dtype
    grouped = corr_test.groupby("size_category")["discrep"]
    means = grouped.mean().sort_index()
    stds = grouped.std().sort_index()

    # Use string labels only for display
    labels = [str(interval) for interval in means.index]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=range(len(means)),  # use integer positions for x
        y=means.values,
        yerr=stds.values,
        fmt='o',
        capsize=5
    )
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.xlabel("Size Category (m²)")
    plt.ylabel("Mean Discrepancy (Model - Human, m²)")
    plt.title("Mean Discrepancy by Size Category")
    plt.grid(False)
    if save_dir:
        plt.savefig(ensure_export_format(save_dir / "discrepancy_by_size_category.png"), dpi=300, bbox_inches='tight', format=cfg.FIG_EXPORT_FORMAT)
    plt.show()

    ## Plot 2: Absolute discrepancy error bars
    grouped_abs = corr_test.groupby("size_category")["abs_discrep"]
    means_abs = grouped_abs.mean().sort_index()
    stds_abs = grouped_abs.std().sort_index()

    labels_abs = [str(interval) for interval in means_abs.index]
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x=range(len(means_abs)),
        y=means_abs,
        yerr=stds_abs,
        fmt='o',
        capsize=5
    )
    plt.xlabel("Size Category (m²)")
    plt.ylabel("Mean Absolute Discrepancy (m²)")
    plt.title("Mean Absolute Discrepancy by Size Category")
    plt.xticks(ticks=range(len(labels_abs)), labels=labels_abs, rotation=45)
    plt.grid(False)
    if save_dir:
        plt.savefig(ensure_export_format(save_dir / "absolute_discrepancy_by_size_category.png"), dpi=300, bbox_inches='tight', format=cfg.FIG_EXPORT_FORMAT)
    plt.show()

    # Optional: Print summary statistics
    print("Median discrepancy:", corr_test["discrep"].median())
    print("Median absolute discrepancy:", corr_test["abs_discrep"].median())
    print("Mean discrepancy:", corr_test["discrep"].mean())
    print("Mean absolute discrepancy:", corr_test["abs_discrep"].mean())
    print(f"Percent with discrepancy > 0: {corr_test['discrep'].gt(0).mean() * 100:.2f}%")
    print(f"Percent with abs discrepancy > 1500m²: {corr_test['abs_discrep'].gt(1500).mean() * 100:.2f}%")


def scatter_plot_three_v_four_band_cluster(
    three_band_clusters: gpd.GeoDataFrame,
    four_band_clusters: gpd.GeoDataFrame,
    CF_clusters: gpd.GeoDataFrame,
    save_path=None,
):
    """Plot the area of the predicted clusters vs the CF clusters.
    Args:
        three_band_clusters: geopandas dataframe of three band model predictions
        four_band_clusters: geopandas dataframe of four band model predictions
        CF_clusters: geopandas dataframe of CF clusters
        save_path: path to save the figure
    """
    # Match the predictions to the CF clusters
    corr_test_three_band, corr_test_four_band = amo.match_predictions_to_labels(
        three_band_clusters, four_band_clusters, CF_clusters
    )
    three_band_rmse = np.sqrt(
        (
            (
                corr_test_three_band["model_cluster_area"]
                - corr_test_three_band["CF_cluster_area"]
            )
            ** 2
        ).mean()
    )
    four_band_rmse = np.sqrt(
        (
            (
                corr_test_four_band["model_cluster_area"]
                - corr_test_four_band["CF_cluster_area"]
            )
            ** 2
        ).mean()
    )

    fig = plotting_utils.scatter_plot_CF_vs_prediction_clusters(
        corr_test_three_band,
        corr_test_four_band,
        fig_save_path=save_path,
        title0="Three Band Model",
        title1="Four Band Model",
        suptitle="Cluster-Level Area Comparison",
        ylims=(-2000, 55000),
        xlims=(-2000, 55000),
        figsize=(16, 6),
        x0label="CF area prediction (m²)",
        y0label="Model area prediction (m²)",
        x1label="CF area prediction (m²)",
        y1label="Model area prediction (m²)",
        dpi=300,
        display=True,
    )

    print("Three band RMSE:", three_band_rmse)
    print("Four band RMSE:", four_band_rmse)

def pixel_level_p_r_IOU_curves(
    model_prediction_path: str,
    vec_thresh: np.array,
    CF_polygons: gpd.GeoDataFrame,
    image_bound_map: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    stats_path: str,
    models: list = ["four_band", "three_band"],
    save_path: str = None,
    recalculate_stats: bool = False,
    suppress_legend: bool = False,
):
    """This function plots the pixel level precision-recall curves for the model predictions.
    Args:
        model_prediction_path: path to the model predictions
        vec_thresh: vector of confidence thresholds
        CF_polygons: GeoDataframe of CF polygons
        image_bound_map: GeoDataframe of image bounds
        counties: GeoDataframe of counties
        stats_path: path to save the precision, recall, IOU statistics for
            faster re-plotting
        models: models to include in the plot
        fig_save_path: path to save the figure
        recalculate_stats: whether to recalculate the statistics, or load from
            `stats_path`. Default False.
        save_path: path to save the figure
    """
    p_r_IOU_all = pd.DataFrame()
    if recalculate_stats:
        for model in models:
            p_r_IOU = []
            labeled_region = amo.find_region_viewed_by_CF(CF_polygons, image_bound_map)
            # load polygons
            print(f"Loading {model} model predictions...")
            all_thresh = amo.load_polygons_from_batches(
                vec_thresh,
                model_prediction_path / f"{model}/{model}_full_state",
                counties,
                model,
            )
            # Compute pixel level precision, recall and IOU for each threshold
            print(f"Computing {model} stats...")
            for thresh in vec_thresh:
                predictions = all_thresh[all_thresh["thresh"] == thresh]
                precision, recall, IOU = amo.pixel_level_IOU_p_r(
                    predictions, CF_polygons, labeled_region, cfg.WI_EPSG
                )
                p_r_IOU.append([thresh, precision, recall, IOU])
            p_r_IOU = pd.DataFrame(
                p_r_IOU, columns=["Threshold", "Precision", "Recall", "IOU"]
            )
            p_r_IOU["model"] = model
            p_r_IOU_all = pd.concat([p_r_IOU_all, p_r_IOU])
        p_r_IOU_all.to_csv(stats_path)
    else:
        p_r_IOU_all = pd.read_csv(stats_path)
        p_r_IOU_all = p_r_IOU_all[p_r_IOU_all["model"].isin(models)]

    # plot figure
    fig = plotting_utils.plot_pixel_level_p_r_iou(
        p_r_IOU_all, figsize=(8, 4), fig_save_path=save_path, suptitle="",
        suppress_legend=suppress_legend

    )


# def pixel_level_p_r_IOU_curves(
#     model_prediction_path: str,
#     vec_thresh: np.array,
#     CF_polygons: gpd.GeoDataFrame,
#     image_bound_map: gpd.GeoDataFrame,
#     counties: gpd.GeoDataFrame,
#     stats_path: str,
#     include_sent_but_no_annot: bool = False,
#     models: list = ["four_band", "three_band"],
#     save_path: str = None,
#     recalculate_stats: bool = False,
# ):
#     """This function plots the pixel level precision-recall curves for the model predictions.
#     Args:
#         model_prediction_path: path to the model predictions
#         vec_thresh: vector of confidence thresholds
#         CF_polygons: GeoDataframe of CF polygons
#         image_bound_map: GeoDataframe of image bounds
#         counties: GeoDataframe of counties
#         stats_path: path to save the precision, recall, IOU statistics for
#             faster re-plotting
#         include_sent_but_no_annot: whether to include images sent to CF but not
#         models: models to include in the plot
#         fig_save_path: path to save the figure
#         recalculate_stats: whether to recalculate the statistics, or load from
#             `stats_path`. Default False.
#         save_path: path to save the figure
#     """
#     p_r_IOU_all = pd.DataFrame()

#     if recalculate_stats:
#         for model in models:
#             p_r_IOU = []
            

#             labeled_region = amo.find_region_viewed_by_CF(image_bound_map,
#                                                         include_sent_but_no_annot=include_sent_but_no_annot,
#                                                         annotated_ims_path=Path(cfg.project_path + "/data/Annotations/all_labeled_images.csv"),
#                                                         CF_polygons=CF_polygons)
            
#             # load polygons
#             print(f"Loading {model} model predictions...")
#             all_thresh = amo.load_polygons_from_batches(
#                 vec_thresh,
#                 model_prediction_path / f"{model}/{model}_full_state",
#                 counties,
#                 model,
#             )
#             # Compute pixel level precision, recall and IOU for each threshold
#             print(f"Computing {model} stats...")
#             for thresh in vec_thresh:
#                 predictions = all_thresh[all_thresh["thresh"] == thresh]
#                 precision, recall, IOU = amo.pixel_level_IOU_p_r(
#                     predictions, CF_polygons, labeled_region, cfg.WI_EPSG
#                 )
#                 p_r_IOU.append([thresh, precision, recall, IOU])
#             p_r_IOU = pd.DataFrame(
#                 p_r_IOU, columns=["Threshold", "Precision", "Recall", "IOU"]
#             )
#             p_r_IOU["model"] = model
#             p_r_IOU_all = pd.concat([p_r_IOU_all, p_r_IOU])
#         p_r_IOU_all.to_csv(stats_path)
#     else:
#         p_r_IOU_all = pd.read_csv(stats_path)
#         p_r_IOU_all = p_r_IOU_all[p_r_IOU_all["model"].isin(models)]

#     # plot figure
#     fig = plotting_utils.plot_pixel_level_p_r_iou(
#         p_r_IOU_all, figsize=(8, 4), fig_save_path=save_path, suptitle=""
#     )


def permit_CF_AU_compare(
    train_clusters: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    include_uncertainty: bool = True,
    figsize: tuple = (5, 4),
    save_path: str = None,
):
    """
    This function creates a scatterplot figure to compare cloud factory animal unit estimates
    of permitted dairy CAFOs in the EWG study region to those facilities' permitted animal units.

    Args:
        train_clusters: dataset of cloud factory annotation clusters in the EWG study region
        permit_data: WDNR CAFO permit data
        include_area_uncertainty: whether or not to plot with uncertainty bounds
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    Returns:
        Summary of discrepancy between the two
    """
    # Merge with permits
    merged_ewg_region, err = caf.merge_clusters_permits(
        train_clusters,
        permit_data[permit_data["AnimalType"] == "Dairy"],
        discrep_analysis=True,
    )

    plotting_utils.plot_permit_v_estimate(
        merged_ewg_region,
        include_uncertainty=include_uncertainty,
        fig_save_path=save_path,
        xlabel="Permit animal units",
        ylabel="Estimated animal units",
        xlims=(0, 16000),
        ylims=(0, 16000),
        dpi=cfg.FIG_DPI,
        figsize=figsize,
    )

    return err

def permit_CF_AU_error_plot(permit_matched_cf_clusters, fig_save_path=None):
    """
    Plot the error between permit-reported and estimated animal units.
    """
    permit_matched_cf_clusters["error"] = (
        permit_matched_cf_clusters["animal_unit_estimate"]
        - permit_matched_cf_clusters["Number ofAnimalUnits"]
    )
    permit_matched_cf_clusters["abs_error"] = abs(permit_matched_cf_clusters["error"])
    permit_matched_cf_clusters["pct_error"] = (
        permit_matched_cf_clusters["error"] / permit_matched_cf_clusters["Number ofAnimalUnits"] * 100
    )
    permit_matched_cf_clusters["abs_pct_error"] = abs(permit_matched_cf_clusters["pct_error"])

    permit_matched_cf_clusters = permit_matched_cf_clusters[permit_matched_cf_clusters["pct_error"] < 1000]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        permit_matched_cf_clusters["Number ofAnimalUnits"],
        permit_matched_cf_clusters["pct_error"],
        alpha=0.5,
    )
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Permit-reported Animal Units")
    ax.set_ylabel("Percent Error (%)")
    ax.set_title("Percent Error vs. Permit-reported Animal Units")
    plt.tight_layout()

    if fig_save_path is not None:
        plt.savefig(ensure_export_format(fig_save_path), format=cfg.FIG_EXPORT_FORMAT)
    plt.show()


def plot_permit_estimate_discrepancy_hist(
    matched_clusters: gpd.GeoDataFrame,
    estimate_col: str = "animal_unit_estimate",
    error_metric: str = "raw",
    bins: int = 30,
    figsize: tuple = (8, 6),
    save_path: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Count",
    include_stats: bool = True,
    fig: plt.figure = None,
    ax: plt.axes = None,
    dpi: float = 300,
    display: bool = True,
    trim_percentiles: tuple = (0, 100),  # (lower, upper) percentiles to keep
    stat_line: str = "median",  # Which statistic to show as vertical line: 'median', 'mean', or None
):
    """
    Plot a histogram of discrepancies between permit-reported and estimated animal units.
    
    Args:
        matched_clusters: GeoDataFrame of clusters matched with permits
        estimate_col: Column name for the estimate to use (e.g. 'animal_unit_estimate', 'animal_units_lower', 'animal_units_upper')
        error_metric: Type of error to plot:
            - 'raw': estimate - permit
            - 'abs': |estimate - permit|
            - 'pct': (estimate - permit) / permit * 100
            - 'abs_pct': |(estimate - permit) / permit| * 100
        bins: Number of histogram bins
        figsize: Figure size tuple
        save_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        include_stats: Whether to include summary statistics in title
        fig: Optional existing figure
        ax: Optional existing axis
        dpi: DPI for saved figure
        display: Whether to display the plot
        trim_percentiles: Tuple of (lower, upper) percentiles to keep. Default (0, 100) keeps all data.
                         Example: (5, 95) keeps middle 90% of data.
        stat_line: Which statistic to show as vertical line. Options:
            - 'median': Show median value (default)
            - 'mean': Show mean value
            - None: Don't show any statistic line
    """
    # Filter out any rows with missing values
    clusters_to_plot = matched_clusters.dropna(
        subset=["Number ofAnimalUnits", estimate_col]
    )
    
    # Calculate discrepancies based on error metric
    if error_metric == "raw":
        discrepancies = clusters_to_plot[estimate_col] - clusters_to_plot["Number ofAnimalUnits"]
        if xlabel is None:
            xlabel = "Error (Estimated - Reported) in Animal Units"
    elif error_metric == "abs":
        discrepancies = abs(clusters_to_plot[estimate_col] - clusters_to_plot["Number ofAnimalUnits"])
        if xlabel is None:
            xlabel = "Absolute Error in Animal Units"
    elif error_metric in ["pct", "abs_pct"]:
        # Filter out rows where permit-reported units are 0 to avoid division by zero
        valid_rows = clusters_to_plot["Number ofAnimalUnits"] > 0
        clusters_to_plot = clusters_to_plot[valid_rows]
        
        if error_metric == "pct":
            discrepancies = (clusters_to_plot[estimate_col] - clusters_to_plot["Number ofAnimalUnits"]) / clusters_to_plot["Number ofAnimalUnits"] * 100
            if xlabel is None:
                xlabel = "Percent Error (%)"
        else:  # abs_pct
            discrepancies = abs((clusters_to_plot[estimate_col] - clusters_to_plot["Number ofAnimalUnits"]) / clusters_to_plot["Number ofAnimalUnits"] * 100)
            if xlabel is None:
                xlabel = "Absolute Percent Error (%)"
    else:
        raise ValueError("error_metric must be one of: 'raw', 'abs', 'pct', 'abs_pct'")
    
    # Trim outliers based on percentiles if specified
    if trim_percentiles != (0, 100):
        lower, upper = np.percentile(discrepancies, trim_percentiles)
        mask = (discrepancies >= lower) & (discrepancies <= upper)
        discrepancies = discrepancies[mask]
        if title is None:
            title = f"Distribution of {error_metric} discrepancies between estimates and permits"
        title += f"\n(Showing {trim_percentiles[1]-trim_percentiles[0]}% of data, excluding extreme values)"
    
    # Create figure if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(discrepancies, bins=bins, alpha=0.7)
    ax.grid(False)
    
    # Add zero line for raw and percent errors
    if error_metric in ["raw", "pct"]:
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add statistic line if requested
    if stat_line is not None:
        if stat_line == "median":
            stat_val = discrepancies.median()
            stat_name = "Median"
        elif stat_line == "mean":
            stat_val = discrepancies.mean()
            stat_name = "Mean"
        else:
            raise ValueError("stat_line must be one of: 'median', 'mean', None")
            
        ax.axvline(stat_val, color='red', linestyle='--', alpha=0.5)
        ax.text(stat_val + 7, ax.get_ylim()[1]*0.95, f'{stat_name}: {stat_val:.1f}', 
                color='red', ha='center', va='top')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title is None:
        title = f"Distribution of {error_metric} discrepancies between estimates and permits"
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=dpi, format=cfg.FIG_EXPORT_FORMAT)
    
    if display:
        plt.show()
    
    return fig, ax

def EWG_cf_count_compare(
    train_clusters: gpd.GeoDataFrame,
    ewg_afos: gpd.GeoDataFrame,
    figsize: tuple = (6.4, 4.8),
    estimate_logic: str = "point",
    save_path: str = None,
):
    """
    This function creates a figure comparing the cloud factory animal count estimates of unpermitted
    AFOs/CAFOs in the EWG study region to EWG's animal count estimates for those facilities.

    Args:
        train_clusters: dataset of cloud factory annotation clusters in the EWG study region
        ewg_afos: dataset of EWG facility labels
        figsize: desired width and height of the figure
        estimate_logic: logic for estimating animal counts. Options are "point", "lower", or "upper"
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    Returns:
        Three-element tuple of the animal density estimates derived from the median facility in
        each EWG size group.
    """
    if estimate_logic == "point":
        estimate_col = "Dairy_count_estimate"
    elif estimate_logic == "lower":
        estimate_col = "Dairy_count_estimate_lower"
    elif estimate_logic == "upper":
        estimate_col = "Dairy_count_estimate_upper"

    # Join clusters to EWG AFOs/CAFOs by centroid
    train_clusters["centroid"] = train_clusters.centroid
    clusters_joined_ewg = train_clusters.set_geometry("centroid").sjoin_nearest(
        ewg_afos, max_distance=400, distance_col="match_distance"
    )
    # Omit swine and poultry AFOs/CAFOs, beef AFOs/CAFOs, and permitted facilities
    clusters_joined_ewg = clusters_joined_ewg[
        (~clusters_joined_ewg["Legend"].isin(["Poultry", "Swine"]))
        & (clusters_joined_ewg["Animal_Typ"] != "Beef")
        & (clusters_joined_ewg["Permitted"] == "Not Permitted / Unknown")
    ]

    # Deduplicate by EWGID, keeping closest match
    clusters_joined_ewg = clusters_joined_ewg.sort_values(
        by="match_distance"
    ).drop_duplicates(subset="EWGID", keep="first")

    # Estimate animal counts
    if "animal_unit_estimate" not in clusters_joined_ewg.columns:
        clusters_joined_ewg = est_au.sample_and_calc_au(
            clusters_joined_ewg, include_area_uncertainty=False
        )

    clusters_joined_ewg_large = clusters_joined_ewg[
        clusters_joined_ewg["Legend"] == "Cattle: Large"
    ]
    clusters_joined_ewg_med = clusters_joined_ewg[
        clusters_joined_ewg["Legend"] == "Cattle: Medium"
    ]
    clusters_joined_ewg_small = clusters_joined_ewg[
        clusters_joined_ewg["Legend"] == "Cattle: Small"
    ]

    largest_annotation = clusters_joined_ewg[estimate_col].max()
    smallest_annotation = clusters_joined_ewg[estimate_col].min()

    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=False)

    # Set up x axis
    ax[0].set_xlim(smallest_annotation, largest_annotation)

    sns.histplot(
        clusters_joined_ewg[clusters_joined_ewg["Legend"] == "Cattle: Small"],
        x=estimate_col,
        ax=ax[0],
        binwidth=50,
    )
    ax[0].axvspan(50, 199, color="green", alpha=0.2, label="EWG estimate range")
    sns.histplot(
        clusters_joined_ewg[clusters_joined_ewg["Legend"] == "Cattle: Medium"],
        x=estimate_col,
        ax=ax[1],
        binwidth=50,
    )
    ax[1].axvspan(200, 499, color="green", alpha=0.2)
    sns.histplot(
        clusters_joined_ewg[clusters_joined_ewg["Legend"] == "Cattle: Large"],
        x=estimate_col,
        ax=ax[2],
        binwidth=50,
    )
    ax[2].axvspan(500, largest_annotation, color="green", alpha=0.2)

    # Add axis labels
    ax[1].set_ylabel("Number of facilities")
    ax[0].set_ylabel("")
    plt.xlabel(f"Cow number {estimate_logic} estimate (annotation area)")
    ax[2].set_ylabel("")

    # Add annotations
    ax[0].text(1700, 50, f"Small (n={clusters_joined_ewg_small.shape[0]})")
    ax[1].text(1700, 20, f"Medium (n={clusters_joined_ewg_med.shape[0]})")
    ax[2].text(1700, 5, f"Large (n={clusters_joined_ewg_large.shape[0]})")

    ax[0].legend()

    for a in ax:
        a.grid(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    # Derive animal density estimates
    small_median = clusters_joined_ewg_small["cluster_area_m2"].median()
    large_median = clusters_joined_ewg_large["cluster_area_m2"].median()
    print(
        "Animal density estimates: ",
        round(small_median / 50, 2),
        round(small_median / 199, 2),
        round(large_median / 500, 2),
    )

    # Calculate overlap percentages with estimate logic selected
    small_overlap = (
        clusters_joined_ewg_small[
            (clusters_joined_ewg_small[estimate_col] >= 50)
            & (clusters_joined_ewg_small[estimate_col] < 200)
        ].shape[0]
        / clusters_joined_ewg_small.shape[0]
    )
    med_overlap = (
        clusters_joined_ewg_med[
            (clusters_joined_ewg_med[estimate_col] >= 200)
            & (clusters_joined_ewg_med[estimate_col] < 500)
        ].shape[0]
        / clusters_joined_ewg_med.shape[0]
    )
    large_overlap = (
        clusters_joined_ewg_large[
            (clusters_joined_ewg_large[estimate_col] >= 500)
        ].shape[0]
        / clusters_joined_ewg_large.shape[0]
    )

    print("Small overlap: ", round(small_overlap, 2))
    print("Medium overlap: ", round(med_overlap, 2))
    print("Large overlap: ", round(large_overlap, 2))
    print(
        "Overall overlap: ",
        round(
            (
                small_overlap * clusters_joined_ewg_small.shape[0]
                + med_overlap * clusters_joined_ewg_med.shape[0]
                + large_overlap * clusters_joined_ewg_large.shape[0]
            )
            / clusters_joined_ewg.shape[0],
            2,
        ),
    )

    # Calculate overall overlap in the full range, using lower and upper bounds
    small_overlap_any = (
        clusters_joined_ewg_small[
            (clusters_joined_ewg_small["Dairy_count_estimate_upper"] >= 50)
            & (clusters_joined_ewg_small["Dairy_count_estimate_lower"] < 200)
        ].shape[0]
        / clusters_joined_ewg_small.shape[0]
    )

    med_overlap_any = (
        clusters_joined_ewg_med[
            (clusters_joined_ewg_med["Dairy_count_estimate_upper"] >= 200)
            & (clusters_joined_ewg_med["Dairy_count_estimate_lower"] < 500)
        ].shape[0]
        / clusters_joined_ewg_med.shape[0]
    )

    large_overlap_any = (
        clusters_joined_ewg_large[
            (clusters_joined_ewg_large["Dairy_count_estimate_upper"] >= 500)
        ].shape[0]
        / clusters_joined_ewg_large.shape[0]
    )

    print("Small overlap (anything in uncertainty bound): ", round(small_overlap_any, 2))
    print("Medium overlap (anything in uncertainty bound): ", round(med_overlap_any, 2))
    print("Large overlap (anything in uncertainty bound): ", round(large_overlap_any, 2))
    print("Overall overlap (anything in uncertainty bound): ", round(
        (
            small_overlap_any * clusters_joined_ewg_small.shape[0]
            + med_overlap_any * clusters_joined_ewg_med.shape[0]
            + large_overlap_any * clusters_joined_ewg_large.shape[0]
        )
        / clusters_joined_ewg.shape[0],
        2,
    ))




    return clusters_joined_ewg


def annotation_prediction_examples(
    human_annotation_cluster: gpd.GeoDataFrame,
    four_band_model_cluster: gpd.GeoDataFrame,
    three_band_model_cluster: gpd.GeoDataFrame = None,
    figsize: tuple = (8, 6),
    save_path=None,
):
    """
    This function creates a figure with an example of a cloud factory annotation cluster next to an example
    of a model prediction cluster of the same facility from both the three and four band models.

    Args:
        human_annotation_cluster: single cluster of human annotations
        four_band_model_cluster: single cluster of four band predictions (one-row slice of clustered human annotations dataset)
        three_band_model_cluster: Optional, single cluster of three band predictions
          (one-row slice of clustered human annotations dataset)
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    Returns: None
    """
    if (
        human_annotation_cluster.shape[0] > 1
        or four_band_model_cluster.shape[0] > 1
        or (
            three_band_model_cluster is not None
            and three_band_model_cluster.shape[0] > 1
        )
    ):
        print("Inputs must only have one cluster.")
        return

    fig, ax = plt.subplots(
        1, 3 if three_band_model_cluster is not None else 2, figsize=figsize
    )
    plotting_utils.plot_cluster_parcel(
        human_annotation_cluster,
        fig=fig,
        ax=ax[0],
        zoom_radius=200,
        title_desc="Human Label",
        display=False,
    )
    if three_band_model_cluster is not None:
        plotting_utils.plot_cluster_parcel(
            three_band_model_cluster,
            fig=fig,
            ax=ax[1],
            zoom_radius=200,
            title_desc="Three-band Model",
            display=False,
        )
    plotting_utils.plot_cluster_parcel(
        four_band_model_cluster,
        fig=fig,
        ax=ax[len(ax) - 1],
        zoom_radius=200,
        title_desc="Four-band Model",
        display=False,
    )

    # Align axis limits
    for subplot in range(1, len(ax)):
        ax[subplot].set_xlim(ax[0].get_xlim())
        ax[subplot].set_ylim(ax[0].get_ylim())

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, bbox_inches="tight", format=cfg.FIG_EXPORT_FORMAT)

    return


def perm_unperm_examples(figsize: tuple = (10, 5), save_path: str = None):
    """
    This figure creates a figure with one example of a permitted dairy CAFO and one example of an unpermitted
    potential dairy CAFO. Not customizable given that the examples were found manually.

    Args:
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    Returns: None
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    # Permitted dairy CAFO  ('Hickory Lawn Dairy', milks 720 cows according to their website, permitted for 1541 AU)
    jpegs = naip_utils.collect_neighboring_tiles(
        ["WI_Sheboygan_7_10240_27648.jpeg", "WI_Sheboygan_7_10240_26624.jpeg"]
    )
    naip_image = naip_utils.download_transform(jpegs)
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[0])
    ax[0].set_xlim([675800 - 350, 675800 + 350])
    ax[0].set_ylim([354551 - 350, 354551 + 350])
    ax[0].set_axis_off()


    # Unpermitted potential dairy CAFO ('Greendale Dairy', milks 560 cows and raises 455 heifers according to their website,
    # estimated by model to have 1778 AU)
    jpegs_2 = naip_utils.collect_neighboring_tiles(["WI_Manitowoc_4_11264_0.jpeg"])
    naip_image = naip_utils.download_transform(jpegs_2)
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[1])
    print(naip_image[0].bounds)
    ax[1].set_xlim([685125 - 350, 685125 + 350])
    ax[1].set_ylim([392000 - 350, 392000 + 350])
    ax[1].set_axis_off()

    # Add scale bars
    fontprops = fm.FontProperties(size=15)
    scalebar = AnchoredSizeBar(
        ax[0].transData,
        100,
        "100 m",
        "lower left",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=2,
        fontproperties=fontprops,
    )
    ax[0].add_artist(scalebar)
    scalebar_2 = AnchoredSizeBar(
        ax[1].transData,
        100,
        "100 m",
        "lower left",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=2,
        fontproperties=fontprops,
    )
    ax[1].add_artist(scalebar_2)

    # Add titles
    ax[0].text(
        675800 - 300,
        354551 + 300,
        "Permitted dairy CAFO",
        color="white",
        fontproperties=fontprops,
    )
    ax[1].text(
        685125 - 300,
        392000 + 300,
        "Unpermitted potential dairy CAFO",
        color="white",
        fontproperties=fontprops,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    return


def cluster_method_compare(
    test_polygons: gpd.GeoDataFrame,
    parcel_data: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    figsize: tuple = (8, 4),
    fig_save_path: str = None,
    table_save_path: str = None,
):
    """
    This function creates a three-panel figure comparing three alternative clustering specifications.

    Args:
        test_polygons: dataset of human annotations outside of the EWG study region
        parcel_data: dataset of WI land parcels
        permit_data: WI permitted CAFO dataset
        figsize: desired width and height of the figure
        fig_save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
        table_save_path: file path where to save summary results table.
    Returns:
        pd.DataFrame of cluster summary statistics
    """
    # Just cluster polygons on the same parcel
    cf_clusters_basic = cluster.cluster(
        parcel_data,
        test_polygons,
        fuzzy_name_match=False,
        cluster_adjacent_parcels_same_name=False,
        same_name_distance_threshold=None,
        cluster_distance_threshold=None,
    )
    # Just cluster polygons within a certain distance of eachother
    cf_clusters_distance_only = cluster.cluster(
        parcel_data,
        test_polygons,
        cluster_same_parcel=False,
        fuzzy_name_match=False,
        cluster_adjacent_parcels_same_name=False,
        same_name_distance_threshold=None,
    )
    # Default clustering method
    cf_clusters = cluster.cluster(parcel_data, test_polygons)

    # Merge with dairy CAFO permits
    cf_clusters_basic_merged, cf_clusters_basic_err = caf.merge_clusters_permits(
        cf_clusters_basic,
        permit_data[permit_data["AnimalType"] == "Dairy"],
        discrep_analysis=True,
    )
    (
        cf_clusters_distance_only_merged,
        cf_clusters_distance_only_err,
    ) = caf.merge_clusters_permits(
        cf_clusters_distance_only,
        permit_data[permit_data["AnimalType"] == "Dairy"],
        discrep_analysis=True,
    )
    cf_clusters_merged, cf_clusters_err = caf.merge_clusters_permits(
        cf_clusters,
        permit_data[permit_data["AnimalType"] == "Dairy"],
        discrep_analysis=True,
    )

    # Label the set of CAFOs with equivalent animal unit estimates across the three methods
    unchanged_CAFOs = (
        cf_clusters_basic_merged[["CAFO_index", "animal_unit_estimate"]]
        .merge(cf_clusters_distance_only_merged[["CAFO_index", "animal_unit_estimate"]])
        .merge(cf_clusters_merged[["CAFO_index", "animal_unit_estimate"]])
    )
    unchanged_CAFOs["unchanged"] = True
    cf_clusters_basic_merged = cf_clusters_basic_merged.merge(
        unchanged_CAFOs, on=["CAFO_index", "animal_unit_estimate"], how="left"
    )
    cf_clusters_distance_only_merged = cf_clusters_distance_only_merged.merge(
        unchanged_CAFOs, on=["CAFO_index", "animal_unit_estimate"], how="left"
    )
    cf_clusters_merged = cf_clusters_merged.merge(
        unchanged_CAFOs, on=["CAFO_index", "animal_unit_estimate"], how="left"
    )

    cf_clusters_basic_merged["unchanged"] = cf_clusters_basic_merged[
        "unchanged"
    ].fillna(False)
    cf_clusters_distance_only_merged["unchanged"] = cf_clusters_distance_only_merged[
        "unchanged"
    ].fillna(False)
    cf_clusters_merged["unchanged"] = cf_clusters_merged["unchanged"].fillna(False)

    # Plot the three methods, including only CAFOs which change estimates between the three
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    plotting_utils.plot_permit_v_estimate(
        cf_clusters_basic_merged[~cf_clusters_basic_merged["unchanged"]],
        fig=fig,
        ax=ax[0],
        display=False,
        xlabel="",
        title="Same parcel only",
        include_legend=False,
    )
    plotting_utils.plot_permit_v_estimate(
        cf_clusters_distance_only_merged[
            ~cf_clusters_distance_only_merged["unchanged"]
        ],
        fig=fig,
        ax=ax[1],
        display=False,
        ylabel="",
        xlabel="Permit animal units",
        title="150m radius only",
        include_legend=False,
    )
    plotting_utils.plot_permit_v_estimate(
        cf_clusters_merged[~cf_clusters_merged["unchanged"]],
        fig=fig,
        ax=ax[2],
        display=False,
        ylabel="",
        xlabel="",
        title="Full method",
        include_legend=False,
    )
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    plt.tight_layout()

    # Save figure
    if fig_save_path is not None:
        plt.savefig(ensure_export_format(fig_path / "fig_appendix_cluster_compare.png"), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    # Create combined dataframe of cluster statistics
    summary_stats = pd.concat(
        [cf_clusters_basic_err, cf_clusters_distance_only_err, cf_clusters_err], axis=0
    )
    summary_stats["cluster_method"] = ["Basic", "Distance only", "Default"]
    summary_stats["n"] = [
        cf_clusters_basic.shape[0],
        cf_clusters_distance_only.shape[0],
        cf_clusters.shape[0],
    ]

    # Save results table
    if table_save_path is not None:
        summary_stats.to_csv(table_save_path)

    return summary_stats


def unperm_au_dist(
    model_clusters: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    county_data: gpd.GeoDataFrame,
    min_AFO_bound: float = 500,
    figsize: tuple = (6, 4),
    save_path: str = None,
):
    """
    This function creates a figure with the distribution of animal unit estimates for unpermitted model
    detections outside of the EWG study region.

    Args:
        model_clusters: dataset of clustered model predictions
        permit_data: WI permitted CAFO dataset
        county_data: WI counties shapefile
        min_AFO_bound: float, minimum size of a cluster in animal units for that cluster to be deemed an AFO
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    Returns: None
    """
    # Join clusters with county data, screen for non-train counties
    model_clusters = model_clusters.sjoin(
        county_data[["geometry", "COUNTY_NAM", "train_counties"]]
    ).drop(columns=["index_right"])
    model_clusters_non_ewg = model_clusters[~model_clusters["train_counties"]]

    # Determine permitted and unpermitted model detections. Here, we don't drop multi matches, in case there
    # are multiple unclustered model detections within the match distance of a permit where either or both may be part of
    # the permit (there are ~4 cases of this for clusters with > 1000 AU).
    matched_clusters, err = caf.merge_clusters_permits(
        model_clusters_non_ewg,
        permit_data,
        drop_multi_matches=False,
        sum_satellite_counts=False,
    )
    unmatched = model_clusters_non_ewg[
        ~model_clusters_non_ewg["polygon_indices"].isin(
            matched_clusters["polygon_indices"]
        )
    ]

    unmatched = est_au.sample_and_calc_au(unmatched, include_area_uncertainty=True)

    fig, ax = plt.subplots(figsize=figsize)
    # Omitting detections so small as to not likely AFOs/CAFOs
    sns.histplot(
        unmatched[unmatched["animal_unit_estimate"] >= min_AFO_bound][
            "animal_unit_estimate"
        ],
        element="step",
        color=cfg.COLOR_UNPERMITTED,
        alpha=0,
        label="Unpermitted detections \n (estimated animal units)",
        bins=40,
    )
    sns.histplot(
        permit_data[permit_data["AnimalType"] == "Dairy"]["Number ofAnimalUnits"],
        element="step",
        color=cfg.COLOR_PERMITTED,
        alpha=0,
        label="Permitted dairy CAFOs \n (permitted animal units)",
        bins=40,
    )
    plt.xlabel("Animal units")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(fig_path / "fig_unperm_au_dist.png"), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    return


def baseline_pred_examples(
    inception_detections: gpd.GeoDataFrame,
    poultry_detections: gpd.GeoDataFrame,
    zoom_radius: float = 300,
    figsize: tuple = (10, 10),
    save_path: str = None,
):
    """
    This function creates a figures with plotted examples of predictions from the baseline models
    (Sandy's model/Inceptionv3 and Caleb's poultry model).

    Args:
        inception_detections: GeoDataFrame of predictions from the Inceptionv3 model
        poultry_detections: GeoDataFrame of predictions from Caleb's poultry model
        zoom_radius: radius around cluster centroid to plot, in meters
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    """
    inception_detections = inception_detections.to_crs(cfg.WI_EPSG)
    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # Inception detection on example 1
    inception_example_1 = 44846
    jpegs = naip_utils.collect_neighboring_tiles(["WI_Lafayette_0_18432_3072.jpeg"])
    naip_image = naip_utils.download_transform(jpegs)
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[0][0])
    inception_detections.loc[[inception_example_1]].plot(
        ax=ax[0][0], color="red", alpha=0.5
    )

    # Poultry detection on example 1
    poultry_example_1 = list(range(1935, 1947))
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[0][1])
    poultry_detections.loc[poultry_example_1].plot(ax=ax[0][1], color="red", alpha=0.5)

    # Inception detection on example 2
    inception_example_2 = 699
    jpegs = naip_utils.collect_neighboring_tiles(["WI_Adams_3_8192_27648.jpeg"])
    naip_image = naip_utils.download_transform(jpegs)
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[1][0])
    inception_detections.loc[[inception_example_2]].plot(
        ax=ax[1][0], color="red", alpha=0.5
    )

    # Poultry detection on example 2
    poultry_example_2 = list(range(3731, 3738))
    rasterio.plot.show(naip_image[0], transform=naip_image[1], ax=ax[1][1])
    poultry_detections.loc[poultry_example_2].plot(ax=ax[1][1], color="red", alpha=0.5)

    # Set axis limits
    for column in [0, 1]:
        ax[0][column].set_xlim(
            [
                inception_detections.centroid.loc[inception_example_1].x - zoom_radius,
                inception_detections.centroid.loc[inception_example_1].x + zoom_radius,
            ]
        )
        ax[0][column].set_ylim(
            [
                inception_detections.centroid.loc[inception_example_1].y - zoom_radius,
                inception_detections.centroid.loc[inception_example_1].y + zoom_radius,
            ]
        )
    for column in [0, 1]:
        ax[1][column].set_xlim(
            [
                inception_detections.centroid.loc[inception_example_2].x - zoom_radius,
                inception_detections.centroid.loc[inception_example_2].x + zoom_radius,
            ]
        )
        ax[1][column].set_ylim(
            [
                inception_detections.centroid.loc[inception_example_2].y - zoom_radius,
                inception_detections.centroid.loc[inception_example_2].y + zoom_radius,
            ]
        )

    # Remove axis labels
    ax[0][0].set_axis_off()
    ax[0][1].set_axis_off()
    ax[1][0].set_axis_off()
    ax[1][1].set_axis_off()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    return


def WI_CAFO_map(
    human_annotation_clusters: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    county_data: gpd.GeoDataFrame,
    CAFO_thresh_logic: str = "point",
    figsize: tuple = (8, 8),
    save_path: str = None,
):
    """
    This function creates a figure with a map of permitted dairy CAFOs and unpermitted potential dairy CAFOs
    across the state of WI.

    Args:
        human_annotation_clusters: dataset of CF annotation clusters
        permit_data: dataset of permitted CAFOs
        county_data: WI county shapefile
        figsize: desired width and height of the figure
        save_path: path where to save the figure. Defaults to None, in which the figure is not saved to disk.
    """
    # Preprocess cluster
    if "animal_unit_estimate" not in human_annotation_clusters.columns:
        human_annotation_clusters = est_au.sample_and_calc_au(human_annotation_clusters)

    permitted_clusters, err = caf.merge_clusters_permits(
        human_annotation_clusters,
        permit_data,
        drop_multi_matches=False,
        sum_satellite_counts=False,
        only_dairy=False
    )
    unpermitted_clusters = human_annotation_clusters[
        ~human_annotation_clusters["polygon_indices"].isin(
            permitted_clusters["polygon_indices"]
        )
    ].copy()

    if CAFO_thresh_logic == "point":
        thresh_col = "animal_unit_estimate"
    elif CAFO_thresh_logic == "lower":
        thresh_col = "animal_units_lower"
    elif CAFO_thresh_logic == "upper":
        thresh_col = "animal_units_upper"

    # Subset to unpermitted human annotation clusters > threshold
    unpermitted_clusters = unpermitted_clusters[
        unpermitted_clusters[thresh_col] >= 1000
    ]

    fig, ax = plt.subplots(figsize=figsize)
    county_data.to_crs(cfg.WI_EPSG).plot(
        ax=ax,
        # column="train_counties",
        edgecolor="black",
        linewidth=1,
        color="lightgray",
        #cmap="YlGn",
        alpha=0.2,
    )
    unpermitted_clusters["centroid"] = unpermitted_clusters.centroid
    unpermitted_clusters.set_geometry("centroid").plot(
        ax=ax,
        legend=True,
        label=f"Unpermitted potential CAFOs (n={unpermitted_clusters.shape[0]})",
        markersize=4,
        c=cfg.COLOR_UNPERMITTED,
    )
    # Plot only main dairy CAFO facilities
    permits_to_plot = permit_data[
        (pd.isna(permit_data["SATELLITE_"])) & (permit_data["AnimalType"] == "Dairy")
    ]
    permits_to_plot.plot(
        ax=ax,
        legend=True,
        label=f"Permitted CAFOs (n={permits_to_plot.shape[0]})",
        markersize=4,
        c=cfg.COLOR_PERMITTED,
    )

    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)

    return


def facility_level_stats_table(
    four_band_clusters: gpd.GeoDataFrame,
    three_band_clusters: gpd.GeoDataFrame,
    poultry_predictions: gpd.GeoDataFrame,
    inception_predictions: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    county_data: gpd.GeoDataFrame,
    ewg_afos: gpd.GeoDataFrame,
    inception_detection_threshold: float = 0.5,
    AFO_lower_bound: float = 150,
    AFO_lower_bound_logic: str = "point_estimate",
    save_path: str = None,
):
    """
    This function creates a table of facility-level performance statistics for the four models.
    Args:
        - four_band_clusters: dataset of clustered four band model predictions
        - three_band_clusters: dataset of clustered three band model predictions
        - poultry_predictions: dataset of poultry model predictions
        - inception_predictions: dataset of Inceptionv3 model predictions
        - permit_data: WDNR CAFO dataset
        - county_data: WI counties dataset
        - ewg_afos: dataset of EWG AFOs
        - inception_detection_threshold: float, confidence threshold for Inceptionv3 model predictions
        - AFO_lower_bound: float, lower bound for a model cluster to be included
        - save_path: file path where to save results
    Returns:
        - Table of model performance statistics
    """
    # Preprocess if necessary
    if "polygon_index" not in poultry_predictions.columns:
        poultry_predictions["polygon_index"] = poultry_predictions.index
    if "polygon_index" not in inception_predictions.columns:
        inception_predictions["polygon_index"] = inception_predictions.index
    poultry_predictions["cluster_area_m2"] = poultry_predictions["geometry"].area

    # Restrict WDNR locations to main Dairy CAFOs
    permit_data = permit_data[
        (pd.isna(permit_data["SATELLITE_"])) & (permit_data["AnimalType"] == "Dairy")
    ]

    # Label by EWG region
    for model_output in [
        four_band_clusters,
        three_band_clusters,
        poultry_predictions,
        inception_predictions,
    ]:
        if "ewg_region" not in model_output.columns:
            model_output = amo.label_by_ewg_region(model_output, county_data)

    # Estimate animal units and filter model predictions by size
    if "animal_unit_estimate" not in four_band_clusters.columns:
        four_band_clusters = est_au.sample_and_calc_au(
            four_band_clusters, include_area_uncertainty=True,
            model_type='four_band', model_error_join='left'
        )
    if "animal_unit_estimate" not in three_band_clusters.columns:
        three_band_clusters = est_au.sample_and_calc_au(
        three_band_clusters, include_area_uncertainty=True,
        model_type='three_band', model_error_join='left'
    ) 
    if "animal_unit_estimate" not in poultry_predictions.columns:
        poultry_predictions = est_au.sample_and_calc_au(
            poultry_predictions, include_area_uncertainty=False 
        )

    # Filter by AFO lower bound
    if AFO_lower_bound_logic == "point_estimate":
        au_col = "animal_unit_estimate"
    elif AFO_lower_bound_logic == "lower_bound":
        au_col = "animal_units_lower"
    elif AFO_lower_bound_logic == "upper_bound":
        au_col = "animal_units_upper"
    elif AFO_lower_bound_logic == "above_10th_percentile":
        au_col = "animal_units_0.1_perc"
    elif AFO_lower_bound_logic == "above_25th_percentile":
        au_col = "animal_units_0.25_perc"
    elif AFO_lower_bound_logic == "above_50th_percentile":
        au_col = "animal_units_0.5_perc"
    elif AFO_lower_bound_logic == "above_75th_percentile":
        au_col = "animal_units_0.75_perc"

    three_band_clusters = three_band_clusters[
        three_band_clusters[au_col] > AFO_lower_bound
    ]
    four_band_clusters = four_band_clusters[
        four_band_clusters[au_col] > AFO_lower_bound
    ]
    poultry_predictions = poultry_predictions[
        poultry_predictions[au_col] > AFO_lower_bound
    ]

    # Split out EWG-labeled Large Dairy AFOs
    ewg_large_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Large")
    ].copy()

    ewg_small_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Cattle") & (ewg_afos["Legend"] == "Cattle: Small")
    ].copy()

    ewg_medium_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Medium")
    ].copy()

    # -------------------- INCEPTION model --------------------
    n_inception_predictions = inception_predictions[
        inception_predictions["conf"] >= inception_detection_threshold
    ].shape[0]
    n_inception_predictions_ewg = inception_predictions[
        (inception_predictions["conf"] >= inception_detection_threshold)
        & (inception_predictions["ewg_region"])
    ].shape[0]

    # A WDNR main match is when the inception tile is within 400m of a WDNR Main Dairy CAFO location

    inception_WDNR_main_matches = permit_data.sjoin_nearest(
        inception_predictions[
            inception_predictions["conf"] >= inception_detection_threshold
        ],
        max_distance=400,
    )
    # An EWG true positive is when the inception tile is within 400m of any EWG AFO location
    EWG_all_true_positives = inception_predictions[
        (inception_predictions["conf"] >= inception_detection_threshold)
        & (inception_predictions["ewg_region"])
    ].sjoin_nearest(ewg_afos, max_distance=400)

    # An EWG large dairy true positive is when the inception tile is within 400m of an EWG Large Dairy AFO location
    EWG_large_dairy_true_positives = inception_predictions[
        (inception_predictions["conf"] >= inception_detection_threshold)
        & (inception_predictions["ewg_region"])
    ].sjoin_nearest(ewg_large_dairy_afos, max_distance=400)

    # An EWG medium dairy true positive is when the inception tile is within 400m of an EWG Large Dairy AFO location
    EWG_medium_dairy_true_positives = inception_predictions[
        (inception_predictions["conf"] >= inception_detection_threshold)
        & (inception_predictions["ewg_region"])
    ].sjoin_nearest(ewg_medium_dairy_afos, max_distance=400)

    # An EWG small dairy true positive is when the inception tile is within 400m of an EWG Large Dairy AFO location
    EWG_small_dairy_true_positives = inception_predictions[
        (inception_predictions["conf"] >= inception_detection_threshold)
        & (inception_predictions["ewg_region"])
    ].sjoin_nearest(ewg_small_dairy_afos, max_distance=400)

    
    # Calculate precision and recall:
    # WDNR recall includes only main facilities
    inception_WDNR_recall = len(
        inception_WDNR_main_matches["CAFO_index"].unique()
    ) / len(permit_data["CAFO_index"].unique())

    # EWG recall includes only large dairies
    inception_ewg_large_recall = (
        EWG_large_dairy_true_positives.drop_duplicates("EWGID").shape[0]
        / ewg_large_dairy_afos.shape[0]
    )

    # Inception EWG medium recall
    inception_ewg_medium_recall = (
        EWG_medium_dairy_true_positives.drop_duplicates("EWGID").shape[0]
        / ewg_medium_dairy_afos.shape[0]
    )

    # Inception EWG small recall
    inception_ewg_small_recall = (
        EWG_small_dairy_true_positives.drop_duplicates("EWGID").shape[0]
        / ewg_small_dairy_afos.shape[0]
    )

    # EWG precision includes all facilities
    inception_ewg_precision = (
        EWG_all_true_positives.drop_duplicates("EWGID").shape[0]
        / inception_predictions[
            (inception_predictions["conf"] >= inception_detection_threshold)
            & (inception_predictions["ewg_region"])
        ].shape[0]
    )

    # Collect results
    results = pd.DataFrame(
        data={
            "Recall of permitted dairy CAFOs outside EWG study region": [
                inception_WDNR_recall
            ],
            "Recall of large dairy farms inside the EWG study region": [
                inception_ewg_large_recall
            ],
            "Recall of medium dairy farms inside the EWG study region": [
                inception_ewg_medium_recall
            ],
            "Recall of small dairy farms inside the EWG study region": [
                inception_ewg_small_recall
            ],
            "Precision over all farms inside the EWG study region": [
                inception_ewg_precision
            ],
            "Number of predictions in the EWG study region": [
                n_inception_predictions_ewg
            ],
            "Number of predictions outside the EWG study region": [
                n_inception_predictions - n_inception_predictions_ewg
            ],
        }
    )

    # -------------------- Segmentation models --------------------
    for model_output in [poultry_predictions, three_band_clusters, four_band_clusters]:
        n_predictions_ewg = model_output[model_output["ewg_region"]].shape[0]

        ewg_large_dairy_matches = ewg_large_dairy_afos.sjoin_nearest(
            model_output, max_distance=400
        )
        ewg_medium_dairy_matches = ewg_medium_dairy_afos.sjoin_nearest(
            model_output, max_distance=400
        )

        ewg_small_dairy_matches = ewg_small_dairy_afos.sjoin_nearest(
            model_output, max_distance=400
        )

        ewg_all_matches = ewg_afos.sjoin_nearest(model_output, max_distance=400)
        WDNR_main_matches = permit_data.sjoin_nearest(model_output, max_distance=400)

        ewg_large_recall = (
            ewg_large_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_large_dairy_afos.shape[0]
        )

        ewg_medium_recall = (
            ewg_medium_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_medium_dairy_afos.shape[0]
        )

        ewg_small_recall = (
            ewg_small_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_small_dairy_afos.shape[0]
        )

        ewg_large_recall = round(ewg_large_recall, 2)
        ewg_medium_recall = round(ewg_medium_recall, 2)
        ewg_small_recall = round(ewg_small_recall, 2)

        WDNR_recall = (
            WDNR_main_matches.drop_duplicates("CAFO_index").shape[0]
            / permit_data.drop_duplicates("CAFO_index").shape[0]
        )
        WDNR_recall = round(WDNR_recall, 2)

        ewg_precision = (
            ewg_all_matches.drop_duplicates("index_right").shape[0]
            / model_output[model_output["ewg_region"]].shape[0]
        )
        ewg_precision = round(ewg_precision, 2)

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    data={
                        "Recall of permitted dairy CAFOs outside EWG study region": [
                            WDNR_recall
                        ],
                        "Recall of large dairy farms inside the EWG study region": [
                            ewg_large_recall
                        ],
                        "Recall of medium dairy farms inside the EWG study region": [
                            ewg_medium_recall
                        ],
                        "Recall of small dairy farms inside the EWG study region": [
                            ewg_small_recall
                        ],
                        "Precision over all farms inside the EWG study region": [
                            ewg_precision
                        ],
                        "Number of predictions in the EWG study region": [
                            n_predictions_ewg
                        ],
                        "Number of predictions outside the EWG study region": [
                            model_output.shape[0] - n_predictions_ewg
                        ],
                    }
                ),
            ]
        )
    results.index = ["Inceptionv3", "Poultry", "Three band", "Four band"]

    # Save results
    if save_path is not None:
        results.to_csv(save_path)

    return results

def facility_level_pr_lowerbound_graph(model_clusters, 
                                       model_type, 
                                       permit_data, 
                                       county_data,
                                       ewg_afos, 
                                       AFO_lower_bound_threshes=np.arange(0, 1000, 100), 
                                       AFO_lower_bound_logic='point_estimate',
                                       save_path=None,
                                       suppress_show=False):
    """
    Creates a graph of various precision/recall stats for model-only clusters, on EWG and outside EWG region
    By varying the AFO lower bound
    """
    # Prep data to match model clusters to
    permit_data = permit_data[
        (pd.isna(permit_data["SATELLITE_"])) & (permit_data["AnimalType"] == "Dairy")
    ]

    # Split out EWG-labeled Large Dairy AFOs
    ewg_large_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Large")
    ].copy()

    ewg_small_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Cattle") & (ewg_afos["Legend"] == "Cattle: Small")
    ].copy()

    ewg_medium_dairy_afos = ewg_afos[
        (ewg_afos["Animal_Typ"] == "Dairy") & (ewg_afos["Legend"] == "Cattle: Medium")
    ].copy()

    # Prep model cluster data
    if "ewg_region" not in model_clusters.columns:
        model_clusters = amo.label_by_ewg_region(model_clusters, county_data)

    if "animal_unit_estimate" not in model_clusters.columns:
        model_clusters = est_au.sample_and_calc_au(
            model_clusters, include_area_uncertainty=True,
            model_type=model_type, model_error_join='left'
        )
    
    # Go through loop and do the precision/recalls
    ewg_large_recalls = []
    ewg_medium_recalls = []
    ewg_small_recalls = []
    permit_recalls = []
    ewg_precisions = []

    for AFO_lower_bound_thresh in AFO_lower_bound_threshes:
        # Filter model clusters
        if AFO_lower_bound_logic == 'point estimate':
            model_clusters_filtered = model_clusters[model_clusters["animal_unit_estimate"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'lower bound 95% confidence interval':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_lower"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'upper bound 95% confidence interval':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_upper"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'above 10th percentile':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_0.1_perc"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'above 25th percentile':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_0.25_perc"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'above 50th percentile':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_0.5_perc"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'above 75th percentile':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_0.75_perc"] > AFO_lower_bound_thresh]
        elif AFO_lower_bound_logic == 'above 90th percentile':
            model_clusters_filtered = model_clusters[model_clusters["animal_units_0.9_perc"] > AFO_lower_bound_thresh]
        else:
            raise ValueError("AFO_lower_bound_logic must be one of: point estimate, lower bound 95% confidence interval, upper bound 95% confidence interval, above 10th percentile, above 25th percentile, above 50th percentile, above 75th percentile, above 90th percentile")
        
        # match data
        ewg_large_dairy_matches = ewg_large_dairy_afos.sjoin_nearest(
            model_clusters_filtered, max_distance=400
        )
        ewg_medium_dairy_matches = ewg_medium_dairy_afos.sjoin_nearest(
            model_clusters_filtered, max_distance=400
        )

        ewg_small_dairy_matches = ewg_small_dairy_afos.sjoin_nearest(
            model_clusters_filtered, max_distance=400
        )

        ewg_all_matches = ewg_afos.sjoin_nearest(model_clusters_filtered, max_distance=400)
        WDNR_main_matches = permit_data.sjoin_nearest(model_clusters_filtered, max_distance=400)

        # Calculate precisions and recalls
        ewg_large_recall = (
            ewg_large_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_large_dairy_afos.shape[0]
        )

        ewg_medium_recall = (
            ewg_medium_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_medium_dairy_afos.shape[0]
        )

        ewg_small_recall = (
            ewg_small_dairy_matches.drop_duplicates("EWGID").shape[0]
            / ewg_small_dairy_afos.shape[0]
        )

        ewg_large_recall = round(ewg_large_recall, 2)
        ewg_medium_recall = round(ewg_medium_recall, 2)
        ewg_small_recall = round(ewg_small_recall, 2)

        WDNR_recall = (
            WDNR_main_matches.drop_duplicates("CAFO_index").shape[0]
            / permit_data.drop_duplicates("CAFO_index").shape[0]
        )
        WDNR_recall = round(WDNR_recall, 2)

        ewg_precision = (
            ewg_all_matches.drop_duplicates("index_right").shape[0]
            / model_clusters_filtered[model_clusters_filtered["ewg_region"]].shape[0]
        )
        ewg_precision = round(ewg_precision, 2)


        # Append to lists
        ewg_large_recalls.append(ewg_large_recall)
        ewg_medium_recalls.append(ewg_medium_recall)
        ewg_small_recalls.append(ewg_small_recall)
        permit_recalls.append(WDNR_recall)
        ewg_precisions.append(ewg_precision)
    
    # Plot the results

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Plot the curves — labels are placed inline via labelLines, not in a legend
    label_x = AFO_lower_bound_threshes[-3] if len(AFO_lower_bound_threshes) >= 3 else AFO_lower_bound_threshes[-1]
    lines = []
    for y_vals, label, ls in [
        (permit_recalls,    "Permitted CAFO recall",           "-"),
        (ewg_large_recalls, "EWG large farm recall",           "-"),
        (ewg_precisions,    "Precision: any farm, EWG region", "--"),
        (ewg_medium_recalls,"EWG medium farm recall",          "-"),
        (ewg_small_recalls, "EWG small farm recall",           "-"),
    ]:
        (line,) = ax.plot(AFO_lower_bound_threshes, y_vals, linewidth=2, linestyle=ls, label=label)
        lines.append(line)

    labelLines(lines, xvals=[label_x] * len(lines), fontsize=9.5,
               yoffsets=0.04, bbox={"pad": 1, "alpha": 0})

    # Chosen threshold line
    ax.axvline(500, color="grey", linewidth=0.8, linestyle="--")
    ax.text(500, 1.02, "Chosen threshold", color="grey",
            ha="center", va="bottom", fontsize=8.5,
            transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel("Minimum Animal Unit Threshold To Keep Detections")
    ax.set_ylabel("Recall or Precision")
    ax.set_yticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_xticks(AFO_lower_bound_threshes)
    ax.grid(False)
    ax.tick_params(length=4)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)
    else:
        if not suppress_show:
            plt.show()

    # stitch all the results into a pandas dataframe and return that
    results = pd.DataFrame({
        "AFO_lower_bound_thresh": AFO_lower_bound_threshes,
        "ewg_large_recall": ewg_large_recalls,
        "ewg_medium_recall": ewg_medium_recalls,
        "ewg_small_recall": ewg_small_recalls,
        "permit_recall": permit_recalls,
        "ewg_precision": ewg_precisions
    })

    return results

def plot_size_unperm_vs_perm(
    all_clusters: gpd.GeoDataFrame,
    save_path: str = None,
    column: str = "animal_unit_estimate",
):
    """
    Creates a KDE plot of the size of permitted vs unpermitted CAFOs, using the var defined in 'column'.
    Assumes the all_clusters dataset is a dataset of facility clusters, with a column 'set' that indicates which type of facility it is (e.g., unpermitted, permitted)
    """
    relevant_clusters = all_clusters[
        all_clusters["set"].isin(
            ["Permitted dairy CAFOs", "Unpermitted potential CAFOs"]
        )
    ]

    plt.figure(figsize=(6, 4))
    sns.kdeplot(
        data=relevant_clusters, x=column, hue="set", common_norm=False, clip=(0, None)
    )
    if column == "animal_unit_estimate":
        plt.xlabel("Animal Unit Estimate")
    if column == "cluster_area_m2":
        plt.xlabel("Cluster Area (m^2)")
    plt.ylabel("Density")
    plt.title("Size Distribution of CAFOs")
    plt.legend(
        title="Type of facility",
        labels=["Potential Unpermitted CAFOs", "Permitted CAFOs"],
    )
    if save_path is not None:
        plt.savefig(ensure_export_format(save_path), dpi=cfg.FIG_DPI, format=cfg.FIG_EXPORT_FORMAT)


def summary_table(
    human_annotation_clusters: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    model_clusters: gpd.GeoDataFrame,
    county_data: gpd.GeoDataFrame,
    milk_producers: gpd.GeoDataFrame,
    slope_data_path: str,
    all_waters: gpd.GeoDataFrame,
    impaired_waters: gpd.GeoDataFrame,
    water_table_depth: gpd.GeoDataFrame,
    snapmaps_layers: dict,
    AFO_lower_bound: float = 150,
    potential_CAFO_thresh: float = 1000,
    just_under_CAFO_thresh: float = 900,
    CAFO_thresh_logic: str = "point",
    omit_ewg_region: bool = False,
    save_path: str = None,
    size_plot_au_save_path: str = None,
    size_plot_m2_save_path: str = None,
    return_unpermitted_potential_CAFOs: bool = False,
):
    """
    This function creates a summary table of four key sets of clusters:
      (1) clustered human annotations of permitted dairy CAFOs statewide (omitting satellite facilities),
      (2) clustered human annotations of unpermitted potential CAFOs (AU>`potential_CAFO_thresh`),
      (3) clustered model predictions of AFOs just under the CAFO limit (AU within `just_under_CAFO_bound`),
      (4) clustered model predictions of smaller AFOs (AU within `AFO_bound`).
    Args:
        - human_annotation_clusters: dataset of CF annotation clusters
        - permit_data: dataset of permitted CAFOs
        - model_clusters: dataset of clustered model predictions
        - county_data: WI county shapefile
        - milk producers: dataset of milk producers, for linking to clusters
        - slope_data_path: path to raster slope data (output of calculate_slope_from_dem)
        - all_waters: all open waters data
        - impaired_waters: impaired waters data
        - water_table_depth: water table depth data
        - snapmaps_layers: dict of snapmaps layers. each dict value is a gpd.GeoDataFrame, with the key referring to the layer name
        - AFO_lower_bound: tuple, lower and upper bounds for defining an AFO (non-CAFO)
        - potential_CAFO_thresh: AU threshold for defining a potential CAFO.
        - just_under_CAFO_thresh: threshold for an afo to count as 'just-under-cafo'
        - CAFO_thresh_logic: str = "point". How to incorporate the uncertainty AU estimation into the threshold decision.
            - If "point", then point estimate >= threshold is a potential CAFO
            - If "lower", then lower bound of 95% CI >= threshold is a potential CAFO
            - If "upper", then upper bound of 95% CI >= threshold is a potential CAFO
        - omit_ewg_region: bool, whether to omit data points from the EWG study region
        - save_path: file path where to save table as .csv
        - size_plot_save_path: file path where to save a plot of the distribution of cluster sizes (that goes along with the table as a supplement)
        - return_unpermitted_potential_CAFOs: bool, whether to return the unpermitted potential CAFOs in addition to the regular summary table.
    Returns:
        -  summary table of the three sets of clusters, as well as a plot
    """
    # Preprocess clusters
    if "animal_unit_estimate" not in human_annotation_clusters.columns:
        human_annotation_clusters = est_au.sample_and_calc_au(human_annotation_clusters)
    if "animal_unit_estimate" not in model_clusters.columns:
        model_clusters = est_au.sample_and_calc_au(
            model_clusters, include_area_uncertainty=True
        )
    
    # Define which column to use based on cafo_thresh_logic
    if CAFO_thresh_logic == "point":
        thresh_col = "animal_unit_estimate"
    elif CAFO_thresh_logic == "lower":
        thresh_col = "animal_units_lower"
    elif CAFO_thresh_logic == "upper":
        thresh_col = "animal_units_upper"
    else:
        raise ValueError(f"Invalid value for CAFO_thresh_logic: {CAFO_thresh_logic}")

    if omit_ewg_region:
        #  Merge in county data to ensure proper screening by EWG region
        human_annotation_clusters = human_annotation_clusters.sjoin(
            county_data[["geometry", "train_counties"]]
        ).drop(columns=["index_right"])
        model_clusters = model_clusters.sjoin(
            county_data[["geometry", "train_counties"]]
        ).drop(columns=["index_right"])
        human_annotation_clusters = human_annotation_clusters[
            ~human_annotation_clusters["train_counties"]
        ]
        model_clusters = model_clusters[~model_clusters["train_counties"]]

    # Filter model clusters based on minimum AFO bound
    print(f"Total model clusters read in: {model_clusters.shape[0]}, unique indices: {model_clusters['polygon_indices'].apply(tuple).nunique()}")
    model_clusters = model_clusters[
        model_clusters["animal_unit_estimate"] >= AFO_lower_bound
    ]
    print(f"Total model clusters after dropping AFO lower bound: {model_clusters.shape[0]}, unique indices: {model_clusters['polygon_indices'].apply(tuple).nunique()}")


    # Print out initial model cluster counts
    matched_model_clusters, err = caf.merge_clusters_permits(
        model_clusters,
        permit_data,
        drop_multi_matches=False, # if more than one cluster matches to a single permit, still retain all of those as 'matched'
        sum_satellite_counts=False, # retain all clusters, even if they are satellite ones according to permit (want to assess environmental risk of facilities)
        only_dairy=False
    )
    unpermitted_model_clusters = model_clusters[
        ~(
            model_clusters["polygon_indices"].isin(
                matched_model_clusters["polygon_indices"]
            )
        )
    ].copy()
    print(
        f"Unpermitted model clusters with model AU >= {AFO_lower_bound}: ",
        unpermitted_model_clusters.shape[0],
    )
    print(
        f"Unpermitted model clusters with model AU >= {potential_CAFO_thresh}: ",
        unpermitted_model_clusters[
            unpermitted_model_clusters[thresh_col] >= potential_CAFO_thresh
        ].shape[0],
    )

    # Define set 1: permitted dairy CAFO clusters
    dairy_matched_clusters, err = caf.merge_clusters_permits(
    human_annotation_clusters, permit_data, sum_satellite_counts=True,
    discrep_analysis=True, only_dairy=True
)

    dairy_matched_clusters["set"] = "Permitted dairy CAFOs"

    # Define set 2: unpermitted potential dairy CAFO clusters
    matched_annotation_clusters, err = caf.merge_clusters_permits(
        human_annotation_clusters,
        permit_data,
        drop_multi_matches=False, # if more than one cluster matches to a single permit, still retain all of those as 'matched'
        sum_satellite_counts=False, # retain all clusters, even if they are satellite ones according to permit (want to assess environmental risk of facilities)
        only_dairy=False
    )
    unpermitted_annotation_clusters = human_annotation_clusters[
        ~(
            human_annotation_clusters["polygon_indices"].isin(
                matched_annotation_clusters["polygon_indices"]
            )
        )
    ].copy()
    unpermitted_potential_CAFOs = unpermitted_annotation_clusters[
        (
            unpermitted_annotation_clusters[thresh_col]
            >= potential_CAFO_thresh
        )
    ].copy()
    # # Record how many unpermitted potential dairy CAFO clusters would have been missed using only model clusters
    # # and the upper AFO AU threshold
    # unperm_annotation_model_clusters = (
    #     unpermitted_annotation_clusters.rename(
    #         columns={"animal_unit_estimate": "human_animal_unit_estimate"}
    #     )
    #     .sjoin_nearest(model_clusters, max_distance=100)
    #     .drop_duplicates("polygon_indices_left")
    # )
    # print(
    #     f"Unpermitted potential dairy CAFOs with human annotation AU >= 1500 but model AU < {AFO_bound[1]}: ",
    #     unperm_annotation_model_clusters[
    #         (unperm_annotation_model_clusters["animal_unit_estimate"] < AFO_bound[1])
    #         & (
    #             unperm_annotation_model_clusters["human_animal_unit_estimate"]
    #             >= potential_CAFO_thresh
    #         )
    #     ].shape[0],
    # )
    unpermitted_potential_CAFOs["set"] = "Unpermitted potential CAFOs"

    # Define set 3: unpermitted clusters between 500 AU and 1000 AU
    afos_under_thresh = unpermitted_annotation_clusters[
        (unpermitted_annotation_clusters[thresh_col] >= 500)
        & (unpermitted_annotation_clusters[thresh_col] < potential_CAFO_thresh)
    ].copy()
    afos_under_thresh["set"] = "AFOs between 500 and 1000 AU"



    # # Define set 4: unpermitted model-detected dairy AFO clusters
    # unperm_model_detected_afos = unpermitted_model_clusters[
    #     (unpermitted_model_clusters[thresh_col] < just_under_CAFO_thresh)
    # ].copy()
    # unperm_model_detected_afos["set"] = "Smaller model-predicted AFOs"

    # Calculate cluster-level variables
    all_clusters = pd.concat(
        [
            dairy_matched_clusters,
            unpermitted_potential_CAFOs,
            afos_under_thresh        ]
    )

    # Print number of rows, and unique polygon_indices, grouped by 'set'
    print(all_clusters.groupby("set").size())
    print("Number of unique polygon indices")
    print(all_clusters.groupby("set")["polygon_indices"].apply(lambda x: x.apply(tuple).nunique()))

    # Plot comparing size of these different cluster sets
    plot_size_unperm_vs_perm(
        all_clusters, save_path=size_plot_au_save_path, column="animal_unit_estimate"
    )

    plot_size_unperm_vs_perm(all_clusters, save_path=size_plot_m2_save_path, column="cluster_area_m2")

    # This function actually adds all the variables at facility-level indicating distance to various water-related areas/relevant dummy variables
    all_clusters = caf.analyze_water_pollution_stats(
        all_clusters,
        slope_data_path,
        all_waters,
        impaired_waters,
        water_table_depth,
        snapmaps_layers,
    )
    all_clusters.reset_index(drop=True, inplace=True)
    # Join clusters with milk producer data
    all_clusters_matched_milk = all_clusters.sjoin_nearest(
        milk_producers.to_crs(cfg.WI_EPSG), max_distance=500
    )
    # polygon_indices are lists (unhashable), so .isin() fails silently.
    # Convert to tuples for correct set-membership test.
    matched_pstrs = set(
        tuple(x) if isinstance(x, list) else x
        for x in all_clusters_matched_milk["polygon_indices"]
    )
    all_clusters["matched_milk"] = all_clusters["polygon_indices"].apply(
        lambda x: (tuple(x) if isinstance(x, list) else x) in matched_pstrs
    )

    all_clusters["n_buildings"] = all_clusters["polygon_indices"].apply(
        lambda x: len(x)
    )
    all_clusters["n_parcels"] = all_clusters["parcel_indices"].apply(lambda x: len(x))

    # Summarize variables by cluster set

    ## Create an aggergation dict that defines how variables should be aggregated at the cluster level
    agg_dict = {}
    mean_cols = [
        "gw_50",
        "gw_20",
        "gw_0",
        "impaired_water_distance",
        "water_distance",
        "closest_water_impaired",
        "mean_slope",
        "cluster_area_m2",
        "n_buildings",
        "n_parcels",
        "animal_unit_estimate",
        "matched_milk",
        "Dairy_count_estimate",
        "bedrock_lt5ft_distance",
        "bedrock_lt5ft_dummy",
        "silurian_2_5_distance",
        "silurian_2_5_dummy",
        "silurian_0_2_distance",
        "silurian_0_2_dummy",
        "shallow_silurian_distance",
        "shallow_silurian_dummy",
        "swqma_300ft_distance",
        "swqma_300ft_dummy",
        "swqma_1000ft_distance",
        "swqma_1000ft_dummy",
        "cafo_w_restrict_distance",
        "cafo_r_restrict_distance",
        "slope_greater_12_distance",
        "slope_greater_12_dummy",
        "hydro_intermit_distance",
        "hydro_perennial_distance"
#        "snapmaps_water_distance",
    ]
    for col in mean_cols:
        # m_ being shorthand for mean
        agg_dict[col] = "mean"

    grouped_results_means = all_clusters.groupby("set").agg(agg_dict)
    grouped_results_means = grouped_results_means.round(2)
    grouped_results_means = grouped_results_means.T

    # Group other cols
    grouped_results_others = all_clusters.groupby("set").agg(
        SE_area=("cluster_area_m2", lambda x: x.sem()),
        SE_n_buildings=("n_buildings", lambda x: x.sem()),
        SE_n_parcels=("n_parcels", lambda x: x.sem()),
        SE_animal_units=("animal_unit_estimate", lambda x: x.sem()),
        n_clusters=("cluster_area_m2", "count"),
        # Add medians
        median_area=("cluster_area_m2", "median"),
        median_n_buildings=("n_buildings", "median"),
        median_n_parcels=("n_parcels", "median"),
        median_animal_units=("animal_unit_estimate", "median"),
    )

    grouped_results_others = grouped_results_others.round(2)
    grouped_results_others = grouped_results_others.T

    # Append the two dataframes
    grouped_results = pd.concat([grouped_results_means, grouped_results_others], axis=0)

    p_vals = []

    # Only doing p-values for mean_cols
    for col in mean_cols:
        p_val = ss.ttest_ind(
            all_clusters[all_clusters["set"] == "Permitted dairy CAFOs"][col],
            all_clusters[all_clusters["set"] == "Unpermitted potential CAFOs"][col],
        ).pvalue
        p_vals.append(p_val.round(3))

    # Insert missing values for SE and median rows
    p_vals.extend([None] * 9)

    p_vals = pd.DataFrame(
        data={"Permitted/unpermitted p_val": p_vals}, index=grouped_results.index
    )

    grouped_results = pd.concat([grouped_results, p_vals], axis=1)

    # Rearrange columns (4-group version — returned to caller for risk analysis)
    grouped_results = grouped_results[
        [
            "Permitted dairy CAFOs",
            "Unpermitted potential CAFOs",
            "AFOs between 500 and 1000 AU",
            "Permitted/unpermitted p_val",
        ]
    ]

    # # For the saved CSV: combine "AFOs just under threshold" and "Smaller model-predicted AFOs"
    # # into a single column using proper weighted means (via relabeling before aggregation).
    # combined_label = "AFOs below CAFO threshold"
    # _table_clusters = all_clusters.copy()
    # _table_clusters.loc[
    #     _table_clusters["set"].isin([
    #         "AFOs just under threshold", "Smaller model-predicted AFOs"
    #     ]),
    #     "set",
    # ] = combined_label
    # _table_means = _table_clusters.groupby("set").agg(agg_dict).round(2).T
    # _table_others = _table_clusters.groupby("set").agg(
    #     SE_area=("cluster_area_m2", lambda x: x.sem()),
    #     SE_n_buildings=("n_buildings", lambda x: x.sem()),
    #     SE_n_parcels=("n_parcels", lambda x: x.sem()),
    #     SE_animal_units=("animal_unit_estimate", lambda x: x.sem()),
    #     n_clusters=("cluster_area_m2", "count"),
    #     median_area=("cluster_area_m2", "median"),
    #     median_n_buildings=("n_buildings", "median"),
    #     median_n_parcels=("n_parcels", "median"),
    #     median_animal_units=("animal_unit_estimate", "median"),
    # ).round(2).T
    # table_for_csv = pd.concat([_table_means, _table_others], axis=0)
    # table_for_csv = pd.concat([table_for_csv, p_vals], axis=1)
    # table_for_csv = table_for_csv[
    #     [
    #         "Permitted dairy CAFOs",
    #         "Unpermitted potential CAFOs",
    #         combined_label,
    #         "Permitted/unpermitted p_val",
    #     ]
    # ]

    # Save results
    if save_path is not None:
        grouped_results.to_csv(save_path)

    return grouped_results, all_clusters

def add_pca_risk_index(all_clusters: gpd.GeoDataFrame,
                       risk_variables: list[str],
                       invert_variables: list[str] = None):
    """
    This function adds the PCA risk index to the all_clusters dataframe.
    Args:
        all_clusters: dataframe of facilities, with risk information columns
        risk_variables: list of columns to use to calculate the PCA risk index
        invert_variables: list of columns where lower raw value = higher risk (e.g. distance
            variables). These are transformed via 1/(x+1) so that higher value = higher risk
            before PCA, ensuring PC1 can be interpreted as a risk axis.
    Returns:
        all_clusters: dataframe of all clusters with PCA risk index
    """

    # Filter to only rows with valid data for all risk variables
    available_cols = [col for col in risk_variables if col in all_clusters.columns]
    print(f"Missing variables: {set(risk_variables) - set(available_cols)}")

    # Use only available columns
    risk_variables = available_cols

    # Create a working dataframe with only the risk variables
    risk_data = all_clusters[risk_variables].copy()

    # Handle missing values - drop rows with any NaN
    print(f"\nOriginal number of facilities: {len(risk_data)}")
    risk_data = risk_data.dropna()
    print(f"Facilities with complete data: {len(risk_data)}")

    # Convert boolean columns to numeric for PCA
    for col in risk_data.columns:
        if risk_data[col].dtype == bool:
            risk_data[col] = risk_data[col].astype(int)

    # Invert distance variables so that higher value = higher risk (closer = more risky)
    if invert_variables:
        for var in invert_variables:
            if var in risk_data.columns:
                risk_data[var] = 1 / (risk_data[var] + 1)

    # Standardize variables (Z-score)
    scaler = StandardScaler()
    risk_data_scaled = scaler.fit_transform(risk_data[risk_variables])
    risk_data_scaled_df = pd.DataFrame(risk_data_scaled, columns=risk_variables, index=risk_data.index)
    
    #  Run PCA
    pca = PCA(random_state=0)
    pca_result = pca.fit_transform(risk_data_scaled_df)
    print("PCA complete")

    # Interpret
    print(f"Explained variance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
        print(f"  PC{i}: {var:.1%}")

    print("\nPC1 Loadings (weights):")
    pc1_loadings = pd.Series(pca.components_[0], index=risk_variables)
    pc1_loadings_sorted = pc1_loadings.abs().sort_values(ascending=False)
    for var in pc1_loadings_sorted.index:
        loading = pc1_loadings[var]
        print(f"  {var:30s}: {loading:7.3f}")

    # Create PC1 risk scores
    pc1_scores = pca_result[:, 0]

    # Normalize PC1 scores to 0-1 range for easier interpretation
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    pc1_normalized = (pc1_scores - pc1_min) / (pc1_max - pc1_min)

    # Add to all_clusters (only for rows with complete data)
    all_clusters['pca_risk_index'] = np.nan
    all_clusters.loc[risk_data.index, 'pca_risk_index'] = pc1_normalized

    print("PC1 Risk Index created")
    print(f"  Min: {pc1_normalized.min():.3f}")
    print(f"  Max: {pc1_normalized.max():.3f}")
    print(f"  Mean: {pc1_normalized.mean():.3f}")
    print(f"  Median: {np.median(pc1_normalized):.3f}")

    return all_clusters

def add_hand_built_risk_index(all_clusters: gpd.GeoDataFrame, 
                              hand_built_weights: dict,
                              invert_variables: list[str]):
    """
    Build a hand-crafted risk index where:
      - Variables in invert_variables are inverted so larger = higher risk
      - All weights are assumed to be positive risk multipliers
      - Each variable is normalized (0–1) AFTER inversion
      - Final weighted index is normalized to 0–1
    """

    # Step 1: Determine usable variables
    risk_variables = list(hand_built_weights.keys())
    available_cols = [col for col in risk_variables if col in all_clusters.columns]

    print(f"Available variables: {available_cols}")
    print(f"Missing variables: {set(risk_variables) - set(available_cols)}")

    # Restrict weights to available cols
    hand_built_weights = {k: hand_built_weights[k] for k in available_cols}
    risk_variables = available_cols

    # Extract data
    risk_data = all_clusters[risk_variables].copy()

    print(f"\nOriginal facilities: {len(risk_data)}")
    risk_data = risk_data.dropna()
    print(f"Facilities with complete data: {len(risk_data)}")

    # Step 2: Convert booleans to int
    for col in risk_variables:
        if risk_data[col].dtype == bool:
            risk_data[col] = risk_data[col].astype(int)

    # Step 3: Invert selected variables BEFORE normalization
    # This ensures that after inversion: larger value = higher risk
    for var in invert_variables:
        if var in risk_data.columns:
            # 1/(x+1) ensures small distances → big inverted values → higher risk
            risk_data[var] = 1 / (risk_data[var] + 1)

    # Step 4: Normalize each variable to [0,1]
    normalized = pd.DataFrame(index=risk_data.index)
    for var in risk_variables:
        col = risk_data[var]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            normalized[var] = (col - cmin) / (cmax - cmin)
        else:
            normalized[var] = 0.5  # No variation → neutral

    # Step 5: Normalize weights to sum to 1
    weight_sum = sum(hand_built_weights.values())
    normalized_weights = {k: w / weight_sum for k, w in hand_built_weights.items()}

    print("\nNormalized weights:")
    for v, w in normalized_weights.items():
        print(f"  {v:30s}: {w:.3f}")

    # Step 6: Weighted sum (risk score)
    risk_index = pd.Series(0.0, index=normalized.index)
    for var, w in normalized_weights.items():
        risk_index += normalized[var] * w

    # Step 7: Normalize final risk index to [0,1]
    rmin, rmax = risk_index.min(), risk_index.max()
    if rmax > rmin:
        risk_index_norm = (risk_index - rmin) / (rmax - rmin)
    else:
        risk_index_norm = risk_index * 0 + 0.5

    # Add back to full df
    all_clusters['hand_built_risk_index'] = np.nan
    all_clusters.loc[normalized.index, 'hand_built_risk_index'] = risk_index_norm

    return all_clusters


def unperm_universe_table(
    human_annotation_clusters: gpd.GeoDataFrame,
    model_clusters: gpd.GeoDataFrame,
    permit_data: gpd.GeoDataFrame,
    county_data: gpd.GeoDataFrame,
    milk_producers: gpd.GeoDataFrame,
    AFO_bound: tuple = (150, 1000),
    potential_CAFO_thresh: float = 1500,
    omit_ewg_region: bool = False,
    save_path: str = None,
):
    """
    This function creates a table summarizing the count of unpermitted facilities of various
     sizes, depending on uncertainty assumptions.
    Args:
        - human_annotation_clusters: dataset of CF annotation clusters
        - model_clusters: dataset of clustered model predictions
        - permit_data: dataset of permitted CAFOs
        - county_data: WI county shapefile
        - AFO_bounds: tuple, lower and upper bounds for defining an AFO (non-CAFO)
        - potential_CAFO_thresh: AU threshold for defining a potential CAFO.
        - omit_ewg_region: bool, whether to omit data points from the EWG study region
        - save_path: file path where to save table as .csv
    Returns:
        - Dataframe of unpermitted facility counts
    """
    # Preprocess clusters
    if "animal_unit_estimate" not in human_annotation_clusters.columns:
        human_annotation_clusters = est_au.sample_and_calc_au(human_annotation_clusters)
    if "animal_unit_estimate" not in model_clusters.columns:
        model_clusters = est_au.sample_and_calc_au(model_clusters)
    if omit_ewg_region:
        #  Merge in county data to ensure proper screening by EWG region
        human_annotation_clusters = (
            human_annotation_clusters.copy()
            .sjoin(county_data[["geometry", "train_counties"]])
            .drop(columns=["index_right"])
        )
        human_annotation_clusters = human_annotation_clusters[
            ~human_annotation_clusters["train_counties"]
        ]

    # Filter model clusters based on minimum AFO bound
    model_clusters = model_clusters.copy()[
        model_clusters["animal_unit_estimate"] >= AFO_bound[0]
    ]
    matched_model_clusters, err = caf.merge_clusters_permits(
        model_clusters,
        permit_data,
        drop_multi_matches=False,
        sum_satellite_counts=False,
    )
    unpermitted_model_clusters = model_clusters[
        ~(
            model_clusters["polygon_indices"].isin(
                matched_model_clusters["polygon_indices"]
            )
        )
    ].copy()

    # Define unpermitted clusters
    matched_annotation_clusters, err = caf.merge_clusters_permits(
        human_annotation_clusters,
        permit_data,
        drop_multi_matches=False,
        sum_satellite_counts=False,
    )
    unpermitted_annotation_clusters = human_annotation_clusters[
        ~(
            human_annotation_clusters["polygon_indices"].isin(
                matched_annotation_clusters["polygon_indices"]
            )
        )
    ].copy()

    all_counts = pd.DataFrame(
        data={
            f"AU >={potential_CAFO_thresh}": [],
            f"{potential_CAFO_thresh}> AU >= {AFO_bound[1]}": [],
            f"{AFO_bound[1]} > AU >= {AFO_bound[0]}": [],
        }
    )
    for classification_method in [
        "animal_unit_estimate",
        "animal_units_lower",
        "animal_units_upper",
    ]:
        unpermitted_annotation_clusters["animal_unit_estimate"] = (
            unpermitted_annotation_clusters[classification_method]
        )
        unpermitted_model_clusters["animal_unit_estimate"] = unpermitted_model_clusters[
            classification_method
        ]

        # Count sets based on size
        n_large = unpermitted_annotation_clusters[
            (
                unpermitted_annotation_clusters["animal_unit_estimate"]
                >= potential_CAFO_thresh
            )
        ].shape[0]
        n_medium = unpermitted_annotation_clusters[
            (unpermitted_annotation_clusters["animal_unit_estimate"] >= AFO_bound[1])
            & (
                unpermitted_annotation_clusters["animal_unit_estimate"]
                < potential_CAFO_thresh
            )
        ].shape[0]
        n_small = unpermitted_model_clusters[
            (unpermitted_model_clusters["animal_unit_estimate"] < AFO_bound[1])
        ].shape[0]

        # Determine proportion with milk license
        n_large_milk = (
            unpermitted_annotation_clusters[
                (
                    unpermitted_annotation_clusters["animal_unit_estimate"]
                    >= potential_CAFO_thresh
                )
            ]
            .sjoin_nearest(milk_producers, max_distance=500)
            .drop_duplicates("polygon_indices")
            .shape[0]
        )
        n_medium_milk = (
            unpermitted_annotation_clusters[
                (
                    unpermitted_annotation_clusters["animal_unit_estimate"]
                    >= AFO_bound[1]
                )
                & (
                    unpermitted_annotation_clusters["animal_unit_estimate"]
                    < potential_CAFO_thresh
                )
            ]
            .sjoin_nearest(milk_producers, max_distance=500)
            .drop_duplicates("polygon_indices")
            .shape[0]
        )
        n_small_milk = (
            unpermitted_model_clusters[
                (unpermitted_model_clusters["animal_unit_estimate"] < AFO_bound[1])
            ]
            .sjoin_nearest(milk_producers, max_distance=500)
            .drop_duplicates("polygon_indices")
            .shape[0]
        )

        all_counts.loc[classification_method] = [
            f"{n_large} ({round(n_large_milk*100/n_large, 3)}%)",
            f"{n_medium} ({round(n_medium_milk*100/n_medium, 3)}%)",
            f"{n_small} ({round(n_small_milk*100/n_small, 3)}%)",
        ]

    # Save results
    if save_path is not None:
        all_counts.to_csv(save_path)

    return all_counts

def get_labeled_images(how="csv", data_path=None):
    """
    Get all labeled images from the CF annotations folder.
    Args:
        how (str): How to get the labeled images. Either "csv" or "json".
        data_path (str): Path to the data folder.
    Returns:
        all_labeled_images (pd.DataFrame): DataFrame containing all labeled images.
    """
    # Check if how is either "csv" or "json"
    if how not in ["csv", "json"]:
        raise ValueError("how must be either 'csv' or 'json'")

    # Check if data_path is provided
    if data_path is None:
        raise ValueError("data_path must be provided")
    # Check if data_path is a valid path
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} does not exist")
    

    ## Either get through CSV
    if how == "csv":
        # Read the CSV file
        all_labeled_images = pd.read_csv(data_path / "Annotations/all_labeled_images.csv")
   
    else:

        # Try and reconstruct all_labeled_images but just using json files in the annotations folder

        # Define the base directory
        base_dir = data_path / "Annotations/Annotations by submission"

        # Initialize a list to store the data
        data = []

        # Loop through each batch (sorted for reproducibility)
        for batch_name in sorted(os.listdir(base_dir)):
            batch_path = os.path.join(base_dir, batch_name, 'raw_jsons')

            if os.path.isdir(batch_path):  # Ensure it's a directory
                # Loop through each image file (sorted for reproducibility)
                for file_name in sorted(os.listdir(batch_path)):
                    if file_name.endswith(".json"):
                        image_name = file_name.replace(".json", ".jpeg")

                        # Check if image_name starts with 'buffered_'
                        buffered = image_name.startswith("buffered_")
                        
                        # Remove 'buffered_' prefix if present
                        if buffered:
                            image_name = image_name.replace("buffered_", "", 1)
                        
                        
                        # Check if the image_name has the 'WI_' prefix
                        if not image_name.startswith("WI_"):
                            image_name = f"WI_{image_name}"
                        
                        # Append to the list
                        data.append((image_name, batch_name, buffered))

        # Create a dataframe
        all_labeled_images = pd.DataFrame(data, columns=['Image name', 'CF submission', 'buffered?'])

    return all_labeled_images

def load_data(analysis_output_path, data_path, land_parcel_path, cluster_path, model_prediction_path, read_snapmaps=True, snapmaps_feather=True):

    # Load image bound map
    image_bound_map = gpd.read_file(analysis_output_path / "image_bound_map.geojson")
    # Load county boundary shapefile
    counties = gpd.read_file(
        data_path / "County_Boundaries_24K/County_Boundaries_24K.shp"
    )
    counties["train_counties"] = counties["COUNTY_NAM"].isin(cfg.EWG_COUNTIES)
    # Load land parcels
    parcels = cluster.load_parcels(
        land_parcel_path / "WI_parcels.feather", feather=True
    )

    # Load milk producers
    milk_producers = gpd.read_file(
        data_path / "milk_producers.geojson", driver="GeoJSON"
    )
    # Coalesce manual geocoded locations with automatic geocoded locations
    milk_producers["location"] = milk_producers.apply(
        lambda x: (
            x.geometry
            if x.manual_location == "None"
            else shapely.from_wkt(x.manual_location)
        ),
        axis=1,
    )
    milk_producers.set_geometry("location", inplace=True)
    milk_producers.set_crs(crs=cfg.WI_EPSG, inplace=True)
    milk_producers.drop(["manual_location"], axis=1, inplace=True)

    # Load known CAFO locations
    WDNR_CAFOs = gpd.read_file(
        data_path / "WDNR_permitted_CAFOs/WDNR_CAFOs.geojson", driver="GeoJSON", crs=cfg.WI_EPSG
    )
    if not WDNR_CAFOs.sindex:
        WDNR_CAFOs.sindex

    # Load EWG AFO location predictions
    ewg_afos = gpd.read_file(
        data_path / "ewg_AFOs_012022.geojson.txt", driver="GeoJSON"
    )
    ewg_afos.to_crs(cfg.WI_EPSG, inplace=True)

    # Load all CF annotations outside EWG study region
    cf_annotations_exc_EWG = cluster.load_polygons(
        data_path / "Annotations/full_state_cf_annotations.geojson",
        counties,
        image_bound_map,
    )

    # Load all CF annotation clusters statewide
    all_cf_clusters = cluster.load_clusters(cluster_path / "all_CF_clusters.csv")
    if "animal_unit_estimate" not in all_cf_clusters.columns:
        all_cf_clusters = est_au.sample_and_calc_au(
            all_cf_clusters, include_area_uncertainty=False
            # Specifying 0 model uncertainty because these are human annotation
        )
    # split by EWG study region
    all_cf_clusters = amo.label_by_ewg_region(all_cf_clusters, counties)
    cf_clusters_EWG = all_cf_clusters[all_cf_clusters["ewg_region"]]
    cf_clusters_exc_EWG = all_cf_clusters[~all_cf_clusters["ewg_region"]]

    # Load poultry and inception model detections
    inception_detections = gpd.read_file(
        model_prediction_path / "inception/inception_dets.geojson"
    ).to_crs(cfg.WI_EPSG)
    poultry_detections = gpd.read_file(
        model_prediction_path / "poultry/all_poultry_predictions_confidence_0.1.geojson"
    ).to_crs(cfg.WI_EPSG)

    # Load pre-created three- and four-band model clusters
    four_band_clusters = cluster.load_clusters(cluster_path / "four_band_clusters.csv")
    # remove clusters in EWG study region
    four_band_clusters = amo.label_by_ewg_region(four_band_clusters, counties)
    if "animal_unit_estimate" not in four_band_clusters.columns:
        four_band_clusters = est_au.sample_and_calc_au(four_band_clusters, include_area_uncertainty=True)
    
    four_band_clusters_exc_EWG = four_band_clusters[~four_band_clusters["ewg_region"]]

    three_band_clusters = cluster.load_clusters(
        cluster_path / "three_band_clusters.csv"
    )
    # add county name and remove clusters in EWG study region
    three_band_clusters = amo.label_by_ewg_region(three_band_clusters, counties)
    three_band_clusters_exc_EWG = three_band_clusters[
        ~three_band_clusters["ewg_region"]
    ]

    all_waters = gpd.read_file(
        data_path / "water_data" / "24k_Hydro_Waterbodies_(Open_Water).geojson"
    ).to_crs(cfg.WI_EPSG)
    impaired_lakes = gpd.read_file(
        data_path / "water_data" / "impaired_lakes.geojson"
    ).to_crs(cfg.WI_EPSG)
    impaired_rivers_streams = gpd.read_file(
        data_path / "water_data" / "impaired_rivers_streams.geojson"
    ).to_crs(cfg.WI_EPSG)
    impaired_waters = gpd.GeoDataFrame(
        pd.concat([impaired_lakes, impaired_rivers_streams], ignore_index=True)
    )
    # Filter to NPS impairments
    impaired_waters = impaired_waters[
        impaired_waters["ALL_SOURCES"] == "Non-Point Source (Rural or Urban)"
    ]
    water_table_depth = gpd.read_file(
        data_path / "water_data" / "GCSM_-_Water_Table_Depth.geojson"
    ).to_crs(cfg.WI_EPSG)

    snapmaps=None
    if read_snapmaps:
        snapmaps = process_snapmaps.load_snapmaps(feather=snapmaps_feather, feather_dir = data_path / "snapmaps_feather",
                                                  raw_gdb = data_path / "NM_590_CAFO_STATEWIDE.gdb", crs=cfg.WI_EPSG,
                                                  simplify_geometries=True, tolerance=10)
        
    return image_bound_map, counties, parcels, milk_producers, WDNR_CAFOs, ewg_afos, cf_annotations_exc_EWG, all_cf_clusters, cf_clusters_EWG, cf_clusters_exc_EWG, inception_detections, poultry_detections, four_band_clusters, four_band_clusters_exc_EWG, three_band_clusters, three_band_clusters_exc_EWG, all_waters, impaired_waters, water_table_depth, snapmaps


# ============================================================================
# Functions extracted from all_key_results.ipynb inline cells
# ============================================================================

def plot_permit_rate_by_size(all_cf_clusters: gpd.GeoDataFrame,
                             save_path=None):
    """Bar chart showing the percentage of facilities that are permitted,
    by animal unit size category (above 1000 AU only).
    Extracted from all_key_results.ipynb cell 99."""
    df = all_cf_clusters.copy()
    df["animal_unit_category"] = pd.cut(
        df["animal_unit_estimate"],
        bins=[500, 800, 1000, 1250, 1500, 2000, 5000, np.inf],
    )
    df["permitted"] = df["set"] == "Permitted dairy CAFOs"

    above_1000 = df[df["animal_unit_estimate"] >= 1000]
    pct_permitted = above_1000.groupby("animal_unit_category")["permitted"].agg(["mean", "count"])
    pct_permitted["pct"] = pct_permitted["mean"] * 100
    pct_permitted = pct_permitted.reset_index()
    pct_permitted = pct_permitted[pct_permitted["count"] > 0]

    categories = sorted(
        pct_permitted["animal_unit_category"],
        key=lambda x: x.left if hasattr(x, "left") else 0,
    )
    category_labels = [
        f"{int(cat.left)}-{int(cat.right)}" if cat.right != np.inf else f"{int(cat.left)}-Inf"
        for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(pct_permitted)), pct_permitted["pct"],
        color=cfg.COLOR_PERMITTED, alpha=cfg.ALPHA_FILL, edgecolor="black",
    )

    for i, (bar, pct, count) in enumerate(
        zip(bars, pct_permitted["pct"], pct_permitted["count"])
    ):
        height = bar.get_height() + 1
        if np.isfinite(height) and height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{pct:.1f}%\n(n={int(count)})",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(range(len(pct_permitted)))
    ax.set_xticklabels(category_labels)
    ax.set_xlabel("Animal Unit Estimate Category")
    ax.set_ylabel("% of facilities")
    ax.set_title("Permit Rate by Estimated Size", pad=20)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(False)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")

    print("\nSummary by Animal Unit Category:")
    print(pct_permitted[["animal_unit_category", "pct", "count"]].to_string(index=False))


def plot_unpermitted_vs_permitted_distribution(
    unperm_potential: gpd.GeoDataFrame,
    permit_matched_cf_clusters: gpd.GeoDataFrame,
    unpermitted_cf_clusters: gpd.GeoDataFrame,
    chosen_logic_col: str,
    save_path=None,
    show_area_distribution: bool = True,
):
    """Stacked histogram (AU distribution) with coarse-bin permit-rate twinx.
    Below-threshold unpermitted farms and above-threshold unpermitted potential
    CAFOs share one series; permitted bars are stacked on top.  A darker teal
    line on a secondary axis shows the permit rate per 500-AU bin above 1000.
    The y-axis is clipped at 95 so the post-1000 overlap is readable; a callout
    annotates the truncated bars.  Optionally also plots the area distribution."""

    C_PERMIT_RATE = "#006868"   # darker teal for permit-rate series

    # ── build histogram arrays ───────────────────────────────────────────────
    perm_au   = permit_matched_cf_clusters["animal_unit_estimate"].values
    unperm_au = unperm_potential["animal_unit_estimate"].values
    under_au  = unpermitted_cf_clusters[
        (unpermitted_cf_clusters["animal_unit_estimate"] > 500)
        & (unpermitted_cf_clusters[chosen_logic_col] < 1000)
    ]["animal_unit_estimate"].values

    bins_au  = np.linspace(500, 6000, 56)
    centers  = (bins_au[:-1] + bins_au[1:]) / 2
    bw       = bins_au[1] - bins_au[0]

    v_under,  _ = np.histogram(under_au,  bins=bins_au)
    v_unperm, _ = np.histogram(unperm_au, bins=bins_au)
    v_perm_r, _ = np.histogram(perm_au,   bins=bins_au)
    v_perm      = np.where(centers >= 1000, v_perm_r, 0)
    v_all_unp   = v_under + v_unperm

    # ── coarse permit-rate bins ──────────────────────────────────────────────
    pr_bins   = [500, 1000, 1500, 2000, 3000, 6001]
    pr_mids   = [(pr_bins[i] + pr_bins[i+1]) / 2 for i in range(len(pr_bins)-1)]
    pr_mids[-1] = 4000
    pr_labels = [
        f"{pr_bins[i]}–{pr_bins[i+1] if pr_bins[i+1] < 6001 else '∞'}"
        for i in range(len(pr_bins)-1)
    ]
    # Use all unpermitted farms ≥500 AU as denominator so the 500-1000 bin
    # reflects # permitted / (all farms in that size range), not just the tiny
    # slice of permitted farms that happen to be under 1000 AU.
    all_unperm_au = unpermitted_cf_clusters["animal_unit_estimate"].values
    all_above = pd.DataFrame({
        "au":      np.concatenate([perm_au[perm_au >= 500], all_unperm_au[all_unperm_au >= 500]]),
        "is_perm": np.concatenate([
            np.ones(len(perm_au[perm_au >= 500])),
            np.zeros(len(all_unperm_au[all_unperm_au >= 500])),
        ]),
    })
    all_above["pr_bin"] = pd.cut(all_above["au"], bins=pr_bins, right=False,
                                 labels=pr_labels)
    pr_stats = (all_above.groupby("pr_bin", observed=True)["is_perm"]
                .agg(["mean", "count"]).reset_index())
    pr_stats["pct"] = pr_stats["mean"] * 100

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (lo, hi) in enumerate(zip(pr_bins[:-1], pr_bins[1:])):
        ax.axvspan(lo, min(hi, 6000), alpha=0.05 if i % 2 == 0 else 0.0,
                   color="#888", zorder=0)

    ax.bar(centers, v_all_unp, width=bw*0.88, color=cfg.COLOR_UNPERMITTED,
           alpha=0.65, label="Unpermitted farms", zorder=2)
    ax.bar(centers, v_perm, width=bw*0.88, color=cfg.COLOR_PERMITTED,
           alpha=0.65, bottom=v_all_unp, label="Permitted dairy CAFOs", zorder=2)

    Y_MAX = 95
    ax.set_ylim(0, Y_MAX)
    ax.grid(False)

    # Truncated-bar callout above the clipped region
    ax.text(0.045, 1.06,
            f"{len(under_au):,} farms\nbelow threshold\n(bars truncated)",
            transform=ax.transAxes, fontsize=13, color="#C47800",
            ha="center", va="bottom", clip_on=False, linespacing=1.4,
            bbox=dict(fc="white", ec="#C47800", lw=0.7,
                      boxstyle="round,pad=0.3"))
    for c, v in zip(centers[centers < 1000], v_all_unp[centers < 1000]):
        if v > Y_MAX:
            ax.bar(c, Y_MAX * 0.04, width=bw*0.88, bottom=Y_MAX * 0.96,
                   color="white", hatch="//", edgecolor=cfg.COLOR_UNPERMITTED,
                   linewidth=0.4, zorder=3, alpha=0.8)

    ax.axvline(1000, color=cfg.COLOR_THRESHOLD_LINE, linestyle="--", linewidth=2.0, zorder=3)
    ax.text(1004, 0.97, "Permit threshold", color=cfg.COLOR_THRESHOLD_LINE,
            ha="left", va="top", fontsize=14, fontweight="bold",
            transform=ax.get_xaxis_transform(),
            bbox=dict(fc="white", ec="none", pad=1))

    # Explicit font sizes: compensate for ~0.65x LaTeX scaling at \textwidth
    _lbl  = cfg.FIG_LABEL_SIZE + 3   # 12 → 15
    _tick = cfg.FIG_TICK_SIZE  + 3   # 11 → 14
    _lgnd = cfg.FIG_LEGEND_SIZE + 2  # 10 → 12

    ax.set_xlabel("Animal Unit Estimate", fontsize=_lbl)
    ax.set_ylabel("Number of facilities", fontsize=_lbl)
    ax.set_xlim(500, 6000)
    ax.set_xticks(np.arange(500, 6001, 500))
    ax.tick_params(axis="both", labelsize=_tick)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=_lgnd)

    ax2 = ax.twinx()
    ax2.plot(pr_mids, pr_stats["pct"].values, color=C_PERMIT_RATE,
             marker="o", markersize=7, linewidth=1.8, linestyle="-",
             zorder=5, clip_on=False)

    label_offsets = [(0, 10), (28, -4), (28, -4), (0, 10), (0, 10)]
    label_ha      = ["center", "left", "left", "center", "center"]
    for (x, y, n), (dx, dy), ha in zip(
        zip(pr_mids, pr_stats["pct"].values, pr_stats["count"].values),
        label_offsets, label_ha,
    ):
        ax2.annotate(f"{y:.0f}%", xy=(x, y),
                     xytext=(dx, dy), textcoords="offset points",
                     ha=ha, fontsize=_tick - 2, color=C_PERMIT_RATE,
                     arrowprops=dict(arrowstyle="-", color=C_PERMIT_RATE,
                                     lw=0.6) if dx != 0 else None)

    for edge in pr_bins[1:-1]:
        ax2.axvline(edge, color=C_PERMIT_RATE, linewidth=0.6, linestyle=":",
                    alpha=0.35, zorder=1)

    ax2.set_ylabel("% permitted (within size group)", color=C_PERMIT_RATE,
                   fontsize=_lbl)
    ax2.tick_params(axis="y", colors=C_PERMIT_RATE, labelsize=_tick)
    ax2.spines["right"].set_edgecolor(C_PERMIT_RATE)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax2.set_ylim(0, 115)
    ax2.set_yticks(np.arange(0, 101, 20))
    ax2.set_xlim(500, 6000)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")

    # ── optional area distribution (original style, kept for reference) ──────
    if show_area_distribution:
        unperm_afos = unpermitted_cf_clusters[
            (unpermitted_cf_clusters[chosen_logic_col] < 1000)
            & (unpermitted_cf_clusters["animal_unit_estimate"] > 500)
        ]
        bins_area = np.linspace(0, 60000, 100)
        fig2, ax3 = plt.subplots(figsize=(8, 4))
        sns.histplot(unperm_potential["cluster_area_m2"], bins=bins_area,
                     color=cfg.COLOR_UNPERMITTED, label="Potential unpermitted CAFOs",
                     ax=ax3, alpha=cfg.ALPHA_FILL)
        sns.histplot(permit_matched_cf_clusters["cluster_area_m2"], bins=bins_area,
                     color=cfg.COLOR_PERMITTED, label="Known permitted CAFOs",
                     ax=ax3, alpha=cfg.ALPHA_FILL)
        sns.histplot(unperm_afos["cluster_area_m2"], bins=bins_area,
                     color=cfg.COLOR_UNPERMITTED,
                     label="Unpermitted farms, under threshold",
                     ax=ax3, alpha=cfg.ALPHA_FILL)
        ax3.set_xlim(4000, 60000)
        ax3.set_xticks(np.arange(4000, 60001, 5000))
        ax3.set_xlabel("Facility area (m²)")
        ax3.set_ylabel("Number of facilities")
        ax3.legend(frameon=False)
        for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
        plt.tight_layout()
        if save_path is not None:
            area_path = str(save_path).replace(
                "unpermitted_vs_permitted_size_distribution",
                "unpermitted_vs_permitted_area_distribution",
            )
            fig2.savefig(area_path, bbox_inches="tight")
        plt.close("all")


def plot_risk_index_by_au_category(all_clusters: gpd.GeoDataFrame,
                                   risk_col: str,
                                   save_path=None,
                                   min_n: int = 10):
    """Scatter + 95% CI error bars of a risk index by AU category,
    comparing permitted vs unpermitted facilities.  Parameterized so it
    works for both hand_built_risk_index and pca_risk_index.
    Extracted from all_key_results.ipynb cells 117 & 125."""
    risk_data = all_clusters[all_clusters[risk_col].notna()].copy()

    risk_by_category = (
        risk_data.groupby(["animal_unit_category", "permitted"])[risk_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    risk_by_category["sem"] = risk_by_category["std"] / np.sqrt(risk_by_category["count"])
    risk_by_category["ci_low"] = risk_by_category["mean"] - 1.96 * risk_by_category["sem"]
    risk_by_category["ci_high"] = risk_by_category["mean"] + 1.96 * risk_by_category["sem"]

    categories = sorted(
        risk_data["animal_unit_category"].dropna().unique(),
        key=lambda x: x.left if hasattr(x, "left") else 0,
    )
    category_labels = [
        f"{int(cat.left)}-{int(cat.right)}" if cat.right != np.inf else f"{int(cat.left)}-Inf"
        for cat in categories
    ]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    alpha = 0.5

    # Two series: permitted (teal) and unpermitted (yellow), spanning all AU categories.
    series = [
        (True,  cfg.COLOR_PERMITTED,  "Permitted CAFOs",   -0.1),
        (False, cfg.COLOR_UNPERMITTED, "Unpermitted farms", +0.1),
    ]

    rng_jitter = np.random.default_rng(42)

    for perm_status, color, label, offset in series:
        data = risk_by_category[risk_by_category["permitted"] == perm_status]
        sub = risk_data[risk_data["permitted"] == perm_status]

        y_means, y_low, y_high, counts = [], [], [], []
        for i, cat in enumerate(categories):
            row = data[data["animal_unit_category"] == cat]
            if len(row) == 1:
                cnt = int(row["count"].values[0])
                mean = row["mean"].values[0]
                low = row["ci_low"].values[0]
                high = row["ci_high"].values[0]
            else:
                cnt, mean, low, high = 0, np.nan, np.nan, np.nan
            if cnt < min_n:
                mean, low, high = np.nan, np.nan, np.nan
            y_means.append(mean)
            y_low.append(low)
            y_high.append(high)
            counts.append(cnt)

            # Jittered strip
            vals = sub.loc[sub["animal_unit_category"] == cat, risk_col].dropna().values
            if len(vals) > 0:
                jitter = rng_jitter.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(i + offset + jitter, vals, color=color,
                           alpha=0.15, s=8, linewidths=0, zorder=1)

        ax.vlines(x + offset, y_low, y_high, color=color, alpha=0.9, linewidth=2, zorder=3)
        ax.scatter(x + offset, y_means, color=color, alpha=1.0,
                   edgecolor="black", s=70, label=label, zorder=4)


    # CAFO threshold line
    threshold_x = None
    for i, cat in enumerate(categories):
        if hasattr(cat, "left") and cat.left >= 1000:
            threshold_x = i - 0.5
            break
    if threshold_x is not None:
        ax.axvline(threshold_x, color=cfg.COLOR_THRESHOLD_LINE, linestyle="--", linewidth=cfg.FIG_LINEWIDTH)
        ax.text(threshold_x + 0.03, 0.95,
                "Permit threshold", color=cfg.COLOR_THRESHOLD_LINE, va="top", fontsize=10,
                transform=ax.get_xaxis_transform())

    pretty_name = risk_col.replace("_", " ").title()
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, ha="right", rotation=30)
    ax.set_xlabel("Animal Unit Category")
    ax.set_ylabel(f"Mean {pretty_name}")
    y_min = max(0, risk_data[risk_col].quantile(0.01) - 0.05)
    y_max = min(1, risk_data[risk_col].quantile(0.99) + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    ax.xaxis.grid(True, color="#cccccc", linewidth=0.6, linestyle=":", zorder=0)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.legend(frameon=True, fontsize=cfg.FIG_LEGEND_SIZE, loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_risk_index_by_au_category_boxplot(all_clusters: gpd.GeoDataFrame,
                                           risk_col: str,
                                           save_path=None,
                                           min_n: int = 10):
    """Boxplot version of plot_risk_index_by_au_category.

    Two side-by-side boxplots per AU category (permitted teal, unpermitted
    orange), showing the full distribution rather than mean ± 95% CI.
    Categories with fewer than min_n observations are omitted.
    """
    risk_data = all_clusters[all_clusters[risk_col].notna()].copy()

    categories = sorted(
        risk_data["animal_unit_category"].dropna().unique(),
        key=lambda x: x.left if hasattr(x, "left") else 0,
    )
    category_labels = [
        f"{int(cat.left)}-{int(cat.right)}" if cat.right != np.inf else f"{int(cat.left)}+"
        for cat in categories
    ]
    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    series = [
        (True,  cfg.COLOR_PERMITTED,  "Permitted CAFOs",   -width / 2),
        (False, cfg.COLOR_UNPERMITTED, "Unpermitted farms", +width / 2),
    ]

    bp_props = dict(linewidth=1.2)

    for perm_status, color, label, offset in series:
        sub = risk_data[risk_data["permitted"] == perm_status]
        box_data = []
        positions = []
        for i, cat in enumerate(categories):
            vals = sub.loc[sub["animal_unit_category"] == cat, risk_col].dropna().values
            if len(vals) >= min_n:
                box_data.append(vals)
                positions.append(i + offset)

        if not box_data:
            continue

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            notch=False,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.6, **bp_props),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color=color, **bp_props),
            capprops=dict(color=color, **bp_props),
        )
        # Add a proxy artist for the legend
        ax.plot([], [], color=color, linewidth=6, alpha=0.6, label=label)

    # CAFO threshold line
    for i, cat in enumerate(categories):
        if hasattr(cat, "left") and cat.left >= 1000:
            threshold_x = i - 0.5
            ax.axvline(threshold_x, color=cfg.COLOR_THRESHOLD_LINE,
                       linestyle="--", linewidth=cfg.FIG_LINEWIDTH)
            ax.text(threshold_x + 0.03, ax.get_ylim()[1] * 0.95,
                    "CAFO permitting threshold", color=cfg.COLOR_THRESHOLD_LINE,
                    rotation=90, va="top", fontsize=10)
            break

    pretty_name = risk_col.replace("_", " ").title()
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, ha="center")
    ax.set_xlabel("Animal Unit Category")
    ax.set_ylabel(pretty_name)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.grid(False)
    ax.xaxis.grid(True, color="#cccccc", linewidth=0.6, linestyle=":", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=True, fontsize=cfg.FIG_LEGEND_SIZE)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def plot_risk_sensitivity_bands(
    all_clusters: gpd.GeoDataFrame,
    weights_dict: dict,
    invert_variables: list[str],
    n_draws: int = 500,
    seed: int = 42,
    save_path=None,
    min_n: int = 10,
):
    """Weight-perturbation sensitivity analysis for the hand-built risk index.

    Draws ``n_draws`` weight vectors uniformly from the Dirichlet simplex
    (alpha=1 for every variable — maximal spread), recomputes the risk index
    for each draw, then shows the 10th–90th percentile band across draws.

    Two-panel figure:
      Top  — overlay of permitted vs unpermitted mean-index ribbons by AU category,
              directly showing whether the two groups are consistently similar.
      Bottom — (permitted mean) – (unpermitted mean) ribbon per AU category,
               with a zero reference line to show direction and magnitude of any gap.

    The original-weight result is overplotted as a filled dot in both panels.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(seed)

    # ── pre-process exactly as in add_hand_built_risk_index ──────────────────
    risk_variables = [k for k in weights_dict if k in all_clusters.columns]
    work = all_clusters[
        risk_variables + ["animal_unit_category", "permitted"]
    ].copy()
    work = work.dropna(subset=risk_variables)

    for col in risk_variables:
        if work[col].dtype == bool:
            work[col] = work[col].astype(int)
    for var in invert_variables:
        if var in work.columns:
            work[var] = 1.0 / (work[var] + 1)

    normed = pd.DataFrame(index=work.index)
    for var in risk_variables:
        col = work[var]
        cmin, cmax = col.min(), col.max()
        normed[var] = (col - cmin) / (cmax - cmin) if cmax > cmin else 0.5

    normed_mat = normed.values  # shape (n_facilities, n_vars)

    categories = sorted(
        work["animal_unit_category"].dropna().unique(),
        key=lambda c: c.left if hasattr(c, "left") else 0,
    )
    cat_labels = [
        f"{int(c.left)}–{int(c.right)}" if c.right != np.inf else f"{int(c.left)}+"
        for c in categories
    ]
    n_cats = len(categories)
    n_vars = len(risk_variables)

    # ── per-draw mean index by (au_cat, permitted) ────────────────────────────
    def _means_for_weights(w_arr):
        w = w_arr / w_arr.sum()
        score = normed_mat @ w
        smin, smax = score.min(), score.max()
        score_n = (score - smin) / (smax - smin) if smax > smin else np.full_like(score, 0.5)
        tmp = work[["animal_unit_category", "permitted"]].copy()
        tmp["s"] = score_n
        out = {}
        for perm in (True, False):
            for cat in categories:
                mask = (tmp["permitted"] == perm) & (tmp["animal_unit_category"] == cat)
                sub = tmp.loc[mask, "s"]
                out[(perm, cat)] = sub.mean() if len(sub) >= min_n else np.nan
        return out

    orig_w = np.array([weights_dict[v] for v in risk_variables])
    orig_means = _means_for_weights(orig_w)

    # Uniform Dirichlet draws — most conservative perturbation (full simplex)
    draws = rng.dirichlet(np.ones(n_vars), size=n_draws)
    all_m = [_means_for_weights(d) for d in draws]

    # Aggregate: arrays shape (n_draws,) per (perm, cat)
    pct_bands = {}
    for perm in (True, False):
        for cat in categories:
            arr = np.array([m[(perm, cat)] for m in all_m])
            valid = arr[~np.isnan(arr)]
            if len(valid) >= 5:
                pct_bands[(perm, cat)] = np.percentile(valid, [10, 50, 90])
            else:
                pct_bands[(perm, cat)] = np.array([np.nan, np.nan, np.nan])

    # Difference arrays: permitted − unpermitted per draw
    diff_bands = {}
    for cat in categories:
        diff = np.array([
            m[(True, cat)] - m[(False, cat)]
            for m in all_m
            if not (np.isnan(m[(True, cat)]) or np.isnan(m[(False, cat)]))
        ])
        diff_bands[cat] = np.percentile(diff, [10, 50, 90]) if len(diff) >= 5 else np.full(3, np.nan)
    orig_diff = {
        cat: orig_means[(True, cat)] - orig_means[(False, cat)]
        for cat in categories
    }

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    x = np.arange(n_cats)

    C_PERM   = cfg.COLOR_PERMITTED
    C_UNPERM = cfg.COLOR_UNPERMITTED
    series = [
        (True,  C_PERM,   "Permitted CAFOs"),
        (False, C_UNPERM, "Unpermitted farms"),
    ]

    for perm_status, color, lbl in series:
        p10s = np.array([pct_bands[(perm_status, c)][0] for c in categories])
        p90s = np.array([pct_bands[(perm_status, c)][2] for c in categories])
        origs = np.array([orig_means[(perm_status, c)] for c in categories])

        # Shaded band between p10 and p90
        ax_top.fill_between(x, p10s, p90s, color=color, alpha=0.25, label=f"{lbl} (10–90th pct)")
        # Original weights dots
        ax_top.scatter(x, origs, color=color, edgecolor="black", s=60, zorder=5)

    # CAFO threshold on top panel
    for i, cat in enumerate(categories):
        if hasattr(cat, "left") and cat.left >= 1000:
            ax_top.axvline(i - 0.5, color=cfg.COLOR_THRESHOLD_LINE,
                           linestyle="--", linewidth=1.5)
            ax_top.text(i - 0.47, 0.96,
                        "Permit threshold", color=cfg.COLOR_THRESHOLD_LINE,
                        rotation=90, va="top", fontsize=cfg.FIG_TICK_SIZE,
                        transform=ax_top.get_xaxis_transform())
            break

    ax_top.set_ylabel("Mean risk index")
    ax_top.set_ylim(0, 0.7)
    for sp in ["top", "right"]: ax_top.spines[sp].set_visible(False)
    ax_top.grid(False)

    # ── bottom: difference panel ─────────────────────────────────────────────
    d_p10 = np.array([diff_bands[c][0] for c in categories])
    d_p90 = np.array([diff_bands[c][2] for c in categories])
    d_orig = np.array([orig_diff[c] for c in categories])

    ax_bot.fill_between(x, d_p10, d_p90, color="#888888", alpha=0.30,
                        label="10–90th pct (random weights)")
    ax_bot.scatter(x, d_orig, color="#222222", edgecolor="black", s=60, zorder=5,
                   label="Original weights")
    ax_bot.axhline(0, color="black", lw=1.0, linestyle="--")

    for i, cat in enumerate(categories):
        if hasattr(cat, "left") and cat.left >= 1000:
            ax_bot.axvline(i - 0.5, color=cfg.COLOR_THRESHOLD_LINE,
                           linestyle="--", linewidth=1.5)
            break

    ax_bot.set_ylim(-0.7, 0.7)
    ax_bot.set_ylabel("Permitted − Unpermitted\nmean risk index")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(cat_labels, ha="center")
    ax_bot.set_xlabel("Animal Unit Category")
    for sp in ["top", "right"]: ax_bot.spines[sp].set_visible(False)
    ax_bot.grid(False)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_els = [
        Patch(fc=C_PERM,   alpha=0.4,  label="Permitted (10–90th pct range)"),
        Patch(fc=C_UNPERM, alpha=0.4,  label="Unpermitted (10–90th pct range)"),
        Patch(fc="#888888", alpha=0.4, label="Difference band (10–90th pct)"),
        Line2D([0],[0], marker="o", lw=0, color="w",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=7, label="Original weights (published index)"),
    ]
    fig.legend(handles=legend_els, loc="upper center", ncol=4,
               frameon=False, fontsize=cfg.FIG_LEGEND_SIZE, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")


def _add_overall_risk_index(all_clusters: gpd.GeoDataFrame,
                            au_relative_weight: float = 0.5,
                            hand_built_col: str = "hand_built_risk_index"):
    """Combine normalized AU estimate with hand-built risk index into a
    single overall risk score. Helper for plot_top_300_breakdown."""
    au_max = all_clusters["animal_unit_estimate"].max()
    au_min = all_clusters["animal_unit_estimate"].min()
    all_clusters["animal_unit_estimate_normalized"] = (
        (all_clusters["animal_unit_estimate"] - au_min) / (au_max - au_min)
    )
    all_clusters["overall_risk_index_hand"] = (
        au_relative_weight * all_clusters["animal_unit_estimate_normalized"]
        + (1 - au_relative_weight) * all_clusters[hand_built_col]
    )
    ov_max = all_clusters["overall_risk_index_hand"].max()
    ov_min = all_clusters["overall_risk_index_hand"].min()
    all_clusters["overall_risk_index_hand_normalized"] = (
        (all_clusters["overall_risk_index_hand"] - ov_min) / (ov_max - ov_min)
    )
    return all_clusters


def plot_top_300_breakdown(all_clusters: gpd.GeoDataFrame,
                           save_path=None,
                           save_table_path=None):
    """Stacked bar chart showing the breakdown of the top-300 riskiest
    facilities (permitted / unpermitted-no-milk / unpermitted-with-milk)
    across different AU-vs-risk weighting schemes.
    Extracted from all_key_results.ipynb cells 129+132."""
    rel_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tot_unperm, tot_unperm_milk = [], []

    for w in rel_weights:
        df = _add_overall_risk_index(all_clusters.copy(), au_relative_weight=w)
        top300 = df.nlargest(300, "overall_risk_index_hand")
        n_unperm = (top300["set"] != "Permitted dairy CAFOs").sum()
        n_unperm_milk = (
            (top300["set"] != "Permitted dairy CAFOs") & (top300["matched_milk"] == True)
        ).sum()
        tot_unperm.append(n_unperm)
        tot_unperm_milk.append(n_unperm_milk)

    tot_perm = [300 - u for u in tot_unperm]
    tot_unperm_no_milk = [u - m for u, m in zip(tot_unperm, tot_unperm_milk)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(rel_weights))
    width = 0.6

    ax.bar(x_pos, tot_perm, width, label="Permitted", color=cfg.COLOR_PERMITTED, alpha=cfg.ALPHA_FILL)
    ax.bar(x_pos, tot_unperm_no_milk, width, bottom=tot_perm,
           label="Unpermitted (no milk license)", color=cfg.COLOR_UNPERMITTED, alpha=cfg.ALPHA_FILL)
    ax.bar(x_pos, tot_unperm_milk, width,
           bottom=[p + u for p, u in zip(tot_perm, tot_unperm_no_milk)],
           label="Unpermitted (with milk license)", color=cfg.COLOR_UNPERMITTED_MILK, alpha=cfg.ALPHA_MILK_LICENSE)

    ax.set_xlabel("Relative Weight of Size")
    ax.set_ylabel("Number of Farms in Top 300")
    #ax.set_title("Breakdown of Top 300 Farms Using Combined Size and Hand-Built Risk Index")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rel_weights)
    ax.grid(False)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.legend(frameon=True, fontsize=cfg.FIG_LEGEND_SIZE)
    ax.set_ylim(0, 300)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close("all")

    breakdown = pd.DataFrame({
        "size_weight": rel_weights,
        "n_permitted": tot_perm,
        "n_unpermitted": tot_unperm,
        "n_unpermitted_no_milk": tot_unperm_no_milk,
        "n_unpermitted_with_milk": tot_unperm_milk,
    })
    print("\n  Top-300 breakdown by AU-vs-risk weight:")
    print(breakdown.to_string(index=False))
    if save_table_path is not None:
        breakdown.to_csv(save_table_path, index=False)


def plot_risk_comparison_examples(all_clusters: gpd.GeoDataFrame,
                                  image_bound_map: gpd.GeoDataFrame,
                                  parcels: gpd.GeoDataFrame,
                                  WDNR_CAFOs: gpd.GeoDataFrame,
                                  save_dir=None,
                                  au_range: tuple = (1500, 2000),
                                  n_examples: int = 3):
    """Plot aerial imagery examples of the lowest-risk permitted facilities
    in a given AU range, for risk comparison case studies.
    Extracted from all_key_results.ipynb cells 122-124."""
    import plotting_utils as pu

    au_subset = all_clusters[
        (all_clusters["animal_unit_estimate"] >= au_range[0])
        & (all_clusters["animal_unit_estimate"] < au_range[1])
    ]

    permitted = au_subset[au_subset["permitted"] == True].sort_values(
        by="hand_built_risk_index"
    )
    if len(permitted) == 0:
        print(f"No permitted facilities in AU range {au_range}")
        return

    cols_to_print = [
        "set", "animal_unit_estimate", "hand_built_risk_index",
        "pca_risk_index", "matched_milk", "water_distance",
        "mean_slope", "gw_0", "bedrock_lt5ft_dummy",
    ]
    available_cols = [c for c in cols_to_print if c in all_clusters.columns]

    for i in range(min(n_examples, len(permitted))):
        idx = permitted.index[i]
        row = permitted.loc[idx]
        print(f"\nRisk example {i+1} (AU {au_range[0]}-{au_range[1]}):")
        print(row[available_cols])

        target_indices = row["polygon_indices"]
        cluster_to_plot = all_clusters[
            all_clusters["polygon_indices"].apply(lambda x: x == target_indices)
        ]
        fig = pu.plot_cluster_parcel(
            cluster_to_plot,
            image_bound_map=image_bound_map,
            permit_data=WDNR_CAFOs,
            parcel_data=parcels,
            annotate_parcels=False,
            zoom_radius=500,
            show_scale_bar=True,
        )
        if save_dir is not None:
            fig.savefig(
                Path(save_dir) / f"risk_example_{i+1}.png",
                bbox_inches="tight", dpi=150,
            )
        plt.close("all")


if __name__ == "__main__":
    # 0. Load paths from config file
    with open(Path().resolve().parent / "afo_vs_cafo/config/config.yml", "r") as file:
        configs = yaml.safe_load(file)
    analysis_output_path = Path(configs["analysis_output_path"])
    fig_path = Path(configs["fig_path"])
    data_path = Path(configs["data_path"])
    land_parcel_path = Path(configs["land_parcel_path"])
    model_prediction_path = Path(configs["model_prediction_path"])
    cluster_path = Path(configs["cluster_path"])
    # Set GCP credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = configs["gcp_cred_path"]
    # Prevent matplotlib from opening new windows
    matplotlib.use("Agg")

    # 1. Load and clean data
    print("Loading data...")
    
    image_bound_map, counties, parcels, milk_producers, WDNR_CAFOs, ewg_afos, cf_annotations_exc_EWG, all_cf_clusters, cf_clusters_EWG, cf_clusters_exc_EWG, inception_detections, poultry_detections, four_band_clusters, four_band_clusters_exc_EWG, three_band_clusters, three_band_clusters_exc_EWG, all_waters, impaired_waters, water_table_depth, snapmaps = load_data(
        analysis_output_path,
        data_path,
        land_parcel_path,
        cluster_path,
        model_prediction_path,
        read_snapmaps=False
    )

    ewg_cf_train_labels = cluster.load_polygons(
        data_path / "Annotations/ewg_region_train_labels.geojson",
        counties,
        image_bound_map
    )

    all_labeled_images = get_labeled_images(how="json", data_path=data_path)

    # Print useful stats about model sending/imagery
    ## Need to add something here about # sent to CF

    # 2. Create figures
    print("Creating figures...")
    plt.ioff()

    # Compare CF AU estimates to permit for all matched clusters
    permit_matched_cf_clusters, err = caf.merge_clusters_permits(
    all_cf_clusters, WDNR_CAFOs, sum_satellite_counts=True,
    discrep_analysis=True, only_dairy=True
)

    print(err)

    permit_CF_AU_error_plot(
        permit_matched_cf_clusters,
        fig_save_path=fig_path / "permit_CF_AU_error_plot.png"
        )

    # Plot histograms of permit-estimate discrepancies
    plot_permit_estimate_discrepancy_hist(
        permit_matched_cf_clusters,
        estimate_col="animal_unit_estimate",
        error_metric="raw",
        save_path=fig_path / "permit_estimate_discrepancy_hist_raw.png"
    )
    
    plot_permit_estimate_discrepancy_hist(
        permit_matched_cf_clusters,
        estimate_col="animal_unit_estimate",
        error_metric="pct",
        save_path=fig_path / "permit_estimate_discrepancy_hist_pct.png"
    )

    plotting_utils.plot_permit_v_estimate(
        permit_matched_cf_clusters, title="Comparison of AU estimates and permit-reported values",
        include_uncertainty=True,
        color_satellite_summed=False,
        fig_save_path=fig_path / "permit_v_estimate_all_clusters.png")

    # Compare CF cow number estimates to EWG size categories for all EWG matched
    EWG_cf_count_compare(
        cf_clusters_EWG, ewg_afos, save_path=fig_path / "ewg_cf_count_compare.png",
        estimate_logic="point"
    )

    # Facility-level precision recall tradeoff with AFO lower bound threshold
    facility_level_pr_lowerbound_graph(four_band_clusters, 
                                       'four_band', 
                                       WDNR_CAFOs, 
                                       counties,
                                       ewg_afos, 
                                       AFO_lower_bound_threshes=np.arange(0, 1000, 100), 
                                       AFO_lower_bound_logic='point estimate',
                                       save_path=fig_path / "facility_level_pr_lowerbound.png")

    # Todo here - show some examples of false positives / small facilities
    # Gas station example
    plotting_utils.plot_cluster_parcel(four_band_clusters[four_band_clusters['parcel_owner1_names'].str.contains('PINK ELEPHANT')],
                                        image_bound_map=image_bound_map, 
                                        zoom_radius=150, 
                                        include_AU_estimate=True,
                                        fig_path=fig_path / "small_false_pos_gas_station.png")
    

    plotting_utils.plot_cluster_parcel(four_band_clusters[four_band_clusters['parcel_owner1_names'].str.contains('BUI PROPERTIES LLC')],
                                        image_bound_map=image_bound_map, 
                                        zoom_radius=150, 
                                        include_AU_estimate=True,
                                        fig_path=fig_path / "small_false_pos_trucking.png")

    # Some examples of true positive permit large picked up by model
    

        # four_band_cluster = four_band_clusters[
    #     four_band_clusters["jpeg_names"].apply(
    #         lambda x: "WI_Brown_2_7168_12288.jpeg" in x
    #     )
    # ]
    
    # Comparison to baseline models - facility level precision/recall
    facility_level_stats_table(
        four_band_clusters,
        three_band_clusters,
        poultry_detections,
        inception_detections,
        WDNR_CAFOs,
        counties,
        ewg_afos,
        AFO_lower_bound=500,
        AFO_lower_bound_logic='point_estimate',
        save_path=fig_path / "model_performance_table.csv"
)
    
    pixel_level_p_r_IOU_curves(
        model_prediction_path=model_prediction_path,
        vec_thresh=[0.25, 0.4, 0.5,0.6, 0.75],
        CF_polygons=cf_annotations_exc_EWG,
        image_bound_map=image_bound_map,
        counties=counties,
        stats_path=analysis_output_path / "pixel_level_p_r_IOU_curves_only_annotated_areas.csv",
        include_sent_but_no_annot=False,
        recalculate_stats=True,
        save_path=fig_path / "pixel_level_p_r_IOU_curves_only_annotated_areas.png",
)
    
    

    # # ------------Grab example clusters for three vs four vs CF clusters ----------
    # CF_cluster = cf_clusters_exc_EWG[
    #     cf_clusters_exc_EWG["jpeg_names"].apply(
    #         lambda x: "WI_Brown_2_7168_12288.jpeg" in x
    #     )
    # ]
    # three_band_cluster = three_band_clusters[
    #     three_band_clusters["jpeg_names"].apply(
    #         lambda x: "WI_Brown_2_7168_12288.jpeg" in x
    #     )
    # ]
    # four_band_cluster = four_band_clusters[
    #     four_band_clusters["jpeg_names"].apply(
    #         lambda x: "WI_Brown_2_7168_12288.jpeg" in x
    #     )
    # ]

    # annotation_prediction_examples(
    #     CF_cluster,
    #     three_band_cluster,
    #     four_band_cluster,
    #     save_path=fig_path / "annotation_prediction_examples.png",
    # )

    # perm_unperm_examples(save_path=fig_path / "perm_unperm_examples.png")
    # cluster_method_compare(
    #     cf_annotations_exc_EWG,
    #     parcels,
    #     WDNR_CAFOs,
    #     fig_save_path=fig_path / "cluster_compare.png",
    #     table_save_path=fig_path / "cluster_compare_table.csv",
    # )
    # unperm_au_dist(
    #     four_band_clusters,
    #     WDNR_CAFOs,
    #     counties,
    #     save_path=fig_path / "unperm_au_dist.png",
    # )
    # baseline_pred_examples(
    #     inception_detections,
    #     poultry_detections,
    #     save_path=fig_path / "baseline_model_example_plots.png",
    # )
    # WI_CAFO_map(
    #     all_cf_clusters,
    #     WDNR_CAFOs,
    #     counties,
    #     save_path=fig_path / "wi_cafo_map.png",
    # )
    # scatter_plot_three_v_four_band_cluster(
    #     three_band_clusters_exc_EWG,
    #     four_band_clusters_exc_EWG,
    #     cf_clusters_exc_EWG,
    #     save_path=fig_path / "scatter_plot_three_v_four_band_cluster.png",
    # )

    # pixel_level_p_r_IOU_curves(
    #     model_prediction_path=model_prediction_path,
    #     vec_thresh=[0.25, 0.4, 0.5, 0.75],
    #     CF_polygons=cf_annotations_exc_EWG,
    #     image_bound_map=image_bound_map,
    #     counties=counties,
    #     stats_path=analysis_output_path / "pixel_level_p_r_IOU_curves.csv",
    #     include_sent_but_no_annot: False,
    #     recalculate_stats=True,
    #     save_path=fig_path / "pixel_level_p_r_IOU_curves.png",
    # )
    # cluster_level_precision_recall_hist_single_threshold(
    #     cf_clusters_exc_EWG,
    #     three_band_clusters_exc_EWG,
    #     four_band_clusters_exc_EWG,
    #     fig_save_path=fig_path / "cluster_p_r_single_threshold.png",
    # )
    # if "WI_slope.tif" not in os.listdir(data_path / "water_data" / "DEM_30m"):
    #     print("Calculating slope from DEM...")
    #     caf.calculate_slope_from_dem(
    #         data_path / "water_data" / "DEM_30m" / "demgw930",
    #         data_path / "water_data" / "DEM_30m",
    #     )

    # 3. Create tables
    # print("Creating tables...")
    # summary_table(
    #     all_cf_clusters,
    #     WDNR_CAFOs,
    #     four_band_clusters,
    #     counties,
    #     milk_producers,
    #     data_path / "water_data" / "DEM_30m" / "WI_slope.tif",
    #     all_waters,
    #     impaired_waters,
    #     water_table_depth,
    #     snapmaps_layers=snapmaps,
    #     CAFO_thresh_logic="point",
    #     save_path=fig_path / "summary_table_point_estimate.csv",
    #     size_plot_au_save_path=fig_path / "size_au_unperm_vs_perm.png",
    #     size_plot_m2_save_path=fig_path / "size_m2_unperm_vs_perm.png",
    # )

    # summary_table(
    #     all_cf_clusters,
    #     WDNR_CAFOs,
    #     four_band_clusters,
    #     counties,
    #     milk_producers,
    #     data_path / "water_data" / "DEM_30m" / "WI_slope.tif",
    #     all_waters,
    #     impaired_waters,
    #     water_table_depth,
    #     snapmaps_layers=snapmaps,
    #     CAFO_thresh_logic="lower",
    #     save_path=fig_path / "summary_table_lower_estimate.csv",
    #     size_plot_au_save_path=fig_path / "size_au_unperm_vs_perm.png",
    #     size_plot_m2_save_path=fig_path / "size_m2_unperm_vs_perm.png",
    # )

    # summary_table(
    #     all_cf_clusters,
    #     WDNR_CAFOs,
    #     four_band_clusters,
    #     counties,
    #     milk_producers,
    #     data_path / "water_data" / "DEM_30m" / "WI_slope.tif",
    #     all_waters,
    #     impaired_waters,
    #     water_table_depth,
    #     snapmaps_layers=snapmaps,
    #     CAFO_thresh_logic="upper",
    #     save_path=fig_path / "summary_table_upper_estimate.csv",
    #     size_plot_au_save_path=fig_path / "size_au_unperm_vs_perm.png",
    #     size_plot_m2_save_path=fig_path / "size_m2_unperm_vs_perm.png",
    # )

    # unperm_universe_table(
    #     all_cf_clusters,
    #     four_band_clusters,
    #     WDNR_CAFOs,
    #     counties,
    #     milk_producers,
    #     save_path=fig_path / "unperm_universe_table.csv",
    # )
    # facility_level_stats_table(
    #     four_band_clusters,
    #     three_band_clusters,
    #     poultry_detections,
    #     inception_detections,
    #     WDNR_CAFOs,
    #     counties,
    #     ewg_afos,
    #     save_path=fig_path / "model_performance_table.csv",
    # )

    plt.ion()
    print("Done")
