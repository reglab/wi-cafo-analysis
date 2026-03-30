# This script contains functions for creating a variety of useful plots for the project.

import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path
import naip_utils
import geopandas as gpd
import estimate_animal_units as est_au
import sys
import seaborn as sns

sys.path.append("..")
import config.config_params as cfg

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


# plotting functions
def plot_cluster_level_precision_recall_histograms(
    four_band_precision: list,
    four_band_recall: list,
    three_band_precision: list,
    three_band_recall: list,
    dpi: float = 300,
    display: bool = True,
    fig_save_path: str = None,
    fig_format: str = None,
):
    """Create histogram of recall and precision for three and four band models at a single threshold
    Args:
        four_band_precision (list): list of precision values for four band model
        four_band_recall (list): list of recall values for four band model
        three_band_precision (list): list of precision values for three band model
        three_band_recall (list): list of recall values for three band model

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(four_band_precision, label="Four band", histtype="step")
    axes[0].hist(three_band_precision, label="Three band", histtype="step")
    axes[0].set_title("Precision")
    axes[0].legend()

    axes[1].hist(four_band_recall, label="Four band", histtype="step")
    axes[1].hist(three_band_recall, label="Three band", histtype="step")
    axes[1].set_title("Recall")
    axes[1].legend()
    plt.suptitle("Cluster Level Precision and Recall")

    plt.tight_layout()
    if display:
        plt.show()
    if fig_save_path is not None:
        if fig_format is None:
            fig_format = cfg.FIG_EXPORT_FORMAT
        fig.savefig(ensure_export_format(fig_save_path, fig_format), dpi=dpi, format=fig_format)


def scatter_plot_CF_vs_prediction_clusters(
    corr_test_three_band: pd.DataFrame,
    corr_test_four_band: pd.DataFrame,
    fig_save_path: str = None,
    title0: str = "Three Band Model",
    title1: str = "Four Band Model",
    suptitle: str = "Cluster-Level Area Comparison",
    ylims: tuple = (-2000, 55000),
    xlims: tuple = (-2000, 55000),
    figsize: tuple = (16, 6),
    x0label: str = "CF area prediction (m²)",
    y0label: str = "Model area prediction (m²)",
    x1label: str = "CF area prediction (m²)",
    y1label: str = "Model area prediction (m²)",
    dpi: float = 300,
    display: bool = True,
    fig_format: str = None,
):
    """Plot the cluster level CF area vs the model predicted area
    Input:
        corr_test_three_band: sjoined prediction and CF clusters in a dataframe
        corr_test_four_band:  sjoined prediction and CF clusters in a dataframe
        fig_save_path: optional, path where to save figure
        title0: title for the first plot
        title1: title for the second plot
        suptitle: title for the whole figure
        ylims: limits of y-axis
        xlims: limits of x-axis
        figsize: figure size, in inches
        x0label: label for x-axis of first plot
        y0label: label for y-axis of first plot
        x1label: label for x-axis of second plot
        y1label: label for y-axis of second plot
        dpi: what dpi to use if saving figure
        display: whether or not to display figure after running
    Returns: A plot of the cluster level CF area vs the model predicted area
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].scatter(
        corr_test_three_band["CF_cluster_area"],
        corr_test_three_band["model_cluster_area"],
        color="darkgreen",
        alpha=0.5,
        s=5,
    )
    ax[0].axline(
        (0, 0),
        slope=1,
        linestyle="--",
        label="model estimate = human annotation estimate",
    )
    ax[0].axhline(y=7300, linestyle="--", color="red")
    ax[0].axvline(x=7300, linestyle="--", color="red")

    r2 = r2_score(
        corr_test_three_band["CF_cluster_area"],
        corr_test_three_band["model_cluster_area"],
    )
    # ax[0].text(8000, 35000, f"r2 = {r2:.3f}")
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_xlabel(x0label)
    ax[0].set_ylabel(y0label)
    ax[0].set_title(title0)

    ax[1].scatter(
        corr_test_four_band["CF_cluster_area"],
        corr_test_four_band["model_cluster_area"],
        color="blue",
        alpha=0.5,
        s=5,
    )
    ax[1].axline(
        (0, 0),
        slope=1,
        linestyle="--",
        label="model estimate = human annotation estimate",
    )
    ax[1].axhline(y=7300, linestyle="--", color="red")
    ax[1].axvline(x=7300, linestyle="--", color="red")
    r2 = r2_score(
        corr_test_four_band["CF_cluster_area"],
        corr_test_four_band["model_cluster_area"],
    )

    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].set_xlabel(x1label)
    ax[1].set_ylabel(y1label)
    ax[1].set_title(title1)
    plt.legend()
    plt.suptitle(suptitle)
    plt.tight_layout()
    if display:  # display the plot
        plt.show()
    if fig_save_path is not None:  # save the plot
        if fig_format is None:
            fig_format = cfg.FIG_EXPORT_FORMAT
        print("saving figure", ensure_export_format(fig_save_path, fig_format))
        fig.savefig(ensure_export_format(fig_save_path, fig_format), dpi=dpi, format=fig_format)
    return fig


def plot_pixel_level_p_r_iou(
    p_r_IOU: pd.DataFrame,
    suptitle: str = "Pixel Level Precision, Recall, and IOU",
    ylims: tuple = (0, 1),
    xlims: tuple = (0, 1),
    figsize: tuple = (10, 5),
    x0label: str = "Recall",
    y0label: str = "Precision",
    title0: str = "Precision-Recall Curve",
    x1label: str = "Confidence Threshold",
    y1label: str = "IOU Score",
    title1: str = "IOU Curve",
    dpi: float = 300,
    display: bool = True,
    fig_save_path: str = None,
    suppress_legend: bool = False,
    fig_format: str = None,
):
    """Plot the pixel level precision, recall and IOU for three and four band inference results
    TODO: ***This plot will have to be updated when we incorporate the three band model inference results***
    Input:
        p_r_IOU: dataframe with pixel level precision, recall and IOU for three and four band inference results
        suptitle: title for the whole figure
        ylims: limits of y-axis
        xlims: limits of x-axis
        figsize: figure size, in inches
        x0label: label for x-axis of first plot
        y0label: label for y-axis of first plot
        title0: title for the first plot
        x1label: label for x-axis of second plot
        y1label: label for y-axis of second plot
        title1: title for the second plot
        dpi: what dpi to use if saving figure
        display: whether or not to display figure after running
        fig_save_path: optional, path where to save figure
    Returns: A plot of the pixel level precision, recall and IOU for three and four band inference results
    """
    # n_thresh = len(p_r_IOU["Threshold"].unique())
    # cmap = ListedColormap(plt.get_cmap("viridis", n_thresh)(range(n_thresh)))

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    sns.lineplot(
        data=p_r_IOU, x="Recall", y="Precision", hue="model", alpha=0.4, ax=ax[0]
    )
    ax[0].set_ylim(ylims)
    ax[0].set_xlim(xlims)
    ax[0].set_xlabel(x0label)
    ax[0].set_ylabel(y0label)
    ax[0].set_title(title0)
    # suppress legend if not needed
    if suppress_legend and ax[0].get_legend() is not None:
        ax[0].get_legend().remove()

    # Plot the IOU curve
    sns.lineplot(data=p_r_IOU, x="Threshold", y="IOU", hue="model", alpha=0.4, ax=ax[1])
    
    #suppress legend if not needed
    if suppress_legend and ax[1].get_legend() is not None:
        ax[1].get_legend().remove()
    
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel(x1label)
    ax[1].set_ylabel(y1label)
    ax[1].set_title(title1)

    # Plot the precision-recall curve
    sns.scatterplot(
        data=p_r_IOU[p_r_IOU["Threshold"] == 0.5],
        x="Recall",
        y="Precision",
        hue="model",
        legend=False,
        ax=ax[0],
    )

    # # Add a color key for the threshold
    # threshold_legend = [
    #     mpatches.Patch(
    #         color=cmap(threshold),
    #         label=f"{threshold:.2f}",
    #     )
    #     for threshold in p_r_IOU["Threshold"]
    # ]

    if not suppress_legend:
        ax[0].legend(
            # handles=threshold_legend, title="Confidence Threshold", loc="lower left"
        )
    plt.suptitle(suptitle)
    plt.tight_layout()
    if display:  # display the plot
        plt.show()
    if fig_save_path is not None:
        if fig_format is None:
            fig_format = cfg.FIG_EXPORT_FORMAT
        fig.savefig(ensure_export_format(fig_save_path, fig_format), dpi=dpi, format=fig_format)
    return fig


def plot_est_v_permit_AU_v_area(test, description):
    """Plot the area vs AU estimates from the permits and out clustering method. Print the slope and the intercepts."""

    test = test.loc[(test["PERMIT"] == True) & (test["PROPOSED_AU"] > 0)]

    areas = list(test["cluster_area_m2"].values)
    proposed_au = list(test["PROPOSED_AU"].values)
    estimated_au = list(test["animal_unit_estimate"].values)

    m_permit, b_permit = np.polyfit(areas, proposed_au, 1)
    m_est, b_est = np.polyfit(areas, estimated_au, 1)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(
        areas,
        proposed_au,
        label="Permit Proposed AU",
    )
    plt.scatter(
        areas,
        estimated_au,
        label="Our Estimated AU",
    )
    plt.plot(
        np.array(areas),
        np.array(areas) * m_permit + b_permit,
        label=f"Permit slope {np.round(m_permit,2)}, intecept {np.round(b_permit,2)}",
    )

    plt.plot(
        np.array(areas),
        np.array(areas) * m_est + b_est,
        label=f"Estimate slope {np.round(m_est, 2)}, intecept {np.round(b_est,2)}",
    )
    plt.legend()
    plt.title(f"Area vs AU count for {description}")
    plt.xlabel("Area - square m")
    plt.ylabel("Animal Units")
    plt.show()


def plot_permit_v_estimate(
    matched_clusters: gpd.GeoDataFrame,
    discrep_data: pd.DataFrame = None,
    include_uncertainty: bool = False,
    color_satellite_summed: bool = False,
    fig_save_path: str = None,
    title: str = "",
    ylims: tuple = (0, 20000),
    xlims: tuple = (0, 20000),
    figsize: tuple = (8, 6),
    xlabel: str = "Permit-reported animal units",
    ylabel: str = "Animal unit estimate",
    include_legend: bool = True,
    fig: plt.figure = None,
    ax: plt.axes = None,
    dpi: float = 300,
    display: bool = True,
    fig_format: str = None,
):
    """
    Input:
        matched_clusters: clusters merged with WDNR CAFO permits (first output of cluster_analysis_functions.merge_clusters_permits(..., discrep_analysis=True))
        discrep_data: Optional, animal unit discrepancy results between clusters and WDNR CAFO permits
             (second output of cluster_analysis_functions.merge_clusters_permits(..., discrep_analysis=True)). If supplied, summary stats will be included in
             title.
        include_uncertainty: bool, whether to include uncertainty (lower and upper 95% ci bounds). Default False.
        fig_save_path: optional, path where to save figure
        title: optional, title for the plot
        ylims: limits of y-axis
        xlims: limits of x-axis
        figsize: figure size, in inches
        xlabel: label for x-axis
        ylabel: label for y-axis
        fig: Optional, existing figure to add to.
        ax: Optional, existing figure axis to plot on.
        dpi: what dpi to use if saving figure
        display: whether or not to display figure after running
    Returns: A plot of the permit animal count vs estimated animal count from our analysis
    """
    clusters_to_plot = matched_clusters.dropna(
        subset=["Number ofAnimalUnits", "animal_unit_estimate"]
    )

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if color_satellite_summed:
        # Plot points with different colors based on 'has_satellite_sites'
        has_satellite = clusters_to_plot['has_satellite_sites'] == True
        ax.plot(
            clusters_to_plot.loc[~has_satellite, "Number ofAnimalUnits"],
            clusters_to_plot.loc[~has_satellite, "animal_unit_estimate"],
            ".",
            label="No satellite sites"
        )
        ax.plot(
            clusters_to_plot.loc[has_satellite, "Number ofAnimalUnits"],
            clusters_to_plot.loc[has_satellite, "animal_unit_estimate"],
            ".",
            label="Has satellite sites",
            color="red"
        )
    else:
        ax.plot(
            clusters_to_plot["Number ofAnimalUnits"],
            clusters_to_plot["animal_unit_estimate"],
            "."
        )

    if include_uncertainty:
        ax.vlines(
            clusters_to_plot["Number ofAnimalUnits"],
            clusters_to_plot["animal_units_lower"],
            clusters_to_plot["animal_units_upper"],
            linewidth=1,
            alpha=0.5,
        )
    ax.axline((0, 0), slope=1, linestyle="--", color="black", label="estimate=permit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if discrep_data is not None:
        mean_discrep = discrep_data["mean_discrep"].iloc[0]
        mean_abs_discrep = discrep_data["mean_abs_discrep"].iloc[0]
        title = f"{title} \n Mean discrepancy {np.round(mean_discrep,2)}, Mean absolute discrepancy: {np.round(mean_abs_discrep, 2)} \n n={len(matched_clusters)}"

    ax.set_title(title)

    ax.set_ylim(ylims[0], ylims[1])
    ax.set_xlim(xlims[0], xlims[1])
    ax.grid(False)
    if include_legend:
        plt.legend()

    # Calculate and display R-squared
    corr_coef = np.corrcoef(clusters_to_plot["Number ofAnimalUnits"], clusters_to_plot["animal_unit_estimate"])[0, 1]
    ax.text(0.95, 0.05, f"Correlation coefficient = {corr_coef:.2f}", transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    if fig_save_path is not None:
        if fig_format is None:
            fig_format = cfg.FIG_EXPORT_FORMAT
        plt.savefig(ensure_export_format(fig_save_path, fig_format), dpi=dpi, format=fig_format)

    if display:
        plt.show()

    return


def plot_cluster_parcel(
    cluster_data: gpd.GeoDataFrame,
    parcel_data: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    permit_data: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    image_bound_map: gpd.GeoDataFrame = None,
    four_band_image_bound_map: gpd.GeoDataFrame = None,
    title_desc: str = "",
    zoom_radius: float = 500,
    include_AU_estimate: bool = False,
    annotate_parcels: bool = True,
    imagery_bands: list = [1, 2, 3],
    cluster_color: str = "red",
    cluster_fill: bool = False,
    figsize: tuple = (6, 6),
    display: bool = True,
    fig: plt.figure = None,
    ax: plt.axes = None,
    fig_path: str = None,
    dpi: float = 300,
    fig_format: str = None,
    show_scale_bar: bool = False,
    show_cluster_outline: bool = True,
):
    """
    Input:
        cluster_data: cluster data
        parcel_data: parcel data
        permit_data: WDNR permit data
        image_bound_map: GeoDataFrame of the boundaries of the small NAIP tiles. Only
            required if plotting four band imagery.
        four_band_image_bound_map: GeoDataFrame of the boundaries of the large four band NAIP tiffs.
            Only required if plotting four band imagery.
        title_desc: title/description of the data
        zoom_radius: radius around cluster centroid to plot, in meters
        include_AU_estimate: whether or not to include our animal unit estimate in the plot header.
            Only works if cluster_data contains a single cluster.
        annotate_parcels: whether or not to label parcels with their owner names
        imagery_bands: which bands to use for the NAIP imagery. Default [1, 2, 3].
        cluster_color: what color the cluster outline should be. Default "red".
        figsize: figure size in inches
        display: whether or not to display the figure. Default True.
        fig_path: path in which to save the figure. If None, figure will not
            be saved. Default None.
        fig: Optional, existing figure to add to.
        ax: Optional, existing figure axis to plot on.
        dpi: resolution of figure to be saved, if fig_path is not None.
        fig_format: format for saving figure (e.g., 'png', 'svg'). If None, uses cfg.FIG_EXPORT_FORMAT.
        show_scale_bar: whether or not to display a scale bar on the plot. Scale bar size is
            automatically calculated based on zoom_radius. Default False.
        show_cluster_outline: whether or not to display the cluster boundary outline. Default True.
            When True, clusters are plotted with red outline by default (or cluster_color if specified).

    Returns:
    Plots of shapes in the clusters, optionally along with our animal estimate
    """
    if len(imagery_bands) == 2:
        print("Warning: visualizing two band imagery is not currently supported.")
        return
    cluster_data = cluster_data.reset_index(drop=True)

    # Collect neighboring NAIP tiles
    jpeg_names = naip_utils.collect_neighboring_tiles(cluster_data.loc[0]["jpeg_names"])

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Optional, label plot with AU estimate
    if include_AU_estimate:
        if "animal_unit_estimate" not in cluster_data.columns:
            cluster_data = est_au.sample_and_calc_au(cluster_data)
        if cluster_data.shape[0] == 1:
            au_estimate = cluster_data.iloc[0]["animal_unit_estimate"]
            title = f"AU estimate: {round(au_estimate, 2)}"
        else:
            print(
                "Warning: unable to include animal unit estimate when multiple clusters \
                  are plotted"
            )
        if ("Number ofAnimalUnits" in cluster_data.keys()) and (
            "PermitteeName" in cluster_data.keys()
        ):
            reported_au = cluster_data.iloc[0]["Number ofAnimalUnits"]
            discrep = au_estimate - reported_au
            permit_name = cluster_data.iloc[0]["PermitteeName"]
            plt.title(
                f"Our estimate for AU: {np.round(au_estimate,2)[0]}, discrep: {np.round(discrep, 2)[0]} \n permit name: {permit_name[0]}",
                fontsize=10,
            )
        else:
            plt.title(title)
    bounds = None
    if 4 in imagery_bands:
        if image_bound_map is None or four_band_image_bound_map is None:
            print("Image bound maps must be supplied to plot four band imagery.")
            return
        # Calculate the boundaries of the relevant small tiles with which to clip the large NAIP tile
        boundary = image_bound_map[
            image_bound_map["filename"].isin(jpeg_names)
        ].dissolve()
        # Note: currently will only pull the first four-band image in the county of the first small tile
        # that intersects the small tiles, as I haven't been able to implement multi-large tiff plotting yet.
        # This may lead to blank areas in some plots.
        county = jpeg_names[0][3 : jpeg_names[0].find("_", 3)]
        four_band_image = (
            four_band_image_bound_map[
                four_band_image_bound_map["filename"].apply(lambda x: county in x)
            ]
            .sjoin(boundary)
            .reset_index()["filename_left"]
            .iloc[0]
        )
        jpeg_names = [four_band_image]
        bounds = boundary.geometry
    naip_image, transform = naip_utils.download_transform(
        jpeg_names, bands=imagery_bands, bounds=bounds
    )
    rasterio.plot.show(naip_image, transform=transform, ax=ax)
    if parcel_data.shape[0] > 0:
        # make sure that the parcel_index is of type int
        parcel_inds = list(
            set(
                [
                    item
                    for sublist in list(cluster_data["parcel_indices"])
                    for item in sublist
                ]
            )
        )
        parcel_inds = list(map(int, parcel_inds))
        parcel_to_plot = parcel_data.loc[parcel_inds]

        if annotate_parcels:
            # add in annotations
            parcel_to_plot.apply(
                lambda x: ax.annotate(
                    text=x["OWNERNME1"],
                    xy=x["geometry"].centroid.coords[0],
                    ha="center",
                    fontsize=10,
                    color="blue",
                ),
                axis=1,
            )
        parcel_to_plot.boundary.plot(ax=ax, color="blue")

    # Plot cluster boundaries with red outline by default
    if show_cluster_outline:
        cluster_data.reset_index(inplace=True)
        if cluster_data.shape[0] > 1:
            # Multiple clusters: use red outline for all
            cluster_data.boundary.plot(ax=ax, edgecolor=cluster_color, linewidth=1.5)
            if cluster_fill:
                cluster_data.plot(ax=ax, alpha=0.2, color=cluster_color)
        else:
            # Single cluster: use red outline
            cluster_data.boundary.plot(ax=ax, edgecolor=cluster_color, linewidth=1.5)
            if cluster_fill:
                cluster_data.plot(ax=ax, alpha=0.2, color=cluster_color)
    if permit_data.shape[0] > 0:
        permit_data.plot(ax=ax, markersize=20, color="orange")

    # Set axis limits
    ax.set_xlim(
        [
            cluster_data.dissolve().centroid.loc[0].x - zoom_radius,
            cluster_data.dissolve().centroid.loc[0].x + zoom_radius,
        ]
    )
    ax.set_ylim(
        [
            cluster_data.dissolve().centroid.loc[0].y - zoom_radius,
            cluster_data.dissolve().centroid.loc[0].y + zoom_radius,
        ]
    )

    # Add scale bar if requested
    if show_scale_bar:
        # Calculate an appropriate scale bar size based on zoom_radius
        # Use approximately 10-20% of the zoom radius, rounded to a nice number
        scale_bar_size = zoom_radius * 0.15
        
        # Round to a nice number (50, 100, 200, 500, 1000, etc.)
        if scale_bar_size < 50:
            scale_bar_size = 50
        elif scale_bar_size < 100:
            scale_bar_size = 100
        elif scale_bar_size < 200:
            scale_bar_size = 200
        elif scale_bar_size < 500:
            scale_bar_size = 500
        elif scale_bar_size < 1000:
            scale_bar_size = 1000
        else:
            scale_bar_size = int(scale_bar_size / 500) * 500
        
        # Format the label
        if scale_bar_size >= 1000:
            scale_bar_label = f"{int(scale_bar_size / 1000)} km"
        else:
            scale_bar_label = f"{int(scale_bar_size)} m"
        
        # Create and add scale bar
        fontprops = fm.FontProperties(size=15)
        scalebar = AnchoredSizeBar(
            ax.transData,
            scale_bar_size,
            scale_bar_label,
            "lower left",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=2,
            fontproperties=fontprops,
        )
        ax.add_artist(scalebar)

    ax.set_axis_off()
    if include_AU_estimate:
        ax.set_title(title)
    else:
        ax.set_title(f"{title_desc}")
    plt.tight_layout()

    if fig_path is not None:
        if fig_format is None:
            fig_format = cfg.FIG_EXPORT_FORMAT
        plt.savefig(ensure_export_format(fig_path, fig_format), dpi=dpi, format=fig_format)

    if display:
        plt.show()

    return


import folium

# Here are some different styles for the map
blue = lambda x: {
    "color": "blue",
    "opacity": 0.80,
    "weight": "1",
}

red = lambda x: {
    "color": "red",
    "opacity": 0.940,
    "weight": "3",
}

green = lambda x: {
    "color": "green",
    "opacity": 0.80,
    "weight": "2",
}
pink = lambda x: {
    "color": "pink",
    "opacity": 0.8,
    "weight": "3",
}

yellow = lambda x: {
    "color": "yellow",
    "opacity": 1,
    "weight": "3",
}
basemaps = {
    "Google Maps": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Maps",
        overlay=True,
        control=True,
    ),
    "Google Satellite": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Google Terrain": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Terrain",
        overlay=True,
        control=True,
    ),
    "Google Satellite Hybrid": folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        overlay=True,
        control=True,
    ),
    "Esri Satellite": folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=True,
        control=True,
    ),
}


def simple_map(object1, object2=None, object3=None, object4=None):
    """A quick function to generate a folium map for any col from a dataframe. This is helpful for a lower resolution map of all of the predictions in the state."""
    # Create a folium map object
    GEO_CRS = "EPSG:4326"
    object1 = object1.to_crs(GEO_CRS)
    map_data = folium.Map(
        location=[
            object1.geometry.centroid.y.mean(),
            object1.geometry.centroid.x.mean(),
        ],
        zoom_start=17,
    )
    basemaps["Google Satellite"].add_to(map_data)
    # Iterate over the missed permits GeoData1Frame
    # Add the GeoDataFrame as a GeoJSON layer
    if object1 is not None:
        folium.GeoJson(object1, style_function=red).add_to(map_data)
    if object2 is not None:
        object2 = object2.to_crs(GEO_CRS)
        folium.GeoJson(object2, style_function=yellow).add_to(map_data)
    if object3 is not None:
        object3 = object3.to_crs(GEO_CRS)
        folium.GeoJson(object3, style_function=green).add_to(map_data)
    if object4 is not None:
        object4 = object4.to_crs(GEO_CRS)
        folium.GeoJson(object4, style_function=pink).add_to(map_data)
    return map_data
