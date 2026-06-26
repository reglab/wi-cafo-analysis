# This script contains functions to estimate animal units (with uncertainty)
# from area estimates of polygon clusters.
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import config.config_params as cfg
import geopandas as gpd
import pandas as pd
import sys
import cluster
from tqdm import tqdm
sys.path.append("../")
import analyze_model_outputs as amo
import os
from pathlib import Path
import yaml





def prep_permit_data(data_path, use_only_dairy_permits=True):
    """Function to load the permit data and compute the AU for each animal type and the percentage of the facility's total AU that
      each type of cattle represents
    Args:
        data_path: str, path to the permit data
    Returns:
        permit_data: pd.DataFrame, the permit data with columns for the AU for each animal type and the percentage of the facility's
         total AU that each type of cattle represents
        use_only_dairy_permits: bool, whether to use/keep only dairy facilities
    """
    # Load the permit data
    permit_data = pd.read_csv(data_path)

    if use_only_dairy_permits:
        permit_data = permit_data[permit_data['AnimalType'] == 'Dairy']

    # Convert the DataFrame to numeric, handling non-numeric entries
    permit_data = permit_data.apply(pd.to_numeric, errors="coerce")
    permit_data = permit_data.replace(np.nan, 0)

    # Define the cattle types and their corresponding column names and AU factors
    cattle_types = {
        "calves": "Calves",
        "milking_cows": "Milking & Dry Cows",
        "heifers_800_1200": "Heifers (800-1200lbs)",
        "heifers_400_800": "Heifers (400-800lbs)",
        "beef_cattle": "Beef cattle",
    }

    # Compute (1) the AU for each animal type and (2) the percentage of the facility's total AU that each type of cattle represents (3) the percentage of the facility's total cattle that each type of cattle represents (head count)
    for cattle, column in cattle_types.items():
        au_column = f"{cattle}_au"
        # convert from animal count to animal type au
        animal_au = permit_data[column] * cfg.au_factors[cattle].values[0]
        permit_data[au_column] = animal_au
        permit_data[f"{cattle}_count"] = permit_data[column]

    # find the total cattle and total au
    permit_data["total_cattle"] = permit_data[
        [f"{cattle}_count" for cattle, column in cattle_types.items()]
    ].sum(axis=1)

    permit_data["total_au"] = permit_data[
        [f"{cattle}_au" for cattle, column in cattle_types.items()]
    ].sum(axis=1)

    # find the percents
    for cattle, column in cattle_types.items():
        au_column = f"{cattle}_au"
        permit_data[f"{cattle}_au_percent"] = (
            permit_data[f"{cattle}_au"] / permit_data["total_au"]
        )
        permit_data[f"{cattle}_count_percent"] = (
            permit_data[f"{cattle}_count"] / permit_data["total_cattle"]
        )

    return permit_data

def prep_model_error_data(cluster_path = Path(cfg.cluster_path),
                          annotated_ims_path = Path(cfg.data_path + "/data/Annotations/all_labeled_images.csv"),
                          size_bins = cfg.size_bins,
                          model_type = "four_band",
                          how='left'):
    all_cf_clusters = cluster.load_clusters(cluster_path / "all_cf_clusters.csv")
    three_band_clusters = cluster.load_clusters(
            cluster_path / "three_band_clusters.csv"
        )
    four_band_clusters = cluster.load_clusters(cluster_path / "four_band_clusters.csv")

    # Left join the model predictions to the human annotations
    matched_three_band, matched_four_band = amo.match_predictions_to_labels(three_band_clusters, four_band_clusters, all_cf_clusters, how=how)

    if model_type == "three_band":
        matched_cf_to_label = matched_three_band.copy()
    elif model_type == "four_band":
        matched_cf_to_label = matched_four_band.copy()

    if how == 'left':
        # Since left join, there may be model predictions that didn't match at all
        matched_cf_to_label['matched_to_CF'] = matched_cf_to_label['CF_cluster_area'].notna()
        matched_cf_to_label['CF_cluster_area'].fillna(0, inplace=True)

        # Drop cases where l = 0 beacuse image wasn't sent to CloudFactory. We want to only capture
        # cases in this distribution where something was sent to cloudfactory
        all_labeled_images = pd.read_csv(annotated_ims_path)

        matched_cf_to_label['jpeg_names_tuple'] = matched_cf_to_label['jpeg_names_left'].apply(lambda x: tuple(x))
        matched_cf_to_label['any_images_sent'] = matched_cf_to_label['jpeg_names_tuple'].apply(lambda x: any(item in all_labeled_images['Image name'].tolist() for item in x))

        # Keep only cases where either we foudn a matching CF annotation, or if we didn't find matching CF annotation, we at least sent an image for that facilitiy
        matched_cf_to_label = matched_cf_to_label[(matched_cf_to_label['matched_to_CF'] == True) | (matched_cf_to_label['any_images_sent'] == True)]


    matched_cf_to_label["l/m"] = (
    matched_cf_to_label["CF_cluster_area"]
    / matched_cf_to_label["model_cluster_area"]
    )

  
    matched_cf_to_label["size_category"] = pd.cut(matched_cf_to_label["model_cluster_area"], bins=size_bins)

    # Convert intervals to strings for plotting
    matched_cf_to_label['size_category_str'] = matched_cf_to_label['size_category'].astype(str)

    return matched_cf_to_label

def truncnorm_rvs(min_val, max_val, mean, std, size):
    """Truncated normal random variable generator
    Args:
        min_val: float, minimum value of the distribution
        max_val: float, maximum value of the distribution
        mean: float, mean of the distribution
        std: float, standard deviation of the distribution
        size: int, number of samples to generate
    Returns:
        np.array, samples from the truncated normal distribution
    """
    a, b = (min_val - mean) / std, (max_val - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def sample_and_calc_au(
    cluster_data,
    sample_size=1000,
    permit_data=None,
    use_only_dairy_permits=True,
    truncnorm_params=cfg.truncnorm_params,
    au_factors=cfg.au_factors,
    include_area_uncertainty=False,
    model_type='four_band',
    model_error_join='left',
    size_bins=cfg.size_bins,
    conf_interval=0.95,
    seed=293847,
    optional_other_perc_threshes=[0.1, 0.25, 0.5, 0.75, 0.9]
):
    """
    This is a function to calculate the AU for each facility in the cluster data and provide uncertainty on the estimates (95% CI). For each facility, estimates are based on:
    - the composition of animal types at a facility, informed from the labeled permits that are randomly sampled with replacement.
    - the spatial requirements of each animal type, sampled from a truncated normal distribution with parameters informed by the industry standards.
    Args:
        cluster_data: pd.DataFrame, the cluster data
        permit_data: pd.DataFrame, the permit data we collected by hand with animal age distirbutions
        use_only_dairy_permits: bool, whether or not to use only dairy permits in the distribution of animal ages
        sample_size: int, the number of samples to take
        truncnorm_params: dict, a dictionary mapping each animal type to its corresponding truncated normal distribution parameters
        au_factors: pd.DataFrame, the AU factors for each animal type
        include_area_uncertainty: bool, whether or not to include model_error uncertainty in the upper and lower confidene intervals.
        model type: which model's errors to use for the uncertainty/model error analysis
        model-error_join: left or inner join between model predictions and human annotations
        size bins: how to bin up the model error distribution
        conf_interval: float, the confidence interval to use for the uncertainty (i.e., retriieving lower and upper bounds for AU estimate)
        
    Returns:
        cluster_data: pd.DataFrame, the cluster data with columns for the mean and 95% confidence intervals
    """
    animal_names = [
        "calves",
        "milking_cows",
        "heifers_800_1200",
        "heifers_400_800",
        "beef_cattle",
    ]
    # Create a dictionary mapping each animal type to a function that samples from the truncated normal distribution
    equation_dict = {
        key: lambda size, params=params: truncnorm_rvs(**params, size=size)
        for key, params in cfg.truncnorm_params.items()
    }
    if permit_data is None:
        permit_data = prep_permit_data(cfg.animal_count_permit_data_path, use_only_dairy_permits)
        print(f"Permit data loaded, length={len(permit_data)}")
        # check that there are no missing values or 0 values
        if (permit_data['Total cattle'].isna().any()) or ((permit_data['Total cattle'] == 0).any()):
            print(f"There are observations with missing or 0 values in the permit data")
            # Dropping these observations from permit data
            permit_data = permit_data.dropna(subset=['Total cattle'])
            permit_data = permit_data[permit_data['Total cattle'] != 0]
            print(f"Permit data cleaned, length={len(permit_data)}")

    # Add columns for the mean and std of sampled density in the cluster DataFrame
    cluster_data = cluster_data.reset_index(drop=True)
    permit_data = permit_data.reset_index(drop=True)
    cluster_data["au_per_animal_mean"] = 0
    cluster_data["au_per_animal_std"] = 0
    cluster_data["area_per_animal_mean"] = 0
    cluster_data["area_per_animal_std"] = 0
    cluster_data["std_sampled_au"] = 0
    cluster_data["Dairy_count_estimate"] = 0
    cluster_data["Dairy_count_estimate_lower"] = 0
    cluster_data["Dairy_count_estimate_upper"] = 0
    
    # Get error data
    error_data = prep_model_error_data(cluster_path = Path(cfg.cluster_path),   
                          annotated_ims_path = Path(cfg.data_path + "/annotations/all_labeled_images.csv"),
                          size_bins = size_bins,
                          model_type = model_type,
                          how=model_error_join)

    np.random.seed(seed)

    # Create a facility area cut, to match to the corresponding conditional error distribution
    cluster_data["size_category"] = pd.cut(cluster_data["cluster_area_m2"], bins=size_bins)

    # Loop through each facility
    for facility_i in tqdm(cluster_data.index):
        # Sample n permits with replacement (seed is set once before the loop;
        # do NOT reset inside the loop so each facility gets independent draws)
        sampled_permit = permit_data.sample(n=sample_size, replace=True)
        facility_area = cluster_data.iloc[facility_i]["cluster_area_m2"]

        # Add uncertainty to the facility area for the model predictions
        if include_area_uncertainty:
            matched_model_data = error_data[error_data["size_category"] == cluster_data["size_category"].iloc[facility_i]]

            # Reading in human annotated area / model area ratio
            sampled_error_factors = np.array(matched_model_data['l/m'].sample(sample_size, replace=True))
            facility_area = facility_area * sampled_error_factors
            assert len(facility_area[facility_area < 0]) == 0, "Error: negative area sampled"

        # Sample the spatial requirements for each animal type and calculate the weighted AU/m^2.
        for animal_name in animal_names:
            # Generate samples for the density (M^2 / animal)*percent of total cattle from the truncated normal
            sampled_permit[f"{animal_name}_sampled_m2_per_animal"] = sampled_permit[
                f"{animal_name}_count_percent"
            ] * equation_dict[animal_name](sample_size, truncnorm_params[animal_name])

            # Weight AU by the percentage of the animal type (AU/animal)*percent of total cattle
            sampled_permit[f"{animal_name}_weighted_animal_unit_per_animal"] = (
                au_factors[animal_name].values[0]
                * sampled_permit[f"{animal_name}_count_percent"]
            )

        # Find the weighted and sampled area density
        sampled_permit["a_s_bar"] = sampled_permit[
            [f"{animal_name}_sampled_m2_per_animal" for animal_name in animal_names]
        ].sum(axis=1)

        # Find the weighted and sampled AU density
        sampled_permit["a_u_bar"] = sampled_permit[
            [
                f"{animal_name}_weighted_animal_unit_per_animal"
                for animal_name in animal_names
            ]
        ].sum(axis=1)

        # store the mean and std area and au per animal for debugging purposes
        cluster_data.at[facility_i, "au_per_animal_mean"] = sampled_permit[
            "a_u_bar"
        ].mean()
        cluster_data.at[facility_i, "au_per_animal_std"] = sampled_permit[
            "a_u_bar"
        ].std()
        cluster_data.at[facility_i, "area_per_animal_mean"] = sampled_permit[
            "a_s_bar"
        ].mean()
        cluster_data.at[facility_i, "area_per_animal_std"] = sampled_permit[
            "a_s_bar"
        ].std()

        # final calculation AU
  

        facility_total_au_est = (
            facility_area * sampled_permit["a_u_bar"] / sampled_permit["a_s_bar"]
        )

        if sampled_permit["a_s_bar"].isna().any() or (sampled_permit["a_s_bar"] == 0).any():
            raise ValueError("There are missing or 0 values in the weighted animal space")

        if facility_total_au_est.isna().any():
            raise ValueError("There are missing or 0 values in the final au estimate array")


        mean_final_au = facility_total_au_est.mean()
        std_final_au = facility_total_au_est.std()

        if conf_interval > 1 or conf_interval < 0:
            raise ValueError("Confidence interval must be between 0 and 1")
        
        conf_lower = np.percentile(facility_total_au_est, ((1 - conf_interval) / 2) * 100)
        conf_upper = np.percentile(facility_total_au_est, ((1 + conf_interval) / 2) * 100)


        # Save the mean and confidence intervals in the cluster DataFrame
        cluster_data.at[facility_i, "animal_unit_estimate"] = mean_final_au
        cluster_data.at[facility_i, "std_sampled_au"] = std_final_au

        cluster_data.at[facility_i, "animal_units_lower"] = conf_lower
        cluster_data.at[facility_i, "animal_units_upper"] = conf_upper
        
        if optional_other_perc_threshes is not None:
            for thresh in optional_other_perc_threshes:
                thresh_value = np.percentile(facility_total_au_est, thresh * 100)
                cluster_data.at[facility_i, f"animal_units_{thresh}_perc"] = thresh_value

        # store the Animal Count Estimates as well
        a_s_bar_non_zero = sampled_permit["a_s_bar"].loc[sampled_permit["a_s_bar"] != 0]
        # force facility_area to be the same length as a_s_bar_non_zero

        if include_area_uncertainty is True:
            if len(facility_area) != len(a_s_bar_non_zero):
                drop_indices = np.random.randint(0, len(facility_area), len(facility_area) - len(a_s_bar_non_zero) + 1)
                facility_area = np.delete(facility_area, drop_indices)
        
        facility_total_a_count_est = facility_area / a_s_bar_non_zero
        mean_final_a_count = facility_total_a_count_est.mean()
        lower_final_a_count = np.percentile(facility_total_a_count_est, ((1 - conf_interval) / 2) * 100)
        upper_final_a_count = np.percentile(facility_total_a_count_est, ((1 + conf_interval) / 2) * 100)
        std_final_a_count = facility_total_a_count_est.std()

        cluster_data.at[facility_i, "Dairy_count_estimate"] = mean_final_a_count
        cluster_data.at[facility_i, "Dairy_count_estimate_lower"] = lower_final_a_count
        cluster_data.at[facility_i, "Dairy_count_estimate_upper"] = upper_final_a_count
        # for animal_name in animal_names:
        #     # add in columns to store the mean and std of each animal stage estimate counts
        #     cluster_data[f"{animal_name}_est_count_mean"] = mean_final_a_count* cluster
        #     cluster_data[f"{animal_name}_est_count_std"] = 0
    cluster_data = cluster_data.reset_index(drop=True)
    return cluster_data


"""THIS FUNCTION IS NOT USED IN THE FINAL IMPLEMENTATION. THIS IS THE OLD VERSION OF THE FUNCTION
   WE KEEP THIS VERSION FOR REFERENCE AND COMPARISONS BETWEEN THE OLD METHOD AND NEW METHOD (above)
"""
def old_estimate_animal_units(
    clusters: gpd.GeoDataFrame,
    include_area_uncertainty: bool = True,
    animal_density: float = cfg.DAIRY_M2_ESTIMATE,
    model_area_uncertainty: float = cfg.MODEL_AREA_STDEV,
    dairy_m2_uncertainty: float = cfg.DAIRY_M2_STDEV,
    au_conversions: dict = cfg.DAIRY_AU_CONVERSIONS,
    dairy_au_uncertainty: float = cfg.DAIRY_AU_STDEV,
    stat: str = "max",
):
    """
    Estimate cattle count and animal units from cluster area.

    Inputs:
        clusters: GeoDataFrame of clusters containing information about polygon areas
                 for each cluster and the information used to calculate that area
        include_area_uncertainty: bool, whether or not to include upper and lower 95% CI bounds on
            the animal count estimate.
        animal_density: float, number of dairy cattle per m2
            (e.g., dairy cattle density)
        au_conversions: dict, dict of different animal unit conversion factors
            for different assumptions of the animal age distribution.
        stat: str, what estimate of animal unit per animal conversion ratio
            to use. If 'max', use the maximum conversion ratio. If 'min',
            use the minimum conversion ration. If 'weighted_mean', use the
            weighted mean conversion ratio based on EPA animal age estimates
            (currently only implemented for animal_type='Dairy').
            Default 'max' in order to capture the long tail of the distribution
            of potential CAFOs.

    Returns: updated version of results dataframe now containing animal count
             estimation column
    """
    if "cluster_area_m2" not in clusters.columns:
        print(
            "Error: cluster dataset must have `cluster_area_m2` column with cluster area."
        )
        return

    clusters_new = clusters.copy()
    # Add uncertainty to cluster area
    if include_area_uncertainty:
        clusters_new["cluster_area_m2_stdev"] = (
            clusters_new["cluster_area_m2"] * model_area_uncertainty
        )

    # Estimate number of dairy cattle
    clusters_new["Dairy_count_estimate"] = (
        clusters_new["cluster_area_m2"] * animal_density
    )
    if include_area_uncertainty:
        clusters_new["dairy_count_stdev"] = clusters_new.apply(
            lambda x: propagate_stdevs(
                [x.cluster_area_m2_stdev, dairy_m2_uncertainty],
                [x.cluster_area_m2, animal_density],
            ),
            axis=1,
        )
        clusters_new["dairy_count_lower"] = (
            clusters_new["cluster_area_m2"] * animal_density
            - 1.96 * clusters_new["dairy_count_stdev"]
        )
        clusters_new["dairy_count_upper"] = (
            clusters_new["cluster_area_m2"] * animal_density
            + 1.96 * clusters_new["dairy_count_stdev"]
        )

    # Estimate animal units
    clusters_new["animal_unit_estimate"] = (
        clusters_new["Dairy_count_estimate"] * au_conversions[stat]
    )

    if include_area_uncertainty:
        # If stat is not a mean estimate, assume 0 uncertainty.
        if stat not in ["EPA_weighted_mean", "CAFO_estimated_mean"]:
            dairy_au_uncertainty = 0
        elif not dairy_au_uncertainty == 0:
            # We need to re-estimate the standard deviation of the AU conversion factor
            # to account for truncations in the distribution at [0.2, 1.4] (the min and max
            # AU conversion factors for dairy cattle according to WI DNR).

            # Standardize bounds
            a, b = (0.2 - au_conversions[stat]) / dairy_au_uncertainty, (
                1.4 - au_conversions[stat]
            ) / dairy_au_uncertainty
            dairy_au = truncnorm(
                loc=au_conversions[stat], scale=dairy_au_uncertainty, a=a, b=b
            )
            dairy_au_uncertainty = np.sqrt(dairy_au.stats(moments="v"))

        clusters_new["animal_unit_stdev"] = clusters_new.apply(
            lambda x: propagate_stdevs(
                [x.dairy_count_stdev, dairy_au_uncertainty],
                [x.Dairy_count_estimate, au_conversions[stat]],
            ),
            axis=1,
        )
        clusters_new["animal_units_lower"] = (
            clusters_new["animal_unit_estimate"]
            - 1.96 * clusters_new["animal_unit_stdev"]
        )
        clusters_new["animal_units_upper"] = (
            clusters_new["animal_unit_estimate"]
            + 1.96 * clusters_new["animal_unit_stdev"]
        )

    return clusters_new

def propagate_stdevs(stdev_list: list, mean_list: list):
    """
    Function to calculate the standard deviation of the product of two normal distributed random variables.
    Args:
        stdev_list: list of standard deviations (floats)
        mean_list: list of means (floats)
    Returns:
        combined standard deviation (float)
    """
    if len(stdev_list) != len(mean_list):
        print("List of stdevs and means must be the same length.")
        return
    if len(stdev_list) > 2:
        print("Currently not implemented for >2 variables.")
        return
    variance = (
        (mean_list[1] ** 2 * stdev_list[0] ** 2)
        + (mean_list[0] ** 2 * stdev_list[1] ** 2)
        + (stdev_list[0] ** 2 * stdev_list[1] ** 2)
    )

    return np.sqrt(variance)


if __name__ == '__main__':

    with open("config/config.yml", "r") as file:
        configs = yaml.safe_load(file)
    
    cluster_path = Path(configs["cluster_path"])

    #  Load all CF annotation clusters statewide
    all_cf_clusters = cluster.load_clusters(cluster_path / "all_cf_clusters.csv")

    # Load pre-created three- and four-band model clusters
    four_band_clusters = cluster.load_clusters(cluster_path / "four_band_clusters.csv")
    three_band_clusters = cluster.load_clusters(
        cluster_path / "three_band_clusters.csv"
    )

    four_band_clusters = sample_and_calc_au(
    four_band_clusters, include_area_uncertainty=True, model_type='four_band', model_error_join='left'
)
    
    all_cf_clusters = sample_and_calc_au(
    all_cf_clusters, include_area_uncertainty=False
)
    