# This script contains code to estimate methane emissions (with uncertainty) from 
# a count of dairy cattle, under different assumptions.
from pathlib import Path
import yaml
from add_asym import add_asym
import geopandas as gpd
import numpy as np

import estimate_animal_units as est_au
import cluster
import cluster_analysis_functions as caf

# Enteric fermentation emissions factors for dairy cattle by age group, kg CH4/head/year
# Source: EPA 2023 GHG Inventory Annexes Table A-146
AGE_SPECIFIC_EF_FACTORS = {'Dairy calves':12,
                                    'Dairy cows': 144,
                                    'Dairy replacement heifers (7-11 mo)': 43,
                                    'Dairy replacement heifers (12-23 mo)': 65}

# Uncertainty in enteric fermentation emissions factors, fraction (lower bound, upper bound)
# Source: EPA 2023 GHG Inventory Annexes Table A-237
EF_UNCERTAINTY = [-0.11, 0.18]

# Population of dairy cattle by age group, in thousands of head
# Source: EPA 2023 GHG Inventory Annexes Table A-126
AGE_SPECIFIC_POPS = {'Dairy calves':655,
                                    'Dairy cows': 1282,
                                    'Dairy replacement heifers (7-11 mo)': 194,
                                    'Dairy replacement heifers (12-23 mo)': 466,
                                    'Dairy total': 655 + 1282 + 194 + 466}

# Coefficient of variation on the USDA's 2020 estimate of total number of cattle in WI
# Source: USDA Cattle Methodology and Quality Measures 2020-21, p.6
# https://www.nass.usda.gov/Publications/Methodology_and_Data_Quality/Cattle/03_2021/cattqm21.pdf 
CATTLE_POP_CV = 3.8

# Total methane emissions from dairy cattle manure management, kt
# Source: EPA 2023 GHG Inventory Annexes Table A-172
TOTAL_MM_EMISSIONS = 130.2341

# Uncertainty in manure management methane emissions factors, fraction (lower bound, upper bound)
# Source: EPA 2023 GHG Inventory Annexes Table A-237
MM_UNCERTAINTY = [-0.18, 0.2]

# Methane GWP conversion factor
# Source: https://ecometrica.com/assets/GHGs-CO2-CO2e-and-Carbon-What-Do-These-Mean-v2.1.pdf
METHANE_GWP = 29.8

def estimate_ef_emissions_factor(method='weighted_mean'):
    """
    This function uses the global constants supplied above to estimate the enteric fermentation
    emissions factor for a head of dairy cattle in WI.
    Args:
        method: how to estimate the enteric fermentation emissions factor. 'weighted mean' 
            calculates an average factor weighted by animal age population, 'assume_old' 
            assumes all cattle are mature dairy cows.
    Returns:
        Point estimate, as well as lower and upper bounds of uncertainty
    """
    # Get weighted average of age-specific EF emissions factors in kg CH4/head/year
    if method=='weighted_mean':
        point_estimate = sum([
            AGE_SPECIFIC_POPS[x] * AGE_SPECIFIC_EF_FACTORS[x] / AGE_SPECIFIC_POPS['Dairy total']
            for x in AGE_SPECIFIC_EF_FACTORS.keys()])
    if method=='assume_old':
        point_estimate = AGE_SPECIFIC_EF_FACTORS['Dairy cows']
    
    # Get uncertainty bounds
    lower_bound = point_estimate+point_estimate*EF_UNCERTAINTY[0]
    upper_bound = point_estimate+point_estimate*EF_UNCERTAINTY[1]
    
    return [point_estimate, lower_bound, upper_bound]

def estimate_mm_emissions_factor():
    """
    This function uses the global constants supplied above to estimate the manure management
    emissions factor for the average head of dairy cattle in WI.
    Args:
    Returns:
        Point estimate, as well as lower and upper bounds of uncertainty
    """
    # Get average MM emissions factor in kg CH4/head/year
    point_estimate = (TOTAL_MM_EMISSIONS * 1e6) / (AGE_SPECIFIC_POPS['Dairy total']*1000)

    # Get uncertainty. Assumes worst case error correlation.
    # CAVEATS: manure management uncertainty is actually the uncertainty on the manure management emissions
    # factors for each type of manure management, not an uncertainty on the total emissions from manure management.
    # This means that uncertainty in the distribution of manure management techniques is not accounted for.
    lower_bound = point_estimate + TOTAL_MM_EMISSIONS * 1e6 * MM_UNCERTAINTY[0] / (
        AGE_SPECIFIC_POPS['Dairy total']*1000 * (1+(CATTLE_POP_CV / 100)))
    upper_bound = point_estimate + TOTAL_MM_EMISSIONS * 1e6 * MM_UNCERTAINTY[1] / (
        AGE_SPECIFIC_POPS['Dairy total']*1000 * (1-(CATTLE_POP_CV / 100)))

    return [point_estimate, lower_bound, upper_bound]


def print_emissions_estimate(estimated_emissions: tuple):
    """
    This function prints out an interpretable summary of estimated emissions 
    from `estimate_combined_emissions`.
    Args:
        estimated_emissions: tuple output from `estimate_combined_emissions`
    Returns: None
    """
        # Print estimate in two units:
    print('Estimated total emissions, kt CH4: ', estimated_emissions[0][0], 
          [estimated_emissions[0][0]-estimated_emissions[0][1],
            estimated_emissions[0][0]+estimated_emissions[0][2]])
    print('Estimated total emissions, Mt CO2e: ', estimated_emissions[1][0], 
          [estimated_emissions[1][0]-estimated_emissions[1][1],
            estimated_emissions[1][0]+estimated_emissions[1][2]])


def estimate_combined_emissions(cow_count: float, verbose=False):
    """
    This function estimates the total methane emissions from both enteric fermentation and 
    manure management for a given number of dairy cattle.
    Args:
        cow_count: number of dairy cattle
        verbose: whether to report estimated emissions factors
    Returns:
        Tuple with first element = point estimate, and lower and upper bounds of uncertainty
            in kt CH4, and second element the same but converted to MT CO2eq.
    """
    # Estimate enteric fermentation emissions factor
    ef_factor = estimate_ef_emissions_factor()
    if verbose:
        print('Estimated enteric fermentation emissions factor: ', round(ef_factor[0], 2))
    # Estimate manure management emissions factor
    mm_factor = estimate_mm_emissions_factor()
    if verbose:
        print('Estimated manure management emissions factor: ', round(mm_factor[0], 2))
    # Combine the two estimates to create an overall emissions factor with uncertainty.
    methane_factor = add_asym([ef_factor[0], mm_factor[0]],
                               [ef_factor[0]-ef_factor[1], mm_factor[0]-mm_factor[1]],
                               [ef_factor[2]-ef_factor[0], mm_factor[2]-mm_factor[0]])
    if verbose:
        print('Estimated combined emissions factor: ', round(methane_factor[0], 2))

    # Estimate total emissions in kt CH4
    total_estimate = [round(x * cow_count / 1000000, 2) for x in methane_factor]

    # Convert estimate to MT CO2eq using methane GWP
    total_estimate_co2e = [round(x * METHANE_GWP / 1000, 2) for x in total_estimate]

    return total_estimate, total_estimate_co2e


if __name__ == "__main__":
    """
    """
    # Find paths from config
    with open(Path().resolve().parent / 'afo_vs_cafo/config/config.yml', 'r') as file:
        configs = yaml.safe_load(file)
    cluster_path = Path(configs['cluster_path'])
    data_path = Path(configs['data_path'])

    # Load and preprocess model cluster file
    four_band_clusters = cluster.load_clusters(cluster_path / "four_band_clusters.csv")
    four_band_clusters =est_au.sample_and_calc_au(four_band_clusters, include_area_uncertainty=True)
    four_band_clusters = four_band_clusters[four_band_clusters['animal_unit_estimate']>150]

    # Estimate upper and lower bounds on total cattle population
    total_dairy_cattle = four_band_clusters['Dairy_count_estimate'].sum()
    total_dairy_cattle_var = four_band_clusters['dairy_count_stdev']**2
    total_dairy_cattle_min = total_dairy_cattle-1.96*np.sqrt(total_dairy_cattle_var.sum())
    total_dairy_cattle_max = total_dairy_cattle+1.96*np.sqrt(total_dairy_cattle_var.sum())
    print('Total dairy cattle count: ', round(total_dairy_cattle),
          f'[{round(total_dairy_cattle_min)}', ', ', f'{round(total_dairy_cattle_max)}]')
    
    # Estimate combined emissions
    estimated_combined_emissions = estimate_combined_emissions(total_dairy_cattle)
    print_emissions_estimate(estimated_combined_emissions)
    
    # Estimate combined emissions from unpermitted dairy cattle in WI
    
    # Load known CAFO locations
    WDNR_CAFOs = gpd.read_file(data_path / "WDNR_CAFOs.geojson", driver="GeoJSON")
    # Match model clusters to CAFO permits 
    matched_model_clusters, err = caf.merge_clusters_permits(
        four_band_clusters,
        WDNR_CAFOs,
        drop_multi_matches=False,
        sum_satellite_counts=False,
    )
    unpermitted_model_clusters = four_band_clusters[
        ~(
            four_band_clusters["polygon_indices"].isin(
                matched_model_clusters["polygon_indices"]
            )
        )
    ].copy()

    # Estimate upper and lower bounds on total unpermitted cattle population
    total_unpermitted_dairy_cattle = unpermitted_model_clusters['Dairy_count_estimate'].sum()
    total_unpermitted_dairy_cattle_var = unpermitted_model_clusters['dairy_count_stdev']**2
    total_unpermitted_dairy_cattle_min = total_unpermitted_dairy_cattle-1.96*np.sqrt(total_unpermitted_dairy_cattle_var.sum())
    total_unpermitted_dairy_cattle_max = total_unpermitted_dairy_cattle+1.96*np.sqrt(total_unpermitted_dairy_cattle_var.sum())
    print('Total unpermitted dairy cattle count: ', round(total_unpermitted_dairy_cattle),
          f'[{round(total_unpermitted_dairy_cattle_min)}', ', ', f'{round(total_unpermitted_dairy_cattle_max)}]')
    
    # Estimate combined emissions
    estimated_combined_emissions = estimate_combined_emissions(total_unpermitted_dairy_cattle)
    print_emissions_estimate(estimated_combined_emissions)


    # Estimate combined emissions from unpermitted potential dairy CAFOs
    human_annotation_clusters = cluster.load_clusters(cluster_path / 'all_CF_clusters.csv')
    human_annotation_clusters =est_au.sample_and_calc_au(human_annotation_clusters, include_area_uncertainty=True)

    matched_annotation_clusters, err = caf.merge_clusters_permits(
            human_annotation_clusters,
            WDNR_CAFOs,
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
    unpermitted_potential_CAFOs = unpermitted_annotation_clusters[
        (
            unpermitted_annotation_clusters["animal_unit_estimate"]
            >= 1500
        )
    ].copy()

    # Estimate upper and lower bounds on total unpermitted cattle population
    total_unpermitted_CAFO_dairy_cattle = unpermitted_potential_CAFOs['Dairy_count_estimate'].sum()
    total_unpermitted_CAFO_dairy_cattle_var = unpermitted_potential_CAFOs['dairy_count_stdev']**2
    total_unpermitted_CAFO_dairy_cattle_min = total_unpermitted_CAFO_dairy_cattle-1.96*np.sqrt(
        total_unpermitted_CAFO_dairy_cattle_var.sum())
    total_unpermitted_CAFO_dairy_cattle_max = total_unpermitted_CAFO_dairy_cattle+1.96*np.sqrt(
        total_unpermitted_CAFO_dairy_cattle_var.sum())
    print('Total unpermitted dairy CAFO cattle count: ', round(total_unpermitted_CAFO_dairy_cattle),
          f'[{round(total_unpermitted_CAFO_dairy_cattle_min)}', ', ', f'{round(total_unpermitted_CAFO_dairy_cattle_max)}]')
    
    # Estimate combined emissions
    estimated_combined_emissions = estimate_combined_emissions(total_unpermitted_CAFO_dairy_cattle)
    print_emissions_estimate(estimated_combined_emissions)
