# List of global constants to use across scripts
#################################################
#################################################
import pandas as pd
import sys
from pathlib import Path
import yaml


# permit file:
# read in configuration file
# with open(Path().resolve().parent / "afo_vs_cafo/config/config.yml", "r") as file:
#     configs = yaml.safe_load(file)
#     project_path = configs["project_path"]

project_path = "/Users/mihirb/gitclones/afo_vs_cafo"


animal_count_permit_data_path = (
    project_path + "/data/permit_animal_type_records_5.14.25.csv"
)
# Geographic constants
#####################
# CRS to use in Wisconsin
WI_EPSG = 3071

# CRS for lat/long coordinates
GEO_EPSG = 4326

# List of counties in the EWG study region
EWG_COUNTIES = [
    "Kewaunee",
    "Adams",
    "Dane",
    "Green",
    "Juneau",
    "Lafayette",
    "Portage",
    "Rock",
    "Wood",
]

# Clustering constants
#####################
WORDS_TO_REMOVE = [
    "LLC",
    "&",
    "FARMS",
    "TRUST",
    "DAIRY",
    "INC",
    "REVOCABLE",
    "IRREVOCABLE",
    "FARM",
    "FAMILY",
    "ACRES",
    "LAND",
    "REAL",
    "ESTATE",
    "RIDGE",
    "DATED",
]
COMMON_NAMES = ["JOHN"]

# Uncertainty constants
#####################

# how to cut up model predicted areas into different categories for conditional error distributions
size_bins = [0, 50, 500, 1000, 2500, 5000, 10000, 10000000]

# standard deviation on the discrepancy between model predicted polygon area and human annotation area
MODEL_AREA_STDEV = 0.26
model_area_percent_error_mean = -0.15
model_area_percent_error_std = 0.51

# estimated standard deviation on head of dairy cattle per m^2 based on industry standards and distribution of

# Square meter per cow (below is the inverse of the square feet per cow)
MEAN_DAIRY_M2_ESTIMATE = 10.7639  # inverse of the DAIRY_M2_ESTIMATE
MIN_DAIRY_M2_ESTIMATE = 8  # from: 1 cow per stall and stall estimates from https://thedairylandinitiative.vetmed.wisc.edu/home/housing-module/adult-cow-housing/stocking-density/
MAX_DAIRY_M2_ESTIMATE = 16
STD_DAIRY_M2_ESTIMATE = 1.4  # from +/- 15square ft per cow stall = 1.4 m^2 per cow


# CAFO footprint area in WI
# dairy cows per m^2 based on industry standards
DAIRY_M2_STDEV = 0.00599  # = (DAIRY_M2_ESTIMATE - (1 / (110 ft2 per cow +/- 15 ft^2 one-sided CI / 10.7639 ft^2 per m2))) / 1.96 stdevs
DAIRY_M2_ESTIMATE = 0.09785  # = 1 / (110 ft2 per cow / 10.7639 ft2 per m2)


# amount of animal units per head of dairy cattle (from WI DNR).
# Weighted mean estimate for Dairy cattle from total animal age distribution estimated by EPA.
DAIRY_AU_CONVERSIONS = {
    "max": 1.40,
    "min": 0.60,
    "EPA_weighted_mean": 0.984,  # = 0.2522 * 0.2 + 0.4936 * 1.4 + 0.0747 * 0.6 + 0.1794 * 1.1
    "CAFO_estimated_mean": 1.205,
}

# These include dead space and feeding space in barns. This is the old params used
# truncnorm_params = {
#     "calves": {"min_val": 2, "max_val": 4, "mean": 3.3, "std": 1},
#     "milking_cows": {"min_val": 8, "max_val": 12, "mean": 10, "std": 1},
#     "heifers_800_1200": {"min_val": 8, "max_val": 10, "mean": 9, "std": 1},
#     "heifers_400_800": {"min_val": 8, "max_val": 10, "mean": 9, "std": 1},
#     "beef_cattle": {"min_val": 8, "max_val": 12, "mean": 10, "std": 1},
# }

# New proposed params (see Mar 18 2025 pre-read for sources)
truncnorm_params = {
    "calves": {"min_val": 3, "max_val": 7, "mean": 4, "std": 0.4},
    "milking_cows": {"min_val": 7, "max_val": 15, "mean": 12, "std": 1.2},
    "heifers_800_1200": {"min_val": 6.5, "max_val": 12, "mean": 9.5, "std": 0.95},
    "heifers_400_800": {"min_val": 4, "max_val": 8, "mean": 6, "std": 0.6},
    "beef_cattle": {"min_val": 4, "max_val": 8, "mean": 5, "std": 0.5},
}

# estimated standard deviation on animal units per head of dairy cattle. See
# https://docs.google.com/spreadsheets/d/1oM1Ux0tv_HlDdGMjBkvStgw_aqbMyAgR4v_u8WSsGa4/edit#gid=0
DAIRY_AU_STDEV = 0.1382
MIN_AU_FACTOR = 0.2
MAX_AU_FACTOR = 1.4

CALVES_AU_FACTOR = 0.2
MILKING_AND_DRY_COWS_AU_FACTOR = 1.4
HEIFERS_800_1200_AU_FACTOR = 1.10
HEIFERS_400_800_AU_FACTOR = 0.6
STEERS_AU_FACTOR = 1


au_factors = {
    "calves": [MILKING_AND_DRY_COWS_AU_FACTOR],
    "milking_cows": [MILKING_AND_DRY_COWS_AU_FACTOR],
    "heifers_800_1200": [MILKING_AND_DRY_COWS_AU_FACTOR],
    "heifers_400_800": [MILKING_AND_DRY_COWS_AU_FACTOR],
    "beef_cattle": [MILKING_AND_DRY_COWS_AU_FACTOR],
}
au_factors = pd.DataFrame(au_factors)


# Figure constants
#####################
# Resolution
FIG_DPI = 300
# Font
FIG_FONT_FAMILY = "sans-serif"
FIG_FONT = "DejaVu Sans"
# Export format ('svg' or 'png')
FIG_EXPORT_FORMAT = "svg"

# Consistent color palette for permitted / unpermitted categories
COLOR_PERMITTED = "teal"
COLOR_UNPERMITTED = "orange"
COLOR_UNPERMITTED_MILK = "darkorange"
COLOR_UNDER_THRESHOLD = "grey"
COLOR_THRESHOLD_LINE = "red"
COLOR_REFERENCE_LINE = "black"

# Default alpha values
ALPHA_FILL = 0.5
ALPHA_UNDER_THRESHOLD = 0.9
ALPHA_MILK_LICENSE = 0.7

# Axes / tick styling
FIG_LABEL_SIZE = 12
FIG_TICK_SIZE = 11
FIG_TITLE_SIZE = 14
FIG_LEGEND_SIZE = 10
FIG_LINEWIDTH = 1.2
