# List of global constants to use across scripts
#################################################
#################################################
import os
import pandas as pd
import sys
from pathlib import Path
import yaml


# Paths
#######
# Every file the pipeline reads is declared once, in the `PATHS` registry below.
# Only two things are user-specific, and both are optional (config/config.yml,
# or environment variables):
#   data_root      folder holding the Hugging Face dataset download
#                  (env: WI_CAFO_DATA_ROOT, default: <repo>/data)
#   gcp_cred_path  GCP key, only needed for NAIP imagery figures
#                  (env: GOOGLE_APPLICATION_CREDENTIALS)
# See README.md ("Data") for the full layout and where each file comes from.
REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE = REPO_ROOT / "config" / "config.yml"


def _load_user_config():
    if not _CONFIG_FILE.exists():
        return {}
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


def _resolve(p):
    """Expand ~ and resolve relative paths against the repo root."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


class DataPaths:
    """Locations of every input file, all derived from one data root.

    Layout under `data_root` mirrors the Hugging Face dataset
    (reglab/wisconsin-dairy-cafo), so `python download_data.py` populates it
    directly. Small hand-curated tables live in git under `reference_data/`.
    """

    # Layers of the DATCP SNAPMAPS geodatabase used for the risk indices.
    SNAPMAPS_LAYERS = [
        "NM_590_DEPTH_BEDROCK_LT5FT", "SILURIAN_0_2", "SILURIAN_2_5",
        "NM_590_SHALLOW_SILURIAN", "NM_590_SWQMA_300FT", "SWQMA1000FT",
        "CAFO_W", "CAFO_R", "SLOPES_GREATER_12", "HYDRO_INTERMIT",
        "HYDRO_PERENNIAL", "WATER",
    ]

    def __init__(self, data_root, parcels_path=None, gcp_cred_path=None):
        self.repo_root = REPO_ROOT
        self.data_root = Path(data_root)
        self.gcp_cred_path = gcp_cred_path

        inp = self.input_dir = self.data_root / "input_data"
        # permit / facility sources
        self.wdnr_cafos = inp / "WDNR_CAFOs.geojson"
        self.ewg_afos = inp / "ewg_AFOs_012022.geojson"
        self.milk_producers = inp / "milk_producers.geojson"
        self.permit_animal_records = inp / "permit_animal_type_records.csv"
        # human annotations and the facility clusters built from them
        self.annotations_dir = inp / "annotations"
        self.full_state_annotations = self.annotations_dir / "full_state_cf_annotations.geojson"
        self.ewg_region_annotations = self.annotations_dir / "ewg_region_train_labels.geojson"
        self.all_labeled_images = self.annotations_dir / "all_labeled_images.csv"
        self.clusters_dir = inp / "clusters"
        self.all_cf_clusters = self.clusters_dir / "all_cf_clusters.csv"
        self.four_band_clusters = self.clusters_dir / "four_band_clusters.csv"
        self.three_band_clusters = self.clusters_dir / "three_band_clusters.csv"
        # baseline detectors used in the model comparison table
        self.inception_dets = inp / "model_predictions" / "inception_dets.geojson"
        self.poultry_dets = inp / "model_predictions" / "poultry_dets_confidence_0.1.geojson"
        # geospatial layers
        geo = self.geospatial_dir = inp / "geospatial"
        self.county_boundaries = geo / "county_boundaries" / "County_Boundaries_24K.shp"
        self.image_bound_map = geo / "image_bound_map.geojson"
        self.urban_areas = geo / "urban_areas" / "tl_2020_55_place20.shp"
        self.slope_tif = geo / "dem" / "WI_slope.tif"
        self.water_dir = geo / "water_data"
        self.open_water = self.water_dir / "24k_Hydro_Waterbodies_(Open_Water).geojson"
        self.impaired_lakes = self.water_dir / "impaired_lakes.geojson"
        self.impaired_rivers = self.water_dir / "impaired_rivers_streams.geojson"
        self.water_table_depth = self.water_dir / "GCSM_-_Water_Table_Depth.geojson"
        self.snapmaps_dir = geo / "snapmaps"
        # NOT on Hugging Face (too large / third-party terms); optional
        self.parcels = Path(parcels_path) if parcels_path else self.data_root / "land_parcels" / "WI_parcels.feather"
        self.raw_model_predictions = self.data_root / "raw_model_predictions"
        # small tables tracked in git
        ref = self.reference_dir = REPO_ROOT / "reference_data"
        self.pixel_curves = ref / "pixel_level_p_r_IOU_curves.csv"
        self.census_acs_county = ref / "census_acs_county.csv"
        self.wdnr_2024_dir = ref / "wdnr_cafos_2024_aug"
        self.manual_review_dir = REPO_ROOT / "manual_review"

    def snapmaps_layer(self, layer):
        return self.snapmaps_dir / f"{layer}.feather"

    def required_files(self, skip_snapmaps=False):
        """Files that must exist for generate_paper_results.py to run."""
        req = [
            self.wdnr_cafos, self.ewg_afos, self.milk_producers,
            self.permit_animal_records, self.full_state_annotations,
            self.ewg_region_annotations, self.all_labeled_images,
            self.all_cf_clusters, self.four_band_clusters,
            self.three_band_clusters, self.inception_dets, self.poultry_dets,
            self.county_boundaries, self.image_bound_map, self.urban_areas,
            self.open_water, self.impaired_lakes, self.impaired_rivers,
            self.water_table_depth, self.pixel_curves, self.census_acs_county,
            self.wdnr_2024_dir / "CAFO Main.shp",
            self.wdnr_2024_dir / "CAFO Satellite.shp",
            self.manual_review_dir / "fp_manual_review_ewg_categorized.csv",
            self.manual_review_dir / "fp_manual_review_cf_unmatched_categorized.csv",
        ]
        if not skip_snapmaps:
            req.append(self.slope_tif)
            req += [self.snapmaps_layer(l) for l in self.SNAPMAPS_LAYERS]
        return req

    def missing_files(self, **kwargs):
        return [p for p in self.required_files(**kwargs) if not p.exists()]


_user_cfg = _load_user_config()
_data_root = os.environ.get("WI_CAFO_DATA_ROOT") or _user_cfg.get("data_root") or "data"
_gcp = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or _user_cfg.get("gcp_cred_path")
PATHS = DataPaths(
    _resolve(_data_root),
    parcels_path=_resolve(_user_cfg["parcels_path"]) if _user_cfg.get("parcels_path") else None,
    gcp_cred_path=str(_resolve(_gcp)) if _gcp else None,
)
if PATHS.gcp_cred_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = PATHS.gcp_cred_path

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
