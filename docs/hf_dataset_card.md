---
license: cc-by-4.0
language:
- en
tags:
- geospatial
- cafo
- dairy
- wisconsin
- environmental
- permits
- remote-sensing
- agriculture
- farm
pretty_name: Wisconsin Large Dairy Farms Dataset (CAFOs)
size_categories:
- 1G<n<10G
configs:
  - config_name: default
    data_files: wi_cafo_facilities.csv
---


# Wisconsin Large Dairy Farms Dataset (CAFOs)

This is a geo-referenced dataset of large dairy farms in Wisconsin compiled using satellite imagery, computer vision, human validation, and merges to key administrative and open geospatial data sources.

The process to generate this data is described in the manuscript: *Blind Spots in Environmental Permitting: An AI-Assisted Assessment of Coverage Gaps and
Environmental Risk among Large Dairy Farms in Wisconsin* (link to be made available upon publication).

We also include the input datasets which, along with the analysis code at [this GitHub repository](https://github.com/reglab/wi-cafo-analysis), can reproduce the paper's key results.

---

## Repository Structure

Download the whole repository into one folder (the *data root*) with the
`download_data.py` script in the analysis repo; every path below is relative to it.

```
wi_cafo_facilities.geojson       <- Primary publication dataset (WGS84)
wi_cafo_facilities.parquet       <- Same data in GeoParquet (WI State Plane EPSG:3071)
wi_cafo_facilities.csv           <- Same data with geometry as WKT column
input_data/
    WDNR_CAFOs.geojson
    ewg_AFOs_012022.geojson
    milk_producers.geojson
    permit_animal_type_records.csv
    annotations/
        full_state_cf_annotations.geojson
        ewg_region_train_labels.geojson
        all_labeled_images.csv
    clusters/
        all_cf_clusters.csv
        all_cf_clusters.geojson
        four_band_clusters.csv
        three_band_clusters.csv
    model_predictions/
        inception_dets.geojson
        poultry_dets_confidence_0.1.geojson
    geospatial/
        image_bound_map.geojson
        county_boundaries/
        urban_areas/
        dem/WI_slope.tif
        water_data/
        snapmaps/
```

---

## Publication Dataset (`wi_cafo_facilities.*`)

The main output of the study: 1,469 large dairy farms across Wisconsin, with permit linkage, animal unit estimates, milk producer license linkage, and environmental risk indices.

The process to generate this data is described in the manuscript (to be linked upon publication). It includes data on:
- All permitted dairy CAFO farms as per 2022 WPDES permit data from Wisconsin Department of Natural Resources
- Dairy farms found by our model and human validation pipeline, estimated to contain 500 animal units or more, that did not match to 2022 WPDES permits.

### Column Descriptions

| Column | Type | Source | Description |
|---|---|---|---|
| `polygon_indices` | int | Constructed | Unique serial identifier for each farm, zero-indexed. |
| `animal_unit_estimate` | float | Constructed | Point estimated animal units (AU) at the farm. Method described in manuscript: mean value of a Monte Carlo simulation using total barn area, space-per-animal and cow stage distributions. |
| `animal_units_lower` | float | Constructed | Lower bound estimated animal units. Calculated as the 2.5th percentile of the simulated uncertainty interval method. |
| `animal_units_upper` | float | Constructed | Upper bound of the 95% simulation interval for the AU estimate. Calculated as the 97.5th percentile of the simulated uncertainty interval method. |
| `type` | string | Constructed | Facility classification based on permit matching and estimated animal units. Categories: `"Permitted dairy CAFOs"` (matched to a WDNR 2022 WPDES CAFO permit), `"AFOs between 500 and 1000 AU"` (estimated 500–1000 AU, no permit match), `"Potential unpermitted CAFOs (>1000 AU)"` (estimated >1000 AU, no permit match — above the threshold requiring a CAFO permit in Wisconsin). |
| `permit_facility_id` | float | WDNR permit data | Wisconsin DNR Facility Identification Number (FIN) for facilities matched to a CAFO permit within 500m. `NaN` for unmatched facilities. Source: `WDNR_CAFOs.geojson`. |
| `permit_allowable_animal_units` | float | WDNR permit data | Maximum animal units allowed under the facility's CAFO permit. `NaN` for unmatched facilities. Source: `WDNR_CAFOs.geojson`. |
| `matched_milk` | bool | WI DATCP milk producer licenses | Whether the facility was spatially matched (within 500m) to an active Wisconsin milk producer license. Source: `milk_producers.geojson`. |
| `type_of_milk` | string | WI DATCP milk producer licenses | Type of milk produced at the matched milk license (`"Bovine"`, `"Goat"`, etc.). `None` if `matched_milk` is False. Source: `milk_producers.geojson`. |
| `water_distance` | float | WI DNR hydrography | Distance (m) from the facility centroid to the nearest surface water body (open water, perennial or intermittent stream). |
| `impaired_water_distance` | float | EPA 303(d) / WI DNR | Distance (m) to the nearest EPA 303(d)-listed impaired water body or stream. |
| `closest_water_impaired` | bool | EPA 303(d) / WI DNR | `True` if the nearest water body is 303(d)-listed impaired (i.e., `impaired_water_distance` ≤ `water_distance`). |
| `hydro_intermit_distance` | float | WI DNR 24K hydrography | Distance (m) to the nearest intermittent stream. |
| `hydro_perennial_distance` | float | WI DNR 24K hydrography | Distance (m) to the nearest perennial stream. |
| `gw_0` | bool | WI DNR groundwater | `True` if the facility is located in an area with a water table depth of 0 ft (i.e., at or above the surface). Source: GCSM water table depth layer. |
| `gw_20` | bool | WI DNR groundwater | `True` if the facility is in an area with water table depth ≤ 20 ft. |
| `gw_50` | bool | WI DNR groundwater | `True` if the facility is in an area with water table depth ≤ 50 ft. |
| `bedrock_lt5ft_distance` | float | SNAPMAPs | Distance (m) to the nearest area with bedrock depth < 5 ft (shallow bedrock). Source: SNAPMAPs soils layer. |
| `bedrock_lt5ft_dummy` | bool | SNAPMAPs | `True` if the facility itself is located on shallow bedrock (depth < 5 ft). |
| `silurian_0_2_distance` | float | SNAPMAPs | Distance (m) to the nearest area with Silurian carbonate bedrock at 0–2 ft depth. |
| `silurian_0_2_dummy` | bool | SNAPMAPs | `True` if the facility sits on Silurian carbonate bedrock at 0–2 ft. |
| `silurian_2_5_distance` | float | SNAPMAPs | Distance (m) to the nearest area with Silurian carbonate bedrock at 2–5 ft depth. |
| `silurian_2_5_dummy` | bool | SNAPMAPs | `True` if the facility sits on Silurian carbonate bedrock at 2–5 ft. |
| `shallow_silurian_distance` | float | SNAPMAPs | Distance (m) to the nearest area with any shallow Silurian carbonate bedrock (0–5 ft). |
| `shallow_silurian_dummy` | bool | SNAPMAPs | `True` if the facility sits on any shallow Silurian carbonate bedrock (0–5 ft). |
| `mean_slope` | float | WI DNR 30m DEM | Mean terrain slope (%) within a 1 km radius of the facility centroid, derived from a 30m digital elevation model. |
| `slope_greater_12_distance` | float | WI DNR 30m DEM | Distance (m) to the nearest area with slope > 12%. |
| `slope_greater_12_dummy` | bool | WI DNR 30m DEM | `True` if the facility is located in an area with slope > 12%. |
| `swqma_300ft_distance` | float | SNAPMAPs | Distance (m) to the nearest Surface Water Quality Management Area (SWQMA) 300 ft buffer zone — a regulatory setback for manure spreading. |
| `swqma_300ft_dummy` | bool | SNAPMAPs | `True` if the facility is within a SWQMA 300 ft zone. |
| `swqma_1000ft_distance` | float | SNAPMAPs | Distance (m) to the nearest SWQMA 1,000 ft buffer zone. |
| `swqma_1000ft_dummy` | bool | SNAPMAPs | `True` if the facility is within a SWQMA 1,000 ft zone. |
| `cafo_w_restrict_distance` | float | SNAPMAPs | Distance (m) to the nearest winter-restricted spreading zone for CAFOs (areas where manure spreading is prohibited in winter). |
| `cafo_r_restrict_distance` | float | SNAPMAPs | Distance (m) to the nearest CAFO regulatory restriction area (general CAFO-specific spreading restriction zone). |
| `hand_built_risk_index` | float | Constructed | Composite environmental risk index (0–1) built by manually weighting proximity to impaired waters, groundwater depth, bedrock depth, and regulatory restriction zones. Higher = greater environmental risk. |
| `pca_risk_index` | float | Constructed | Composite environmental risk index (0–1) derived from principal component analysis of the same geospatial risk factors used in `hand_built_risk_index`. Higher = greater environmental risk. |
| `n_buildings` | int | Constructed | Number of individual barn polygons that make up this facility cluster. |
| `n_parcels` | int | Constructed | Number of distinct land parcels associated with this facility cluster (via spatial join to WI statewide parcel data). |
| `cluster_area_m2` | float | Constructed | Total footprint area of all barn polygons in the facility cluster, in square meters. |
| `geometry` | geometry | Constructed | MultiPolygon of all human-annotated barn footprints comprising the facility, in WGS84 (EPSG:4326) for GeoJSON/CSV and WI State Plane (EPSG:3071) for Parquet. |

### Quick Start

```python
import geopandas as gpd

# Load from local file
gdf = gpd.read_file("wi_cafo_facilities.geojson")

# Or load from Hugging Face Hub
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="reglab/wisconsin-dairy-cafo",
    filename="wi_cafo_facilities.geojson",
    repo_type="dataset"
)
gdf = gpd.read_file(path)
```

---

## Input Data (`input_data/`)

These are the raw and intermediate datasets used to generate the publication dataset and all paper figures. Download this folder and follow the setup instructions at [this GitHub repository](https://github.com/reglab/wi-cafo-analysis) to reproduce all results.

### `input_data/annotations/`

Human-labeled barn polygon annotations produced by CloudFactory workers on NAIP aerial imagery.

| File | Description |
|---|---|
| `full_state_cf_annotations.geojson` | All human-annotated polygon annotations outside the 9-county EWG study region. |
| `ewg_region_train_labels.geojson` | All human-annotated polygon annotations within the 9-county EWG study region (Brown, Calumet, Fond du Lac, Green Lake, Manitowoc, Outagamie, Sheboygan, Waupaca, Winnebago). |
| `all_labeled_images.csv` | Every NAIP image tile (`Image name`) that was sent to CloudFactory for labeling, with the submission batch (`CF submission`) and whether the tile was buffered (`buffered?`). Used to tell "model prediction with no human label" apart from "never sent for labeling". |

---

### `input_data/WDNR_CAFOs.geojson`

**Source:** Wisconsin Department of Natural Resources (WDNR)
**Acquired:** 2022-01-25

Locations and full permit metadata for all CAFO sites permitted by WDNR in Wisconsin. This is a full join of CAFO main permit sites (2018 database) and satellite sites (2022 geocoded permits). 

---

### `input_data/ewg_AFOs_012022.geojson`

**Source:** Environmental Working Group (EWG)
**Acquired:** January 2022

Point locations and metadata for all Animal Feeding Operations (AFOs) labeled by EWG across Wisconsin. See description and methodology [here](https://www.ewg.org/research/double-trouble-wisconsins-land-and-water-are-inundated-pollution-animal-manure-and-excess)

---

### `input_data/permit_animal_type_records.csv`

**Source:** WDNR permit website (manual compilation)
**Date:** 2025-05-14

Animal type counts manually compiled from WDNR CAFO permit application PDF documents. Each row corresponds to one permitted facility and contains counts of each animal type on-site (dairy cows, heifers, calves, dry cows, swine, poultry, etc.). Sourced from the WDNR e-permitting portal website [here](https://dnr.wisconsin.gov/permits/water).

**Key columns:**
- Facility identifiers linking to `WDNR_CAFOs.geojson` via FIN
- Per-animal-type counts (dairy cows, heifers, calves, dry cows, beef, swine, poultry)

---

### `input_data/milk_producers.geojson`

**Source:** Wisconsin Department of Agriculture, Trade and Consumer Protection (DATCP)
**Acquired:** 2023-07-01, updated 2024-03-04
**URL:** [DATCP Milk Producer Licenses](https://mydatcp.wi.gov/SiteMap/ServiceDetails/60670424-cf8c-eb11-8141-0050568c4f26)

All active permitted milk producer licenses in Wisconsin, geocoded to coordinate points using the Google Geocoding API. 75 instances of inaccurate geocoding were identified and manually corrected in the March 2024 update. 

---

### `input_data/clusters/`

Facility-level clusters: groups of overlapping/adjacent model-predicted barn polygons (or human-annotated polygons) merged into single facility records using the clustering algorithm in `cluster.py`. All model clusters use confidence threshold 0.5 unless otherwise noted.

| File | Size | Description |
|---|---|---|
| `all_cf_clusters.csv` | ~9 MB | Facility clusters derived from all CloudFactory human annotations statewide, in the same CSV schema as the model cluster files (geometry as WKT in `geometry2`). This is the file the analysis code loads. |
| `all_cf_clusters.geojson` | ~14 MB | The same human-annotation clusters as GeoJSON, for use in GIS tools. |
| `four_band_clusters.csv` | ~224 MB | Facility clusters from the 4-band (RGB + near-infrared) segmentation model, run at full state coverage. |
| `three_band_clusters.csv` | ~180 MB | Facility clusters from the 3-band (RGB only) segmentation model, run at full state coverage. |

---

### `input_data/model_predictions/`

Detections from two baseline models, used only for the model-comparison table.

| File | Description |
|---|---|
| `inception_dets.geojson` | Image-tile detections (tile bounds `lon_min`..`lat_max`, confidence `conf`) from an earlier InceptionV3 tile classifier. |
| `poultry_dets_confidence_0.1.geojson` | Poultry-barn segmentation polygons at confidence >= 0.1. |

The full-state segmentation outputs of the three- and four-band models (~9 GB) are not
hosted here; the pipeline uses the cluster files above, and the cached pixel-level
precision/recall curves are checked into the analysis repo (`reference_data/`).

---

### `input_data/geospatial/`

Ancillary geospatial layers.

| Path | Source | Description |
|---|---|---|
| `image_bound_map.geojson` | Constructed | Footprints of the NAIP image tiles covering Wisconsin (`filename` column), used to map annotations to imagery. |
| `county_boundaries/` | WDNR Open Data Portal | Wisconsin county boundary shapefile (1:24,000). |
| `urban_areas/tl_2020_55_place20.*` | [U.S. Census TIGER/Line 2020](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Incorporated places in Wisconsin. Used as the proxy for "urban area" when filtering likely false positives within 500 m of a city or village. |
| `dem/WI_slope.tif` | Derived from the [WI DNR 30 m DEM](https://geodata.wisc.edu/catalog/WIDNR-b131bd2b397f467499575afa5c22f994) | Terrain slope (%) on the 30 m grid (float32, nodata -9999, losslessly compressed). Used for mean slope within 1 km of each facility. |
| `snapmaps/*.feather` | [DATCP SNAPMAPS](https://gis-widatcp.opendata.arcgis.com/datasets/47ff7f962ad8415aa82938e9df392f21/about) (NM 590 statewide geodatabase) | The 12 layers used for the risk indices (shallow bedrock, Silurian carbonate, SWQMA buffers, winter/regulatory restrictions, steep slopes, intermittent/perennial streams, water), as geofeather files in EPSG:3071. |

#### `geospatial/water_data/`

| File | Source | Description |
|---|---|---|
| `GCSM_-_Water_Table_Depth.geojson` | [WI DNR](https://data-wi-dnr.opendata.arcgis.com/datasets/wi-dnr::gcsm-water-table-depth/about) | Depth to groundwater table (m) across Wisconsin |
| `impaired_rivers_streams.geojson` | [EPA 303(d) / WI DNR](https://data-wi-dnr.opendata.arcgis.com/search?q=303&tags=water) | Rivers and streams listed as impaired under the Clean Water Act |
| `impaired_lakes.geojson` | [EPA 303(d) / WI DNR](https://data-wi-dnr.opendata.arcgis.com/search?q=303&tags=water) | Lakes listed as impaired under the Clean Water Act |
| `24k_Hydro_Waterbodies_(Open_Water).geojson` | [WI DNR](https://data-wi-dnr.opendata.arcgis.com/datasets/wi-dnr::24k-hydro-waterbodies-open-water) | All open water bodies in Wisconsin (24K hydrography) |

---

## Not Included

These are not needed to run `generate_paper_results.py`, but are used by other parts of the workflow.

| Dataset | Source | Needed for |
|---|---|---|
| Wisconsin land parcels (V8.0.0) | [WI SCO Parcel Data](https://www.sco.wisc.edu/parcels/data/) | Parcel overlays on imagery figures; re-running `cluster.py` |
| USDA NAIP aerial imagery | GCS bucket (credentials required; contact the authors) | Imagery case-study figures (`--skip-imagery` omits them) |
| Raw SNAPMAPS geodatabase | [DATCP GIS Data](https://datcp.wi.gov/Pages/Programs_Services/GISData.aspx) | Re-deriving `snapmaps/*.feather` with `process_snapmaps.py` |
| Raw 30 m DEM | [geodata.wisc.edu](https://geodata.wisc.edu/catalog/WIDNR-b131bd2b397f467499575afa5c22f994) | Re-deriving `WI_slope.tif` (`cluster_analysis_functions.calculate_slope_from_dem`) |
| Full-state model prediction polygons | [cafo-segmentation](https://github.com/reglab/cafo-segmentation) | `generate_paper_results.py --recalculate-pixel-stats` |

---

## Reproducing the Paper

```bash
git clone https://github.com/reglab/wi-cafo-analysis && cd wi-cafo-analysis
pip install -r requirements.txt
python download_data.py            # downloads this repository into ./data
python generate_paper_results.py --skip-imagery
```

See the [analysis repo README](https://github.com/reglab/wi-cafo-analysis) for details. All paper figures and tables are written to `paper_results/`.

---

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt for any purpose with attribution.

Source data (WDNR permits, EWG AFOs, DATCP milk licenses, WI DNR hydrography) are from public government sources.

---

## Citation

*Citation to be added upon publication.*

---

## Contact

RegLab, Stanford Law School — [reglab.stanford.edu](https://reglab.stanford.edu)