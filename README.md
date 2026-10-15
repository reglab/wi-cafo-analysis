# wi-cafo-analysis

Analysis code for detecting and characterizing dairy Concentrated Animal Feeding Operations (CAFOs) in Wisconsin using satellite imagery segmentation models and government permit data.

**Research questions:**
- How well can an image segmentation model detect dairy CAFOs and predict their barn footprints?
- How well can we predict dairy cattle herd size using barn footprint predictions?
- Can we use these results to identify potentially unpermitted dairy CAFOs?

## Quick start

```bash
git clone https://github.com/reglab/wi-cafo-analysis && cd wi-cafo-analysis

uv venv --python 3.11 wicafo
uv pip install -r requirements.txt --python wicafo/bin/python3.11
source wicafo/bin/activate

python download_data.py                         # fetch inputs from Hugging Face into ./data (~4 GB)
python generate_paper_results.py --skip-imagery
```

Requires Python 3.11 (any environment manager works). `python check_data.py` lists any missing input files.

## Data

All inputs live under one **data root**, and every file location is derived from it. Set it with
`data_root` in `config/config.yml` (copy `config/config_example.yml`) or the `WI_CAFO_DATA_ROOT`
environment variable. The default is `./data`, so no config is needed if you download there.
`config.yml` has two other optional settings: `gcp_cred_path` (NAIP imagery figures) and
`parcels_path` (see below). Every path is declared once, in `DataPaths` in
[config/config_params.py](config/config_params.py), next to the analysis constants.

Data comes from three places:

| Where | What | How you get it |
|---|---|---|
| [Hugging Face: reglab/wisconsin-dairy-cafo](https://huggingface.co/datasets/reglab/wisconsin-dairy-cafo) | Everything large: annotations, clusters, permits, water, SNAPMAPS layers, slope raster, urban areas, image bounds | `python download_data.py` |
| `reference_data/` in this repo | Small cached tables: pixel-level P/R/IOU curves, Census ACS county covariates, Aug 2024 WDNR permit shapefiles | `git clone` |
| `manual_review/` in this repo | Human-judgment columns from the manual false-positive review and online validation | `git clone` |

Layout of the data root after `download_data.py`. It is the same tree as the Hugging Face repo, whose
dataset card describes every file:

```
data/
├── wi_cafo_facilities.{geojson,parquet,csv}     # published facility dataset
└── input_data/
    ├── WDNR_CAFOs.geojson, ewg_AFOs_012022.geojson, milk_producers.geojson
    ├── permit_animal_type_records.csv
    ├── annotations/         full_state_cf_annotations.geojson, ewg_region_train_labels.geojson, all_labeled_images.csv
    ├── clusters/            all_cf_clusters.csv, four_band_clusters.csv, three_band_clusters.csv
    ├── model_predictions/   inception_dets.geojson, poultry_dets_confidence_0.1.geojson
    └── geospatial/
        ├── image_bound_map.geojson
        ├── county_boundaries/, urban_areas/, dem/WI_slope.tif
        ├── water_data/
        └── snapmaps/        12 SNAPMAPS layers as .feather
```

Not hosted, and not required for `generate_paper_results.py`:
- **Wisconsin land parcels** (V8.0.0, [WI SCO](https://www.sco.wisc.edu/parcels/data/)): only used for parcel overlays on imagery figures and for re-running `cluster.py`. Set `parcels_path` in `config.yml` if you have them.
- **NAIP imagery** (GCS, credentials required; contact the authors): needed for the case-study imagery figures. Use `--skip-imagery` otherwise.
- **Raw model prediction polygons** (~9 GB, see [cafo-segmentation](https://github.com/reglab/cafo-segmentation)): only for `--recalculate-pixel-stats`; place them under `<data_root>/raw_model_predictions/`.
- **Raw SNAPMAPS geodatabase and 30 m DEM**: only to re-derive `snapmaps/*.feather` (`python process_snapmaps.py <gdb>`) and `WI_slope.tif` (`cluster_analysis_functions.calculate_slope_from_dem`).

Model weights and training/inference code are in [cafo-segmentation](https://github.com/reglab/cafo-segmentation).

## Reproducing paper results

```bash
python generate_paper_results.py
```

This generates all paper figures and tables in `paper_results/`:

| Subfolder | Contents |
|---|---|
| `01_model_validation/` | Permit vs. AU estimate comparisons |
| `02_segmentation_quality/` | Pixel-level P/R/IOU curves, annotation error distributions |
| `03_error_analysis/` | Facility-level P/R curves, case study imagery (requires GCP) |
| `04_unpermitted_analysis/` | Statewide CAFO map, permit rate by size, size distributions |
| `05_risk_assessment/` | Multi-factor risk indices by animal unit category |
| `tables/` | CSV tables for publication |
| `publication_dataset/` | Final publication-ready facility dataset |

Optional flags:

```bash
python generate_paper_results.py --skip-pixel-stats   # skip recomputing pixel P/R (use cached)
python generate_paper_results.py --skip-snapmaps      # skip SNAPMAPs layers (if not downloaded)
python generate_paper_results.py --skip-imagery       # skip NAIP tile case-study figures (no GCP needed)
```

## Repository contents

| File | Description |
|---|---|
| `generate_paper_results.py` | Master pipeline — runs all analyses and generates all figures |
| `create_figures.py` | Analysis functions called by the pipeline |
| `plotting_utils.py` | Plotting and visualization utilities |
| `cluster.py` | Polygon clustering from model predictions or human labels |
| `cluster_analysis_functions.py` | Cluster-level analysis (water risk, parcel matching, etc.) |
| `estimate_animal_units.py` | Animal unit estimation with uncertainty quantification |
| `estimate_methane_emissions.py` | Methane emissions estimation from dairy cattle counts |
| `analyze_model_outputs.py` | Model output loading and processing |
| `process_snapmaps.py` | Load/process SNAPMAPs geodatabase layers |
| `naip_utils.py` | NAIP imagery utilities (requires GCP credentials) |
| `add_asym.py` | Asymmetric uncertainty calculations |
| `config/config_params.py` | Single config module: input-file paths (`PATHS`) plus global constants (EPSG codes, AU factors, figure styling) |
| `config/config_example.yml` | Optional overrides (data root, GCP key, parcels) — copy to `config.yml` |
| `download_data.py` / `check_data.py` | Fetch inputs from Hugging Face / report missing files |
| `scripts/stage_hf_upload.py` | Maintainers: assemble the Hugging Face upload tree from a local data folder |

## Citation

*Citation to be added upon publication.*
