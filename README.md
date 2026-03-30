# wi-cafo-analysis

Analysis code for detecting and characterizing dairy Concentrated Animal Feeding Operations (CAFOs) in Wisconsin using satellite imagery segmentation models and government permit data.

**Research questions:**
- How well can an image segmentation model detect dairy CAFOs and predict their barn footprints?
- How well can we predict dairy cattle herd size using barn footprint predictions?
- Can we use these results to identify potentially unpermitted dairy CAFOs?

## Data

All data required to reproduce the paper results are available on Hugging Face at [reglab/wisconsin-dairy-cafo](https://huggingface.co/datasets/reglab/wisconsin-dairy-cafo).

Model weights and training/inference code are available in the [cafo-segmentation](https://github.com/reglab/cafo-segmentation) repository.

## Setup

**Requirements:** Python 3.11

1. Clone this repository.

2. Copy `config/config_example.yml` to `config/config.yml` and update each path to match your local filesystem.

3. Download the data from [reglab/wisconsin-dairy-cafo](https://huggingface.co/datasets/reglab/wisconsin-dairy-cafo) (see below for directory structure).

4. Create a virtual environment and install dependencies:

```bash
uv venv --python 3.11 wicafo
uv pip install -r requirements.txt --python wicafo/bin/python3.11
source wicafo/bin/activate
```

Any environment manager targeting Python 3.11 works; replace the `uv` commands with your preferred tool.

## Expected data directory layout

After downloading from Hugging Face, your `data/` and `analysis_outputs/` directories should look like:

```
data/
├── Annotations/
│   ├── full_state_cf_annotations.geojson
│   └── ewg_region_train_labels.geojson
├── WDNR_permitted_CAFOs/
│   └── WDNR_CAFOs.geojson
├── ewg_AFOs_012022.geojson
├── permit_animal_type_records_*.csv
├── milk_producers.geojson
├── County_Boundaries_24K/
├── WI_urban_areas/
└── water_data/

analysis_outputs/
└── Clusters/
    ├── all_CF_clusters.csv
    ├── all_cf_clusters.geojson
    ├── four_band_clusters.csv
    ├── three_band_clusters.csv
    └── hyper_parameter_best_model_clusters.csv
```

Large ancillary datasets not included in the HF repository:
- **Wisconsin land parcels** (V8.0.0): [WI SCO](https://www.sco.wisc.edu/parcels/data/)
- **NAIP imagery**: stored on Google Cloud (GCP credentials required; contact the authors)
- **SNAPMAPs nutrient management layers**: [DATCP open data portal](https://gis-widatcp.opendata.arcgis.com/datasets/47ff7f962ad8415aa82938e9df392f21/about)

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
| `config/config_params.py` | Global constants (EPSG codes, AU factors, figure styling) |
| `config/config_example.yml` | Configuration template — copy to `config.yml` and edit paths |

## Citation

*Citation to be added upon publication.*
