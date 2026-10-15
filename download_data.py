#!/usr/bin/env python3
"""Download the analysis inputs from Hugging Face into the configured data root.

    python download_data.py                   # everything needed by generate_paper_results.py
    python download_data.py --skip-snapmaps   # omit SNAPMAPS layers + slope raster (~2.5 GB)

Data root is `data_root` in config/config.yml (default ./data) or $WI_CAFO_DATA_ROOT.
Re-running only fetches files that are missing or changed.
"""
import argparse

from huggingface_hub import snapshot_download

import config.config_params as cfg

REPO_ID = "reglab/wisconsin-dairy-cafo"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-snapmaps", action="store_true",
                        help="Do not download SNAPMAPS layers and the slope raster")
    args = parser.parse_args()

    P = cfg.PATHS
    ignore = []
    if args.skip_snapmaps:
        ignore += ["input_data/geospatial/snapmaps/*", "input_data/geospatial/dem/*"]

    print(f"Downloading {REPO_ID} -> {P.data_root}")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=P.data_root,
        ignore_patterns=ignore or None,
    )

    missing = P.missing_files(skip_snapmaps=args.skip_snapmaps)
    if missing:
        print("\nDownloaded, but these required files are still missing:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)
    print("\nAll required input files are present. Run: python generate_paper_results.py")


if __name__ == "__main__":
    main()
