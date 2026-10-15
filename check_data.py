#!/usr/bin/env python3
"""Report which input files generate_paper_results.py needs and which are missing.

    python check_data.py                  # full pipeline
    python check_data.py --skip-snapmaps  # without SNAPMAPS / slope layers
"""
import argparse
import sys

import config.config_params as cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-snapmaps", action="store_true")
    args = parser.parse_args()

    P = cfg.PATHS
    print(f"data_root: {P.data_root}")
    missing = P.missing_files(skip_snapmaps=args.skip_snapmaps)
    total = len(P.required_files(skip_snapmaps=args.skip_snapmaps))
    for m in missing:
        print(f"  MISSING  {m}")
    print(f"{total - len(missing)}/{total} required files present.")
    print("Optional: land parcels", "found" if P.parcels.exists() else f"not found ({P.parcels})",
          "| GCP credentials", "set" if P.gcp_cred_path else "not set")
    if missing:
        print("\nRun `python download_data.py` to fetch the missing files from Hugging Face.")
        sys.exit(1)


if __name__ == "__main__":
    main()
