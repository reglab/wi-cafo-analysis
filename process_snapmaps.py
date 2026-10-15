"""
Functions for read/write snapmaps
"""
import geopandas as gpd
import os
from pathlib import Path
import geofeather
import fiona
import config.config_params as cfg


def save_snapmaps_to_feather(raw_gdb, out_dir, crs=cfg.WI_EPSG, layers=None):
    """Convert layers of the raw DATCP SNAPMAPS geodatabase (NM_590_CAFO_STATEWIDE.gdb,
    https://gis-widatcp.opendata.arcgis.com/) into the per-layer .feather files that
    load_snapmaps() reads. Defaults to the layers the analysis uses."""
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    layers = layers or cfg.PATHS.SNAPMAPS_LAYERS

    print("Reading SNAPMAPS data...")
    for layer in layers:
        gdf = gpd.read_file(raw_gdb, layer=layer).to_crs(crs)
        geofeather.to_geofeather(gdf, out_dir / f"{layer}.feather")
        print(f"Read {layer} from GDB, n={gdf.shape[0]} and saved as Feather")


def load_snapmaps(feather_dir=None, layers=None, crs=cfg.WI_EPSG, simplify_geometries=False, tolerance=10):
    """Returns a dict {layer name: GeoDataFrame} of SNAPMAPS layers read from feather files."""
    feather_dir = Path(feather_dir or cfg.PATHS.snapmaps_dir)
    layers = layers or cfg.PATHS.SNAPMAPS_LAYERS

    snapmaps_data = {}
    print("Reading SNAPMAPS data...")
    for layer in layers:
        layer_feather = geofeather.from_geofeather(feather_dir / f"{layer}.feather")
        gdf = gpd.GeoDataFrame(layer_feather, geometry="geometry", crs=crs)

        if simplify_geometries:
            gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)

        # Set a spatial index to make joins quicker
        if not gdf.sindex:
            gdf.sindex
        snapmaps_data[layer] = gdf
        print(f"Read {layer} from Feather, n={gdf.shape[0]}")

    return snapmaps_data


if __name__ == '__main__':
    # Rebuild the feather layers from the raw geodatabase:
    #   python process_snapmaps.py /path/to/NM_590_CAFO_STATEWIDE.gdb
    import sys
    save_snapmaps_to_feather(sys.argv[1], cfg.PATHS.snapmaps_dir)
