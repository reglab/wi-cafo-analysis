"""
Functions for read/write snapmaps
"""
import geopandas as gpd
import os
from pathlib import Path
import geofeather
import yaml
import fiona
import config.config_params as cfg

def save_snapmaps_to_feather(raw_gdb, out_dir, crs=cfg.WI_EPSG):
    
    # Make directory out_dir if doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    print("Reading SNAPMAPS data...")
    layers = fiona.listlayers(raw_gdb)
    for layer in layers:
        feather_path = out_dir / f"{layer}.feather"

        # Read from geodatabase and save as Feather
        gdf = gpd.read_file(
            raw_gdb, layer=layer
        ).to_crs(crs)
        geofeather.to_geofeather(gdf,feather_path) 
        print(f"Read {layer} from GDB, n={gdf.shape[0]} and saved as Feather")

def load_snapmaps(feather=True, feather_dir=None, raw_gdb=None, crs=cfg.WI_EPSG, simplify_geometries=False, tolerance=10):
    """Returns a dict of snapmaps layers loaded as a gdb"""
  
    layers = fiona.listlayers(raw_gdb)

    snapmaps_data = {}
    print("Reading SNAPMAPS data...")

    if feather:
        for layer in layers:
            # Read from Feather
            layer_feather = geofeather.from_geofeather(feather_dir / f"{layer}.feather")
            gdf = gpd.GeoDataFrame(layer_feather, geometry="geometry", crs=crs)

            # Simplify geometries
            if simplify_geometries:
                gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)

            # Set a spatial index to make joins quicker
            if not gdf.sindex:
                gdf.sindex
            snapmaps_data[layer] = gdf
            print(f"Read {layer} from Feather, n={gdf.shape[0]}")
    
    return snapmaps_data

if __name__ == '__main__':

    with open(Path().resolve().parent / 'afo_vs_cafo/config/config.yml', 'r') as file:
        configs = yaml.safe_load(file)
        data_path = Path(configs['data_path'])
    
    save_snapmaps_to_feather(raw_gdb = data_path / "NM_590_CAFO_STATEWIDE.gdb", 
                                out_dir = data_path / "snapmaps_feather", crs=cfg.WI_EPSG)
    
    # Time how long the below function takes to run
    import time
    start = time.time()

    snapmaps =load_snapmaps(feather=True, feather_dir = data_path / "snapmaps_feather",
                   raw_gdb = data_path / "NM_590_CAFO_STATEWIDE.gdb", crs=cfg.WI_EPSG)
    print(snapmaps)
    print(snapmaps['SILURIAN_2_5'].head())

    
    print(f"Time taken: {time.time() - start}")
    

