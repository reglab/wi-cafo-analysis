# This script contains functions to download and process NAIP imagery from Google Cloud storage
# for use in visualization and analysis.

import os
from google.cloud import storage
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import yaml
from tqdm import tqdm
from shapely import Polygon
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import json
from datetime import datetime
from multiprocessing import Pool


def collect_neighboring_tiles(filename_list, tilesize=1024):
    """
    Given a list of NAIP double-tiled images, grab all neighboring NAIP double-tiled
      images within the same super-tile and county.
    """
    # Handle cases with a single filename not in list format
    if type(filename_list) == str:
        filename_list = [filename_list]

    final_list = []
    for filename in filename_list:
        final_list.append(filename)

        county = filename.split("_")[1]
        supertile = filename.split("_")[2]
        column_pixel = int(filename.split("_")[3])
        row_pixel = int(filename[:-5].split("_")[4])

        # Next photo above
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county, supertile, str(column_pixel), str(max(0, row_pixel - tilesize))
            )
        )

        # Next photo below
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county, supertile, str(column_pixel), str(max(0, row_pixel + tilesize))
            )
        )

        # Next photo left
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county, supertile, str(max(0, column_pixel - tilesize)), str(row_pixel)
            )
        )

        # Next photo right
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county, supertile, str(max(0, column_pixel + tilesize)), str(row_pixel)
            )
        )

        # Diagonals
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county,
                supertile,
                str(max(0, column_pixel + tilesize)),
                str(max(0, row_pixel - tilesize)),
            )
        )
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county,
                supertile,
                str(max(0, column_pixel + tilesize)),
                str(max(0, row_pixel + tilesize)),
            )
        )
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county,
                supertile,
                str(max(0, column_pixel - tilesize)),
                str(max(0, row_pixel - tilesize)),
            )
        )
        final_list.append(
            "WI_{}_{}_{}_{}.jpeg".format(
                county,
                supertile,
                str(max(0, column_pixel - tilesize)),
                str(max(0, row_pixel + tilesize)),
            )
        )

    return list(set(final_list))


def download_transform(filename_list: list,
                       bands: list = [1, 2, 3],
                       bounds: gpd.GeoSeries = None,
                        target_crs_epsg=3071,
                        verbose: bool=False):
    """
    Function to download one or more NAIP tiles and transform them to a specified CRS.
    Requires GCP credentials in .json form to be stored in a location specified
    in the user's local config.yml file.
    Note: this function was initially written such that each tile would be reprojected
    before creating a mosaic (merging), but this created dotted lines along the tile 
    seams for unknown reasons. The fix I found was to create a mosaic (merge the tiles)
    before reprojecting.
    Inputs:
        - filename_list: str list, names of .jpeg NAIP tiles
        - bands: list of bands to download (default [1, 2, 3])
        - bounds: GeoSeries, optional, polygon(s) to bound the image
        - target_crs_epsg: int, EPSG CRS to transform tile to (default 3071)
        - verbose: bool, whether to produce verbose output
    Returns:
        rasterio MemoryFile containing the downloaded and transformed .tif.
    """
    # If GCP credentials are not loaded yet, reload them
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        with open(Path().resolve().parent / "config/config.yml", "r") as file:
            configs = yaml.safe_load(file)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = configs["gcp_cred_path"]
    # Handle cases with a single filename not in list format
    if type(filename_list) == str:
        filename_list = [filename_list]
    reprojected_rasters = []
    if verbose:
        print('Downloading tiles...')
    for filename in filename_list:
        try:
            # Open the GeoTIFF file from GCP
            if 4 in bands:
                file_path = "gs://image-hub/NAIP-RGB-2022/four_band/WI/" \
                + filename[0 : filename.find("0")] \
                + "/initial_tiff/" + filename
            else:
                file_path = "gs://image-hub/NAIP-RGB-2022/WI/" \
                    + filename[3 : filename.find("_", 3)] \
                    + "/tiled_tif/" \
                    + filename[0 : filename.find(".jpeg")] \
                    + ".tif"
            src = rasterio.open(file_path)
            reprojected_rasters.append(src)
        except:
            if verbose:
                print("Error reading file: " + filename)
            continue
    
    if verbose:
        print('Merging tiles...')
    # Merge rasters
    merged, transform = rasterio.merge.merge(reprojected_rasters)

    # Create an in-memory dataset for the merged data
    merged_data = rasterio.MemoryFile().open(
        driver="GTiff",
        width=merged.shape[2],
        height=merged.shape[1],
        count=len(bands),
        dtype=merged.dtype,
        nodata=0,
        crs=4326,
        transform=transform
    )
    for i in range(0, len(bands)):
        merged_data.write_band(i+1, merged[bands[i]-1])
    
    if verbose:
        print('Calculating transform...')
    # Define the transform parameters for the target CRS
    transform, width, height = calculate_default_transform(
        4326, target_crs_epsg, merged_data.width, merged_data.height, *merged_data.bounds
    )
    
    # Create an in-memory dataset for the reprojected data
    reproj_data = rasterio.MemoryFile().open(
        driver="GTiff",
        width=width,
        height=height,
        count=len(bands),
        dtype=merged.dtype,
        nodata=0,
        crs=f"EPSG:{target_crs_epsg}",
        transform=transform
    )

    if verbose:
        print('Reprojecting...')
    # Perform the reprojection
    for i in range(0, len(bands)):
        reproject(
            source=rasterio.band(merged_data, i+1),
            destination=rasterio.band(reproj_data, i+1),
            src_transform=merged_data.transform,
            src_crs=merged_data.crs,
            dst_transform=transform,
            dst_crs=target_crs_epsg
        )    
    
    # If bounds are supplied, mask the image data to fit within the bounds
    if bounds is not None:
        bounds = bounds.to_crs(target_crs_epsg)
        reproj_data, transform = rasterio.mask.mask(reproj_data, bounds, crop=True, filled=False)
    else:
        reproj_data = reproj_data.read()
        
    return reproj_data, transform


def get_image_bound(file_name: str, target_crs_epsg=3071, verbose: bool=False):
    """Creates and returns the bounding polygon for the NAIP image tile.
    Args:
        file_name: image tile to analyze
        target_crs_epsg: target CRS
        four_band: whether the image is four-band (default False)
        verbose: whether to produce verbose output
    Returns:
        Bounding polygon.
    """
    # Heuristic for whether image is a small three-band tile or large
    # four-band initial tiff
    four_band = file_name[0 : 2] != "WI"
    try:
        if four_band:
            file_path = "gs://image-hub/NAIP-RGB-2022/four_band/WI/" \
            + file_name[0 : file_name.find("0")] \
            + "/initial_tiff/" + file_name
        else:
            file_path = "gs://image-hub/NAIP-RGB-2022/WI/" \
                + file_name[3 : file_name.find("_", 3)] \
                + "/tiled_tif/" \
                + file_name[0 : file_name.find(".jpeg")] \
                + ".tif"
        with rasterio.open(file_path) as src:
            # Define the transform parameters for the target CRS
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs_epsg, src.width, src.height, *src.bounds
            )
        bounds = box(*src.bounds)
        bounding_box = Polygon(bounds)
    except:
        if verbose:
            print("Error with file: " + file_name)
        return

    return bounding_box

def create_image_bound_map(counties: gpd.GeoDataFrame, target_crs_epsg=3071,
                           four_band: bool=False):
    """Creates a GeoDataFrame of NAIP tile bounding polygons for the specified counties.
    Args:
        counties: GeoDataframe of county polygons
        target_crs_epsg: target CRS
        four_band: whether the images are four-band (default False)
    Returns:
        GeoDataFrame of NAIP tile bounding polygons.
    """
    # This is the list of county names as they appear in the image-hub NAIP 2022 WI folder
    county_names = [
        "Adams",
        "Ashland",
        "Barron",
        "Bayfield",
        "Brown",
        "Buffalo",
        "Burnett",
        "Calumet",
        "Chippewa",
        "Clark",
        "Columbia",
        "Crawford",
        "Dane",
        "Dodge",
        "Door",
        "Douglas",
        "Dunn",
        "Eau Claire",
        "Florence",
        "Fond du Lac",
        "Forest",
        "Grant",
        "Green Lake",
        "Green",
        "Iowa",
        "Iron",
        "Jackson",
        "Jefferson",
        "Juneau",
        "Kenosha",
        "Kewaunee",
        "La Crosse",
        "Lafayette",
        "Langlade",
        "Lincoln",
        "Manitowoc",
        "Marathon",
        "Marinette",
        "Marquette",
        "Menominee",
        "Milwaukee",
        "Monroe",
        "Oconto",
        "Oneida",
        "Outagamie",
        "Ozaukee",
        "Pepin",
        "Pierce",
        "Polk",
        "Portage",
        "Price",
        "Racine",
        "Richland",
        "Rock",
        "Rusk",
        "Sauk",
        "Sawyer",
        "Shawano",
        "Sheboygan",
        "St. Croix",
        "Taylor",
        "Trempealeau",
        "Vernon",
        "Vilas",
        "Walworth",
        "Washburn",
        "Washington",
        "Waukesha",
        "Waupaca",
        "Waushara",
        "Winnebago",
        "Wood",
    ]
    all_NAIP_tiles = []
    print("Collecting tile filenames...")
    if four_band:
        for county in tqdm(county_names):
            all_NAIP_tiles.extend(get_all_NAIP_tiles(county, four_band=True))
        # Clean up garbage files (e.g., 'completed.txt')
        all_NAIP_tiles = [tile for tile in all_NAIP_tiles if tile[-4:] == ".tif"]
    else:
        for county in tqdm(county_names):
            all_NAIP_tiles.extend(get_all_NAIP_tiles(county))
        # Clean up garbage files (e.g., 'completed.txt')
        all_NAIP_tiles = [tile for tile in all_NAIP_tiles if tile[-5:] == ".jpeg"]
    
    print("Creating image bound map...")
    with Pool() as pool:
            image_bounds = list(tqdm(pool.imap(get_image_bound, all_NAIP_tiles), total=len(all_NAIP_tiles)))
    
    image_bound_map = gpd.GeoDataFrame(data={'filename': pd.Series(all_NAIP_tiles),
                                             'geometry': gpd.GeoSeries(image_bounds)}, crs=4326)
    image_bound_map = image_bound_map.to_crs(target_crs_epsg)
    
    
    print("Labeling masked tiles...")
    if four_band:
        image_bound_map['origin_county'] = image_bound_map['filename'].apply(lambda x: x[0 : x.find("0")])
    else:
        image_bound_map['origin_county'] = image_bound_map['filename'].apply(lambda x: x[3 : x.find("_", 3)])
        

    image_bound_map_joined = image_bound_map.sjoin(counties)
    image_bound_map_joined = image_bound_map_joined[image_bound_map_joined['origin_county'] == image_bound_map_joined['COUNTY_NAM']]
    image_bound_map['all_black'] = ~image_bound_map['filename'].isin(image_bound_map_joined['filename'])

    image_bound_map.drop('origin_county', axis=1, inplace=True)

    return image_bound_map


def get_all_NAIP_tiles(county: str, bucket_name="image-hub",
                       four_band: bool=False):
    """Lists all the NAIP tile images for the specified county.
    Args:
        county: county name
        bucket_name: name of the GCP bucket
        four_band: whether the images are four-band (default False)
    Returns:
        list of file names (as .jpeg, not .tif)
    """

    storage_client = storage.Client('law-cafo')

    # Note: Client.list_blobs requires at least package version 1.17.0.
    if four_band:
        prefix="NAIP-RGB-2022/four_band/WI/" + county + "/initial_tiff/"
    else:
        prefix="NAIP-RGB-2022/WI/" + county + "/tiled_tif/"
    
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix)

    filename_list = []
    # Note: The call returns a response only when the iterator is consumed.
    for blob in blobs:
        if four_band:
            filename_list.append(blob.name[blob.name.rindex("/") + 1 :])
        else:
            filename_list.append(
                blob.name[blob.name.rindex("/") + 1 :].replace(".tif", ".jpeg")
            )

    return filename_list

def georeference_cf_annotations(annotations_path: str,
                                geotiffs_path: str="gs://image-hub/NAIP-RGB-2022/WI/",
                                border_data: pd.DataFrame=None,
                                  save_path: str=None, target_crs=3071):
    """Extracts and georeferences individual annotations from cloud factory output .json files.
    Args:
        annotations_path: path to the folder containing .json annotation files
        geotiff_path: path to folder containing georeferenced .tif files
        border_data: Optional, dataframe containing width of left, right, top, and bottom whitespace
            of each jpeg image originally submitted to CF. To be used if the jpegs contain white border
            (not advised) as a result of using matplotlib to create them.
        save_path: Optional, path where to save output .geojson file
        target_crs: CRS in which to return the output .geojson file, default EPSG:3071
    Returns:
        GeoDataFrame of annotation polygons in the target CRS.
    """
    geodata_all = gpd.GeoDataFrame()
    file_names = os.listdir(annotations_path)
    file_names = [i for i in file_names if i not in ['legend.json', '.DS_Store']]

    # If GCP credentials are not loaded yet, reload them
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        with open(Path().resolve().parent / "config/config.yml", "r") as file:
            configs = yaml.safe_load(file)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = configs["gcp_cred_path"]
    
    progress = tqdm(file_names)

    for file_name in progress:
        progress.set_postfix_str(file_name)
        # Load new annotations
        image = open(annotations_path / file_name)
        image_labels = json.load(image)
        # Annotated image width
        original_width = image_labels['metadata']['system']['width']
        original_height = image_labels['metadata']['system']['height']

        # Locate GeoTIFF for georeferencing
        if geotiffs_path == 'gs://image-hub/NAIP-RGB-2022/WI/':
            # If file_name has old format, append "WI_"
            if file_name[0:2] != 'WI':
                file_name = 'WI_' + file_name
            geotiff_path = (geotiffs_path
                        + file_name[3 : file_name.find("_", 3)]
                        + "/tiled_tif/"
                        + file_name[0 : file_name.find(".json")]
                        + ".tif")
        else: 
            geotiff_path = geotiffs_path + file_name[0 : file_name.find(".json")] + ".tif"
        
        # Load image file to determine transform
        with rasterio.open(geotiff_path) as src:
            # Define the transform parameters for the target CRS
            transform, width, height = calculate_default_transform(
                src.crs, 4326, src.width, src.height, *src.bounds
            )

        # Collect all new annotations
        for annotation in image_labels['annotations']:
            if 'coordinates' in annotation.keys():
                if len(annotation['coordinates']) > 0:
                    updatedAt = annotation['updatedAt']
                    # If annotation coords are a nested list, unnest by one level
                    if len(annotation['coordinates'])==1:
                        coords = annotation['coordinates'][0]
                    x_coords = []
                    y_coords = []
                    if border_data is not None:
                        border = border_data[border_data['file_name']==file_name.replace('.json', '.jpeg').replace('buffered_', '')]
                        for point in coords:
                            x_coords.append((point["x"]-border['left_offset'])*width/(original_width-border['left_offset']-border['right_offset']))
                            y_coords.append((point["y"]-border['top_offset'])*height/(original_height-border['top_offset']-border['bottom_offset']))
                    else:
                        for point in coords:
                            x_coords.append(point['x'])
                            y_coords.append(point['y'])
                    # Transform from pixel to lat/long coordinates
                    transformed_coords = rasterio.transform.xy(transform, y_coords, x_coords)
                    transformed_points = [(transformed_coords[0][i], transformed_coords[1][i]) for i in range(0, len(transformed_coords[0]))]
                    polygon = Polygon(transformed_points)
                    # Save as geodataframe
                    geodata = gpd.GeoDataFrame(data={'jpeg_name': [file_name[0 : file_name.find(".json")] + '.jpeg'],
                                                    'geometry': gpd.GeoSeries(polygon),
                                                    'upDatedAt': updatedAt}, crs=3071)
                    geodata_all = pd.concat([geodata_all, geodata], axis=0)
    
    geodata_all = geodata_all.to_crs(target_crs)
    geodata_all['upDatedAt'] = geodata_all['upDatedAt'].apply(
        lambda x: datetime.strptime(x.replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S.%f'))
    if save_path is not None:
        geodata_all.to_file(save_path, driver='GeoJSON')

    return(geodata_all)


if __name__ == "__main__":
    with open(Path().resolve().parent / "afo_vs_cafo/config/config.yml", "r") as file:
        configs = yaml.safe_load(file)
    analysis_output_path = Path(configs["analysis_output_path"])
    data_path = Path(configs['data_path'])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = configs["gcp_cred_path"]

    # Load county boundary shapefile for determining masked tiles
    counties = gpd.read_file(data_path / 'County_Boundaries_24K/County_Boundaries_24K.shp')
    counties['COUNTY_NAM'] = counties['COUNTY_NAM'].apply(lambda x: 'St. Croix' if x == 'Saint Croix' else x)

    image_bound_map = create_image_bound_map(counties)
    
    image_bound_map.to_file(
        analysis_output_path / "image_bound_map.geojson", driver="GeoJSON"
    )
