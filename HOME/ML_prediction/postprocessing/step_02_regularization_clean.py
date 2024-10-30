# %%
from osgeo import gdal
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from shapely.geometry import shape
import geopandas as gpd
import os
from shapely.ops import unary_union
from pathlib import Path
import pandas as pd
import pickle
import glob
import re
from tqdm import tqdm
import json
import folium
import numpy as np
import math
from rasterio.warp import transform_bounds
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, shape
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


# %%
def process_image(
    processed_img_path: Path, project_coverage: gpd.GeoJSON
) -> tuple[gpd.GeoDataFrame, gpd.GeoJSON]:
    # Open the processed image with rasterio to access both data and metadata
    with rasterio.open(str(processed_img_path)) as src:
        # Read the first band
        myarray = src.read(1)

        # Extract polygons from the array
        mypoly = [
            shape(vec[0])
            for vec in rasterio.features.shapes(myarray, transform=src.transform)
        ]

        # Filter and simplify polygons
        simplified_polygons = [
            polygon.simplify(5, preserve_topology=True) for polygon in mypoly
        ]
        rounded_polygons = [
            polygon.buffer(1, join_style=3, single_sided=True)
            for polygon in simplified_polygons
        ]

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame({"geometry": rounded_polygons}, crs=src.crs)

        # Exclude polygons that have at least one vertex on the edge (within two pixels) of the image
        def is_within_bounds(geom, width, height):
            if isinstance(geom, Polygon):
                coords = geom.exterior.coords
                return all(
                    2 <= coord[0] < width - 2 and 2 <= coord[1] < height - 2
                    for coord in coords
                )
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:
                    coords = list(polygon.exterior.coords)
                    if not all(
                        2 <= coord[0] < width - 2 and 2 <= coord[1] < height - 2
                        for coord in coords
                    ):
                        return False
                return True
            else:
                return False

        # Filter the GeoDataFrame manually
        filtered_polygons = [
            geom
            for geom in gdf["geometry"]
            if is_within_bounds(geom, src.width, src.height)
        ]
        gdf = gpd.GeoDataFrame({"geometry": filtered_polygons}, crs=src.crs)
        tile_geom = box(*src.bounds)
        coverage = tile_coverage_area(tile_geom, project_coverage)
    return gdf, coverage


def tile_coverage_area(
    tile_geometry: Polygon, project_geometry: gpd.GeoJSON
) -> gpd.GeoJSON:
    """
    Create a GeoDataFrame with the geographic area of prediction
    that is covered within a tile.

    Args:
        tile_geometry: Polygon, geometry of the tile
        project_geometry: gpd.GeoJSON, geometry of the project

    Returns:
        covered_area: gpd.GeoJSON, the geographic are that we have predictions for
    """
    # Get the intersection of the tile and the project geometry
    covered_area = tile_geometry.intersection(project_geometry)
    return covered_area


def process_project_tiles(tile_dir: Path, output_dir: Path, project_name: str) -> None:
    """
    Process all images in the tile directory and save a
    GeoDataFrame with the polygons to a pickle file (one per tile).
    Also save the output dir to the log, and add a metadata
    file detailing the geographic area of prediction that
    is covered by each GeoDataFrame.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    project_coverage = 0

    # Process all images in the project directory
    for processed_img_path in tqdm(glob.glob(str(tile_dir / "*.tif"))):
        polygons, covered_area = process_image(processed_img_path, project_coverage)
        # add an ID column:
        polygons["ID"] = range(len(polygons))
        # Save the GeoDataFrame to a gjson. one file per large tile
        tile_name = Path(processed_img_path).stem

        polygon_file_name = output_dir / f"polygons_{tile_name}.gjson"
        # save the polygons to a gjson file
        polygons.to_file(polygon_file_name, driver="GeoJSON")

        covered_area_file_name = output_dir / f"covered_area_{tile_name}.gjson"
        # save the covered area to a gjson file
        covered_area.to_file(covered_area_file_name, driver="GeoJSON")
        """
        polygon_pickle_file_path = output_dir / f"polygons_{tile_name}_gdf.pkl"
        with open(polygon_pickle_file_path, "wb") as f:
            pickle.dump(polygons, f)
        area_pickle_file_path = output_dir / f"area_{tile_name}_gjson.pkl"
        with open(area_pickle_file_path, "wb") as f:
            pickle.dump(covered_area, f)
        """
    return


# %%

if __name__ == "__main__":
    # Set the paths
    root_dir = Path(__file__).parents[3]
    data_path = root_dir / "data"

    # Parameters
    assemblies = [
        30001,
    ]

    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "r"
    ) as file:
        assembly_log = json.load(file)

    with open(data_path / "metadata_log/polygon_gdfs.json", "r") as file:
        polygon_gdfs_log = json.load(file)
    highest_polygon_key = max([int(key) for key in polygon_gdfs_log.keys()])
    polygon_gdf_key = highest_polygon_key + 1

    for assembly_id in assemblies:
        assembly_metadata = assembly_log[str(assembly_id)]
        project_name = assembly_metadata["project_name"]
        prediction_id = assembly_metadata["prediction_id"]
        tile_directory = assembly_metadata["tile_directory"]

        output_dir = (
            data_path
            / f"ML_prediction/polygons"
            / get_download_str(download_id)
            / f"prediction_{prediction_id}"
            / f"tiling_{assembly_id}"
            / f"polygons_{polygon_gdf_key}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # TODO: add more paramters to the processing and write it in dictionary
        process_project_tiles(tile_directory, output_dir)

        # TODO: dump the metadata to the log

    print("Processing complete.")
# %%
# Main script

if __name__ == "__main__1":
    # Set the paths
    root_dir = Path(__file__).parents[3]

    # Parameters
    project_list = [
        # "trondheim_2019",
        "trondheim_kommune_2020",
        "trondheim_kommune_2021",
        "trondheim_kommune_2022",
    ]

    # Get necessary data
    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, "r") as file:
        project_dict = json.load(file)

    for project_name in project_list:
        # Get the resolution and other details
        resolution = project_dict[project_name]["resolution"]
        compression_name = project_dict[project_name]["compression_name"]
        compression_value = project_dict[project_name]["compression_value"]

        # Path to the reassembled tiles
        tile_dir = (
            root_dir / f"data/ML_prediction/large_tiles/res_{resolution}/{project_name}"
        )
        # Output directory for pickled GeoDataFrames with timestamp
        timestamp = datetime.now().strftime("%m%d%Y")
        output_dir = (
            root_dir
            / f"data/ML_prediction/polygons/{project_name}/{project_name}_ran_{timestamp}"
        )

        # Run the function
        process_project_tiles(tile_dir, output_dir)

    print("Processing complete.")
# %%
