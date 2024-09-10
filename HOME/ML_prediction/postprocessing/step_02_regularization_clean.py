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


def process_project_tiles(tile_dir: str, output_dir: Path):
    """
    Process all images in the project directory.
    """

    def process_image(processed_img_path):
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
        return gdf, processed_img_path

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the project directory
    for processed_img_path in tqdm(glob.glob(str(tile_dir / "*.tif"))):
        gdf, processed_img_path = process_image(processed_img_path)

        # Save the GeoDataFrame to a pickle file. one file per large tile
        tile_name = Path(processed_img_path).stem
        pickle_file_path = output_dir / f"{tile_name}_geodata.pkl"
        with open(pickle_file_path, "wb") as f:
            pickle.dump(gdf, f)


# %%
# Main script

if __name__ == "__main__":
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
