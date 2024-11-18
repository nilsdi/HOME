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
from shapely.geometry import Polygon, MultiPolygon, shape, box
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


from HOME.utils.project_coverage_area import (
    project_coverage_area,
)
from HOME.utils.project_paths import get_tiling_details

# %%
root_dir = Path(__file__).parents[3]
data_path = root_dir / "data"


# Exclude polygons that have at least one vertex on the edge (within two pixels) of the image
def is_within_bounds(geom, width, height):
    if isinstance(geom, Polygon):
        coords = geom.exterior.coords
        return all(
            2 <= coord[0] < width - 2 and 2 <= coord[1] < height - 2 for coord in coords
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


def process_image(
    processed_img_path: Path,
    project_geometry: gpd.GeoDataFrame,
    simplification_tolerance: int,
    buffer_distance: int,
    buffer_join_style: int,
    buffer_single_sided: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
            polygon.simplify(simplification_tolerance, preserve_topology=True)
            for polygon in mypoly
        ]
        rounded_polygons = [
            polygon.buffer(
                buffer_distance,
                join_style=buffer_join_style,
                single_sided=buffer_single_sided,
            )
            for polygon in simplified_polygons
        ]

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame({"geometry": rounded_polygons}, crs=src.crs)

        # Filter the GeoDataFrame manually
        filtered_polygons = [
            geom
            for geom in gdf["geometry"]
            if is_within_bounds(geom, src.width, src.height)
        ]
        gdf = gpd.GeoDataFrame({"geometry": filtered_polygons}, crs=src.crs)
        tile_geom = box(*src.bounds)
        coverage = tile_geom.intersection(project_geometry)
        coverage.crs = src.crs
        # tile_coverage_area(tile_geom, project_coverage)
    return gdf, coverage


def process_project_tiles(
    tile_dir: Path,
    output_dir: Path,
    project_name: str,
    simplification_tolerance: int,
    buffer_distance: int,
    buffer_join_style: int,
    buffer_single_sided: bool,
    crs: int,
) -> None:
    """
    Process all images in the tile directory and save a
    GeoDataFrame with the polygons to a pickle file (one per tile).
    Also save the output dir to the log, and add a metadata
    file detailing the geographic area of prediction that
    is covered by each GeoDataFrame.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    project_coverage = project_coverage_area(project_name, crs=crs)
    project_geometry = gpd.GeoSeries(project_coverage.unary_union, crs=crs)

    # Process all images in the project directory
    for processed_img_path in tqdm(glob.glob(str(Path(tile_dir) / "*.tif"))):
        polygons, covered_area = process_image(
            processed_img_path,
            project_geometry,
            simplification_tolerance,
            buffer_distance,
            buffer_join_style,
            buffer_single_sided,
        )
        # add an ID column:
        polygons["ID"] = range(len(polygons))
        # Save the GeoDataFrame to a gjson. one file per large tile
        tile_name = Path(processed_img_path).stem

        polygon_file_name = output_dir / f"polygons_{tile_name}.fgb"

        # save the polygons to a gjson file
        polygons.to_file(polygon_file_name, engine="pyogrio")

        covered_area_file_name = output_dir / "coverage" / f"coverage_{tile_name}.fgb"
        # save the covered area to a gjson file
        covered_area.to_file(covered_area_file_name, engine="pyogrio")
        """
        polygon_pickle_file_path = output_dir / f"polygons_{tile_name}_gdf.pkl"
        with open(polygon_pickle_file_path, "wb") as f:
            pickle.dump(polygons, f)
        area_pickle_file_path = output_dir / f"area_{tile_name}_gjson.pkl"
        with open(area_pickle_file_path, "wb") as f:
            pickle.dump(covered_area, f)
        """
    return


def regularize(
    project_name: str,
    assembly_id: int,
    simplification_tolerance: int = 5,
    buffer_distance: int = 1,
    buffer_join_style: int = 3,
    buffer_single_sided: bool = True,
) -> None:

    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "r"
    ) as file:
        assembly_log = json.load(file)
        assembly_metadata = assembly_log[str(assembly_id)]

    with open(data_path / "metadata_log/polygon_gdfs.json", "r") as file:
        polygon_gdfs_log = json.load(file)

    highest_polygon_key = int(max([int(key) for key in polygon_gdfs_log.keys()]))
    polygon_gdf_key = highest_polygon_key + 1

    assert assembly_metadata["project_name"] == project_name, "Project name mismatch."

    prediction_id = assembly_metadata["prediction_id"]
    tile_id = assembly_metadata["tile_id"]
    tile_directory = assembly_metadata["tile_directory"]

    tiling_detail = get_tiling_details(tile_id)
    crs = tiling_detail["crs"]

    output_dir = (
        data_path
        / f"ML_prediction/polygons"
        / f"tiles_{tile_id}"
        / f"prediction_{prediction_id}"
        / f"assembly_{assembly_id}"
        / f"polygons_{polygon_gdf_key}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "coverage", exist_ok=True)

    process_project_tiles(
        tile_directory,
        output_dir,
        project_name,
        simplification_tolerance,
        buffer_distance,
        buffer_join_style,
        buffer_single_sided,
        crs=crs,
    )

    polygon_gdfs_log[polygon_gdf_key] = {
        "tile_id": tile_id,
        "prediction_id": prediction_id,
        "assembly_id": assembly_id,
        "project_name": project_name,
        "gdf_directory": str(output_dir),
        "gdf_coverage_directory": str(output_dir / "coverage"),
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "simplication_tolerance": simplification_tolerance,
        "buffer_distance": buffer_distance,
        "buffer_join_style": buffer_join_style,
        "buffer_single_sided": buffer_single_sided,
    }

    with open(data_path / "metadata_log/polygon_gdfs.json", "w") as file:
        json.dump(polygon_gdfs_log, file, indent=4)

    print("Polygonization complete.")

    return polygon_gdf_key


# %%

if __name__ == "__main__":
    # Set the paths

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
    highest_polygon_key = int(max([int(key) for key in polygon_gdfs_log.keys()]))
    polygon_gdf_key = highest_polygon_key + 1

    # how to process:

    simplification_tolerance: int = 5
    buffer_distance: int = 1
    buffer_join_style: int = 3
    buffer_single_sided: bool = True

    for assembly_id in assemblies:
        assembly_metadata = assembly_log[str(assembly_id)]
        project_name = assembly_metadata["project_name"]
        prediction_id = assembly_metadata["prediction_id"]
        tile_id = assembly_metadata["tile_id"]
        tile_directory = assembly_metadata["tile_directory"]

        output_dir = (
            data_path
            / f"ML_prediction/polygons"
            / f"tiles_{tile_id}"
            / f"prediction_{prediction_id}"
            / f"assembly_{assembly_id}"
            / f"polygons_{polygon_gdf_key}"
        )
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "coverage", exist_ok=True)

        process_project_tiles(
            tile_directory,
            output_dir,
            project_name,
            simplification_tolerance,
            buffer_distance,
            buffer_join_style,
            buffer_single_sided,
        )

        polygon_gdfs_log[polygon_gdf_key] = {
            "tile_id": tile_id,
            "prediction_id": prediction_id,
            "assembly_id": assembly_id,
            "project_name": project_name,
            "gdf_directory": str(output_dir),
            "gdf_coverage_directory": str(output_dir / "coverage"),
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "simplication_tolerance": simplification_tolerance,
            "buffer_distance": buffer_distance,
            "buffer_join_style": buffer_join_style,
            "buffer_single_sided": buffer_single_sided,
        }

        with open(data_path / "metadata_log/polygon_gdfs.json", "w") as file:
            json.dump(polygon_gdfs_log, file)

        for key in polygon_gdfs_log.keys():
            print(
                f"{key} (type: {type(key)}): polygon_gdfs_log[key] (type: {type(polygon_gdfs_log[key])})"
            )
    print("Processing complete.")
# %%
# Main script

# %%
