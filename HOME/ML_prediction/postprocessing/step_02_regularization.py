# %%
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import geopandas as gpd
import os
from pathlib import Path
import glob
from tqdm import tqdm
import json
import rasterio
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon, shape, box
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime


from HOME.utils.project_coverage_area import (
    project_coverage_area,
)
from HOME.utils.project_paths import get_tiling_details, get_assembling_details
from HOME.get_data_path import get_data_path

# %%
root_dir = Path(__file__).parents[3]
data_path = get_data_path(root_dir)


# %%
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
    project_coverage: gpd.GeoDataFrame,
    geotiff_extend: dict,
    simplification_tolerance: int,
    buffer_distance: int,
    buffer_join_style: int,
    buffer_single_sided: bool,
    destination_crs=25833,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # Open the processed image with rasterio to access both data and metadata
    with rasterio.open(str(processed_img_path)) as src:
        # Read the first band
        myarray = src.read(1)
        # Extract polygons from the array
        mypoly = []
        for vec in rasterio.features.shapes(
            myarray, mask=myarray == 255, transform=src.transform
        ):
            polygon = shape(vec[0])
            # remove polygons under 2.5m2
            if polygon.area > 2.5:
                mypoly.append(polygon)

        # Filter and simplify polygons
        simplified_polygons = [
            polygon.simplify(simplification_tolerance, preserve_topology=True)
            for polygon in mypoly
        ]

        # Create a GeoDataFrame with the correct CRS
        gdf = gpd.GeoDataFrame({"geometry": simplified_polygons}, crs=src.crs)

        # Filter the GeoDataFrame manually
        filtered_polygons = [
            geom
            for geom in gdf["geometry"]
            if is_within_bounds(geom, src.width, src.height)
        ]
        gdf = gpd.GeoDataFrame({"geometry": filtered_polygons}, crs=src.crs)
        coverage = project_coverage.loc[
            (
                slice(geotiff_extend["grid_x_min"], geotiff_extend["grid_x_max"] - 1),
                slice(geotiff_extend["grid_y_min"] + 1, geotiff_extend["grid_y_max"]),
            ),
            :,
        ]
        # tile_coverage_area(tile_geom, project_coverage)
    return gdf.to_crs(destination_crs), coverage.to_crs(destination_crs)


def process_project_tiles(
    tile_dir: Path,
    output_dir: Path,
    project_name: str,
    simplification_tolerance: int,
    buffer_distance: int,
    buffer_join_style: int,
    buffer_single_sided: bool,
    geotiff_extends: dict,
    original_crs: int = 25833,
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
    project_coverage = project_coverage_area(project_name, crs=original_crs)

    # Process all images in the project directory
    for processed_img_path in tqdm(glob.glob(str(Path(tile_dir) / "*.tif"))):
        geotiff_extend = geotiff_extends[processed_img_path.split("/")[-1]]
        polygons, covered_area = process_image(
            processed_img_path,
            project_coverage,
            geotiff_extend,
            simplification_tolerance,
            buffer_distance,
            buffer_join_style,
            buffer_single_sided,
        )

        # Save the GeoDataFrame to a gjson. one file per large tile
        tile_name = Path(processed_img_path).stem

        # add an ID column:
        xy_grid = "_".join(tile_name.split("_")[-2:])
        polygons["ID"] = [f"{xy_grid}_{i}" for i in range(len(polygons))]
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
    geotiff_extends: dict,
    simplification_tolerance: int = 2,
    buffer_distance: int = 0.5,
    buffer_join_style: int = 3,
    buffer_single_sided: bool = True,
) -> None:

    assembly_metadata = get_assembling_details(assembly_id, data_path)

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
        / project_name
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
        geotiff_extends=geotiff_extends,
        original_crs=crs,
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

    simplification_tolerance: int = 0.5
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
