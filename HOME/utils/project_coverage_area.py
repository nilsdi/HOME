# %%
from pathlib import Path
import os
import scipy.sparse
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from HOME.get_data_path import get_data_path
from HOME.utils.get_project_metadata import get_project_geometry
import pandas as pd

root_dir = Path(__file__).parents[2]
data_dir = get_data_path(root_dir)


# %%
def project_coverage_area(
    project_name: str,
    res=0.3,
    tile_size=512,
    overlap_rate=0,
    crs=25833,
    dilation=0,
    prediction_type: str = "buildings",
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with the geographic area of prediction
    that is covered by the GeoDataFrame in the input file.

    Args:
        project_name: str, name of the project
        prediction_mask: Path, path to the prediction mask used (csv for tiles)

    Returns:
        covered_are: gpd.GeoDataFrame, the geographic are that we have predictions for
    """
    # Open the prediction mask
    folderpath = (
        data_dir
        / f"ML_prediction/prediction_mask/{prediction_type}/crs_{crs}/dilation_{dilation}"
        / f"overlap_{overlap_rate}/res_{res}/tile_{tile_size}"
    )
    filepaths = [file for file in os.listdir(folderpath) if file.endswith(".npz")]
    assert len(filepaths) == 1, "Several masks correspond to the specified parameters"
    filepath = filepaths[0]
    prediction_mask = scipy.sparse.load_npz(folderpath / filepath)
    parts = filepath.split("_")
    min_x = int(parts[-2])
    min_y = int(parts[-1].split(".")[0])

    # project area
    project_geometry = get_project_geometry(project_name).to_crs(crs)[0]

    # Create a GeoDataFrame for the mask, within the project area
    bounds = project_geometry.bounds

    grid_size_m = tile_size * res
    coordgrid_min_x = int(np.floor(bounds[0] / grid_size_m))
    coordgrid_max_x = int(np.ceil(bounds[2] / grid_size_m))
    coordgrid_min_y = int(np.floor(bounds[1] / grid_size_m))
    coordgrid_max_y = int(np.ceil(bounds[3] / grid_size_m))

    rectangles = []
    tile_coords = []

    for x_grid in range(coordgrid_min_x, coordgrid_max_x):
        for y_grid in range(coordgrid_max_y, coordgrid_min_y, -1):
            if prediction_mask[y_grid - min_y, x_grid - min_x]:
                # Coordinates top left corner in EPSG:25833 (meters)
                coordgrid_x_m = (x_grid) * grid_size_m
                coordgrid_y_m = (y_grid) * grid_size_m

                bbox = box(
                    coordgrid_x_m,
                    coordgrid_y_m - grid_size_m,
                    coordgrid_x_m + grid_size_m,
                    coordgrid_y_m,
                )

                if bbox.intersects(project_geometry):
                    rectangles.append(bbox.intersection(project_geometry))
                    tile_coords.append((x_grid, y_grid))

    coords_index = pd.MultiIndex.from_tuples(tile_coords, names=["x_grid", "y_grid"])
    covered_area = gpd.GeoDataFrame(geometry=rectangles, crs=crs, index=coords_index)

    return covered_area.sort_index()


# %%
if __name__ == "__main__":
    covered_area = project_coverage_area("trondheim_2023")

# %%
