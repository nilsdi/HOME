# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from scipy import ndimage
import argparse

def road_grid(grid_size=512, res=0.3):
    # Read the veg.geojson file
    root_dir = Path(__file__).parents[3]
    road_dir = root_dir / "data/raw/FKB_veg/Basisdata_0000_Norge_5973_FKB-Veg_FGDB.gdb"
    roads = gpd.read_file(road_dir, layer="fkb_veg_omrade")
    bounds = roads.total_bounds

    pixel_size = res * grid_size

    min_grid_x = int(np.floor(bounds[0] / pixel_size))  # minimum grid x-coordinate
    max_grid_x = int(np.ceil(bounds[2] / pixel_size))  # maximum grid x-coordinate
    min_grid_y = int(np.floor(bounds[1] / pixel_size))  # minimum grid y-coordinate
    max_grid_y = int(np.ceil(bounds[3] / pixel_size))  # maximum grid y-coordinate

    # Calculate the number of grids in x and y directions
    num_grids_x = max_grid_x - min_grid_x
    num_grids_y = max_grid_y - min_grid_y

    transform = from_bounds(
        min_grid_x * pixel_size,
        min_grid_y * pixel_size,
        max_grid_x * pixel_size,
        max_grid_y * pixel_size,
        num_grids_x,
        num_grids_y,
    )

    # Create an empty array of the same size as the GeoTIFF

    geometries = roads.geometry.to_list()
    mask = geometry_mask(
        geometries,
        transform=transform,
        out_shape=(num_grids_y, num_grids_x),
        invert=True,
        all_touched=True,
    )

    # Dilate the mask
    dilated_mask = ndimage.binary_dilation(mask)

    # Create a DataFrame with the grid coordinates
    road_presence = pd.DataFrame(
        columns=np.arange(min_grid_x, max_grid_x),
        index=np.arange(max_grid_y, min_grid_y, -1),
        data=dilated_mask,
    )

    road_presence.to_csv(
        root_dir / f"data/ML_prediction/prediction_mask/prediction_mask_{res}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=512)
    parser.add_argument("--res", type=float, default=0.3)
    args = parser.parse_args()
    road_grid(grid_size=args.grid_size, res=args.res)