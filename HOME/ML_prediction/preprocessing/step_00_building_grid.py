# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from scipy import ndimage
import argparse
from HOME.get_data_path import get_data_path
import scipy
import pickle
import os

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)


def road_grid(tile_size=512, res=0.3, overlap_rate=0, crs=25833, dilation=0):
    # Read the veg.geojson file
    building_path = (
        data_path / "raw/FKB_bygning" / "Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
    )
    with open(building_path, "rb") as f:
        buildings = pickle.load(f).to_crs(crs)
    bounds = buildings.total_bounds

    effective_tile_size = int(tile_size * (1 - overlap_rate))
    pixel_size = res * effective_tile_size

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

    geometries = buildings.geometry.to_list()
    mask = geometry_mask(
        geometries,
        transform=transform,
        out_shape=(num_grids_y, num_grids_x),
        invert=True,
        all_touched=True,
    )

    # Dilate the mask
    for i in range(dilation):
        mask = ndimage.binary_dilation(mask)

    # Create a DataFrame with the grid coordinates
    building_presence = pd.DataFrame(
        columns=np.arange(min_grid_x, max_grid_x),
        index=np.arange(max_grid_y, min_grid_y, -1),
        data=mask,
    )

    output_dir = (
        data_path
        / f"ML_prediction/prediction_mask/buildings/crs_{crs}/dilation_{dilation}"
        / f"overlap_{overlap_rate}/res_{res}/tile_{tile_size}"
    )
    os.makedirs(output_dir, exist_ok=True)
    # building_presence.to_csv(output_dir / f"prediction_mask_{res}_{grid_size}.csv")

    data_flat = building_presence.values.flatten()

    x_coords, y_coords = np.meshgrid(
        np.arange(min_grid_x, max_grid_x), np.arange(max_grid_y, min_grid_y, -1)
    )
    y_coords = y_coords.flatten()
    x_coords = x_coords.flatten()

    min_x = x_coords.min()
    min_y = y_coords.min()

    x_coords_sparse = (x_coords[data_flat] - min_x).astype(int)
    y_coords_sparse = (y_coords[data_flat] - min_y).astype(int)
    data_sparse = data_flat[data_flat].astype(int)

    mask_sparse = scipy.sparse.coo_matrix(
        (data_sparse, (y_coords_sparse, x_coords_sparse))
    )
    mask_sparse_csr = mask_sparse.tocsr()

    scipy.sparse.save_npz(
        output_dir / f"masksparse_{min_x}_{min_y}.npz",
        mask_sparse_csr,
    )


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--res", type=float, default=0.3)
    parser.add_argument("--overlap_rate", type=float, default=0)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--crs", type=int, default=25833)
    args = parser.parse_args()
    road_grid(
        tile_size=args.tile_size,
        res=args.res,
        overlap_rate=args.overlap_rate,
        crs=args.crs,
        dilation=args.dilate,
    )

# %%
