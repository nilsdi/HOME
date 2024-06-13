import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from pathlib import Path


def road_grid(grid_size=512, res=0.3):
    # Read the veg.geojson file
    root_dir = str(Path(__file__).parents[3])
    road_dir = root_dir / "data/raw/FKB_veg/Basisdata_0000_Norge_5973_FKB-Veg_FGDB.gdb"
    roads = gpd.read_file(road_dir)
    bounds = roads.total_bounds

    min_grid_x = np.floor(bounds[0] / res)  # minimum grid x-coordinate
    max_grid_x = np.ceil(bounds[2] / res)  # maximum grid x-coordinate
    min_grid_y = np.floor(bounds[1] / res)  # minimum grid y-coordinate
    max_grid_y = np.ceil(bounds[3] / res)  # maximum grid y-coordinate

    # Calculate the number of grids in x and y directions
    num_grids_x = max_grid_x - min_grid_x
    num_grids_y = max_grid_y - min_grid_y

    # Create an empty dataframe with grid coordinates as rows and columns
    road_presence = pd.DataFrame(
        False, index=range(num_grids_y), columns=range(num_grids_x), dtype=bool
    )

    # Iterate over the roads and fill the dataframe with True for each grid containing a road
    # Iterate over each cell in the grid
    for y in range(min_grid_y, num_grids_x):
        for x in range(min_grid_x, num_grids_y):
            # Create a box for the current cell
            cell = box(
                x * grid_size * res + roads.bounds["minx"].min(),
                y * grid_size * res + roads.bounds["miny"].min(),
                (x + 1) * grid_size * res + roads.bounds["minx"].min(),
                (y + 1) * grid_size * res + roads.bounds["miny"].min(),
            )
            # Check if the road intersects with the cell
            road_presence.loc[y, x] = roads.geometry.intersects(cell).any()

    road_presence.to_csv("/path/to/road_presence.csv")
