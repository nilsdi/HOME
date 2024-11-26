# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import os

from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    load_project_details,
)
from HOME.get_data_path import get_data_path

from HOME.visualization.footprint_changes.methods_figure.plot_skewed_footprints import (
    plot_footprint,
    plot_skewed_footprints,
    skew_flatten_verts,
    get_extend_boxes,
    stacked_skewed_footprints,
)

# %%
root_dir = Path(__file__).parents[4]
data_path = get_data_path(root_dir)
project_details = load_project_details(data_path)

# %%

res = 0.3
tile_size = 512
overlap = 0
grid_size = tile_size * res * (1 - overlap)

# %% import some tiles


def find_projects_coords(project_list, x_tile, y_tile):
    polygon_ids = {project: get_polygon_ids(project)[-1] for project in project_list}

    project_dates = []
    project_coords = []

    for i, project in enumerate(project_list):
        polygon_directory = get_polygon_details(polygon_ids[project])["gdf_directory"]
        if i == 0:
            large_tiles = os.listdir(polygon_directory)
            for large_tile in large_tiles:
                x_grid = int(large_tile.split("_")[-2])
                y_grid = int(large_tile.split("_")[-1].split(".")[0])
                if x_tile in range(x_grid, x_grid + 10) and y_tile in range(
                    y_grid - 10, y_grid
                ):
                    break

        t1_path = (
            polygon_directory
            + f"/polygons_{project}_resolution{res}_{x_grid}_{y_grid}.fgb"
        )
        project_dates.append(project_details[project]["date"])
        t1 = (
            gpd.read_file(t1_path)
            .to_crs(25832)
            .cx[
                x_tile * grid_size : (x_tile + 1) * grid_size,
                (y_tile - 1) * grid_size : (y_tile) * grid_size,
            ]
        )
        t1 = t1.loc[t1.area.sort_values(ascending=False)[1:].index]
        project_coords.append(
            [
                [[x, y] for x, y in list(polygon.exterior.coords)]
                for polygon in t1.geometry
            ]
        )
    return project_dates, project_coords, polygon_ids


def plot_orthophoto_tiles(project_list, polygon_ids, x_tile, y_tile):

    fig, axes = plt.subplots(1, len(project_list), figsize=(20, 5))
    for i, project in enumerate(project_list):
        tile_id = get_polygon_details(polygon_ids[project])["tile_id"]
        tile_dir = (
            data_path
            / "ML_prediction"
            / "topredict"
            / "image"
            / project
            / f"tiles_{tile_id}"
        )
        tile_path = tile_dir / f"{project}_{x_tile}_{y_tile}.tif"
        orthophoto = cv2.imread(str(tile_path))

        ax = axes[i]
        ax.imshow(orthophoto)
        ax.axis("off")
        ax.set_title(project)

    plt.show()

    return fig, axes


# %%
project_list = [
    "trondheim_1991",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    "trondheim_2016",
    "trondheim_kommune_2022",
]
manual_dates = [1991, 1999, 2006, 2011, 2016, 2022]

# %%

x_tile = 3696
y_tile = 45797

project_dates, project_coords, polygon_ids = find_projects_coords(
    project_list, x_tile, y_tile
)

fig, ax = plt.subplots()
stacked_skewed_footprints(
    project_coords, manual_dates, ax=ax, skew=0.4, flatten=0.5, overlap=-0.2
)
plt.axis("off")
ax.set_aspect("equal")
plt.show()

plot_orthophoto_tiles(project_list, polygon_ids, x_tile, y_tile)
# %%

x_tile = 3694
y_tile = 45798

project_dates, project_coords, polygon_ids = find_projects_coords(
    project_list, x_tile, y_tile
)

fig, ax = plt.subplots()
stacked_skewed_footprints(
    project_coords, manual_dates, ax=ax, skew=0.4, flatten=0.5, overlap=-1
)
plt.axis("off")
ax.set_aspect("equal")
plt.show()

plot_orthophoto_tiles(project_list, polygon_ids, x_tile, y_tile)

# %%
