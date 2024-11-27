# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path
import copy
import geopandas as gpd
import os
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt

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


def get_tile_paths(project_list, x_tile, y_tile):
    polygon_ids = {project: get_polygon_ids(project)[-1] for project in project_list}

    tile_paths = []
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
        tile_paths.append(tile_path)
    return tile_paths


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
tile_paths = get_tile_paths(project_list, x_tile, y_tile)
fig, ax = plt.subplots()
stacked_skewed_footprints(
    project_coords, manual_dates, tile_paths, ax=ax, skew=0.4, flatten=0.5, overlap=-0.2
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


def plot_orthophoto_tile(ax, tile_path, x_min, x_max, y_min, y_max):
    with rasterio.open(tile_path) as src:
        # Read the RGB channels
        img = np.dstack([src.read(i) for i in (1, 2, 3)])

        # Plot the image with specified extent
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max])


def stacked_skewed_footprints(
    footprints_t: list[list[list[float]]],
    t: list[float],
    tif_paths: list[str] = None,
    ax: plt.Axes = None,
    skew: float = 0.5,
    flatten: float = 0.9,
    overlap: float = -0.1,
    cmap: str = "tab20",
    plot_connecting_lines: bool = False,
    figsize: tuple = None,
):
    """
    Plot a series of footprints stacked on top of each other in a skewed coordinate system
    Args:
    footprints_t: list of list  of footprints, each list of footprints represent a time, each
                    footprint is a list of vertices, each vertex is a list of x and y coordinates
    t: list of time values, the time values should be in increasing order
    ax: matplotlib axis object, the axis to plot the footprints on
    skew: float, the skew factor to apply to the x coordinates
    flatten: float, the flatten factor to apply to the y coordinates
    overlap: by how much the closest (!) footprints should overlap,
                 negative values mean distance instead of overlap
    """

    # prepare colors for each layer
    colormap = colormaps[cmap]
    colors = colormap(np.arange(len(footprints_t) % colormap.N))

    if not figsize:
        figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    # offset the y coordinates of each footprint by the corresponding t value
    min_t_dist = min([t1 - t0 for t0, t1 in zip(t[:-1], t[1:])])
    max_ys = []
    min_ys = []
    for i, footprints in enumerate(footprints_t):
        try:
            max_ys.append(max([max([v[1] for v in fp]) for fp in footprints]))
            min_ys.append(min([min([v[1] for v in fp]) for fp in footprints]))
        except ValueError:  # if there is an empty footprint layer,
            pass
    y_dist = max(max_y - min_y for max_y, min_y in zip(max_ys, min_ys))
    y_mid = (max(max_ys) + min(min_ys)) / 2
    """y_dist = min(
        [
            max([max([v[1] for v in fp]) for fp in footprints])
            - min([min([v[1] for v in fp]) for fp in footprints])
            for footprints in footprints_t
        ]
    )"""
    t_dist = [tx - t[0] for tx in t]
    # we want to offset enough to make the footprints not overlap and even have some distance
    # so we assign the y offset per y distance in a way that the overlap is as asked for
    y_offset_factor = y_dist / (min_t_dist) * (1 - overlap) * flatten

    # we make a rectangle for each list of footprints that will show the layer:
    boxes = get_extend_boxes(footprints_t)
    # we first skew and flatten all coordinates
    skewed_flattened_footprints = [
        [skew_flatten_verts(fp, skew=skew, flatten=flatten) for fp in footprints]
        for footprints in footprints_t
    ]
    skew_flattened_boxes = [
        skew_flatten_verts(box, skew=skew, flatten=flatten) for box in boxes
    ]
    # we make a copy of the footprint verts of all but the first time step to later make the
    # connecting lines down

    connecting_lines_bottom_verts = copy.deepcopy(
        [[fp for fp in footprints] for footprints in skewed_flattened_footprints[1:]]
    )

    # then we offset the y coordinates of boxes and footprints
    for footprints, t_offset in zip(skewed_flattened_footprints, t_dist):
        for fp in footprints:
            for v in fp:
                v[1] += t_offset * y_offset_factor
    for bottom_fps, t_m1_offset in zip(connecting_lines_bottom_verts, t_dist[:-1]):
        for fp in bottom_fps:
            for v in fp:
                v[1] += t_m1_offset * y_offset_factor
    for box, t_offset in zip(skew_flattened_boxes, t_dist):
        for v in box:
            v[1] += t_offset * y_offset_factor

    # then we plot the boxes and  the footprints of each time step
    for i, (footprints, box) in enumerate(
        zip(skewed_flattened_footprints, skew_flattened_boxes)
    ):
        plot_footprint(
            box,
            ax=ax,
            color="gray",
            ls="--",
            lw=1,
            fill=True,
            fill_color="gray",
            fill_alpha=0.1,
        )
        # print the date for each layer
        ax.text(
            min([box[0][0] for box in skew_flattened_boxes]),
            np.mean([v[1] for v in skew_flattened_boxes[i]]),
            f"{t[i]}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )
        # if given, print the tifs for each layer on the right hand side
        if tif_paths:
            # put the tif on the right hand side of the box
            box_max_x = max([v[0] for v in box])
            box_min_y = min([v[1] for v in box])
            box_max_y = max([v[1] for v in box])
            box_y_extent = box_max_y - box_min_y - 10

            tile_min_x = box_max_x + 20
            tile_max_x = tile_min_x + box_y_extent
            tile_min_y = box_min_y + 5
            tile_max_y = box_max_y - 5

            plot_orthophoto_tile(
                ax,
                tif_paths[i],
                tile_min_x,
                tile_max_x,
                tile_min_y,
                tile_max_y,
            )
            pass

        for fp in footprints:
            plot_footprint(fp, ax=ax, color=colors[i])
        # then we plot the connecting lines
        if i > 0 and plot_connecting_lines:
            bottom_verts = connecting_lines_bottom_verts[i - 1]
            for top_fp, bottom_fp in zip(footprints, bottom_verts):
                for v1, v2 in zip(top_fp, bottom_fp):
                    ax.plot(
                        [v1[0], v2[0]], [v1[1], v2[1]], color="gray", ls="--", lw=0.3
                    )

    plt.axis("off")
    ax.set_aspect("equal")
    return fig, ax


# %%
x_tile = 3694
y_tile = 45798

project_dates, project_coords, polygon_ids = find_projects_coords(
    project_list, x_tile, y_tile
)
tile_paths = get_tile_paths(project_list, x_tile, y_tile)

stacked_skewed_footprints(
    project_coords,
    manual_dates,
    tile_paths,
    skew=0.4,
    flatten=0.5,
    overlap=-0.3,
    cmap="tab10",
    figsize=(20, 20),
)

# %%
x_tile = 3695
y_tile = 45788

project_dates, project_coords, polygon_ids = find_projects_coords(
    project_list, x_tile, y_tile
)
tile_paths = get_tile_paths(project_list, x_tile, y_tile)

stacked_skewed_footprints(
    project_coords,
    manual_dates,
    tile_paths,
    skew=0.4,
    flatten=0.5,
    overlap=-0.3,
    cmap="tab10",
    figsize=(20, 20),
)
# %%
