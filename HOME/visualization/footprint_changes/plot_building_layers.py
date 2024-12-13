"""
Plotting any number of building layers for any location.
"""

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
import pandas as pd
import warnings

from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, Point
from shapely.ops import unary_union

from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    load_project_details,
    get_assembling_details,
    get_tiling_details,
)
from HOME.utils.get_project_metadata import get_project_details
from HOME.get_data_path import get_data_path

from HOME.visualization.footprint_changes.methods_figure.plot_skewed_footprints import (
    plot_footprint,
    plot_skewed_footprints,
    skew_flatten_verts,
    get_extend_boxes,
)

# %% prepare for getting the projects
root_dir = Path(__file__).parents[3]
data_path = get_data_path(root_dir)
project_details = load_project_details(data_path)

# %% general paramters - maybe should come frome metadata

res = 0.3
tile_size = 512
overlap = 0
grid_size = tile_size * res * (1 - overlap)

# %% finding the relevant large tiles for a given area


def find_polygonisations(
    project_list: list[str],
    res: float = 0.3,
    tile_size: int = 512,
    overlap: int = 0,
    reassembly_edge: int = 10,
    reassembly_overlap: int = 1,
    simplication_tolerance: float = 2,
    buffer_distance: float = 0.5,
) -> list[str]:
    """
    Find the last polygonisation for each project that fullfills the given parameters
    """
    all_polygon_ids = {project: get_polygon_ids(project) for project in project_list}
    fitting_polygon_ids = {}
    for project in project_list:
        possible_polygon_ids = all_polygon_ids[project]
        fitting_polygon_ids[project] = None
        for (
            polygon_id
        ) in possible_polygon_ids:  # will overwrite fitting with the latest fitting one
            polygon_details = get_polygon_details(polygon_id)
            assembly_details = get_assembling_details(polygon_details["assembly_id"])
            tiling_details = get_tiling_details(assembly_details["tile_id"])
            if (
                tiling_details["res"] == res
                and tiling_details["tile_size"] == tile_size
                and tiling_details["overlap_rate"] == overlap
                and assembly_details["n_tiles_edge"] == reassembly_edge
                and assembly_details["n_overlap"] == reassembly_overlap
                and polygon_details["simplication_tolerance"] == simplication_tolerance
                and polygon_details["buffer_distance"] == buffer_distance
            ):
                fitting_polygon_ids[project] = polygon_id
            elif True:  # for debugging why there isn't a polygon that fits
                # print where it failed:
                print(f"Failed for {project} with polygon_id {polygon_id}")
                # because of:
                if tiling_details["res"] != res:
                    print(f"Resolution: {tiling_details['res']} != {res}")
                if tiling_details["tile_size"] != tile_size:
                    print(f"Tile size: {tiling_details['tile_size']} != {tile_size}")
                if tiling_details["overlap_rate"] != overlap:
                    print(
                        f"Overlap rate: {tiling_details['overlap_rate']} != {overlap}"
                    )
                if assembly_details["n_tiles_edge"] != reassembly_edge:
                    print(
                        f"n_tiles_edge: {assembly_details['n_tiles_edge']} != {reassembly_edge}"
                    )
                if assembly_details["n_overlap"] != reassembly_overlap:
                    print(
                        f"n_overlap: {assembly_details['n_overlap']} != {reassembly_overlap}"
                    )
                if polygon_details["simplication_tolerance"] != simplication_tolerance:
                    print(
                        f"Simplication tolerance: {polygon_details['simplication_tolerance']} != {simplication_tolerance}"
                    )
                if polygon_details["buffer_distance"] != buffer_distance:
                    print(
                        f"Buffer distance: {polygon_details['buffer_distance']} != {buffer_distance}"
                    )

    return fitting_polygon_ids


project_list = [
    "trondheim_1991",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    "trondheim_2016",
    "trondheim_kommune_2022",
]
if False:
    polygon_ids = find_polygonisations(project_list)
    print(polygon_ids)
# %%


def find_large_tiles(
    large_tiles: list[str],
    bshape: Polygon,
    res: float = 0.3,
    tile_size: int = 512,
    overlap_rate: float = 0,
    reassembly_edge: int = 10,
    reassembly_overlap: int = 1,
):
    """
    Find the large tiles that contain the given bounding shape (box) which
    can be any polygon.
    """
    # max extent of the bounding shape
    min_x, min_y, max_x, max_y = bshape.bounds

    grid_size = tile_size * res * (1 - overlap_rate)

    overlapping_large_tiles = []
    for large_tile in large_tiles:
        x_grid = int(large_tile.split("_")[-2])
        y_grid = int(large_tile.split("_")[-1].split(".")[0])
        lt_min_x = x_grid * grid_size
        lt_max_x = (x_grid + reassembly_edge) * grid_size
        lt_min_y = (y_grid - reassembly_edge) * grid_size
        lt_max_y = y_grid * grid_size
        min_x_overlap = min_x > lt_min_x and min_x < lt_max_x
        min_y_overlap = min_y > lt_min_y and min_y < lt_max_y
        max_x_overlap = max_x > lt_min_x and max_x < lt_max_x
        max_y_overlap = max_y > lt_min_y and max_y < lt_max_y
        if (  # any of the cornerpoints within the tile
            (max_x_overlap and max_y_overlap)
            or (max_x_overlap and min_y_overlap)
            or (min_x_overlap and min_y_overlap)
            or (min_x_overlap and max_y_overlap)
        ):
            overlapping_large_tiles.append(large_tile)
        elif (  # if the tile is completely within the bounding box
            lt_min_x >= min_x
            and lt_max_x <= max_x
            and lt_min_y >= min_y
            and lt_max_y <= max_y
        ):
            overlapping_large_tiles.append(large_tile)
    return overlapping_large_tiles


if False:
    polygon_directory = get_polygon_details("40003")["gdf_directory"]
    all_entries = os.listdir(polygon_directory)
    large_tiles = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(polygon_directory, entry))
    ]
    print(large_tiles[-2:])

    x_tile = 3754  # 3696
    y_tile = 45755  # 45796

    grid_size = tile_size * res * (1 - overlap)

    bshape = Polygon(
        [
            [x_tile * grid_size, (y_tile) * grid_size],
            [x_tile * grid_size, (y_tile - 1) * grid_size],
            [(x_tile + 1) * grid_size, (y_tile - 1) * grid_size],
            [(x_tile + 1) * grid_size, (y_tile) * grid_size],
            [x_tile * grid_size, (y_tile) * grid_size],
        ]
    )
    # plt.plot(*bshape.exterior.xy)
    selected_large_tiles = find_large_tiles(large_tiles, bshape)
    print(selected_large_tiles)

    # with a little spin we can even find the small tiles:
    small_tile_directory = get_tiling_details("10003")["tile_directory"]
    all_entries = os.listdir(small_tile_directory)
    small_tiles = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(small_tile_directory, entry))
    ]
    print(small_tiles[-2:])
    selected_small_tiles = find_large_tiles(small_tiles, bshape, reassembly_edge=1)
    print(selected_small_tiles)


# %%
def combine_geometries(
    geometry_fgbs: list[str], polygon_directory: str, bshape: Polygon = None
):
    """
    Combine the geometries of the given geometry_gdbs that are within the bshape
    """
    combined_geometries = []
    for geometry_fgb in geometry_fgbs:
        gdf = gpd.read_file(f"{polygon_directory}/{geometry_fgb}").to_crs(25832)
        if bshape:
            gdf = gdf[gdf.within(bshape)]
        # gdf = gdf.loc[gdf.area.sort_values(ascending=False)[1:].index]
        # remove the biggest one if it is as big as a tile
        gdf = gdf.loc[gdf.area != grid_size**2]  # TODO: internalize grid_size
        # if there are shapes left in the gdf, we append it to the combined geometries
        if not gdf.empty:
            combined_geometries.append(gdf)
    # Concatenate all GeoDataFrames
    if len(combined_geometries) > 1:
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(combined_geometries, ignore_index=True)
        )
    elif len(combined_geometries) == 1:
        combined_gdf = combined_geometries[0]
    else:
        combined_gdf = gpd.GeoDataFrame()
    # Remove duplicate geometries
    combined_gdf = combined_gdf.drop_duplicates(subset="geometry")

    return combined_gdf


def reassemble_and_cut_small_tiles(
    small_tiles: list[str], small_tile_directory: str, bshape: Polygon, BW: bool = False
):
    """
    Reassemble the small tiles that are within the bshape and cut them to the bshape.
    Should return a single geotiff.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )

        if len(small_tiles) >= 1:
            # open the last tile
            with rasterio.open(f"{small_tile_directory}/{small_tiles[-1]}") as src:
                if BW:
                    return src.read(1)
                else:
                    return np.dstack([src.read(i) for i in (1, 2, 3)])
    return None


if False:
    dir_40003 = "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/polygons/trondheim_kommune_2022/tiles_10003/prediction_20003/assembly_30003/polygons_40003"
    print(combine_geometries(selected_large_tiles, dir_40003, bshape))

    dir_10003 = "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/topredict/image/trondheim_kommune_2022/tiles_10003"
    img = reassemble_and_cut_small_tiles(selected_small_tiles, dir_10003, bshape)
    plt.imshow(img)


# %%
def stacked_combined_plot(
    footprints_t: list[list[list[float]]],
    t: list[float],
    coverages_t: dict[str, Polygon],
    b_shape: Polygon,
    tifs: dict[str, str] = None,
    skew: float = 0.5,
    flatten: float = 0.9,
    overlap: float = -0.1,
    cmap: str = "tab20",
    figsize: tuple = None,
):
    """
    Plot a series of footprints stacked on top of each other in a skewed coordinate system
    Args:
    footprints_t: list of list  of footprints, each list of footprints represent a time, each
                    footprint is a list of vertices, each vertex is a list of x and y coordinates
    t: list of time values, the time values should be in increasing order
    b_shape: shapely Polygon, the bounding shape for the footprints
    tifs: dict of tif images (one per layer) to plot on the right hand side of the footprints
    ax: matplotlib axis object, the axis to plot the footprints on
    skew: float, the skew factor to apply to the x coordinates
    flatten: float, the flatten factor to apply to the y coordinates
    overlap: by how much the closest (!) footprints should overlap,
                 negative values mean distance instead of overlap
    cmap: str, the colormap to use for the footprints
    figsize: tuple, the size of the figure
    """

    # prepare colors for each layer
    colormap = colormaps[cmap]
    colors = colormap(np.arange(len(footprints_t) % colormap.N))

    if not figsize:
        figsize = (20, 20)
    fig, ax = plt.subplots(figsize=figsize)

    t_dist = [tx - t[0] for tx in t]
    min_t_dist = min([t1 - t0 for t0, t1 in zip(t[:-1], t[1:])])

    x_min, y_min, x_max, y_max = b_shape.bounds
    y_dist = y_max - y_min

    # the y offset factor is calculated from the distance between the closest time steps,
    # the max distance between layers, and the overlap factor.
    y_offset_factor = y_dist * (1 - overlap) / min_t_dist * flatten

    # coverage for each project, plus one that is max extent
    extend_boxes = [
        [[x, y] for x, y in list(bshape.exterior.coords)] for _ in range(len(t) + 1)
    ]

    # we first skew and flatten all coordinates
    skewed_flattened_footprints = [
        [skew_flatten_verts(fp, skew=skew, flatten=flatten) for fp in footprints]
        for footprints in footprints_t
    ]
    extend_boxes_skewed = [
        skew_flatten_verts(box, skew=skew, flatten=flatten) for box in extend_boxes
    ]
    # then we offset the y coordinates of boxes and footprints
    for footprints, t_offset in zip(skewed_flattened_footprints, t_dist):
        for fp in footprints:
            for v in fp:
                v[1] += t_offset * y_offset_factor
    for box, t_offset in zip(extend_boxes_skewed, t_dist):
        for v in box:
            v[1] += t_offset * y_offset_factor

    # then we plot the boxes and  the footprints of each time step
    for i, (year, footprints, extend_box) in enumerate(
        zip(t, skewed_flattened_footprints, extend_boxes_skewed)
    ):
        if str(year) in coverages_t.keys():
            for coverage_box in coverages_t[str(year)]:
                coverage_box = [[x, y] for x, y in list(coverage_box.exterior.coords)]
                coverage_box_skewed = skew_flatten_verts(
                    coverage_box, skew=skew, flatten=flatten
                )
                for v in coverage_box_skewed:
                    v[1] += t_dist[i] * y_offset_factor
                plot_footprint(
                    coverage_box_skewed,
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
            min([v[0] for v in extend_box]),
            np.mean([v[1] for v in extend_box]),
            f"{t[i]}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )
        # if given, print the tifs for each layer on the right hand side
        if tifs:
            y_offset = (0.05 + overlap / 2) * y_dist * flatten
            # put the tif on the right hand side of the max extent box :
            box_max_x = max([v[0] for v in extend_box])
            box_min_y = min([v[1] for v in extend_box])
            box_max_y = max([v[1] for v in extend_box])
            # box_y_extent = box_max_y - box_min_y - _offset

            tile_min_y = box_min_y + y_offset
            tile_max_y = box_max_y - y_offset
            tile_min_x = box_max_x + 0.2 * y_dist * flatten
            tile_max_x = tile_min_x + tile_max_y - tile_min_y

            img = tifs[str(t[i])]
            if img is not None:
                # if rgb:
                if len(img.shape) == 3:
                    ax.imshow(
                        img,
                        extent=[tile_min_x, tile_max_x, tile_min_y, tile_max_y],
                    )
                elif len(img.shape) == 2:
                    ax.imshow(
                        img,
                        extent=[tile_min_x, tile_max_x, tile_min_y, tile_max_y],
                        cmap="gray",
                    )
        for fp in footprints:
            plot_footprint(fp, ax=ax, color=colors[i])
    plt.axis("off")
    ax.set_aspect("equal")
    return fig, ax


# %%
def plot_building_layers(
    project_list: list[str],
    b_shape: Polygon,
    layer_overlap: float = 0.1,
    figsize: tuple = (10, 10),
    res: float = 0.3,
    tile_size: int = 512,
    tile_overlap: int = 0,
    reassembly_edge: int = 10,
    reassembly_overlap: int = 1,
    simplication_tolerance: float = 2,
    buffer_distance: float = 0.5,
    cmap: str = "tab10",
):
    """
    Plot the building layers for a given area
    """
    # find the fitting polygonisations
    polygon_ids = find_polygonisations(
        project_list,
        res=res,
        tile_size=tile_size,
        overlap=tile_overlap,
        reassembly_edge=reassembly_edge,
        reassembly_overlap=reassembly_overlap,
        simplication_tolerance=simplication_tolerance,
        buffer_distance=buffer_distance,
    )
    projects_details = get_project_details(project_list)
    # capture dates
    t = [pd["capture_date"].year for pd in projects_details]
    geometries_t = []
    coverages_t = {}
    tifs = {}
    for year, project, project_details in zip(t, project_list, projects_details):
        # print(project)
        # print(project_details)
        polygon_id = polygon_ids[project]
        polygon_details = get_polygon_details(polygon_id)
        polygon_directory = polygon_details["gdf_directory"]
        all_entries = os.listdir(polygon_directory)
        large_tiles = [
            entry
            for entry in all_entries
            if os.path.isfile(os.path.join(polygon_directory, entry))
        ]
        selected_large_tiles = find_large_tiles(large_tiles, b_shape)
        print(selected_large_tiles)
        geometries = combine_geometries(
            selected_large_tiles, polygon_directory, b_shape
        )
        geometries_t.append(geometries)
        if len(selected_large_tiles) > 0:
            # coverage
            coverage_fgbs = [
                "_".join(["coverage/coverage"] + t.split("_")[1:])
                for t in selected_large_tiles
            ]
            # merge the polygons of the coverage
            coverages_gdb = combine_geometries(coverage_fgbs, polygon_directory)
            coverage = unary_union(coverages_gdb.geometry)
            intersect_coverage = coverage.intersection(b_shape)
            print(intersect_coverage)
            if isinstance(intersect_coverage, GeometryCollection):
                intersect_coverage = [e for e in intersect_coverage.geoms if e.area > 0]
            else:
                intersect_coverage = [intersect_coverage]
            print(f" the coverage object is of type {type(intersect_coverage)}")
            coverages_t[str(year)] = intersect_coverage
        # small tiles
        tiling_details = get_tiling_details(polygon_details["tile_id"])
        small_tile_directory = tiling_details["tile_directory"]
        all_entries = os.listdir(small_tile_directory)
        small_tiles = [
            entry
            for entry in all_entries
            if os.path.isfile(os.path.join(small_tile_directory, entry))
        ]
        BW = project_details["bandwidth"] == "BW"
        selected_small_tiles = find_large_tiles(small_tiles, b_shape, reassembly_edge=1)
        img = reassemble_and_cut_small_tiles(
            selected_small_tiles, small_tile_directory, b_shape, BW=BW
        )
        tifs[str(year)] = img

    footprints_t = [
        (
            [
                [[x, y] for x, y in list(polygon.exterior.coords)]
                for polygon in geo.geometry
            ]
            if not geo.empty
            else []
        )
        for geo in geometries_t
    ]
    # if all footprints are empty, we can't plot anything:
    if all([len(footprints) == 0 for footprints in footprints_t]):
        print("No footprints found for the given area.")
        return

    print(t)
    fig, ax = stacked_combined_plot(
        footprints_t,
        t,
        coverages_t,
        bshape,
        overlap=layer_overlap,
        tifs=tifs,
        cmap=cmap,
        figsize=figsize,
    )
    return


def bshape_from_tile_coords(x_tile, y_tile):
    grid_size = tile_size * res * (1 - overlap)
    bshape = Polygon(
        [
            [x_tile * grid_size, (y_tile) * grid_size],
            [x_tile * grid_size, (y_tile - 1) * grid_size],
            [(x_tile + 1) * grid_size, (y_tile - 1) * grid_size],
            [(x_tile + 1) * grid_size, (y_tile) * grid_size],
            [x_tile * grid_size, (y_tile) * grid_size],
        ]
    )
    return bshape


print_examples = True
if print_examples:
    warnings.warn("plot_building_layers.py is running in example mode.")
# %% first example

project_list = [
    "trondheim_1991",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    "trondheim_2016",
    "trondheim_kommune_2022",
]
bshape = bshape_from_tile_coords(3696, 45796)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(10, 50))
# %% second example
bshape = bshape_from_tile_coords(3754, 45755)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %% third example
bshape = bshape_from_tile_coords(3695, 45788)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %%
bshape = bshape_from_tile_coords(3696, 45796)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %%
bshape = bshape_from_tile_coords(3695, 45798)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(20, 20))

# %%
project_list1 = [
    "trondheim_1969",
    "trondheim_1977",
    "trondheim_1983",
    "trondheim_1988",
    # "trondheim_1991",
    "trondheim_1994",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    # "trondheim_2016",
    "trondheim_2017",
    "trondheim_kommune_2022",
]
bshape = bshape_from_tile_coords(3715, 45798)
plot_building_layers(
    project_list1, bshape, layer_overlap=0.1, figsize=(10, 40), cmap="tab20"
)
# %%

bshape = bshape_from_tile_coords(3725, 45798)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(20, 20))
