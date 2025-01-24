"""
Plotting any number of building layers for any location.
"""

# %%
import os

from pathlib import Path
from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import unary_union

from HOME.visualization.footprint_changes.get_plot_data import (
    find_polygonisations,
    find_large_tiles,
)
from HOME.visualization.footprint_changes.combine_plot_data import (
    combine_geometries,
    reassemble_and_cut_small_tiles,
)
from HOME.visualization.footprint_changes.stacked_combined_plot import (
    stacked_combined_plot,
)
from HOME.visualization.footprint_changes.utils import bshape_from_tile_coords
from HOME.utils.get_project_metadata import get_project_details
from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    get_tiling_details,
)


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
    t = {p: pd["capture_date"].year for p, pd in zip(project_list, projects_details)}
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
        print(f"dir: {polygon_directory}, tiles: {selected_large_tiles}")
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
            # print(intersect_coverage)
            if isinstance(intersect_coverage, GeometryCollection):
                intersect_coverage = [e for e in intersect_coverage.geoms if e.area > 0]
            else:
                intersect_coverage = [intersect_coverage]
            print(f" the coverage object is of type {type(intersect_coverage)}")
            coverages_t[project] = intersect_coverage
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

    footprints_t = {
        project: (
            {
                pid: [[x, y] for x, y in list(polygon.exterior.coords)]
                for pid, polygon in zip(geo["ID"], geo.geometry)
            }
            if not geo.empty
            else {}
        )
        for project, geo in zip(project_list, geometries_t)
    }
    for project, footprints in footprints_t.items():
        print(project)
        print(footprints)
    # if all footprints are empty, we can't plot anything:
    if all([len(footprints.values()) == 0 for footprints in footprints_t.values()]):
        print("No footprints found for the given area.")
        return

    print(t)
    fig, ax = stacked_combined_plot(
        project_list,
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


# %% first example
if __name__ == "__main__":
    root_dir = Path(__file__).parents[3]
    data_path = root_dir / "data"
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
    bshape = bshape_from_tile_coords(3706, 45797)
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

# %%
