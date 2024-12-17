# %%
import geopandas as gpd

from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import unary_union
from HOME.footprint_analysis.overlap_analysis.shape_similarity import (
    bounding_box_overlap,
)
from HOME.visualization.footprint_changes.plot_building_layers import (
    find_polygonisations,
    bshape_from_tile_coords,
    find_large_tiles,
)
from HOME.utils.get_project_metadata import get_project_details
from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    get_tiling_details,
)
from HOME.visualization.footprint_changes.combine_plot_data import (
    combine_geometries,
    reassemble_and_cut_small_tiles,
)
from HOME.visualization.footprint_changes.stacked_combined_plot import (
    stacked_combined_plot,
)
from HOME.visualization.footprint_changes.utils import bshape_from_tile_coords


def access_shape(shape_id, shape_layer: str) -> Polygon:
    """
    Access the shape from the given location.

    Args:
        shape_id: gdf ID of the shape
        shape_layer: The layer of the shape - path to the gdf

    Returns:
        Polygon: The shape
    """
    # gdf_path = ".".join(shape_loc.split(".")[:-1])
    # polygon_id = shape_loc.split(".")[-1]
    gdf = gpd.read_file(shape_layer)
    return Polygon(gdf[gdf["shapeID"] == polygon_id].geometry)


def overlap_tree(
    shape_id: str,
    shape_layer: str,
    layers: list[str],
    layer_times: dict[str],
    layer_cover: dict[str, str],
    layer_shapes: dict[dict[str, str]],
    extension: float = 5,
):
    """
    Create a tree of overlapping shapes rooted in the given shape.

    Args:
        shape (Polygon): ID of the shape to start from. "fgb_file.shapeID"
        sourrounding_shapes_layers (dict[str, str]): A dictionary of layers/gdf with shapes
        extension (float, optional): The extension of the bounding box. Defaults to 5.
    """
    shape_gdf = gpd.read_file(layer_shapes[shape_layer])
    # print(shape_gdf)
    shape_geometry = shape_gdf.loc[shape_gdf["ID"] == shape_id].geometry.iloc[0]
    # print(
    #    f"the type of shape_geometry is {type(shape_geometry)}, and it looks like {shape_geometry}"
    # )
    root_shape = Polygon(shape_geometry)
    tree = {shape_layer: {shape_id: {}}}

    # for quick access to the gdfs
    project_gdfs = {}
    for project_name, project_layer in layer_shapes.items():
        project_gdfs[project_name] = gpd.read_file(project_layer)

    for project_name, project_layer in layered_shapes.items():
        capture_date = project_layer["capture_date"]
        project_gdf = project_gdfs[project_name]
        # check if the project has coverage for the area with that shape:
        project_covers: bool = True
        if project_covers:
            tree[project_name] = {}
            for any_shape in project_gdf.geometry:
                any_shape = Polygon(any_shape)
                for tree_project in tree.keys():
                    tree_gdf = project_gdfs[tree_project]
                    for s_id in gdf.keys():
                        tree_shape = Polygon(tree_gdf[tree_gdf["ID"] == s_id].geometry)
                        if bounding_box_overlap(tree_shape, any_shape):
                            # calculate similarities and enter results in dictionary for the tree

                            tree[project_name][any_shape] = {}

    return tree


def overlap_trees(
    layers: list[str],
    layer_times: dict[str],
    layer_cover: dict[str, str],
    layer_shapes: dict[dict[str, str]],
    extension: float = 5,
):
    """
    Create a tree of overlapping shapes rooted in the given shape.

    Args:
        layers (list[str]): A list of layers (named by projects)
        layer_times (dict[str]): A dictionary of the times of the layers
        layer_cover (dict[str, str]): A dictionary of the path to the cover of the layers
        layer_shapes (dict[dict[str, str]]): A dictionary of the paths to the shapes of the layers
        extension (float, optional): The extension of the bounding box. Defaults to 5.
    """
    # find the first layer
    for layer, time in layer_times.items():
        if time == min(layer_times.values()):
            first_layer = layer
            break

    first_shapes_loc = layer_shapes[first_layer]
    # read in the first layer
    first_shapes = gpd.read_file(first_shapes_loc)

    trees = {}
    for shape in first_shapes.geometry:
        tree = overlap_tree(shape, layers, layer_times, layer_cover, layer_shapes)
        trees["f"] = tree

    return tree


def prepare_projects_for_trees(
    project_list: list[str],
    b_shape: Polygon,
):
    """
    Prepare the projects for the overlap analysis.

    Args:
        pro
    """
    polygon_ids = find_polygonisations(
        project_list,
        res=0.3,
        tile_size=512,
        overlap=0,
        reassembly_edge=10,
        reassembly_overlap=1,
        simplication_tolerance=2,
        buffer_distance=0.5,
    )
    projects_details = get_project_details(project_list)
    # capture dates
    times = [pd["capture_date"].year for pd in projects_details]
    project_layer_gdfs = {}
    project_coverages = {}
    project_times = {}
    polygon_detail_dict = {}
    for year, project, project_details in zip(times, project_list, projects_details):
        project_times[project] = year
        polygon_id = polygon_ids[project]
        polygon_details = get_polygon_details(polygon_id)
        polygon_detail_dict[project] = polygon_details
        polygon_directory = polygon_details["gdf_directory"]
        all_entries = os.listdir(polygon_directory)
        large_tiles = [
            entry
            for entry in all_entries
            if os.path.isfile(os.path.join(polygon_directory, entry))
        ]
        selected_large_tiles = find_large_tiles(large_tiles, b_shape)
        assert len(selected_large_tiles) < 2, "More than one large tile selected"
        project_layer_gdfs[project] = f"{polygon_directory}/{selected_large_tiles[0]}"

        coverage_fgb = "_".join(
            ["coverage/coverage"] + selected_large_tiles[0].split("_")[1:]
        )
        project_coverages[project] = coverage_fgb

    return (
        project_layer_gdfs,
        project_coverages,
        project_times,
        polygon_detail_dict,
        project_details,
    )


def prepare_projects_for_plots(
    project_list: list[str],
    b_shape: Polygon,
    project_layer_gdfs: dict[str, str],
    project_coverages: dict[str, str],
    project_times: dict[str, int],
    polygon_detail_dict: dict[str, dict[str, str]],
    project_details,
):
    """
    Prepare footprints and overlaps for the plotting

    Args:
        pro
    """
    geometries = {}
    coverage_shapes = {}
    tifs = {}
    for project in project_list:
        project_gdf = project_layer_gdfs[project]
        polygon_directory = "/".join(project_gdf.split("/")[:-1])
        gdf_name = project_gdf.split("/")[-1]
        geoms = combine_geometries([gdf_name], polygon_directory, b_shape)
        geometries[project] = {
            pid: [[x, y] for x, y in list(polygon.exterior.coords)]
            for pid, polygon in zip(geoms["ID"], geoms.geometry)
        }

        coverage_fgb = project_coverages[project]
        coverage_name = "/".join(coverage_fgb.split("/")[-2:])
        coverage_gdb = combine_geometries([coverage_name], polygon_directory)
        coverage = unary_union(coverage_gdb.geometry)
        intersect_coverage = coverage.intersection(b_shape)
        if isinstance(intersect_coverage, GeometryCollection):
            intersect_coverage = [e for e in intersect_coverage.geoms if e.area > 0]
        else:
            intersect_coverage = [intersect_coverage]
        coverage_shapes[project] = intersect_coverage

        polygon_details = polygon_detail_dict[project]
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
        tifs[project] = img

    return geometries, coverage_shapes, tifs


# %%
project_list = [
    "trondheim_1991",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    "trondheim_2016",
    "trondheim_kommune_2022",
]
bshape = bshape_from_tile_coords(3696, 45796)
(
    project_layer_gdfs,
    project_coverages,
    project_times,
    polygon_detail_dict,
    project_details,
) = prepare_projects_for_trees(project_list, bshape)

print(project_layer_gdfs)
print(project_coverages)
print(project_times)
# %%
tree1 = overlap_tree(
    244,
    "trondheim_1991",
    project_list,
    project_times,
    project_coverages,
    project_layer_gdfs,
)
print(tree1)

# %%
geometries, coverage_shapes, tifs = prepare_projects_for_plots(
    project_list,
    bshape,
    project_layer_gdfs,
    project_coverages,
    project_times,
    polygon_detail_dict,
    project_details,
)
print(geometries["trondheim_1991"].keys())

special_footprint_ids = {p: [] for p in project_list}
special_footprint_ids["trondheim_1991"] = [244, 235]
stacked_combined_plot(
    project_list,
    geometries,
    project_times,
    coverage_shapes,
    bshape,
    tifs=tifs,
    figsize=(3, 10),
    special_footprint_ids=special_footprint_ids,
)
# %%
