# %%
import geopandas as gpd
import math
import json
import os

from pathlib import Path
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
from HOME.footprint_analysis.overlap_analysis.shape_similarity import (
    calculate_similarity_measures,
)

root_dir = Path(__file__).resolve().parents[3]


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
    keep_shapes: bool = False,
):
    """
    Create a tree of overlapping shapes rooted in the given shape.

    Args:
        shape (Polygon): ID of the shape to start from. "fgb_file.shapeID"
        sourrounding_shapes_layers (dict[str, str]): A dictionary of layers/gdf with shapes
        extension (float, optional): The extension of the bounding box. Defaults to 5.
    """
    # for quick access to the gdfs we open them once and store them in a dictionary
    project_gdfs = {}
    for project_name, project_layer in layer_shapes.items():
        project_gdfs[project_name] = gpd.read_file(project_layer)

    shape_gdf = project_gdfs[shape_layer]
    shape_geometry = shape_gdf.loc[shape_gdf["ID"] == shape_id].geometry.iloc[0]
    # print(
    #    f"the type of shape_geometry is {type(shape_geometry)}, and it looks like {shape_geometry}"
    # )
    root_shape = Polygon(shape_geometry)

    tree = {shape_layer: {shape_id: {"shape": root_shape, "comparisons": {}}}}
    # go through the layers by time and check for overlaps
    sorted_layers = sorted(layer_times.keys(), key=lambda layer: layer_times[layer])
    for project_name in sorted_layers:
        capture_date = layer_times[project_name]
        project_gdf = project_gdfs[project_name]
        # TODO: check if the project has coverage for the entire area around all current shapes.
        project_covers: bool = True
        if project_covers:
            # print(f"checking coverage for {project_name}")
            # print(f"curren tree: {tree}")
            if project_name not in tree.keys():
                # print(f"adding {project_name} to the tree")
                tree[project_name] = {}
            # go through all shapes in the project and check for overlaps
            for any_id, any_shape in zip(project_gdf["ID"], project_gdf.geometry):
                # print(f"checking overlaps for id {any_id} ({project_name})")
                any_shape = Polygon(any_shape)
                # if tthe shape covers a whole tile, we exclude it
                if any_shape.area > 0.9 * 512**2 * 0.3**2:
                    continue
                # check overlap of each shape with all items already in the tree
                for tree_project in tree.keys():
                    tree_gdf = project_gdfs[tree_project]
                    for tree_id in list(
                        tree[tree_project].keys()
                    ):  # list needed because of some weird dict of ints stuff
                        tree_shape = tree[tree_project][tree_id]["shape"]
                        # tree_shape = Polygon(
                        #    tree_gdf[tree_gdf["ID"] == tree_id].iloc[0].geometry
                        # )
                        if bounding_box_overlap(tree_shape, any_shape):
                            # print(
                            #     f"overlap between {tree_id} (layer: {tree_project}) and {any_id} ({project_name})"
                            # )
                            # calculate similarities and enter results in dictionary for the tree
                            if any_id in tree[project_name].keys():
                                if (
                                    tree_project
                                    not in tree[project_name][any_id][
                                        "comparisons"
                                    ].keys()
                                ):
                                    tree[project_name][any_id]["comparisons"][
                                        tree_project
                                    ] = {}
                                tree[project_name][any_id]["comparisons"][tree_project][
                                    tree_id
                                ] = calculate_similarity_measures(any_shape, tree_shape)
                            else:
                                tree[project_name][any_id] = {
                                    "shape": any_shape,
                                    "comparisons": {
                                        tree_project: {
                                            tree_id: calculate_similarity_measures(
                                                any_shape, tree_shape
                                            )
                                        },
                                    },
                                }
    # after all layers have been checked, we delete the actual shapes from the tree
    if not keep_shapes:
        for project_name in tree.keys():
            for shape_id in tree[project_name].keys():
                del tree[project_name][shape_id]["shape"]
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
def comparison_heatmap(
    tree: dict,
    project_list: list[str],
    comparison: str,
    cmap="hot_r",
    default_value=False,
    vertical_labels=False,
):
    """
    Create a heatmap of the comparison values in the tree.

    Args:
        tree (dict): The tree of overlap comparisons
        project_list (list[str]): The list of projects
        comparison (str): The comparison metric to put in the heatmap
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    # we want to create a matrix of the comparison values.
    # the matrix should in itself be n_projects x n_projects, but we need to extend it
    # so that each project row can contain multiple shapes

    n_projects = len(project_list)
    n_shapes = [len(tree[project].keys()) for project in project_list]
    n_project_pixels = math.prod(set(n_shapes))
    if default_value:
        comparison_matrix = (
            np.ones((n_projects * n_project_pixels, n_projects * n_project_pixels))
            * default_value
        )
    else:
        comparison_matrix = np.ones(
            (n_projects * n_project_pixels, n_projects * n_project_pixels)
        ) * -(10**6)

    # fill the matrix with values
    pixel_indices = {project: {} for project in project_list}
    pixel = 0
    for project, shapes in tree.items():
        n_project_shapes = len(list(shapes.keys()))
        n_shape_pixels = int(n_project_pixels / n_project_shapes)
        for shape, data in shapes.items():
            pixel_indices[project][shape] = [pixel, pixel + n_shape_pixels]
            pixel += n_shape_pixels

    for project, shapes in tree.items():
        for shape, data in shapes.items():
            for comparison_project, comparisons in data["comparisons"].items():
                for comparison_shape, comparison_data in comparisons.items():
                    pixel_range_x = pixel_indices[project][shape]
                    pixel_range_y = pixel_indices[comparison_project][comparison_shape]
                    comparison_matrix[
                        pixel_range_x[0] : pixel_range_x[1],
                        pixel_range_y[0] : pixel_range_y[1],
                    ] = comparison_data[comparison]
    comparison_matrix[comparison_matrix == -(10**6)] = np.nan
    minor_offset = -pixel / 6
    major_offset = -pixel / 3
    # print a heatmap
    fig, ax = plt.subplots()
    cax = ax.matshow(comparison_matrix, cmap=cmap)
    fig.colorbar(cax)

    # add labels to the matrix
    for project, shapes in pixel_indices.items():
        project_min_pixel = 10**6
        project_max_pixel = 0
        for shape, pixel_range in shapes.items():
            shape_min_pixel = pixel_range[0]
            shape_max_pixel = pixel_range[1]
            ax.text(
                x=minor_offset,
                y=(shape_min_pixel + shape_max_pixel) / 2,
                s=shape,
                ha="center",
                va="center",
            )
            if vertical_labels:
                ax.text(
                    x=(shape_min_pixel + shape_max_pixel) / 2,
                    y=minor_offset,
                    s=shape,
                    ha="center",
                    va="center",
                    rotation=90,
                )
            if shape_min_pixel < project_min_pixel:
                project_min_pixel = shape_min_pixel
            if shape_max_pixel > project_max_pixel:
                project_max_pixel = shape_max_pixel
        ax.text(
            x=major_offset,
            y=(project_min_pixel + project_max_pixel) / 2,
            s=project,
            ha="right",
            va="center",
        )
        if vertical_labels:
            ax.text(
                x=(project_min_pixel + project_max_pixel) / 2,
                y=major_offset,
                s=project,
                ha="center",
                va="bottom",
                rotation=90,
            )
    if vertical_labels:
        ax.text(x=major_offset, y=major_offset, s="project", ha="right", va="top")
        ax.text(x=minor_offset, y=minor_offset, s="shape", ha="center", va="center")
    else:
        ax.text(x=major_offset, y=0, s="project", ha="right", va="center")
        ax.text(x=minor_offset, y=0, s="shape", ha="center", va="center")
    plt.axis("off")
    ax.set_aspect("equal")
    ax.set_title(comparison)
    plt.show()

    return


# %%
def save_overlap_tree(
    shape_id: str,
    shape_layer: str,
    layers: list[str],
    layer_times: dict[str],
    layer_cover: dict[str, str],
    layer_shapes: dict[dict[str, str]],
    extension: float = 5,
    keep_shapes: bool = False,
    subfolder: str = "default",
    data_path: Path = None,
):
    """
    Save the overlap tree to a file.

    Args:
        tree (dict): The overlap tree
        shape_id (str): The ID of the shape
        shape_layer (str): The layer of the shape
        layers (list[str]): The layers of the tree
        subfolder (str): The subfolder to save the tree in
    """
    if data_path is None:
        data_path = root_dir / "data"
    save_path = data_path / "footprint_analysis/overlap_trees" / subfolder
    os.makedirs(save_path, exist_ok=True)
    # print(f"Saving tree to {save_path / f'{shape_id}_{shape_layer}.json'}")
    tree = overlap_tree(
        shape_id,
        shape_layer,
        layers,
        layer_times,
        layer_cover,
        layer_shapes,
        extension=extension,
        keep_shapes=keep_shapes,
    )

    def convert_shapes_to_serializable(data):
        """
        Recursively convert Polygon objects to serializable formats in a data structure.

        Args:
            data: The data structure to convert.

        Returns:
            The converted data structure.
        """
        if isinstance(data, dict):
            return {k: convert_shapes_to_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_shapes_to_serializable(item) for item in data]
        elif isinstance(data, Polygon):
            return list(data.exterior.coords)
        else:
            return data

    # print(f"tree: {tree}")
    tree_data = {
        "shape_id": shape_id,
        "shape_layer": shape_layer,
        "layers": layers,
        "layer_times": layer_times,
        "extension": extension,
        "keep_shapes": keep_shapes,
        "tree": convert_shapes_to_serializable(tree),
    }
    with open(save_path / f"{shape_id}_{shape_layer}.json", "w") as f:
        json.dump(tree_data, f)
    return


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
bshape = bshape_from_tile_coords(3695, 45798)
(
    project_layer_gdfs,
    project_coverages,
    project_times,
    polygon_detail_dict,
    project_details,
) = prepare_projects_for_trees(project_list, bshape)

# print(project_layer_gdfs)
# print(project_coverages)
# print(project_times)

# tree1 = overlap_tree(
#     248,  # 215,  # 248,
#     "trondheim_1991",
#     project_list,
#     project_times,
#     project_coverages,
#     project_layer_gdfs,
# )

# for project, shapes in tree1.items():
#     print(f"\nproject: {project}, shapes: {list(shapes.keys())}")
#     for shape, data in shapes.items():
#         print(f"shape: {shape}")
#         for comparison_project, comparisons in data["comparisons"].items():
#             print(f"comparison project: {comparison_project}")
#             for comparison_shape, comparison_data in comparisons.items():
#                 print(f"comparison shape: {comparison_shape}")
#                 for key, value in comparison_data.items():
#                     print(f"{key}: {value}")
# print(tree1)

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


# %% start plotting some cases to get a feeling for the data
tree = overlap_tree(
    117,  # ,#256,  # 244,  # 257,  # 244,  # 215,# 257,# 248,
    "trondheim_1991",
    project_list,
    project_times,
    project_coverages,
    project_layer_gdfs,
)
print(tree)
special_footprint_ids = {p: [] for p in project_list}
for project, shapes in tree.items():
    special_footprint_ids[project] = list(shapes.keys())
# special_footprint_ids["trondheim_1991"] = [
#     277,
#     282,
#     257,
#     237,
#     248,
#     215,
#     199,
#     221,
#     236,
#     244,
#     239,
#     235,
#     233,
#     222,
#     256,
#     272,
#     276,
#     280,
#     279,
#     250,
# ]

stacked_combined_plot(
    project_list,
    geometries,
    project_times,
    coverage_shapes,
    bshape,
    tifs=tifs,
    figsize=(5, 16),
    special_footprint_ids=special_footprint_ids,
)
comparison_heatmap(tree, project_list, "IoU")
comparison_heatmap(tree, project_list, "Hausdorff_distance", cmap="hot")
# %% save a tree
save_overlap_tree(
    256,  # 221,  # 257,  # 257,  # 244,  # 215,# 257,# 248,
    "trondheim_1991",
    project_list,
    project_times,
    project_coverages,
    project_layer_gdfs,
    extension=5,
    keep_shapes=True,
    subfolder="testing3",
)

# %%


def filter_tree(tree):
    """
    Filter the tree so we get sets of shapes that belong to the same building or an interesting case.

    Args:
        tree (dict): The tree of overlap comparisons

    Returns:
        subtrees (list(dict)): The filtered subtrees
    """
    subtrees = []

    def single_branch_tree(tree: dict) -> bool:
        single_branch = True
        for project, shapes in tree.items():
            if len(shapes.keys()) > 1:
                single_branch = False
                break
        return single_branch

    # first option: single branch tree (one or zero shapes in each layer)
    if single_branch_tree(tree):
        return tree

    # second option: single branch tree after only considering overlaps
    def construct_subtree(tree: dict, starting_shape: int) -> Tuple[dict]:
        """
        Construct a subtree that only includes shapes with actual overlaps,
        starting with any initial shape. This can be used to cut a large tree into
        multiple subtrees that are more interesting to analyze.

        Args:
            tree (dict): The tree of overlap comparisons - assumes the
                            tree is ordered by time already (order of projects)
            starting_shape (int): The ID of the starting shape

        Returns:
            Tuple[dict]: The subtree and the remaining tree
        """
        subtree = {}
        remaining_tree = {}
        for project, shapes in tree.items():
            if starting_shape in shapes.keys():
                subtree[project] = {starting_shape: shapes[starting_shape]}
            else:
                remaining_tree[project] = shapes
        return subtree, remaining_tree

    return subtrees
