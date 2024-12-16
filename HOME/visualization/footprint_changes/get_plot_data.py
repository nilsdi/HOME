# %%
from pathlib import Path
from shapely.geometry import Polygon

from HOME.get_data_path import get_data_path
from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    load_project_details,
    get_assembling_details,
    get_tiling_details,
)


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


if __name__ == "__main__":
    root_dir = Path(__file__).parents[3]
    data_path = get_data_path(root_dir)
    project_details = load_project_details(data_path)

    project_list = [
        "trondheim_1991",
        "trondheim_1999",
        "trondheim_2006",
        "trondheim_2011",
        "trondheim_2016",
        "trondheim_kommune_2022",
    ]

    polygon_ids = find_polygonisations(project_list)
    print(polygon_ids)

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

    tile_size = 512
    res = 0.3
    overlap = 0

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
