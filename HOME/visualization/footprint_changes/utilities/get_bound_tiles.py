# %%
import os
from pathlib import Path
from shapely.geometry import Polygon

from HOME.utils.project_paths import (
    get_polygon_ids,
    get_polygon_details,
    load_project_details,
    get_assembling_details,
    get_tiling_details,
)
from HOME.visualization.footprint_changes.utilities.get_bounding_shape import (
    get_bounding_shape,
)


def find_polygonisations(
    project_list: list[str],
    res: float = 0.3,
    tile_size: int = 512,
    overlap: float = 0,
    reassembly_edge: int = 10,
    reassembly_overlap: int = 1,
    simplication_tolerance: float = 2,
    buffer_distance: float = 0.5,
) -> list[str]:
    """
    Find the last polygonisation for each project that fullfills the given parameters.

    Args:
        project_list: list of project names
        res: resolution of the tiles in m/px
        tile_size: size of the tiles in pixels
        overlap: rate of overlapp between small tiles (might not be used atm.)
        reassembly_edge: number of tiles to reassemble
        reassembly_overlap: overlapping reassembled tiles
        simplication_tolerance: tolerance for simplification of the polygons
        buffer_distance: distance for the buffer around the polygons

    Returns:
        fitting_polygonisation_ids: dict with project names as keys and the fitting
            polygonization id as value. Can be used to find the polygonization in the logs.

    Raises:
        Technically nothing: There are debugging prints if no polygonisation fits.
    """
    all_polygonisation_ids = {
        project: get_polygon_ids(project) for project in project_list
    }
    fitting_polygonisation_ids = {}
    for project in project_list:
        possible_polygonisation_ids = all_polygonisation_ids[project]
        fitting_polygonisation_ids[project] = None
        for (
            polygonisation_id
        ) in (
            possible_polygonisation_ids
        ):  # will overwrite fitting with the latest fitting one
            polygon_details = get_polygon_details(polygonisation_id)
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
                fitting_polygonisation_ids[project] = polygonisation_id
            elif True:  # for debugging why there isn't a polygon that fits
                # print where it failed:
                print(
                    f"Failed for {project} with polygonisation_id {polygonisation_id}"
                )
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

    return fitting_polygonisation_ids


def get_overlapping_tiles(
    tiles: list[str],
    bshape: Polygon,
    tile_size: float = 512 * 0.3 * (1 - 0),
    res: float = 0.3,
    small_tile_size: int = 512,
    overlap_rate: float = 0,
):
    """
    Find the tiles in a list that overlap with the given bounding shape (box) which
    can be any polygon. Requires that the tiles are named with their left top corner
    coordinates in the name. (x second to last, y last.)

    Args:
        tiles: list of tile names
        bshape: shapely Polygon
        tile_size: size of each tile in meters
        res (optional): resolution of the small tiles in m/px
        small_tile_size (optional): size of the small tiles in pixels
        overlap_rate (optional): rate of overlapp between small tiles

    Returns:
        overlapping_tiles: list of tile names that overlap with the bounding shape
    """
    # max extent of the bounding shape
    min_x, min_y, max_x, max_y = bshape.bounds
    grid_size = small_tile_size * res * (1 - overlap_rate)

    overlapping_tiles = []
    for tile in tiles:
        x_grid = int(tile.split("_")[-2])
        y_grid = int(tile.split("_")[-1].split(".")[0])
        t_min_x = x_grid * grid_size
        t_max_x = t_min_x + tile_size
        t_max_y = y_grid * grid_size
        t_min_y = t_max_y - tile_size

        min_x_overlap = min_x > t_min_x and min_x < t_max_x
        min_y_overlap = min_y > t_min_y and min_y < t_max_y
        max_x_overlap = max_x > t_min_x and max_x < t_max_x
        max_y_overlap = max_y > t_min_y and max_y < t_max_y
        if (  # any of the cornerpoints within the tile
            (max_x_overlap and max_y_overlap)
            or (max_x_overlap and min_y_overlap)
            or (min_x_overlap and min_y_overlap)
            or (min_x_overlap and max_y_overlap)
        ):
            overlapping_tiles.append(tile)
        elif (  # if the tile is completely within the bounding box
            t_min_x >= min_x
            and t_max_x <= max_x
            and t_min_y >= min_y
            and t_max_y <= max_y
        ):
            overlapping_tiles.append(tile)
    return overlapping_tiles


def find_large_tiles(
    polygonisation_id: str,
    bshape: Polygon,
    res: float = 0.3,
    tile_size: int = 512,
    overlap_rate: float = 0,
    reassembly_edge: int = 10,
    reassembly_overlap: int = 1,
) -> list[str or Path]:
    """
    Find the large tiles that contain the given bounding shape (box) based
    on the concrete polygonisation and bshape.

    Args:
        polygonisation_id: id of the polygonisation
        bshape: shapely Polygon
        res: resolution of the small tiles in m/px
        tile_size: size of the small tiles in pixels
        overlap_rate: rate of overlapp between small tiles
        reassembly_edge: number of tiles to reassemble
        reassembly_overlap: overlapping reassembled tiles

    Returns:
        large_tiles: path to tiles that contain the bounding shape
    """
    polygonisation_details = get_polygon_details(polygonisation_id)
    polygonisation_directory = polygonisation_details["gdf_directory"]
    all_files = os.listdir(polygonisation_directory)
    all_tiles = [
        file
        for file in all_files
        if os.path.isfile(os.path.join(polygonisation_directory, file))
    ]
    # print(f'we start with {len(all_tiles)} tiles')
    grid_size = tile_size * res * (1 - overlap_rate)
    large_tile_size = tile_size * res * reassembly_edge
    overlapping_large_tiles = get_overlapping_tiles(
        all_tiles,
        bshape,
        tile_size=large_tile_size,
        res=res,
        small_tile_size=tile_size,
        overlap_rate=overlap_rate,
    )
    overlapping_tile_paths = [
        os.path.join(polygonisation_directory, tile) for tile in overlapping_large_tiles
    ]
    # print(f'we end with {len(overlapping_tile_paths)} tiles')
    return overlapping_tile_paths


def find_small_tiles(
    polygonisation_id: str,
    bshape: Polygon,
    res: float = 0.3,
    tile_size: int = 512,
    overlap: float = 0,
) -> list[str or Path]:
    tiling_id = get_polygon_details(polygonisation_id)["tile_id"]
    tif_directory = get_tiling_details(tiling_id)["tile_directory"]
    all_entries = os.listdir(tif_directory)
    tiles = [
        entry
        for entry in all_entries
        if os.path.isfile(os.path.join(tif_directory, entry))
    ]
    overlapping_tiles = get_overlapping_tiles(
        tiles,
        bshape,
        tile_size=res * tile_size,
        res=res,
        small_tile_size=tile_size,
        overlap_rate=overlap,
    )
    overlapping_tile_paths = [
        os.path.join(tif_directory, tile) for tile in overlapping_tiles
    ]
    return overlapping_tile_paths


if __name__ == "__main__":
    root_dir = Path(__file__).parents[4]
    data_path = root_dir / "data"

    project_list = [
        "trondheim_1991",
        "trondheim_1999",
        "trondheim_2006",
        "trondheim_2011",
        "trondheim_2016",
        "trondheim_kommune_2022",
    ]
    tile_size = 512
    res = 0.3
    overlap = 0
    reassembly_edge = 10
    reassembly_overlap = 1
    simplification_tolerance = 2
    buffer_distance = 0.5

    polygonisation_ids = find_polygonisations(
        project_list,
        res=res,
        tile_size=tile_size,
        overlap=overlap,
        reassembly_edge=reassembly_edge,
        reassembly_overlap=reassembly_overlap,
        simplication_tolerance=simplification_tolerance,
        buffer_distance=buffer_distance,
    )
    print(f"we get the following most recent polygonisations:{polygonisation_ids}")

    grid_size = tile_size * res * (1 - overlap)
    x_tile = 3653  # 3696
    y_tile = 45811  # 45796
    bshape = get_bounding_shape(x_tile, y_tile)

    selected_large_tiles = find_large_tiles(
        polygonisation_ids["trondheim_kommune_2022"],
        bshape,
        res=res,
        tile_size=tile_size,
        overlap_rate=overlap,
        reassembly_edge=reassembly_edge,
        reassembly_overlap=reassembly_overlap,
    )
    print(f"we get the following large tiles:{selected_large_tiles}")

    overlapping_small_tiles = find_small_tiles(
        polygonisation_ids["trondheim_kommune_2022"], bshape
    )
    print(f"we get the following small tiles:{overlapping_small_tiles}")

# %%
