""" Reassembly of the 512x512 tifs into larger geotifs.

Ready to take any set of tiles that are named according to the tile grid
and reassemble them into larger tiles with proper georeferencing.
"""

# %% imports
# general tools
import numpy as np
import json
import cv2
import rasterio
from rasterio.transform import from_origin

# file handling
from pathlib import Path
import os

# analysis of code
from tqdm import tqdm
from HOME.utils.project_paths import (
    get_prediction_details,
    get_tiling_details,
)
from HOME.get_data_path import get_data_path

# %%

root_dir = Path(__file__).resolve().parents[3]
data_path = get_data_path(root_dir)

# %% functions


def extract_tile_numbers(filename: str) -> list[int, int]:
    """
    Extracts the x/col and y/row (coordinates) from a filename
    of pattern '_x_y'.

    Args:
        filename (str): name of a tile cut from a larger image.

    Returns:
        tuple: row, col number of the tile  in the absolute tile system,
        meaning a tile with the name '_0_0' is the top left tile.
    """
    parts = filename.split("_")
    col = int(parts[-2])  # x_coord
    row = int(parts[-1].split(".")[0])  # y_coord
    return [col, row]


def get_max_min_extend(tiles: list[str]) -> list[int, int, int, int]:
    """
    Get the extend of all tiles given in tile coordinates

    Args:
    - tiles: list of strings with the names of all the tiles we want to reassemble

    Returns:
    - min_x: minimum x coordinate of the tiles
    - max_x: maximum x coordinate of the tiles
    - min_y: minimum y coordinate of the tiles
    - max_y: maximum y coordinate of the tiles
    """
    tile_coords = [extract_tile_numbers(tile) for tile in tiles]
    min_x = min([coord[0] for coord in tile_coords])
    max_x = max([coord[0] for coord in tile_coords])
    min_y = min([coord[1] for coord in tile_coords])
    max_y = max([coord[1] for coord in tile_coords])
    return min_x, max_x, min_y, max_y


def get_large_tiles(
    extend_tile_coords: list[int], n_tiles_edge: int, n_overlap: int
) -> dict[list[int]]:
    """
    Checks the number of large tiles needed to host all given tiles assuming
    we want a grid of uniform reassambled tiles with a given number of small tiles
    as the overlap, and square assembled tiles with n_tiles_edge on each side.

    Args:
    - extend_tile_coords: list of the minx, maxx, miny, maxy of the tiles
    - n_tiles_edge: number of tiles on each side of the large tiles
    - n_overlap: number of tiles to overlap (typically 1)

    Returns:
    - large_tile_coords: dictionary based on tile name (based on  grid position)
        and coordinates of the large tiles - top left & bottom right (in global tile grid)
    """
    # get the extend of the tiles given in tile coordinates
    min_x, max_x, min_y, max_y = extend_tile_coords
    # round the minimum points down with n_edge - n-overlap
    # so that all large tiles are aligned across projects
    edge_coverage = n_tiles_edge - n_overlap
    min_x = (min_x // edge_coverage) * edge_coverage
    min_y = (min_y // edge_coverage) * edge_coverage
    x_dist = max_x - min_x
    y_dist = max_y - min_y

    # get the area that is uniquely covered by each large tile
    edge_coverage = n_tiles_edge - n_overlap
    center_coverage = n_tiles_edge - 2 * n_overlap

    # fit a grid of n_tiles_edge x n_tiles_edge tiles with n_overlap overlap
    n_large_tiles_x = 3 + (x_dist - 2 * edge_coverage) // center_coverage
    n_large_tiles_y = 3 + (y_dist - 2 * edge_coverage) // center_coverage

    # get the coordinates of the large tiles
    # n_large_tiles = n_large_tiles_x * n_large_tiles_y
    large_tile_coords = {}
    for x_column in range(n_large_tiles_x):
        if x_column == 0:
            x_coord_l = min_x
            x_coord_r = min_x + n_tiles_edge
        else:
            x_coord_l = x_coord_l + n_tiles_edge - n_overlap
            x_coord_r = x_coord_l + n_tiles_edge
        for y_row in range(n_large_tiles_y):
            if y_row == 0:
                y_coord_u = min_y - 1 + n_tiles_edge
                y_coord_d = y_coord_u - n_tiles_edge
            else:
                y_coord_u = y_coord_u + n_tiles_edge - n_overlap
                y_coord_d = y_coord_u - n_tiles_edge
            # save the coordinates of the large tile (named "x_y") with top left, bottom right
            large_tile_coords[f"{x_coord_l}_{y_coord_u}"] = [
                [x_coord_l, y_coord_u],
                [x_coord_r, y_coord_d],
            ]

    return large_tile_coords


def match_small_tiles_to_large_tiles_old(
    tiles: list[str], large_tile_coords: dict[list[list[int]]]
) -> dict[list[str]]:
    """
    Matches the small tiles to the large tiles given the coordinates of the large tiles.
    returns a list with the names of all the small tiles that belong to each large tile.

    Args:
    - tiles: list of strings with the names of all the tiles we want to reassemble
    - large_tile_coords: dictionary with the coordinates of the large tiles - first the top left
    and then the bottom right corner.

    Returns:
    - large_tile_tiles: dictionary to assign small tiles to large tile name (see above)
    """
    large_tile_tiles = {lt_name: [] for lt_name in large_tile_coords.keys()}
    for tile in tiles:
        tile_coords = extract_tile_numbers(tile)
        # find the large tile that the small tile belongs to
        for lt_name, coords in large_tile_coords.items():
            if (
                tile_coords[0] >= coords[0][0]
                and tile_coords[0] < coords[1][0]
                and tile_coords[1] <= coords[0][1]
                and tile_coords[1] > coords[1][1]
            ):
                large_tile_tiles[lt_name].append(tile)
    return large_tile_tiles


from scipy.spatial import KDTree
import numpy as np


def match_small_tiles_to_large_tiles(tiles, large_tile_coords):
    """
    Args:
    - tiles: list of strings with the names of all the tiles we want to reassemble
    - large_tile_coords: dictionary with the coordinates of the large tiles - first the top left
    and then the bottom right corner.

    Returns:
    - large_tile_tiles: dictionary to assign small tiles to large tile name (see above)
    """
    large_tile_tiles = {lt_name: [] for lt_name in large_tile_coords.keys()}

    # Prepare data for KDTree
    large_tile_centers = []
    large_tile_names = []
    for lt_name, coords in large_tile_coords.items():
        center_x = (coords[0][0] + coords[1][0]) / 2
        center_y = (coords[0][1] + coords[1][1]) / 2
        large_tile_centers.append((center_x, center_y))
        large_tile_names.append(lt_name)

    # Build KDTree
    kdtree = KDTree(large_tile_centers)

    for tile in tiles:
        tile_coords = extract_tile_numbers(tile)
        tile_center = (tile_coords[0] + 0.5, tile_coords[1] - 0.5)

        # Query KDTree to find the 4 nearest large tiles
        dists, indices = kdtree.query(tile_center, k=4)

        for idx in indices:
            lt_name = large_tile_names[idx]

            # Check if the tile is within the bounds of the large tile
            coords = large_tile_coords[lt_name]
            if (
                tile_coords[0] >= coords[0][0]
                and tile_coords[0] < coords[1][0]
                and tile_coords[1] <= coords[0][1]
                and tile_coords[1] > coords[1][1]
            ):
                large_tile_tiles[lt_name].append(tile)

    return large_tile_tiles


def assemble_large_tile(
    coords: list[list[int]],
    small_tiles: list[str],
    tile_size_px: int = 512,
) -> tuple[np.array, bool]:
    """
    Assembles a large tile from the small tiles (without coordinates)

    Args:
    - coords: list of the top left and bottom right coordinates of the large tile
    - small_tiles: list of the names of the small tiles that belong to the large tile
    - tile_size_px: size of the small tiles in pixels

    Returns:
    - large_tile: np.array with the large tile
    - contains_data: boolean indicating if the large tile contains data
    """
    if len(small_tiles) == 0:
        return np.array([]), False
    # sort the small tiles by their coordinates in x and y
    small_tiles.sort(key=lambda x: extract_tile_numbers(x)[0])
    small_tiles.sort(key=lambda x: extract_tile_numbers(x)[1])

    top_left = coords[0]
    bottom_right = coords[1]
    # create a large tile with the right size (in pixels)
    extend_x_t = coords[1][0] - coords[0][0]
    extend_y_t = coords[0][1] - coords[1][1]
    extend_x_t_px = extend_x_t * tile_size_px
    extend_y_t_px = extend_y_t * tile_size_px
    large_tile = np.full((extend_x_t_px, extend_y_t_px), 0, dtype=np.uint8)

    # fill in the large tile with the small tiles
    for tile in small_tiles:
        [col, row] = extract_tile_numbers(tile)
        # now the pixel coordinates within the large tile:
        # px_x_tl = (bottom_right[0] - col - 1) * tile_size_px # tested to work, don't know why
        px_x_tl = (col - top_left[0]) * tile_size_px  # tested to work, don't know why
        px_y_tl = (top_left[1] - row + 1) * tile_size_px
        # px_y_tl = (row - bottom_right[1]) * tile_size_px
        # read the small tile
        small_tile = cv2.imread(tile, cv2.IMREAD_GRAYSCALE)
        # add the small tile to the large tile
        large_tile[
            px_y_tl - tile_size_px : px_y_tl,
            px_x_tl : px_x_tl + tile_size_px,
        ] += small_tile
    large_tile = np.clip(large_tile, 0, 255)
    return large_tile, True


def get_coords_m(
    row, col, tile_size: int, res: float, overlap_rate: int = 0
) -> tuple[list[int], list[int]]:
    """
    Get the coordinates of the top left corner and bottom right corner of a tile in
    EPSG:25833, based on its  row and column in the grid of tiles.
    """
    # get the coordinates of the top left corner of the tile
    grid_size_m = tile_size * (1 - overlap_rate) * res
    x_tl = col * grid_size_m
    y_tl = row * grid_size_m
    x_br = (x_tl + 1) * grid_size_m
    y_br = (y_tl - 1) * grid_size_m
    return [x_tl, y_tl], [x_br, y_br]


def get_transform(large_tile_coords: list[list[int]], tile_size: int, res: float):
    """
    Get the affine transformation to go from pixel coordinates to EPSG:25833
    """
    # get the coordinates of the top left corner of the tile

    # get the affine transformation
    return


def reassemble_tiles(
    project_name: str,
    prediction_id: int,
    n_tiles_edge: int,
    n_overlap: int,
) -> dict[dict]:
    """
    Reassembles a list of tiles into a smaller number of larger tiles with overlap.
    Args:
        project_name: name of the project
        prediction_id: id of the prediction
        n_tiles_edge: number of tiles on each side of the large tile
        n_overlap: number of tiles to overlap

    Returns:
        None
    """

    # load the assembly metadata log
    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "r"
    ) as file:
        reassembled_tiles_log = json.load(file)
    highest_key = max([int(k) for k in reassembled_tiles_log.keys()])
    assembly_key = highest_key + 1

    # Read metadata from tiling and prediction
    prediction_details = get_prediction_details(prediction_id, data_path)
    prediction_folder = prediction_details["prediction_folder"]
    tile_id = prediction_details["tile_id"]
    tiling_detail = get_tiling_details(tile_id, data_path)
    res = np.round(tiling_detail["res"], 1)
    tile_size = int(tiling_detail["tile_size"])
    crs = tiling_detail["crs"]

    assert (
        prediction_details["project_name"] == project_name
    ), "Prediction id does not correspond to the project name"

    # get the tiles
    tiles = [str(tile) for tile in Path(prediction_folder).rglob("*.tif")]

    # extend of all tiles we want to reassemble
    extend_tile_coords = get_max_min_extend(tiles)

    # get the large tiles
    large_tile_coords = get_large_tiles(extend_tile_coords, n_tiles_edge, n_overlap)

    # match the small tiles to the large tiles
    large_tile_tiles = match_small_tiles_to_large_tiles(tiles, large_tile_coords)

    save_path = (
        data_path
        / "ML_prediction/large_tiles"
        / project_name
        / f"tiles_{tile_id}"
        / f"prediction_{prediction_id}"
        / f"assembly_{assembly_key}"
    )
    os.makedirs(save_path, exist_ok=True)

    geotiff_extends = {"directory": str(save_path)}
    geotiff_id = 0
    # assemble the large tiles
    tile_name_base = project_name + "_resolution" + str(res)
    for (
        lt_name,
        coords,
    ) in tqdm(large_tile_coords.items(), desc="Assembling large tiles"):
        matched_tiles = large_tile_tiles[lt_name]
        assembled_tile, contains_data = assemble_large_tile(
            coords,
            matched_tiles,
        )
        if contains_data:  # if no small tiles, skip it, don't save it etc.
            # add georeference to assembled tile
            top_left, bottom_right = get_coords_m(
                coords[0][1], coords[0][0], tile_size, res
            )
            # get the affine transformation to go from pixel coordinates to EPSG:25833
            transform = from_origin(top_left[0], top_left[1], res, res)
            metadata = {
                "driver": "GTiff",
                "dtype": "uint8",
                "height": assembled_tile.shape[0],
                "width": assembled_tile.shape[1],
                "transform": transform,
                "crs": crs,
                "count": 1,
            }
            # save the assembled tile
            tile_name = f"{tile_name_base}_{lt_name}.tif"
            geotiff_extends[tile_name] = {
                "grid_x_min": coords[0][0],
                "grid_x_max": coords[1][0],
                "grid_y_min": coords[1][1],
                "grid_y_max": coords[0][1],
            }
            geotiff_id += 1
            # write the assembled tile to disk
            with rasterio.open(save_path / tile_name, "w", **metadata) as dst:
                dst.write(assembled_tile, 1)

    reassembled_tiles_log[str(assembly_key)] = {
        "project_name": project_name,
        "prediction_id": prediction_id,
        "tile_id": tile_id,
        "tile_directory": str(save_path),
        "n_tiles_edge": n_tiles_edge,
        "n_overlap": n_overlap,
        "tile_size": tile_size,
    }

    # dump the logs
    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "w"
    ) as file:
        json.dump(reassembled_tiles_log, file, indent=4)
    return assembly_key, geotiff_extends


# %% test the entire thing

# get the tiles
if __name__ == "__main__":
    print(f"data_path: {data_path}")
    # read in the json for gdfs:

    predictions = [
        20001,
    ]

    for prediction_id in predictions:

        n_tiles_edge = 10
        n_overlap = 1

        geotiff_extends = reassemble_tiles(
            project_name="test_project",
            prediction_id=prediction_id,
            n_tiles_edge=n_tiles_edge,
            n_overlap=n_overlap,
        )

# %%
