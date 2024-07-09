""" Reassembly of the 512x512 tifs into larger geotifs.

Ready to take any set of tiles that are named according to the tile grid 
and reassemble them into larger tiles with proper georeferencing.
"""

# %% imports
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
import json

from pathlib import Path

import matplotlib.pyplot as plt
from typing import Dict

# %% functions


def extract_tile_numbers(filename: str) -> list[int, int]:
    """
    Extracts the x/col and y/row (coordinates) from a filename
    of pattern '_x_y'.

    Args:
        filename (str): name of a tile cut from a larger image.

    Returns:
        tuple: row, col number of the tile  in the absolute system of , meaning a tile with the name '_0_0' is the top left tile.
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
    x_dist = max_x - min_x
    y_dist = max_y - min_y

    # get the area that is uniquely covered by each large tile
    edge_coverage = n_tiles_edge - n_overlap
    center_coverage = n_tiles_edge - 2 * n_overlap

    # fit a grid of n_tiles_edge x n_tiles_edge tiles with n_overlap overlap
    n_large_tiles_x = 2 + (x_dist - 2 * edge_coverage) // center_coverage
    n_large_tiles_y = 2 + (y_dist - 2 * edge_coverage) // center_coverage

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
            large_tile_coords[f"{x_column}_{y_row}"] = [
                [x_coord_l, y_coord_u],
                [x_coord_r, y_coord_d],
            ]

    return large_tile_coords


def match_small_tiles_to_large_tiles(
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
        for lt_name, coords in large_tile_coords.items():
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
    channels: int = 3,
) -> np.array:
    """
    Assembles a large tile from the small tiles (without coordinates)
    
    Args:
    - coords: list of the top left and bottom right coordinates of the large tile
    - small_tiles: list of the names of the small tiles that belong to the large tile
    - tile_size_px: size of the small tiles in pixels
    - channels: number of channels in the small tiles

    Returns:
    - large_tile: np.array with the large tile
    """
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
    large_tile = np.full((extend_x_t_px, extend_y_t_px, channels), 0, dtype=np.uint8)

    # fill in the large tile with the small tiles
    for tile in small_tiles:
        [col, row] = extract_tile_numbers(tile)
        # now the pixel coordinates within the large tile:
        # px_x_tl = (bottom_right[0] - col - 1) * tile_size_px # tested to work, don't know why
        px_x_tl = (col - top_left[0]) * tile_size_p  # tested to work, don't know why
        px_y_tl = (top_left[1] - row + 1) * tile_size_px
        # px_y_tl = (row - bottom_right[1]) * tile_size_px
        # read the small tile
        small_tile = cv2.imread(tile)
        # add the small tile to the large tile
        large_tile[
            px_y_tl - tile_size_px : px_y_tl,
            px_x_tl : px_x_tl + tile_size_px,
        ] = small_tile
    return large_tile


def get_EPSG25833_coords(
    row, col, tile_size: int, res: float
) -> tuple[list[int], list[int]]:
    """
    Get the coordinates of the top left corner and bottom right corner of a tile in
    EPSG:25833, based on its  row and column in the grid of tiles.
    """
    # get the coordinates of the top left corner of the tile
    x_tl = col * tile_size * res
    y_tl = row * tile_size * res
    x_br = (x_tl + 1) * tile_size * res
    y_br = (y_tl - 1) * tile_size * res
    return [x_tl, y_tl], [x_br, y_br]


def get_transform(large_tile_coords: list[list[int]], tile_size: int, res: float):
    """
    Get the affine transformation to go from pixel coordinates to EPSG:25833
    """
    # get the coordinates of the top left corner of the tile

    # get the affine transformation
    return


def reassemble_tiles(
    tiles: list[str],
    n_tiles_edge: int,
    n_overlap: int,
    tile_size: int,
    res: float,
    large_tile_loc: str,
    project_name: str,
    project_details: dict,
    save_path: str,
):
    """
    Reassembles a list of tiles into a smaller number of larger tiles with overlap.
    Args:
    - tiles: list of strings with the names of all (!) the tiles we want to reassemble
    - n_tiles_edge: number of tiles on each side of the large tile
    - n_overlap: number of tiles to overlap
    - tile_size: size of the small tiles in pixels
    - res: resolution of the tiles in m/px
    - large_tile_loc: location to save the large tiles
    - project_name: name of the project (for naming the large tiles)
    - project_details: dictionary with the details of the project (for naming the large tiles)
    """
    project_channels = project_details["channels"]
    tif_channels = 3
    if project_channels == "BW":
        tif_channels = 1
    # extend of all tiles we want to reassemble
    extend_tile_coords = get_max_min_extend(tiles)
    # get the large tiles
    large_tile_coords = get_large_tiles(extend_tile_coords, n_tiles_edge, n_overlap)
    # match the small tiles to the large tiles
    large_tile_tiles = match_small_tiles_to_large_tiles(tiles, large_tile_coords)
    # assemble the large tiles
    tile_name_base = project_name + "resolution" + str(project_details["resolution"])
    for (
        lt_name,
        coords,
    ) in large_tile_coords.items():
        matched_tiles = large_tile_tiles[lt_name]
        assembled_tile = assemble_large_tile(
            coords, matched_tiles, channels=tif_channels
        )
        # add georeference to assembled tile
        top_left = get_EPSG25833_coords(coords[0][1], coords[0][0], tile_size, res)[0]
        # get the affine transformation to go from pixel coordinates to EPSG:25833
        transform = from_origin(top_left[0], top_left[1], res, res)
        metadata = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": tif_channels,
            "height": assembled_tile.shape[0],
            "width": assembled_tile.shape[1],
            "transform": transform,
            "crs": "EPSG:25833",
        }
        # save the assembled tile
        tile_name = f"{tile_name_base}_{lt_name}.tif"
        # write the assembled tile to disk
        with rasterio.open(save_path / tile_name, "w", **metadata) as dst:
            if tif_channels == 1:
                # For single-band images (not yet tested!)
                dst.write(assembled_tile, 1)
            else:
                # For multi-band images (e.g., RGB)
                for i in range(1, tif_channels + 1):
                    dst.write(assembled_tile[:, :, i - 1], i)
    return


# %% test the entire thing
# get the tiles
from HOME.get_data_path import get_data_path

if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = Path(__file__).resolve().parents[3]
    # print(root_dir)
    # get the data path (might change)
    data_path = get_data_path(root_dir)
    data_path = root_dir / "data"
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "r"
    ) as file:
        project_details = json.load(file)

    project_name = "trondheim_kommune_2021"
    project_details = project_details[project_name]
    tiles = [
        str(tile)
        for tile in Path(
            data_path / f"ML_prediction/topredict/image/res_0.3/{project_name}/i_lzw_25"
        ).rglob("*.tif")
    ]

    n_tiles_edge = 10
    n_overlap = 1
    large_tile_loc = data_path / "ML_prediction/predicted_tiles"
    save_loc = data_path / "temp/test_assembly"
    res = 0.3
    tile_size = 512
    reassemble_tiles(
        tiles,
        n_tiles_edge,
        n_overlap,
        tile_size,
        res,
        large_tile_loc,
        project_name,
        project_details,
        save_loc,
    )

# %%