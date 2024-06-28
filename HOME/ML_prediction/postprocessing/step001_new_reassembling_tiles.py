"""
Minimal version of how to reassemble tiles (e.g. from a prediction) into larger
tiles with some overlap. This is useful for large images that are too big to
process in one go.
"""

# %% imports


# %% functions


def extract_tile_numbers(filename: str) -> tuple[int, int]:
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
    return col, row


def get_large_tiles(
    tiles: list[str], n_tiles_edge: int, n_overlap: int
) -> dict[list[int]]:
    """
    Checks the number of large tiles needed to host all given tiles assuming
    we want a grid of uniform reassambled tiles with a given number of small tiles
    as the overlap, and square assembled tiles with n_tiles_edge on each side.
    """
    # get the extend of the tiles given in tile coordinates
    # fit a grid of n_tiles_edge x n_tiles_edge tiles with n_overlap overlap

    return  # a number of large tiles with their minimum and maximum coordinates


def match_small_tiles_to_large_tiles(
    tiles: list[str], large_tile_coords: list[list[int]]
) -> list[list[str]]:
    """
    Matches the small tiles to the large tiles given the coordinates of the large tiles.
    returns a list with the names of all the small tiles that belong to each large tile.
    """
    return  # a list of lists of tiles for each large tile


def assemble_large_tile(large_tile_coords: list[int], small_tiles: list[str]):
    """
    Assembles a large tile from the small tiles (without coordinates)
    """
    return  # a large tile


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


def get_transform(large_tile_coords: list[int], tile_size: int, res: float):
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
    large_tile_loc: str,
    project_name: str,
    project_details: dict,
):
    """
    Reassembles a list of tiles into a smaller number of larger tiles with overlap.
    Args:
    - tiles: list of strings with the names of all the tiles we want to reassemble
    - n_tiles_edge: number of tiles on each side of the large tile
    - n_overlap: number of tiles to overlap
    - large_tile_loc: location to save the large tiles
    - project_name: name of the project (for naming the large tiles)
    - project_details: dictionary with the details of the project (for naming the large tiles)
    """
    # get the large tiles
    large_tiles = get_large_tiles(tiles, n_tiles_edge, n_overlap)
    # match the small tiles to the large tiles
    matched_tiles = match_small_tiles_to_large_tiles(tiles, large_tiles.values())
    # assemble the large tiles
    tile_name_base = project_name + "resolution"
    for i, coords in large_tiles.items():
        assembled_tile = assemble_large_tile(coords, matched_tiles[i])
        # add georeference to assembled tile
        # write the assembled tile to disk
    return
