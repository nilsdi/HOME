from shapely.geometry import Polygon

tile_size = 512
res = 0.3
overlap = 0
grid_size = tile_size * res * (1 - overlap)


def get_bounding_shape(x_tile: int, y_tile: int):
    """
    Create a bounding shape identical to the tile which
    is named by the given tile coordinates

    Args:
        x_tile (int): x-coordinate of the tile
        y_tile (int): y-coordinate of the tile

    Returns:
        bshape (Polygon): the bounding shape of the tile
    """
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
