from shapely.geometry import Polygon

tile_size = 512
res = 0.3
overlap = 0
grid_size = tile_size * res * (1 - overlap)


def bshape_from_tile_coords(x_tile, y_tile):
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
