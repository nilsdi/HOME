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
import time

from HOME.utils.project_paths import get_prediction_details, get_download_str

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
            large_tile_coords[f"{x_column}_{y_row}"] = [
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
    channels: int = 3,
) -> tuple[np.array, bool]:
    """
    Assembles a large tile from the small tiles (without coordinates)

    Args:
    - coords: list of the top left and bottom right coordinates of the large tile
    - small_tiles: list of the names of the small tiles that belong to the large tile
    - tile_size_px: size of the small tiles in pixels
    - channels: number of channels in the small tiles

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
    large_tile = np.full((extend_x_t_px, extend_y_t_px, channels), 0, dtype=np.uint8)

    # fill in the large tile with the small tiles
    for tile in small_tiles:
        [col, row] = extract_tile_numbers(tile)
        # now the pixel coordinates within the large tile:
        # px_x_tl = (bottom_right[0] - col - 1) * tile_size_px # tested to work, don't know why
        px_x_tl = (col - top_left[0]) * tile_size_px  # tested to work, don't know why
        px_y_tl = (top_left[1] - row + 1) * tile_size_px
        # px_y_tl = (row - bottom_right[1]) * tile_size_px
        # read the small tile
        small_tile = cv2.imread(tile)
        # add the small tile to the large tile
        large_tile[
            px_y_tl - tile_size_px : px_y_tl,
            px_x_tl : px_x_tl + tile_size_px,
        ] = small_tile
    return large_tile, True


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
    project_name: str,
    project_details: dict,
    download_details: dict,
    save_path: str,
) -> dict[dict]:
    """
    Reassembles a list of tiles into a smaller number of larger tiles with overlap.
    Args:
        tiles: list of strings with the names of all (!) the tiles we want to reassemble
        n_tiles_edge: number of tiles on each side of the large tile
        n_overlap: number of tiles to overlap
        tile_size: size of the small tiles in pixels
        res: resolution of the tiles in m/px
        project_name: name of the project (for naming the large tiles)
        project_details: dictionary with the details of the project (for naming the large tiles)
        download_details: dictionary with the details of the download (for naming the large tiles)
        save_path: path to save the large tiles

    Returns:
        None
    """
    start_time = time.time()

    project_channels = project_details["bandwidth"]
    tif_channels = 3
    if project_channels == "BW":
        tif_channels = 1
    # extend of all tiles we want to reassemble
    extend_tile_coords = get_max_min_extend(tiles)
    lab1_time = time.time()
    print(f"Getting set up took {lab1_time - start_time:.2f} seconds")
    # get the large tiles
    large_tile_coords = get_large_tiles(extend_tile_coords, n_tiles_edge, n_overlap)
    # match the small tiles to the large tiles
    lab2_time = time.time()
    print(f"Getting the large tile layout took {lab2_time - lab1_time:.2f} seconds")
    large_tile_tiles = match_small_tiles_to_large_tiles(tiles, large_tile_coords)
    lab3_time = time.time()
    print(f"matching small and large tiles took {lab3_time - lab2_time:.2f} seconds")
    geotiff_extends = {"directory": str(save_path)}
    geotiff_id = 0
    # assemble the large tiles
    tile_name_base = project_name + "resolution" + str(download_details["resolution"])
    for (
        lt_name,
        coords,
    ) in tqdm(large_tile_coords.items(), desc="Assembling large tiles"):
        matched_tiles = large_tile_tiles[lt_name]
        assembled_tile, contains_data = assemble_large_tile(
            coords, matched_tiles, channels=tif_channels
        )
        if not contains_data:  # if no small tiles, skip it, don't save it etc.
            continue
        # add georeference to assembled tile
        top_left, bottom_right = get_EPSG25833_coords(
            coords[0][1], coords[0][0], tile_size, res
        )
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
        geotiff_extends[geotiff_id] = {
            "filename": tile_name,
            "bounding_box": {
                "min_x": top_left[0],
                "min_y": bottom_right[1],
                "max_x": bottom_right[0],
                "max_y": top_left[1],
            },
            "width": bottom_right[0] - top_left[0],
            "height": top_left[1] - bottom_right[1],
        }
        geotiff_id += 1
        # write the assembled tile to disk
        with rasterio.open(save_path / tile_name, "w", **metadata) as dst:
            if tif_channels == 1:
                # For single-band images (not yet tested!)
                dst.write(assembled_tile, 1)
            else:
                # For multi-band images (e.g., RGB)
                for i in range(1, tif_channels + 1):
                    dst.write(assembled_tile[:, :, i - 1], i)
    lab4_time = time.time()
    print(f"Assembling the large tiles took {lab4_time - lab3_time:.2f} seconds")
    return geotiff_extends


def get_tiles(project_name: str, project_details: dict, data_path: Path) -> list[str]:
    """
    Get the tiles for a project.
    """
    tiles = [
        str(tile)
        for tile in Path(
            data_path
            / f"ML_prediction"  # /topredict/image/res_0.3/{project_name}/i_lzw_25"
            / "predictions"
            # / get_downloproject_details, project_name)
        ).rglob("*.tif")
    ]
    return tiles


# %% test the entire thing
from HOME.utils.project_paths import get_prediction_details, get_download_details
from HOME.utils.get_project_metadata import get_project_details

# get the tiles
if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[3]
    data_path = root_dir / "data"
    print(f"data_path: {data_path}")
    # read in the json for gdfs:
    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "r"
    ) as file:
        reassembled_tiles_log = json.load(file)
    highest_key = max([int(k) for k in reassembled_tiles_log.keys()])
    assembly_key = highest_key

    predictions = [
        20001,
    ]

    for prediction_id in predictions:
        assembly_key += 1
        prediction_details = get_prediction_details(prediction_id, data_path)
        prediction_folder = prediction_details["prediction_folder"]
        download_id = prediction_details["download_id"]
        project_name = prediction_details["project_name"]
        project_details = get_project_details(project_name)
        download_details = get_download_details(download_id, data_path)
        resolution = np.round(download_details["resolution"], 1)

        tiles = [
            str(tile) for tile in Path(data_path / prediction_folder).rglob("*.tif")
        ]
        # print(f"the data_path is {data_path}")
        # print(f"the folder we are looking into is {data_path / prediction_folder}")
        print(f"Found {len(tiles)} tiles for project {project_name}")
        if len(tiles) == 0:
            print(
                f"No tiles found for project {project_name} - we move on to the next!"
            )
            continue

        n_tiles_edge = 10
        n_overlap = 1
        tile_size = 512
        save_loc = (
            data_path
            / "ML_prediction/large_tiles"
            / get_download_str(download_id)
            / f"prediction_{prediction_id}"
            / f"tiling_{assembly_key}"
        )
        os.makedirs(save_loc, exist_ok=True)

        geotiff_extends = reassemble_tiles(
            tiles,
            n_tiles_edge,
            n_overlap,
            tile_size=tile_size,
            res=resolution,
            project_name=project_name,
            project_details=project_details,
            download_details=download_details,
            save_path=save_loc,
        )

        reassembled_tiles_log[str(assembly_key)] = {
            "project_name": project_name,
            "prediction_id": prediction_id,
            "download_id": download_id,
            "tile_directory": str(save_loc),
            "n_tiles_edge": n_tiles_edge,
            "n_overlap": n_overlap,
            "tile_size": tile_size,
        }

    # dump the logs
    with open(
        data_path / "metadata_log/reassembled_prediction_tiles.json", "w"
    ) as file:
        json.dump(reassembled_tiles_log, file, indent=4)

    with open(
        data_path
        / "metadata_log/reassembled_prediction_tiles"
        / f"assembly_{assembly_key}.json",
        "w",
    ) as file:
        json.dump(geotiff_extends, file, indent=4)


if __name__ == "__main__1":
    # Get the root directory of the project
    root_dir = Path(__file__).resolve().parents[3]
    # print(root_dir)
    # get the data path (might change)
    data_path = get_data_path(root_dir)
    data_path = root_dir / "data"

    projects = [
        # "trondheim_kommune_2020",
        # "trondheim_kommune_2021",
        "trondheim_kommune_2022",
        # "trondheim_2019",
    ]
    for project in projects:
        # project_details = get_project_details(root_dir, project)
        # TODO: adjust to new method
        tiles = get_tiles(project, project_details, data_path)
        print(f"Found {len(tiles)} tiles for project {project}")

        n_tiles_edge = 10
        n_overlap = 1
        save_loc = (
            data_path
            / "ML_prediction/large_tiles"
            # / "test_topredict_tiles"
            / get_project_str_res_name(project_details, project)
        )
        # save_loc = data_path / "temp/test_assembly/"
        os.makedirs(save_loc, exist_ok=True)
        res = 0.3
        tile_size = 512
        reassemble_tiles(
            tiles,
            n_tiles_edge,
            n_overlap,
            tile_size,
            res,
            project,
            project_details,
            save_loc,
        )

# %%
