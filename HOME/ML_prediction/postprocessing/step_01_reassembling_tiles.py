# %%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import json

# %%

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Parameters
project_name = "trondheim_kommune_2020"  # Example project name

overlap_rate = 0  # 0% overlap (prediction tiles)
x_nb_tiles = 20  # number of tiles in x direction larger tiles
y_overlap_nb_tiles = 0  # number of tiles in y direction overlap

"""
- still some issues with the naming of the tiles but works + coordinates in the geotifs should be right

"""


# %%


# Function to parse filename and extract its coordinates: row and col
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


def get_top_left(filenames: list[str]) -> tuple[int, int]:
    # Initialize minimum column and max row values (top left is actually smallest col and largest row)
    min_col, max_row = np.inf, 0
    for filename in filenames:
        col, row = extract_tile_numbers(filename)
        # Update minimum column and row values
        if col < min_col:
            min_col = col
        if row > max_row:
            max_row = row
    return min_col, max_row


def get_min_row(filenames: list[str]) -> int:
    # Initialize minimum row value
    min_row = np.inf
    for filename in filenames:
        row = extract_tile_numbers(filename)[1]
        # Update minimum row value: bottom of the image
        if row < min_row:
            min_row = row
    return min_row


def get_nb_row_col(filenames: list[str]) -> tuple[int, int]:
    # Determine grid size
    cols, rows = 0, 0
    col0 = get_top_left(filenames)[0]
    minrow = get_min_row(filenames)
    for filename in filenames:
        col, row = extract_tile_numbers(filename)
        if row - minrow + 1 > rows:
            rows = rows + 1
        if col - col0 + 1 > cols:
            cols = cols + 1
    return cols, rows


def order_files_by_xy(files):
    """
    Orders a list of filenames in the pattern '_1_0_x_y' by growing x and y.

    Args:
        files (list of str): List of filenames to order.

    Returns:
        list of str: List of filenames ordered by growing x and y.
    """
    # Extract x and y numbers from each filename using the extract_tile_numbers function
    file_info = [extract_tile_numbers(filename) for filename in files]

    # Sort the filenames by x and then by y
    ordered_files = sorted(
        files,
        key=lambda x: (file_info[files.index(x)][0], file_info[files.index(x)][1]),
    )

    return ordered_files


# %%

# here I want to make a function that would create the corresponding indices for the tiles sets.
"""let's say the big tiff is a large matrix A, size (rows, cols). We add padding to make it square.
Then we create submatrices of size (x_nb_tiles, x_nb_tiles) with overlap of y_nb_tiles.
We create a list of corresponding indices for each submatrix. these are the tiles that shoulod be in the 
same set, if they exist. If none of the tiles exist, we leave it empty. If some, we add them and fill out the 
rest with black tiles. We keep in mind the top left corner of each set, to get the coordinates"""


def get_indices_for_tile_sets_with_top_left(
    row0, col0, rows, cols, x_nb_tiles, y_nb_tiles
):
    # Calculate padding needed to make rows and cols a multiple of tile size
    row_padding = (x_nb_tiles - (rows % x_nb_tiles)) % x_nb_tiles
    col_padding = (x_nb_tiles - (cols % x_nb_tiles)) % x_nb_tiles

    # Adjusted matrix size to include padding
    padded_rows = rows + row_padding
    padded_cols = cols + col_padding

    # Use the padded sizes to calculate max_size
    max_size = max(padded_rows, padded_cols)
    list_of_submatrices_indexes = []
    top_left_indices = []  # List to store top left indices of each submatrix
    step_size = x_nb_tiles - y_nb_tiles  # Effective step size after considering overlap

    for row_start in range(row0 - y_nb_tiles, row0 + max_size, step_size):
        for col_start in range(col0 - y_nb_tiles, col0 + max_size, step_size):
            # Ensure row_end and col_end do not exceed padded matrix dimensions
            row_end = min(row_start + x_nb_tiles + 2 * y_nb_tiles, row0 + padded_rows)
            col_end = min(col_start + x_nb_tiles + 2 * y_nb_tiles, col0 + padded_cols)
            # Generate submatrix indices, clamping to the padded matrix boundaries
            submatrix_indexes = [
                (col, row)
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            ]
            list_of_submatrices_indexes.append(submatrix_indexes)
            top_left_indices.append((col_start, row_end))  # Capture the top left index

    return list_of_submatrices_indexes, top_left_indices


# %%

"""here we should modify this one to integrate get_indices_for_tile_sets"""


def make_tile_sets(ordered_files_og, ordered_files, x_nb_tiles, y_overlap_nb_tiles):

    # determine the number of rows and columns
    num_cols, num_rows = get_nb_row_col(ordered_files_og)
    # find the top left corner of the first tile
    col0, row_max = get_top_left(ordered_files_og)
    row0 = row_max - num_rows
    # get all indices possible for the tile sets + top left coordinates
    indices_tile_sets, top_left_coords = get_indices_for_tile_sets_with_top_left(
        row0, col0, num_rows, num_cols, x_nb_tiles, y_overlap_nb_tiles
    )
    # determine how many sets of tiles we need
    nb_sets = len(indices_tile_sets)

    # initialize the list of tile sets
    tile_sets = [[] for _ in range(nb_sets)]

    for set_index, indices in enumerate(indices_tile_sets):
        for col, row in indices:
            tile_name = f"{project_name}_b_{col}_{row}.tif"
            if tile_name in ordered_files:
                tile_sets[set_index].append(tile_name)
            else:
                tile_sets[set_index].append(f"black_tile_{col}_{row}")
    tile_sets_dict = {
        top_left: tile_set
        for top_left, tile_set in zip(top_left_coords, tile_sets)
        if not all(tile.startswith("black_tile") for tile in tile_set) and tile_set
    }
    return tile_sets_dict


# %%
def stitch_tiles_together(
    ordered_files, top_left, tile_files, output_file, input_dir, tile_size_px=512
):
    sample_path = os.path.join(input_dir, ordered_files[0])
    sample_dataset = gdal.Open(sample_path)
    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")
    original_tile_width = tile_size_px
    original_tile_height = tile_size_px
    # get the number of rows and columns
    # set up the new tiles
    overlap_width = int(original_tile_width * overlap_rate)
    overlap_height = int(original_tile_height * overlap_rate)
    # small tiles
    tile_width = original_tile_width - overlap_width
    tile_height = original_tile_height - overlap_height
    # larger tiles
    total_width = (x_nb_tiles + 2 * y_overlap_nb_tiles) * tile_width
    total_height = (x_nb_tiles + 2 * y_overlap_nb_tiles) * tile_height

    # create the output file
    driver = gdal.GetDriverByName("GTiff")
    # col0, row0 = get_top_left(tile_files)
    col0, row_max = top_left
    row0 = row_max - (x_nb_tiles - 2 * y_overlap_nb_tiles)

    # calculate the coordinates (top left)
    x_coord = col0 * resolution * tile_size_px
    y_coord = (row_max - 1) * resolution * tile_size_px
    # now the coordinates in the metadata are the top left corner
    # but the names indicate bottom left still so maybe I should change that

    # set the metadata of the new tif to include the coordinates
    geo_transform = (x_coord, resolution, 0, y_coord, 0, -resolution)

    combined_dataset = driver.Create(
        str(output_file),
        total_width,
        total_height,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType,
    )
    combined_dataset.SetGeoTransform(geo_transform)
    combined_dataset.SetProjection(sample_dataset.GetProjection())

    # create a black tile
    black_tile = np.full(
        (tile_height, tile_width, sample_dataset.RasterCount), 0, dtype=np.uint8
    )

    for tile_name in tile_files:
        if "black_tile" in tile_name:
            # Handle black tiles
            parts = tile_name.split("_")
            row = int(parts[-1]) - row0
            col = int(parts[-2]) - col0
            x_offset = col * tile_width
            y_offset = total_height - ((row + 1) * tile_height)
            # add the black tile in the dataset
            for band in range(1, sample_dataset.RasterCount + 1):
                combined_dataset.GetRasterBand(band).WriteArray(
                    black_tile[:, :, band - 1], x_offset, y_offset
                )
            # continue  # Skip the rest of the loop for this iteration
        else:
            # Handle regular tiles
            tile_path = os.path.join(input_dir, tile_name)

            tile_dataset = gdal.Open(tile_path)
            if tile_dataset:
                col, row = extract_tile_numbers(tile_name)
                col, row = col - col0, row - row0
                x_offset = col * tile_size_px
                y_offset = total_height - ((row + 1) * tile_height)
                for band in range(1, tile_dataset.RasterCount + 1):
                    data = tile_dataset.GetRasterBand(band).ReadAsArray()
                    combined_dataset.GetRasterBand(band).WriteArray(
                        data, x_offset, y_offset
                    )

    print(f"Final stitched image saved as {output_file}")
    combined_dataset = None


# %%

# Main script

if __name__ == "__main__":

    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, "r") as file:
        project_dict = json.load(file)

    # Get the resolutiom and other details
    resolution = project_dict[project_name]["resolution"]
    compression_name = project_dict[project_name]["compression_name"]
    compression_value = project_dict[project_name]["compression_value"]

    # path to the prediction tiles
    # input_dir = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}"
    # path to the og tiles
    input_dir_og = (
        root_dir
        / f"data/ML_prediction/topredict/image/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}"
    )
    input_dir = input_dir_og
    # output location for the reassembled tile: in a diff folder
    # output for og tiles (for testing purposes)
    output_dir = (
        root_dir
        / f"data/ML_prediction/topredict/image/res_{resolution}/{project_name}/reassembled_tiles"
    )
    # output for prediction tiles
    # output_dir = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/reassembled_tiles"

    # assuming we keep the same format for tile names (include row and column)
    # Use glob to find all .tif files in the directory
    tif_files = list(input_dir.glob("*.tif"))
    tif_filenames = [file.name for file in tif_files]
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ordered_files = order_files_by_xy(tif_filenames)

    tif_files_og = list(input_dir_og.glob("*.tif"))
    tif_filenames_og = [file.name for file in tif_files_og]
    ordered_files_og = order_files_by_xy(tif_filenames_og)

    tile_sets_dict = make_tile_sets(
        ordered_files_og, ordered_files, x_nb_tiles, y_overlap_nb_tiles
    )

    for top_left, tile_set in tile_sets_dict.items():
        # Define the output file for the stitched image
        final_output_file = (
            output_dir / f"stitched_tif_{project_name}_{top_left[0]}_{top_left[1]}.tif"
        )
        # Stitch back together the tile
        stitch_tiles_together(
            ordered_files,
            top_left,
            tile_set,
            final_output_file,
            input_dir,
            tile_size_px=512,
        )


# %%
