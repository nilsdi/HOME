# %%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import cv2

# %%

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Parameters
project_name = "trondheim_1979"  # Example project name
x_km = 5  # Size of each small tile in kilometers
y_overlap = 500  # Overlap in meters for the bigger tiles
overlap_rate = 0  # 0% overlap (prediction tiles)


"""
- we want to do all of this given only the project name
- there will be one folder per project with all the tiles of prediction
- + one text file with the path + names of the tiles?
- need the original ortophoto geotif (to get the coordinates + resolution), this I have yet to do

"""


# %%

"""
- first we extract the nb of rows and cols
- then we order the files
- then we create a list of ordered columns
- then we combine the tiles into columns + stitch the columns together
"""


# Function to parse filename and extract its coordinates: row and col
def extract_tile_numbers(filename):
    """
    Extracts the x and y (row and col) from a filename
    of pattern '_1_0_x_y'.

    Args:
        filename (str): The filename to extract the tile info from.

    Returns:
        tuple: row, col part extracted from the filename.
    """
    parts = filename.split("_")
    row = int(parts[-2])
    col = int(parts[-1].split(".")[0])
    return row, col


def get_nb_row_col(file_list_path):
    # Read filenames from the text file
    with open(file_list_path, "r") as file:
        filenames = [line.strip() for line in file]
    # Determine grid size
    cols, rows = 0, 0
    for filename in filenames:
        col, row = extract_tile_numbers(filename)
        if row + 1 > rows:
            rows = row + 1
        if col + 1 > cols:
            cols = col + 1
    return rows, cols


rows, cols = get_nb_row_col(file_list_path)


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


def get_columns_from_ordered_files(ordered_files):
    """
    Creates lists of columns of tiles from the ordered files.

    Args:
        ordered_files (list of str): List of ordered filenames.

    Returns:
        list of list of str: A list where each element is a list of filenames in a column.
    """
    columns = {}
    for filename in ordered_files:
        x, y = extract_tile_numbers(filename)
        if x not in columns:
            columns[x] = []
        columns[x].append(filename)

    # Sort columns by their keys (y indices)
    sorted_columns = [columns[key] for key in sorted(columns.keys())]
    return sorted_columns


def stitch_tiles_together(tile_files, output_file, num_columns, num_rows, input_dir):
    # Open one of the files to get the tile size and georeference info
    sample_path = os.path.join(input_dir, tile_files[0])
    sample_dataset = gdal.Open(sample_path)
    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")

    original_tile_width = sample_dataset.RasterXSize
    original_tile_height = sample_dataset.RasterYSize

    # Calculate overlap in pixels
    overlap_width = int(original_tile_width * overlap_rate)
    overlap_height = int(original_tile_height * overlap_rate)

    # Adjust tile dimensions to remove overlap
    tile_width = original_tile_width - overlap_width
    tile_height = original_tile_height - overlap_height

    # Calculate the total size of the output image
    total_width = tile_width * num_columns + overlap_width
    total_height = tile_height * num_rows + overlap_height

    # Create a new raster dataset for the combined image
    driver = gdal.GetDriverByName("GTiff")
    combined_dataset = driver.Create(
        str(output_file),
        total_width,
        total_height,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType,
    )
    combined_dataset.SetGeoTransform(sample_dataset.GetGeoTransform())
    combined_dataset.SetProjection(sample_dataset.GetProjection())

    # Define a white tile
    white_tile = np.full(
        (tile_height, tile_width, sample_dataset.RasterCount), 255, dtype=np.uint8
    )

    # Create a dictionary to map tile positions to filenames
    tile_dict = {extract_tile_numbers(filename): filename for filename in tile_files}

    # Paste each tile into the combined dataset
    for column in range(num_columns):
        for row in range(num_rows):
            tile_pos = (column, row)
            if tile_pos in tile_dict:
                tile_path = os.path.join(input_dir, tile_dict[tile_pos])
                tile_dataset = gdal.Open(tile_path)
                if tile_dataset:
                    for band in range(1, tile_dataset.RasterCount + 1):
                        # Read the tile with overlap removed
                        data = tile_dataset.GetRasterBand(band).ReadAsArray(
                            overlap_width // 2,
                            overlap_height // 2,
                            tile_width,
                            tile_height,
                        )
                        x_offset = column * tile_width
                        y_offset = row * tile_height
                        combined_dataset.GetRasterBand(band).WriteArray(
                            data, x_offset, y_offset
                        )
            else:
                # If no tile, fill the area with white (or any other placeholder)
                for band in range(1, sample_dataset.RasterCount + 1):
                    combined_dataset.GetRasterBand(band).WriteArray(
                        white_tile[:, :, band - 1],
                        column * tile_width,
                        row * tile_height,
                    )

    print(f"Final stitched image saved as {output_file}")
    # Close the datasets
    combined_dataset = None


# %%


def split_large_tile_into_small_tiles(
    large_tile_path, output_dir, x_km, y_overlap, resolution, project_name
):
    large_tile = gdal.Open(str(large_tile_path))

    if not large_tile:
        raise FileNotFoundError(
            f"Unable to open large tile TIFF file: {large_tile_path}"
        )

    large_tile_width = large_tile.RasterXSize
    large_tile_height = large_tile.RasterYSize
    tile_size_pixels = int((x_km * 1000) / resolution)
    overlap_pixels = int(y_overlap / resolution)

    # Adjust loop to account for overlap on all sides
    for y in range(
        0, large_tile_height - overlap_pixels, tile_size_pixels - overlap_pixels
    ):
        for x in range(
            0, large_tile_width - overlap_pixels, tile_size_pixels - overlap_pixels
        ):
            x_end = min(x + tile_size_pixels + overlap_pixels, large_tile_width)
            y_end = min(y + tile_size_pixels + overlap_pixels, large_tile_height)

            # Ensure we have a valid tile size after accounting for overlap
            if x_end - x > overlap_pixels and y_end - y > overlap_pixels:
                row_index = y // (tile_size_pixels - overlap_pixels)
                col_index = x // (tile_size_pixels - overlap_pixels)
                output_tile_path = os.path.join(
                    output_dir,
                    f"{project_name}_tile_{x_km}km_{row_index}_{col_index}.tif",
                )
                driver = gdal.GetDriverByName("GTiff")
                output_tile = driver.Create(
                    str(output_tile_path),
                    x_end - x,
                    y_end - y,
                    large_tile.RasterCount,
                    large_tile.GetRasterBand(1).DataType,
                )
                output_tile.SetGeoTransform(
                    (
                        large_tile.GetGeoTransform()[0] + x * resolution,  # top left x
                        resolution,  # pixel width
                        0,  # rotation, 0 if image is "north up"
                        large_tile.GetGeoTransform()[3] + y * resolution,  # top left y
                        0,  # rotation, 0 if image is "north up"
                        -resolution,  # pixel height (negative)
                    )
                )
                output_tile.SetProjection(large_tile.GetProjection())

                for band in range(1, large_tile.RasterCount + 1):
                    band_data = large_tile.GetRasterBand(band).ReadAsArray(
                        x, y, x_end - x, y_end - y
                    )
                    output_tile.GetRasterBand(band).WriteArray(band_data)

                output_tile = None
                print(f"Saved tile: {output_tile_path}")

    large_tile = None


# %%

# Main script

if __name__ == "__main__":

    # path to the prediction tiles
    input_dir = root_dir / f"data/ML_model/{project_name}/predictions/test"

    # path to the original orthophoto UPDATE with project name
    # ortho_path = root_dir / 'data/ML_prediction/{project_name}'
    ortho_path = (
        root_dir / "data/temp/test_zoe/images/archive/trondheim_0.3_1979_1_0.tif"
    )  # for testing
    # retrieve metadata from it (resolution + coordinates of the top left pixel)
    orig_ortho = gdal.Open(str(ortho_path))

    metadata_ortho = orig_ortho.GetGeoTransform()
    # resolution in meters
    resolution = metadata_ortho[1]
    # coord of the upper left corner of the upper left pixel
    x_coord = metadata_ortho[0]
    y_coord = metadata_ortho[3]
    # coord system
    coord_sstem = orig_ortho.GetProjection()

    # output location for the reassembled tile: in a diff folder
    # bc we make multiple reassembled tiles per project (5km x 5km with 500m overlap)
    """here add that we create the file if it doesnt exist yet"""
    output_dir = root_dir / f"data/ML_model/{project_name}/predictions/reassembled_tile"
    output_file = output_dir / f"full_tif_{project_name}.tif"

    # path to text file with (hopefully) all the file names of the tiles of the project
    # assuming we keep the same format for tile names (include row and column)
    file_list_path = root_dir / f"data/ML_model/{project_name}/dataset/test.txt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read filenames from the text file

    with open(file_list_path, "r") as file:
        file_list = [line.strip() for line in file]

    # Append '.tif' extension if not present
    file_list = [
        filename if filename.endswith(".tif") else f"{filename}.tif"
        for filename in file_list
    ]

    # Assuming number of columns (cols) and number of rows (rows) are known or can be calculated
    # No need to order files by x and y indices or sort them into columns for the new approach

    # Define the output file for the stitched image
    final_output_file = output_dir / f"full_tif_{project_name}.tif"

    # Call the improved stitch_tiles_together function directly
    stitch_tiles_together(file_list, final_output_file, cols, rows, input_dir)
    # split into smaller tiles
    split_large_tile_into_small_tiles(
        final_output_file, output_dir, x_km, y_overlap, resolution, project_name
    )
