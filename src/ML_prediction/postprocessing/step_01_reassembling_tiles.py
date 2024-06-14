# %%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import cv2
import json
# %%

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Parameters
project_name = "trondheim_2019"  # Example project name
x_km = 1  # Size of each small tile in kilometers
y_overlap = 50  # Overlap in meters for the bigger tiles
overlap_rate = 0  # 0% overlap (prediction tiles)


"""
- just need to sort out the coordinates and add them in the big tiles
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


def get_nb_row_col(filenames):
    # Determine grid size
    cols, rows = 0, 0
    for filename in filenames:
        col, row = extract_tile_numbers(filename)
        if row + 1 > rows:
            rows = row + 1
        if col + 1 > cols:
            cols = col + 1
    return rows, cols



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

    # Define a black tile
    black_tile = np.full(
        (tile_height, tile_width, sample_dataset.RasterCount), 0, dtype=np.uint8
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
                # If no tile, fill the area with black (or any other placeholder)
                for band in range(1, sample_dataset.RasterCount + 1):
                    combined_dataset.GetRasterBand(band).WriteArray(
                        black_tile[:, :, band - 1],
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
        raise FileNotFoundError(f"Unable to open large tile TIFF file: {large_tile_path}")

    large_tile_width = large_tile.RasterXSize
    large_tile_height = large_tile.RasterYSize
    tile_size_pixels = int((x_km * 1000) / resolution)
    overlap_pixels = int(y_overlap / resolution)

    # Calculate the number of tiles needed in both dimensions
    num_tiles_x = (large_tile_width + tile_size_pixels - overlap_pixels - 1) // (tile_size_pixels - overlap_pixels)
    num_tiles_y = (large_tile_height + tile_size_pixels - overlap_pixels - 1) // (tile_size_pixels - overlap_pixels)

    # Open a text file to write the tile names
    tiles_list_path = os.path.join(output_dir, f"{project_name}_tiles_list.txt")
    with open(tiles_list_path, 'w') as tiles_list_file:
        for y_tile_index in range(num_tiles_y):
            for x_tile_index in range(num_tiles_x):
                x_start = x_tile_index * (tile_size_pixels - overlap_pixels)
                y_start = y_tile_index * (tile_size_pixels - overlap_pixels)
                x_end = x_start + tile_size_pixels
                y_end = y_start + tile_size_pixels

                # Define the output tile path
                output_tile_path = os.path.join(
                    output_dir,
                    f"{project_name}_tile_{x_km}km_{y_tile_index}_{x_tile_index}.tif",
                )
                driver = gdal.GetDriverByName("GTiff")
                output_tile = driver.Create(
                    str(output_tile_path),
                    tile_size_pixels,
                    tile_size_pixels,
                    large_tile.RasterCount,
                    large_tile.GetRasterBand(1).DataType,
                )
                output_tile.SetGeoTransform(
                    (
                        large_tile.GetGeoTransform()[0] + x_start * resolution,
                        resolution,
                        0,
                        large_tile.GetGeoTransform()[3] + y_start * resolution,
                        0,
                        -resolution,
                    )
                )
                output_tile.SetProjection(large_tile.GetProjection())

                # Initialize tile with black pixels
                for band in range(1, large_tile.RasterCount + 1):
                    black_tile_data = np.full((tile_size_pixels, tile_size_pixels), 0, dtype=np.uint8)
                    output_tile.GetRasterBand(band).WriteArray(black_tile_data)

                # Read and write actual data within the bounds of the large tile
                if x_end > large_tile_width:
                    x_end = large_tile_width
                if y_end > large_tile_height:
                    y_end = large_tile_height

                for band in range(1, large_tile.RasterCount + 1):
                    band_data = large_tile.GetRasterBand(band).ReadAsArray(
                        x_start, y_start, x_end - x_start, y_end - y_start
                    )
                    if band_data is not None:
                        output_tile.GetRasterBand(band).WriteArray(band_data, 0, 0)
                tiles_list_file.write(f"{output_tile_path}\n")

                print(f"Saved tile: {output_tile_path}")
                output_tile = None  # Close and save the tile
                print(f"Saved tile: {output_tile_path}")

    large_tile = None  # Close the large tile


# %%

# Main script


if __name__ == "__main__":

    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, 'r') as file:
        project_dict = json.load(file)

    # Get the resolutiom and other details
    resolution = project_dict[project_name]['resolution']
    compression_name = project_dict[project_name]['compression_name']
    compression_value = project_dict[project_name]['compression_value']
    
    """    # path to the original orthophoto UPDATE with project name
    # ortho_path = root_dir / 'data/ML_prediction/{project_name}'
    ortho_path = (
        root_dir / f"data/raw/orthophoto/res_{resolution}/{project_name}/Eksport-nib.tif"
    )  # for testing
    # retrieve metadata from it (resolution + coordinates of the top left pixel)
    orig_ortho = gdal.Open(str(ortho_path))

    metadata_ortho = orig_ortho.GetGeoTransform()

    # coord of the upper left corner of the upper left pixel
    x_coord = metadata_ortho[0]
    y_coord = metadata_ortho[3]
    # coord system
    coord_sstem = orig_ortho.GetProjection()"""


    # path to the prediction tiles
    input_dir = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}"

    # output location for the reassembled tile: in a diff folder
    # bc we make multiple reassembled tiles per project (5km x 5km with 500m overlap)
    """here add that we create the file if it doesnt exist yet"""
    output_dir = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}/reassembled_tiles"
    output_file = output_dir / f"full_tif_{project_name}.tif"

    # path to text file with (hopefully) all the file names of the tiles of the project
    # assuming we keep the same format for tile names (include row and column)
    # Use glob to find all .tif files in the directory
    tif_files = list(input_dir.glob('*.tif'))
    tif_filenames = [file.name for file in tif_files]
    rows, cols = get_nb_row_col(tif_filenames)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the output file for the stitched image
    final_output_file = output_dir / f"full_tif_{project_name}.tif"

    # stitch back together the full tif
    stitch_tiles_together(tif_filenames, final_output_file, cols, rows, input_dir)
    # split into smaller tiles
    split_large_tile_into_small_tiles(
        final_output_file, output_dir, x_km, y_overlap, resolution, project_name
    )
