#%%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import cv2

#%%

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Parameters
project_name = 'trondheim_1979'  # Example project name
x_km = 1  # Size of each small tile in kilometers
y_overlap = 500  # Overlap in meters
overlap_rate = 0.01  # 1% overlap (prediction tiles)


resolution = 0.3  # Resolution in meters (adjust as needed)

""" 
- we want to do all of this given only the project name
- there will be one folder per project with all the tiles of prediction 
- + one text file with the path + names of the tiles? 
- need the original ortophoto geotif (to get the coordinates + resolution), this I have yet to do

"""

# path to the prediction tiles
input_dir = root_dir / f'data/ML_model/{project_name}/predictions/test'

#path to the original orthophoto UPDATE with project name
#ortho_path = root_dir / 'data/ML_prediction/{project_name}'
ortho_path = root_dir / 'data/temp/test_zoe/images/archive/trondheim_0.3_1979_1_0.tif' #for testing
# retrieve metadata from it (resolution + coordinates of the top left pixel)
orig_ortho = gdal.Open(str(ortho_path))

metadata_ortho = orig_ortho.GetGeoTransform()
#resolution in meters
resolution = metadata_ortho[1]
#coord of the upper left corner of the upper left pixel
x_coord = metadata_ortho[0]
y_coord = metadata_ortho[3]
# coord system
coord_sstem = orig_ortho.GetProjection()

#resolution = 0.3  # Resolution in meters (adjust as needed)

# output location for the reassembled tile: in a diff folder 
# bc we make multiple reassembled tiles per project (5km x 5km with 500m overlap)
"""here add that we create the file if it doesnt exist yet"""
output_dir = root_dir / f'data/ML_model/{project_name}/predictions/reassembled_tile'
output_file = output_dir / f'full_tif_{project_name}.tif'

# path to text file with (hopefully) all the file names of the tiles of the project
# assuming we keep the same format for tile names (include row and column)
file_list_path = root_dir / f'data/ML_model/{project_name}/dataset/test.txt'


#%%

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
    parts = filename.split('_')
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])
    return row, col

def get_nb_row_col(file_list_path):
    # Read filenames from the text file
    with open(file_list_path, 'r') as file:
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
    ordered_files = sorted(files, key=lambda x: (file_info[files.index(x)][0], file_info[files.index(x)][1]))
    
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

"""def combine_column_tiles(file_list, output_file, num_rows, num_cols, overlap_rate=0.01):
    sample_path = os.path.join(input_dir, file_list[0])
    sample_dataset = gdal.Open(sample_path)

    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    geo_transform = sample_dataset.GetGeoTransform()
    projection = sample_dataset.GetProjection()

    effective_tile_height = int(tile_height * (1 - overlap_rate))
    effective_tile_width = int(tile_width * (1 - overlap_rate))
    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        effective_tile_width * num_cols,
        effective_tile_height * num_rows,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    combined_dataset.SetGeoTransform(geo_transform)
    combined_dataset.SetProjection(projection)

    white_tile = np.full((tile_height, tile_width, sample_dataset.RasterCount), 255, dtype=np.uint8)
    tile_dict = {extract_tile_numbers(filename): filename for filename in file_list}

    for y in range(num_rows):
        for x in range(num_cols):
            tile_pos = (x, y)
            x_offset = x * effective_tile_width
            y_offset = y * effective_tile_height

            if tile_pos in tile_dict:
                tile_path = str(predictions_dir / tile_dict[tile_pos])
                tile_dataset = gdal.Open(tile_path)
                if tile_dataset:
                    for band in range(1, tile_dataset.RasterCount + 1):
                        data = tile_dataset.GetRasterBand(band).ReadAsArray()
                        effective_data = data[:effective_tile_height, :effective_tile_width]
                        combined_dataset.GetRasterBand(band).WriteArray(effective_data, x_offset, y_offset)
            else:
                for band in range(1, sample_dataset.RasterCount + 1):
                    combined_dataset.GetRasterBand(band).WriteArray(white_tile[:effective_tile_height, :effective_tile_width, band - 1], x_offset, y_offset)

    sample_dataset = None
    combined_dataset = None

    print(f"Combined image saved as {output_file}")"""

def combine_column_tiles(file_list, column, output_file, num_rows=rows):
    """
    Combines tiles in one column into a single image, filling missing tiles with white squares.
    
    Args:
        file_list (list of str): List of filenames representing tiles.
        column (int): The column number to combine.
        output_file (str): Path to save the combined image.
        num_rows (int): Number of rows each column should have.
    """
    # Open one of the files to get the tile size and georeference info

    sample_path = os.path.join(input_dir, file_list[0])
    sample_dataset = gdal.Open(sample_path)

    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    geo_transform = sample_dataset.GetGeoTransform()
    projection = sample_dataset.GetProjection()

    # Create a new raster dataset for the combined image
    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        tile_width,
        tile_height * num_rows,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    # Copy georeference information
    combined_dataset.SetGeoTransform(geo_transform)
    combined_dataset.SetProjection(projection)

    # Define a white tile (255 for each channel)
    white_tile = np.full((tile_height, tile_width, sample_dataset.RasterCount), 255, dtype=np.uint8)

    # Create a dictionary to map tile positions to filenames
    tile_dict = {extract_tile_numbers(filename): filename for filename in file_list}

    # Paste each tile into the combined dataset
    for y in range(num_rows):
        tile_pos = (column, y)
        y_offset = y * tile_height

        if tile_pos in tile_dict:
            tile_path = tile_dict[tile_pos]
            tile_path = os.path.join(input_dir, tile_path)
            tile_dataset = gdal.Open(tile_path)
            if tile_dataset:
                for band in range(1, tile_dataset.RasterCount + 1):
                    data = tile_dataset.GetRasterBand(band).ReadAsArray()
                    combined_dataset.GetRasterBand(band).WriteArray(data, 0, y_offset)
        else:
            # Write the white tile where there is no tile file
            for band in range(1, sample_dataset.RasterCount + 1):
                combined_dataset.GetRasterBand(band).WriteArray(white_tile[:, :, band - 1], 0, y_offset)

    # Close the datasets
    sample_dataset = None
    combined_dataset = None

    print(f"Combined column image saved as {output_file}")

def stitch_columns_together(column_files, output_file):
    """
    Stitches together multiple column images into a single image.
    
    Args:
        column_files (list of str): List of paths to the column images.
        output_file (str): Path to save the final stitched image.
    """
    # Open the first column image to get size and other information
    sample_dataset = gdal.Open(column_files[0])

    # Determine the width and height of the final stitched image
    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    final_width = tile_width * len(column_files)
    final_height = tile_height

    # Create a new raster dataset for the final stitched image
    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        final_width,
        final_height,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    # Copy georeference information
    combined_dataset.SetGeoTransform(sample_dataset.GetGeoTransform())
    combined_dataset.SetProjection(sample_dataset.GetProjection())

    # Paste each column image into the final stitched image
    for i, column_file in enumerate(column_files):
        column_dataset = gdal.Open(column_file)
        if column_dataset:
            for band in range(1, column_dataset.RasterCount ): #used to be +1 here
                data = column_dataset.GetRasterBand(band).ReadAsArray()
                x_offset = i * tile_width
                print(x_offset)
                y_offset = 0
                combined_dataset.GetRasterBand(band).WriteArray(data, x_offset, y_offset)

    # Close the datasets
    sample_dataset = None
    combined_dataset = None

    print(f"Final stitched image saved as {output_file}")


def split_large_tile_into_small_tiles(large_tile_path, output_dir, x_km, y_overlap, resolution, project_name):
    large_tile = gdal.Open(str(large_tile_path))

    if not large_tile:
        raise FileNotFoundError(f"Unable to open large tile TIFF file: {large_tile_path}")

    large_tile_width = large_tile.RasterXSize
    large_tile_height = large_tile.RasterYSize
    tile_size_pixels = int((x_km * 1000) / resolution)
    overlap_pixels = int(y_overlap / resolution)

    for y in range(0, large_tile_height, tile_size_pixels - overlap_pixels):
        for x in range(0, large_tile_width, tile_size_pixels - overlap_pixels):
            x_end = min(x + tile_size_pixels, large_tile_width)
            y_end = min(y + tile_size_pixels, large_tile_height)
            if x_end - x > overlap_pixels and y_end - y > overlap_pixels:
                output_tile_path = os.path.join(output_dir, f"{project_name}_large_tile_{y // tile_size_pixels}_{x // tile_size_pixels}.tif")
                driver = gdal.GetDriverByName('GTiff')
                output_tile = driver.Create(
                    str(output_tile_path),
                    x_end - x,
                    y_end - y,
                    large_tile.RasterCount,
                    large_tile.GetRasterBand(1).DataType
                )
                output_tile.SetGeoTransform(large_tile.GetGeoTransform())
                output_tile.SetProjection(large_tile.GetProjection())

                for band in range(1, large_tile.RasterCount + 1):
                    band_data = large_tile.GetRasterBand(band).ReadAsArray(x, y, x_end - x, y_end - y)
                    output_tile.GetRasterBand(band).WriteArray(band_data)
                
                output_tile = None
                print(f"Saved tile: {output_tile_path}")

    large_tile = None
#%%
# Main script

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read filenames from the text file

with open(file_list_path, 'r') as file:
    file_list = [line.strip() for line in file]

    
# Append '.tif' extension if not present
file_list = [filename if filename.endswith('.tif') else f"{filename}.tif" for filename in file_list]

# Order the files by x and y indices
ordered_files = order_files_by_xy(file_list)
sorted_columns = get_columns_from_ordered_files(ordered_files)

for i, column in enumerate(sorted_columns):
    output_file = output_dir / f'reassembled_tile_{i}.tif'
    
    if not output_file.exists():
        combine_column_tiles(column,i, output_file)
        print(f"Created {output_file}")
# Assuming number of rows (num_rows) is known or can be calculated

column_files = [str(output_dir / f'reassembled_tile_{i}.tif') for i in range(cols)]

print(column_files)

stitch_columns_together(column_files, output_file)

split_large_tile_into_small_tiles(output_file, output_dir, x_km, y_overlap, resolution, project_name)