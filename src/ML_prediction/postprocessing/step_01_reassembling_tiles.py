#%%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import cv2

#%%

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

overlap_rate = 0.01

""" 
- we want to do all of this given only the project name
- there will be one folder per project with all the tiles of prediction 
- + one text file with the path + names of the tiles? 
- need the original ortophoto geotif (to get the coordinates + resolution), this I have yet to do

"""

# path to the prediction tiles
input_dir = root_dir / 'data/ML_model/trondheim_1979/predictions/test'

"""#path to the original orthophoto
ortho_path = root_dir / 'nananinana'

# retrieve metadata from it (resolution + coordinates of the top left pixel)
orig_ortho = gdal.Open(ortho_path)

metadata_ortho = orig_ortho.GetGeoTransform()
#resolution in meters
resolution = metadata_ortho[1]
#coord of the upper left corner of the upper left pixel
x_coord = metadata_ortho[0]
y_coord = metadata_ortho[3]
# coord system
coord_sstem = orig_ortho.GetProjection()"""

# output location for the reassembled tile: in a diff folder 
# bc we make multiple reassembled tiles per project (5km x 5km with 500m overlap)
"""here add that we create the file if it doesnt exist yet"""
output_file = root_dir / 'data/ML_model/trondheim_1979/predictions/reassembled_tile/final_stitched_tif.tif'

# path to text file with (hopefully) all the file names of the tiles of the project
# assuming we keep the same format for tile names (include row and column)
file_list_path = root_dir / 'data/ML_model/trondheim_1979/dataset/test.txt'

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

def combine_columns_and_stitch(file_list, output_file, num_rows, overlap_rate=0.01):
    sample_path = os.path.join(input_dir,file_list[0])
    sample_dataset = gdal.Open(sample_path)

    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    geo_transform = sample_dataset.GetGeoTransform()
    projection = sample_dataset.GetProjection()

    effective_tile_width = int(tile_width * (1 - overlap_rate))
    final_width = effective_tile_width * num_rows
    final_height = tile_height * len(file_list) // num_rows

    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        final_width,
        final_height,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    combined_dataset.SetGeoTransform(geo_transform)
    combined_dataset.SetProjection(projection)

    white_tile = np.full((tile_height, tile_width, sample_dataset.RasterCount), 255, dtype=np.uint8)
    tile_dict = {extract_tile_numbers(filename): filename for filename in file_list}

    for y in range(len(file_list) // num_rows):
        for x in range(num_rows):
            tile_pos = (x, y)
            x_offset = x * effective_tile_width
            y_offset = y * tile_height

            if tile_pos in tile_dict:
                tile_path = os.path.join(input_dir,tile_dict[tile_pos])
                tile_dataset = gdal.Open(tile_path)
                if tile_dataset:
                    for band in range(1, tile_dataset.RasterCount + 1):
                        data = tile_dataset.GetRasterBand(band).ReadAsArray()
                        effective_data = data[:, :effective_tile_width]
                        combined_dataset.GetRasterBand(band).WriteArray(effective_data, x_offset, y_offset)
            else:
                for band in range(1, sample_dataset.RasterCount + 1):
                    combined_dataset.GetRasterBand(band).WriteArray(white_tile[:, :, band - 1], x_offset, y_offset)

    sample_dataset = None
    combined_dataset = None

    print(f"Final stitched image saved as {output_file}")

#%%
# Example usage:
# Read filenames from the text file

with open(file_list_path, 'r') as file:
    file_list = [line.strip() for line in file]
    
# Append '.tif' extension if not present
file_list = [filename if filename.endswith('.tif') else f"{filename}.tif" for filename in file_list]

# Order the files by x and y indices
ordered_files = order_files_by_xy(file_list)

# Assuming number of rows (num_rows) is known or can be calculated

combine_columns_and_stitch(ordered_files, output_file, rows)
