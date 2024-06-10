#%%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path
import cv2
#%%

#test_tif = '/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/ML_model/trondheim_1979/tiles/images/trondheim_0.3_1979_1_0_18_2.tif'
test_tif = '/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/prepred/images/archive/trondheim_0.3_2023_1_0.tif'
test_test = gdal.Open(test_tif)
#print(test_test.GetGeoTransform())
print(test_test.GetProjection())
image = cv2.imread(test_tif)
#image.shape

#%%

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

overlap_rate = 0.01

""" 
- there will be one folder per project with all the tiles of prediction 
- + one text file with the path + names of the tiles? 
- need the original ortophoto geotif (to get the coordinates + resolution), this I have yet to do

"""
# path to the prediction tiles
input_dir = root_dir / 'data/model/trondheim_1979/predictions/test'

#path to the original orthophoto
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
orig_ortho.GetProjection()

# output location for the reassembled tile: in a diff folder 
# bc we make multiple reassembled tiles per project (5km x 5km with 500m overlap)
"""here add that we create the file if it doesnt exist yet"""
output_file = root_dir / 'data/model/trondheim_1979/predictions/reassembled_tile/test.tif'

# path to text file with (hopefully) all the file names of the tiles of the project
# assuming we keep the same format for tile names (include row and column)
file_list_path = root_dir / 'data/model/trondheim_1979/dataset/test.txt'

#%%

"""
- first we extract the nb of rows and cols 
- then we order the files
- then we create a list of ordered columns
- then we combine the tiles into columns + stitch the columns together
"""

#%%

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

#%%

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

#%%

def combine_column_tiles(file_list, column, output_file, num_rows=rows, overlap_rate=0.01):
    """
    Combines tiles in one column into a single image, removing vertical overlap and filling missing tiles with white squares.
    
    Args:
        file_list (list of str): List of filenames representing tiles.
        column (int): The column number to combine.
        output_file (str): Path to save the combined image.
        num_rows (int): Number of rows each column should have.
        overlap_rate (float): The overlap rate to account for.
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

    # Calculate the effective height of each tile without the overlap
    effective_tile_height = int(tile_height * (1 - overlap_rate))

    # Create a new raster dataset for the combined image
    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        tile_width,
        effective_tile_height * num_rows,
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
        y_offset = y * effective_tile_height

        if tile_pos in tile_dict:
            tile_path = tile_dict[tile_pos]
            tile_path = os.path.join(input_dir, tile_path)
            tile_dataset = gdal.Open(tile_path)
            if tile_dataset:
                for band in range(1, tile_dataset.RasterCount + 1):
                    data = tile_dataset.GetRasterBand(band).ReadAsArray()
                    # Remove the overlap by slicing the array
                    effective_data = data[:effective_tile_height, :]
                    combined_dataset.GetRasterBand(band).WriteArray(effective_data, 0, y_offset)
        else:
            # Write the white tile where there is no tile file
            for band in range(1, sample_dataset.RasterCount + 1):
                combined_dataset.GetRasterBand(band).WriteArray(white_tile[:effective_tile_height, :, band - 1], 0, y_offset)

    # Close the datasets
    sample_dataset = None
    combined_dataset = None

    print(f"Combined column image saved as {output_file}")



#%%

def stitch_columns_together(column_files, output_file, overlap_rate=0.01):
    """
    Stitches together multiple column images into a single image, removing horizontal overlap.
    
    Args:
        column_files (list of str): List of paths to the column images.
        output_file (str): Path to save the final stitched image.
        overlap_rate (float): The overlap rate to account for.
    """
    # Open the first column image to get size and other information
    sample_dataset = gdal.Open(column_files[0])

    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {column_files[0]}")

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize

    # Calculate the effective width of each tile without the overlap
    effective_tile_width = int(tile_width * (1 - overlap_rate))

    # Determine the width and height of the final stitched image
    final_width = effective_tile_width * len(column_files)
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
            for band in range(1, column_dataset.RasterCount + 1):
                data = column_dataset.GetRasterBand(band).ReadAsArray()
                # Remove the overlap by slicing the array
                effective_data = data[:, :effective_tile_width]
                x_offset = i * effective_tile_width
                y_offset = 0
                combined_dataset.GetRasterBand(band).WriteArray(effective_data, x_offset, y_offset)

    # Close the datasets
    sample_dataset = None
    combined_dataset = None

    print(f"Final stitched image saved as {output_file}")


#%%
# Example usage on the prediction


# Read filenames from the text file
with open(file_list_path, 'r') as file:
    file_list = [line.strip() for line in file]

# Append '.tif' extension if not present
file_list = [filename if filename.endswith('.tif') else f"{filename}.tif" 
             for filename in file_list]


#%%
# Order the files by x and y indices
ordered_files = order_files_by_xy(file_list)

sorted_columns = get_columns_from_ordered_files(ordered_files)

output_base_path = root_dir / 'data/model/trondheim_1979/predictions/reassembled_tile'

for i, column in enumerate(sorted_columns):
    output_file = output_base_path / f'reassembled_tile_{i}.tif'
    
    if not output_file.exists():
        combine_column_tiles(column,i, output_file)
        print(f"Created {output_file}")
    #else:
    #    combine_tiles(column, output_file)
     #   print(f"{output_file} already exists, overwriting.")

column_files = [str(output_base_path / f'reassembled_tile_{i}.tif') for i in range(cols)]


print(column_files)
output_file = output_base_path/ f'final_stitched_image.tif'
stitch_columns_together(column_files, output_file)
