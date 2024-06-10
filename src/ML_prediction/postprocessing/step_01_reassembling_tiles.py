#%%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path

#%%

# Function to parse filename and extract its coordinates
def parse_filename(filename):
    parts = filename.split('_')
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])
    return row, col

#%%

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]
test_file = open(root_dir / 'data/model/trondheim_1979/test.txt', 'w')
data_path = root_dir / 'data/model/trondheim_1979/tiles/images'
# Directory containing the TIFF files
input_dir = root_dir / 'data/model/trondheim_1979/predictions/test'
output_file = root_dir / 'data/model/trondheim_1979/predictions/reassembled_tile/test.tif'
file_list_path = root_dir / 'data/model/trondheim_1979/dataset/test.txt'

# Read filenames from the text file
with open(file_list_path, 'r') as file:
    filenames = [line.strip() for line in file]

# Determine grid size
cols, rows = 0, 0
for filename in filenames:
    col, row = parse_filename(filename)
    if row + 1 > rows:
        rows = row + 1
    if col + 1 > cols:
        cols = col + 1

#%%

def extract_tile_numbers(filename):
    """
    Extracts the x and y (row and col) from a filename 
    of pattern '_1_0_x_y'.
    
    Args:
        filename (str): The filename to extract the tile info from.
    
    Returns:
        tuple: row, col part extracted from the filename.
    """
    # Split the filename by underscores
    parts = filename.split('_')
    
    # Extract the numbers and convert them to integers
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])

    return row, col

# Example usage
filename = 'trondheim_0.3_2023_1_0_1_1.tif'
tile_info = extract_tile_numbers(filename)
print(tile_info)

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

# Example usage
file_list = ['trondheim_0.3_2023_1_0_18_19.tif','trondheim_0.3_2023_1_0_17_15.tif','trondheim_0.3_2023_1_0_17_1.tif', 'trondheim_0.3_2023_1_0_18_15.tif', 'trondheim_0.3_2023_1_0_18_20.tif']
ordered_files = order_files_by_xy(file_list)

print(ordered_files)


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

def combine_tiles(ordered_files, output_file):
    """
    Combines tiles in the right order into a single image.
    
    Args:
        file_list (list of str): List of filenames representing tiles.
        output_file (str): Path to save the combined image.
    """
    # Open one of the files to get the tile size and georeference info
    sample_path = os.path.join(input_dir, ordered_files[0])
    # Open the first tile to get size and other information
    sample_dataset = gdal.Open(sample_path)

    # Create a new raster dataset for the combined image
    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        sample_dataset.RasterXSize * len(set(x for x, _ in map(extract_tile_numbers, ordered_files))),
        sample_dataset.RasterYSize * len(set(y for _, y in map(extract_tile_numbers, ordered_files))),
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    # Copy georeference information
    combined_dataset.SetGeoTransform(sample_dataset.GetGeoTransform())
    combined_dataset.SetProjection(sample_dataset.GetProjection())

    # Paste each tile into the combined dataset
    for i, filename in enumerate(ordered_files):
        tile_dataset = gdal.Open(filename)
        if tile_dataset:
            for band in range(1, tile_dataset.RasterCount + 1):
                data = tile_dataset.GetRasterBand(band).ReadAsArray()
                x_offset = (i % len(set(x for x, _ in map(extract_tile_numbers, ordered_files)))) * sample_dataset.RasterXSize
                y_offset = (i // len(set(x for x, _ in map(extract_tile_numbers, ordered_files)))) * sample_dataset.RasterYSize
                combined_dataset.GetRasterBand(band).WriteArray(data, x_offset, y_offset)

    # Close the datasets
    sample_dataset = None
    combined_dataset = None

    print(f"Combined image saved as {output_file}")
#%%

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


#%%

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
            for band in range(1, column_dataset.RasterCount + 1):
                data = column_dataset.GetRasterBand(band).ReadAsArray()
                x_offset = i * tile_width
                y_offset = 0
                combined_dataset.GetRasterBand(band).WriteArray(data, x_offset, y_offset)

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
