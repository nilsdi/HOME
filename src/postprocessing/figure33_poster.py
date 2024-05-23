#%%

import os
import numpy as np
from osgeo import gdal
from pathlib import Path

#%%
#FOR PREDICTION
root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]
test_file = open(root_dir / 'data/model/test_full_pic/test.txt', 'w')
data_path = root_dir / 'data/model/test_full_pic/tiles/images'
# Directory containing the TIFF files
input_dir = root_dir / 'data/model/test_full_pic/predictions/reassembled_tile/'
output_file = root_dir / 'data/model/test_full_pic/predictions/test_33/test_33.tif'
file_list_path = root_dir / 'data/model/test_full_pic/dataset/test.txt'
#%%
#FOR ORIGINAL TIF
root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]
test_file = open(root_dir / 'data/model/test_full_pic/test.txt', 'w')
data_path = root_dir / 'data/model/test_full_pic/tiles/images'
# Directory containing the TIFF files
input_dir = root_dir / 'data/model/test_full_pic/tiles/reassembled_tile'
output_file = root_dir / 'data/model/test_full_pic/tiles/reassembled_tile/test.tif'
file_list_path = root_dir / 'data/model/test_full_pic/dataset/test.txt'
#%%
# Read filenames from the text file
with open(file_list_path, 'r') as file:
    filenames = [line.strip() for line in file]

def parse_filename(filename):
    parts = filename.split('_')
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])
    return row, col

# Determine grid size
cols, rows = 0, 0
for filename in filenames:
    col, row = parse_filename(filename)
    if row + 1 > rows:
        rows = row + 1
    if col + 1 > cols:
        cols = col + 1

def extract_tile_numbers(filename):
    """
    Extracts the x and y tile numbers from the filename.
    
    Args:
        filename (str): The filename to extract tile numbers from.
    
    Returns:
        tuple: A tuple (x, y) representing the tile coordinates.
    """
    parts = filename.split('_')
    x = int(parts[-2])
    y = int(parts[-1].split('.')[0])
    return (x, y)

def extract_3x3_tiles(full_image_path, top_left_tile_coords, output_file):
    """
    Extracts a 3x3 tile section from the full image.
    
    Args:
        full_image_path (str): Path to the full image.
        top_left_tile_coords (tuple): (x, y) coordinates of the top-left tile of the 3x3 section.
        output_file (str): Path to save the extracted 3x3 section.
    """
    x_start, y_start = top_left_tile_coords

    # Open the full image
    full_dataset = gdal.Open(full_image_path)
    if not full_dataset:
        raise FileNotFoundError(f"Unable to open full TIFF file: {full_image_path}")

    tile_width = full_dataset.RasterXSize // cols
    tile_height = full_dataset.RasterYSize // rows

    # Create a new raster dataset for the 3x3 section
    driver = gdal.GetDriverByName('GTiff')
    section_dataset = driver.Create(
        str(output_file),
        tile_width * 3,
        tile_height * 3,
        full_dataset.RasterCount,
        full_dataset.GetRasterBand(1).DataType
    )

    # Copy georeference information (modify the origin for the new dataset)
    geo_transform = list(full_dataset.GetGeoTransform())
    geo_transform[0] += x_start * tile_width * geo_transform[1]
    geo_transform[3] += y_start * tile_height * geo_transform[5]
    section_dataset.SetGeoTransform(tuple(geo_transform))
    section_dataset.SetProjection(full_dataset.GetProjection())

    # Extract and write the 3x3 section
    for row in range(3):
        for col in range(3):
            x_offset = (x_start + col) * tile_width
            y_offset = (y_start + row) * tile_height
            for band in range(1, full_dataset.RasterCount + 1):
                data = full_dataset.GetRasterBand(band).ReadAsArray(x_offset, y_offset, tile_width, tile_height)
                section_dataset.GetRasterBand(band).WriteArray(data, col * tile_width, row * tile_height)
    """# Draw the white grid on the extracted section
    thickness = 5
    for band in range(1, section_dataset.RasterCount + 1):
        data = section_dataset.GetRasterBand(band).ReadAsArray()

        # Draw vertical lines
        for i in range(1, 3):
            data[:, i * tile_width - thickness // 2: i * tile_width + thickness // 2] = 255

        # Draw horizontal lines
        for i in range(1, 3):
            data[i * tile_height - thickness // 2: i * tile_height + thickness // 2, :] = 255

        section_dataset.GetRasterBand(band).WriteArray(data)"""
    # Close the datasets
    full_dataset = None
    section_dataset = None

    print(f"3x3 section saved as {output_file}")

# Example usage
full_image_path = os.path.join(input_dir,'final_stitched_image.tif')
top_left_tile_coords = (1, 10)  # Example coordinates

extract_3x3_tiles(full_image_path, top_left_tile_coords, output_file)
