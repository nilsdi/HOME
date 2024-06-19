# %%
from osgeo import gdal
from osgeo.gdalnumeric import *  
from osgeo.gdalconst import * 
from shapely.geometry import shape
import rasterio.features
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from shapely.ops import unary_union
from pathlib import Path
import pandas as pd
import pickle
import glob
import re
from tqdm import tqdm
import json

#%%
# Set the paths
root_dir = Path(__file__).parents[3]

# Parameters
project_name = "trondheim_2019"  # Example project name
x_km = 1  # Size of each small tile in kilometers
y_overlap = 50  # Overlap in meters for the bigger tiles
overlap_rate = 0  # 0% overlap (prediction tiles)


def process_project_tiles(project_name):
    # Set the path to the project directory
    root_dir = Path(__file__).parents[3]
    path_data = root_dir / 'data/ML_model'
    project_dir = path_data / project_name
    tile_dir = project_dir / 'predictions/reassembled_tile'

    # Function to process a single image
    def process_image(img_path):
        src_ds = gdal.Open(str(img_path), gdal.GA_ReadOnly)
        srcband = src_ds.GetRasterBand(1)
        myarray = srcband.ReadAsArray()

        mypoly = []
        for vec in rasterio.features.shapes(myarray):
            mypoly.append(shape(vec[0]))

        filtered_polygons = [polygon for polygon in mypoly if polygon.area < 450*450 and polygon.area > 15*15]
        simplified_polygons = [polygon.simplify(5, preserve_topology=True) for polygon in filtered_polygons]
        rounded_polygons = [polygon.buffer(1, join_style=3, single_sided=True) for polygon in simplified_polygons]
        rounded_polygons_ext = [polygon.exterior for polygon in rounded_polygons]
        geoms = list(rounded_polygons_ext)
        gdf = gpd.GeoDataFrame({'geometry': geoms})
        return gdf

    # Process all images in the project directory
    all_gdfs = []
    
    for info in tqdm(files_info, desc="Processing tiles"):
        img_path = info['file_path']
        gdf = process_image(img_path)
        all_gdfs.append(gdf)
        
        # Plotting the GeoDataFrame for the current image
        plt.figure(figsize=(10, 10))
        gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
        plt.axis('off')
        plt.title(f"Polygons for {img_path}")
        plt.show()

    #Combine all GeoDataFrames into one
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    return combined_gdf

#%%
# Main script

if __name__ == '__main__':

    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, 'r') as file:
        project_dict = json.load(file)

    # Get the resolutiom and other details
    resolution = project_dict[project_name]['resolution']
    compression_name = project_dict[project_name]['compression_name']
    compression_value = project_dict[project_name]['compression_value']
    
    folder_path = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}/reassembled_tiles"

    # Regular expression to match the file format
    file_pattern = re.compile(r'^(?P<project_name>.+)_tile_(?P<x_km>\d+)km_(?P<row>\d+)_(?P<col>\d+)\.tif$')

    # List to store file paths and parsed information
    files_info = []

    #maybe instead of this I should just write a text file with the file names and then read it

    # Iterate over all .tif files in the folder
    for file_path in glob.glob(os.path.join(folder_path, '*.tif')):
        # Get the file name from the file path
        file_name = os.path.basename(file_path)
        
        # Match the file name against the pattern
        match = file_pattern.match(file_name)
        if match:
            # Extract information from the file name
            project_name = match.group('project_name')
            x_km = int(match.group('x_km'))
            row = int(match.group('row'))
            col = int(match.group('col'))
            
            # Add the file information to the list
            files_info.append({
                'file_path': file_path,
                'project_name': project_name,
                'row': row,
                'col': col
            })
    

    project_name = "trondheim_1979"
    combined_gdf = process_project_tiles(project_name)



#%%
# Optionally, save the combined GeoDataFrame
pickle_file_path = root_dir / f"{project_name}_combined_geodata.pkl"
with open(pickle_file_path, 'wb') as f:
    pickle.dump(combined_gdf, f)

print(f"Combined GeoDataFrame for {project_name} has been pickled to {pickle_file_path}")

