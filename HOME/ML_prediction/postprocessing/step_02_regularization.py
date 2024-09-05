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
import folium
import numpy as np
import matplotlib.pyplot as plt
#%%
# Set the paths
root_dir = Path(__file__).parents[3]

# Parameters
project_name = "trondheim_kommune_2020"  # Example project name

project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
# Open and read the JSON file
with open(project_dict_path, 'r') as file:
    project_dict = json.load(file)

# Get the resolutiom and other details
resolution = project_dict[project_name]['resolution']
compression_name = project_dict[project_name]['compression_name']
compression_value = project_dict[project_name]['compression_value']

# path to the reassembled tiles
tile_dir = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/reassembled_tiles"

def process_project_tiles(files_info):

    # # Function to process a single image
    # def process_image(img_path):
    #     src_ds = gdal.Open(str(img_path), gdal.GA_ReadOnly)
    #     srcband = src_ds.GetRasterBand(1)
    #     myarray = srcband.ReadAsArray()

    #     mypoly = []
    #     for vec in rasterio.features.shapes(myarray):
    #         mypoly.append(shape(vec[0]))

    #     filtered_polygons = [polygon for polygon in mypoly if polygon.area < 450*450 and polygon.area > 15*15]
    #     simplified_polygons = [polygon.simplify(5, preserve_topology=True) for polygon in filtered_polygons]
    #     rounded_polygons = [polygon.buffer(1, join_style=3, single_sided=True) for polygon in simplified_polygons]
    #     #rounded_polygons_ext = [polygon.exterior for polygon in rounded_polygons]
    #     geoms = list(rounded_polygons_ext)
    #     gdf = gpd.GeoDataFrame({'geometry': geoms})
    #     return gdf

    def process_image(img_path):
        # Open the image with rasterio to access both data and metadata
        with rasterio.open(str(img_path)) as src:
            # Read the first band
            myarray = src.read(1)

            # Extract polygons from the array
            mypoly = [shape(vec[0]) for vec in rasterio.features.shapes(myarray, transform=src.transform)]
            
            # Filter and simplify polygons
            filtered_polygons = [polygon for polygon in mypoly if 5*5 < polygon.area < 450*450]
            simplified_polygons = [polygon.simplify(5, preserve_topology=True) for polygon in filtered_polygons]
            rounded_polygons = [polygon.buffer(1, join_style=3, single_sided=True) for polygon in simplified_polygons]
            
            # Create a GeoDataFrame with the correct CRS
            gdf = gpd.GeoDataFrame({'geometry': rounded_polygons}, crs=src.crs)
            
        return gdf
    # Process all images in the project directory
    all_gdfs = []
    
    for info in tqdm(files_info, desc="Processing tiles"):
        img_path = info['file_path']
        gdf = process_image(img_path)
        all_gdfs.append(gdf)
        
        # # Plotting the GeoDataFrame for the current image
        plt.figure(figsize=(10, 10))
        gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
        plt.axis('off')
        plt.title(f"Polygons for {img_path}")
        plt.show()

    #Combine all GeoDataFrames into one
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    return combined_gdf

#%%

def process_project_tiles(files_info):
        
    def process_image(processed_img_path, original_img_path):
        # Open the processed image with rasterio to access both data and metadata
        with rasterio.open(str(processed_img_path)) as src:
            # Read the first band
            myarray = src.read(1)

            # Extract polygons from the array
            mypoly = [shape(vec[0]) for vec in rasterio.features.shapes(myarray, transform=src.transform)]
            
            # Filter and simplify polygons
            filtered_polygons = [polygon for polygon in mypoly if 5*5 < polygon.area < 450*450]
            simplified_polygons = [polygon.simplify(5, preserve_topology=True) for polygon in filtered_polygons]
            rounded_polygons = [polygon.buffer(1, join_style=3, single_sided=True) for polygon in simplified_polygons]
            
            # Create a GeoDataFrame with the correct CRS
            gdf = gpd.GeoDataFrame({'geometry': rounded_polygons}, crs=src.crs)
            
        return gdf, original_img_path

    # Process all images in the project directory
    all_gdfs = []

    for info in tqdm(files_info, desc="Processing tiles"):
        processed_img_path = info['file_path']
        original_img_path = info['original_file_path']  # Assuming this key exists in info
        gdf, original_img_path = process_image(processed_img_path, original_img_path)
        all_gdfs.append(gdf)
        
        # Plotting the GeoDataFrame over the original image
        plt.figure(figsize=(10, 10))
        with rasterio.open(original_img_path) as src:
            # Read the first band of the image
            img_array = src.read()
            # Plot the image
            plt.imshow(img_array[0], cmap='gray', extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        
        # Plot the polygons on top of the image
        gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
        plt.axis('off')
        plt.title(f"Polygons for {processed_img_path}")
        plt.show()

    # Combine all GeoDataFrames into one
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
    
    folder_path = root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/reassembled_tiles"
    original_folder_path = root_dir / f"data/ML_prediction/topredict/image/res_{resolution}/{project_name}/reassembled_tiles"
    # Regular expression to match the file format
    file_pattern = re.compile(r'^stitched_tif_(?P<project_name>.+)_(?P<col>\d+)_(?P<row>\d+)\.tif$')

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
            row = int(match.group('row'))
            col = int(match.group('col'))
            
            original_file_path = os.path.join(original_folder_path, file_name)


            # Add the file information to the list
            files_info.append({
                'file_path': file_path,
                'original_file_path': original_file_path, 
                'project_name': project_name,
                'row': row,
                'col': col
            })

#%%

    files_info = files_info[1:2]
    combined_gdf = process_project_tiles(files_info)


#%%

from pyproj import Transformer
import folium
from shapely.ops import transform

# Set the CRS for combined_gdf to UTM 33N if it's not already set
if combined_gdf.crs is not({'init': 'epsg:32633'}):
    combined_gdf.set_crs('epsg:32633', inplace=True)

# Create a transformer to convert from UTM 33N (EPSG:32633) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs('epsg:32633', 'epsg:4326', always_xy=True)

# Function to apply the transformation to a geometry
def apply_transformation(geom):
    return transform(lambda x, y: transformer.transform(x, y), geom)

# Apply the transformation to each geometry in the GeoDataFrame
combined_gdf['geometry'] = combined_gdf['geometry'].apply(apply_transformation)

def plot_gdf_on_map(gdf, polygon_index=0):
    polygon = gdf.iloc[polygon_index].geometry
    minx, miny, maxx, maxy = polygon.bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=14)
    
    for _, row in gdf.iterrows():
        sim_geo = row['geometry']
        folium.GeoJson(sim_geo, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange'}).add_to(m)
    
    return m

# Plotting the map
map = plot_gdf_on_map(combined_gdf, polygon_index=0)
map

#%%
# Optionally, save the combined GeoDataFrame
pickle_file_path = root_dir / f"{project_name}_combined_geodata.pkl"
with open(pickle_file_path, 'wb') as f:
    pickle.dump(combined_gdf, f)

print(f"Combined GeoDataFrame for {project_name} has been pickled to {pickle_file_path}")

