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
import math
from rasterio.warp import transform_bounds
import rasterio


#%%
def process_project_tiles(tile_dir):
        
    def process_image(processed_img_path):
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
            #now we exclude the polygons that have at least one vertice on the edge (within two pixels) of the image
            gdf = gdf[~gdf['geometry'].apply(lambda x: any([x.exterior.coords[i][0] in [0, 1, src.width-1, src.width] or x.exterior.coords[i][1] in [0, 1,src.height-1, src.height] for i in range(len(x.exterior.coords))]))]
        return gdf, processed_img_path

    # Process all images in the project directory
    all_gdfs = []
    
    for processed_img_path in tqdm(glob.glob(str(tile_dir / "*.tif"))):
        gdf, processed_img_path = process_image(processed_img_path)
        all_gdfs.append(gdf)

    # Combine all GeoDataFrames into one
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    # Drop duplicates based on the geometry column
    combined_gdf = combined_gdf.drop_duplicates(subset='geometry')

    return combined_gdf

#%%
# Main script

if __name__ == '__main__':
    # Set the paths
    root_dir = Path(__file__).parents[3]

    # Parameters
    project_name = "trondheim_2019"  

    # get necessary data
    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, 'r') as file:
        project_dict = json.load(file)

    # Get the resolutiom and other details
    resolution = project_dict[project_name]['resolution']
    compression_name = project_dict[project_name]['compression_name']
    compression_value = project_dict[project_name]['compression_value']

    # path to the reassembled tiles
    tile_dir = root_dir / f"data/ML_prediction/large_tiles/res_{resolution}/{project_name}"
    
    # run the function
    combined_gdf = process_project_tiles(tile_dir)
    
    # save the combined GeoDataFrame (into a pickle file)
    pickle_file_path = root_dir / f"data/ML_prediction/polygons/{project_name}_combined_geodata.pkl"
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(combined_gdf, f)

# %%
