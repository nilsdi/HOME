# %%
from osgeo import ogr, gdal, osr
from osgeo.gdalnumeric import *  
from osgeo.gdalconst import * 
import fiona
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

#%%

#TEST FOR ONE 3*3 TILE 

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

# Directory containing the TIFF files
path_data = root_dir / 'data/model'

dir_2023 = path_data / 'test_full_pic'
dir_1937 = path_data / 'trondheim_1937'
dir_1979 = path_data / 'trondheim_1979'

#img_select = 'trondheim_0.3_1979_1_0_2_14.tif'
img_select = 'trondheim_0.3_2023_1_0_2_14.tif'
#img_select = 'trondheim_0.3_1937_1_1_2_14.tif'

orig_tif_2023 = dir_2023 / 'tiles/images' / img_select
prediction_2023 = dir_2023 / 'predictions/test' / img_select

orig_tif_1937 = dir_1937 / 'tiles/images' / img_select
prediction_1937 = dir_1937 / 'predictions/test' / img_select

orig_tif_1979 = dir_1979 / 'tiles/images' / img_select
prediction_1979 = dir_1979 / 'predictions/test' / img_select

segimg = prediction_2023
orig_tif = orig_tif_2023
#segimg = root_dir / 'data/model/test_full_pic/predictions/test_33/test_33.tif'
src_ds = gdal.Open( str(segimg), GA_ReadOnly )

srcband=src_ds.GetRasterBand(1)
myarray=srcband.ReadAsArray() 
print(myarray)
print(src_ds)

#%%
#mueller_andco/demolition_footprints/demolition_footprints/data/model/original/predictions/BW/oslo_0_0.3_2023_4_5.tif
#these lines use gdal to import an image. 'myarray' can be any numpy array

mypoly=[]

for vec in rasterio.features.shapes(myarray):
    mypoly.append(shape(vec[0]))

#filtering out the very large polygons (just to avoid that the frame is 
# considered as a polygon, not necessary when we will apply on full pic)
filtered_polygons = [polygon for polygon in mypoly if polygon.area < 450*450 and polygon.area>15*15]  

#%%
print(len(mypoly))
print(len(filtered_polygons))

geoms = list(mypoly)
gdf = gpd.GeoDataFrame({'geometry': geoms})

# Plot the GeoDataFrame
# Plot the GeoDataFrame with custom style
plt.figure(figsize=(10, 10))  # Set the figure size and background color
plt.gca().set_facecolor('black')  # Set the background color

# Plot filled polygons with red color
gdf.plot(ax=plt.gca(),facecolor='none', edgecolor='red')

# Remove axes
plt.axis('off')

# Show plot
plt.show()

#plt.savefig('figname.png', facecolor=fig.get_facecolor(), transparent=True)

#%%
# Simplify the shapes to minimize the edges
simplified_polygons = [polygon.simplify(5, preserve_topology=True) for polygon in filtered_polygons]  # Adjust the tolerance value as needed


# Enforce right angles
rounded_polygons = [polygon.buffer(1, join_style=3,single_sided=True) for polygon in simplified_polygons]

rounded_polygons_ext = [polygon.exterior for polygon in rounded_polygons]
# Filter out small polygons (e.g., the frame of the picture)
#merged_polygons = unary_union(simplified_polygons)
# Convert polygons to GeoPandas GeoDataFrame
geoms = list(rounded_polygons_ext)
gdf = gpd.GeoDataFrame({'geometry': geoms})

# Plot the GeoDataFrame
# Plot the GeoDataFrame with custom style
plt.figure(figsize=(10, 10))  # Set the figure size and background color
#plt.gca().set_facecolor('black')  # Set the background color

# Plot filled polygons with red color
gdf.plot(ax=plt.gca(),facecolor='none', edgecolor='red')

# Remove axes
plt.axis('off')

# Show plot
plt.show()

#%%
# Overlay with the TIFF file (prediction)

# Overlay with the TIFF file (prediction)

with rasterio.open(segimg) as src:
    # Read the raster data
    data = src.read()

    # Plot the TIFF file as a background image
    plt.figure(figsize=(10, 10))
    plt.imshow(data[0], cmap='gray', alpha=0.5)#, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top), alpha=0.5)

    # Plot the filled polygons on top
    gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    plt.axis('off')
    plt.gca().set_facecolor('black')

    # Show plot
    plt.show()
#%%
# Overlay with the TIFF file (original pic)
with rasterio.open(orig_tif) as src:
    # Read the raster data
    data = src.read()

    # Plot the TIFF file as a background image
    plt.figure(figsize=(10, 10))
    plt.imshow(data[0], cmap='gray', alpha = 0.4 )#, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top), alpha=0.5)

    # Plot the filled polygons on top
    gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='gold',linewidth=5)
    plt.axis('off')
   
    # Set background color
    #plt.gca().set_facecolor('black')

    # Show plot
    plt.show()


#%%
"""segimg = 'data/model/original/predictions/BW_2023/oslo_0_0.3_2023_0_29.tif'
src_ds = gdal.Open(segimg, GA_ReadOnly )
srcband=src_ds.GetRasterBand(1)
myarray=srcband.ReadAsArray() 
#these lines use gdal to import an image. 'myarray' can be any numpy array


mypoly=[]
for vec in rasterio.features.shapes(myarray):
    mypoly.append(shape(vec[0]))


geoms = list(mypoly)
p = gpd.GeoSeries(mypoly[0])
p.plot()"""


# %%
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import rasterio.features
from shapely.geometry import shape
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

#%%
# Set the paths
root_dir = Path(__file__).parents[2]
path_data = root_dir / 'data/model'

# Directories containing the TIFF files
dir_2023 = path_data / 'test_full_pic'
dir_1937 = path_data / 'trondheim_1937'
dir_1979 = path_data / 'trondheim_1979'

# List of image selections
img_selects = [
    'trondheim_0.3_2023_1_0_2_14.tif',
    'trondheim_0.3_1937_1_1_2_14.tif',
    'trondheim_0.3_1979_1_0_2_14.tif'
]

# Function to process a single image
def process_image(img_select, dir_path):
    segimg = dir_path / 'predictions/test' / img_select
    src_ds = gdal.Open(str(segimg), GA_ReadOnly)
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

# Process all images and combine GeoDataFrames
all_gdfs = []
for img_select, dir_path in zip(img_selects, [dir_2023, dir_1937, dir_1979]):
    gdf = process_image(img_select, dir_path)
    all_gdfs.append(gdf)

# Combine all GeoDataFrames into one
combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))

# Pickle the combined GeoDataFrame
pickle_file_path = root_dir / "combined_geodata.pkl"
with open(pickle_file_path, 'wb') as f:
    pickle.dump(combined_gdf, f)

print(f"Combined GeoDataFrame has been pickled to {pickle_file_path}")

# Plot the combined GeoDataFrame for visualization
plt.figure(figsize=(10, 10))
combined_gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
plt.axis('off')
plt.show()

#%%

folder_path = '/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/ML_model/trondheim_1979/predictions/reassembled_tile'

# Regular expression to match the file format
file_pattern = re.compile(r'^(?P<project_name>.+)_tile_1m_(?P<row>\d+)_(?P<col>\d+)\.tif$')

# List to store file paths and parsed information
files_info = []

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
        
        # Add the file information to the list
        files_info.append({
            'file_path': file_path,
            'project_name': project_name,
            'row': row,
            'col': col
        })

#%%

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

# Example usage
project_name = "trondheim_1979"
combined_gdf = process_project_tiles(project_name)


#%%
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

# Assuming combined_gdf is your GeoDataFrame containing the polygons

# Create a blank image array
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
ax.margins(0,0)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

# Plot the polygons onto the blank array
combined_gdf.plot(ax=ax, facecolor='none', edgecolor='red')

# Save the plot to a buffer
fig.canvas.draw()
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# Create a transform (adjust as necessary)
transform = from_origin(0, 10, 1, 1)  # Example values, adjust as necessary

#%%
# Optionally, save the combined GeoDataFrame
pickle_file_path = root_dir / f"{project_name}_combined_geodata.pkl"
with open(pickle_file_path, 'wb') as f:
    pickle.dump(combined_gdf, f)

print(f"Combined GeoDataFrame for {project_name} has been pickled to {pickle_file_path}")

