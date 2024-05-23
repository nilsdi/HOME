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

#%%

#TEST FOR ONE 3*3 TILE 

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

# Directory containing the TIFF files
orig_tif = root_dir / 'data/model/test_full_pic/tiles/reassembled_tile/test.tif'

segimg = root_dir / 'data/model/test_full_pic/predictions/test_33/test_33.tif'
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
filtered_polygons = [polygon for polygon in mypoly if polygon.area < 450*450]  

#%%
print(len(mypoly))
print(len(filtered_polygons))
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
#plt.figure(figsize=(10, 10))  # Set the figure size and background color
plt.gca().set_facecolor('black')  # Set the background color

# Plot filled polygons with red color
gdf.plot(ax=plt.gca(),facecolor='none', edgecolor='red')

# Remove axes
plt.axis('off')

# Show plot
plt.show()

#%%
# Assuming gdf is your GeoDataFrame containing the polygons

print(gpd.GeoDataFrame({'geometry':mypoly}).area.max())

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
    plt.imshow(data[0], cmap='gray', alpha = 0.8 )#, extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top), alpha=0.5)

    # Plot the filled polygons on top
    gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    plt.axis('off')
    # Set background color
    #plt.gca().set_facecolor('black')

    # Show plot
    plt.show()

#%%
segimg = 'data/model/original/predictions/BW_2023/oslo_0_0.3_2023_0_29.tif'
src_ds = gdal.Open(segimg, GA_ReadOnly )
srcband=src_ds.GetRasterBand(1)
myarray=srcband.ReadAsArray() 
#these lines use gdal to import an image. 'myarray' can be any numpy array


mypoly=[]
for vec in rasterio.features.shapes(myarray):
    mypoly.append(shape(vec[0]))


geoms = list(mypoly)
p = gpd.GeoSeries(mypoly[0])
p.plot()


