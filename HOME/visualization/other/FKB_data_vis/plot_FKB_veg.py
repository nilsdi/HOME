'''
printing some roads and buildings on the FKB data
'''
#%% imports
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import fiona

from pathlib import Path

root_dir = Path(__file__).resolve().parents[4]
print(root_dir)
# load the FKB data
FKB_bygning_path = os.path.join(root_dir, 'data/raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb')
FKB_veg_path = os.path.join(root_dir, 'data/raw/FKB_veg/Basisdata_0000_Norge_5973_FKB-Veg_FGDB.gdb')
# list all layers
layers_bygning = fiona.listlayers(FKB_bygning_path)
layers_veg = fiona.listlayers(FKB_veg_path)
print(f' the layers in the FKB bygning data are: {layers_bygning}')
print(f' the layers in the FKB veg data are: {layers_veg}')

#%% load the data
veg_grenser = gpd.read_file(FKB_veg_path, layer='fkb_veg_grense')
veg_omrader = gpd.read_file(FKB_veg_path, layer='fkb_veg_omrade')
#bygg_grenser = gpd.read_file(FKB_bygning_path, layer='fkb_bygning_grense')

#%% simple plot in small bounding box
import folium
from shapely.geometry import box
bbox = [10.4081, 63.4305, 10.4101, 63.4325]
#bbox = [270202.737422,7041627.464458, 270250.921554,7041681.901292 ] # _25833
veg_grenser_4326 = veg_grenser.to_crs( 'EPSG:4326')
bbox_veg_grenser =veg_grenser_4326.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

# plot in simple folium map
m = folium.Map(location=[63.4305, 10.3951], zoom_start=13)
# Add a rectangle for the bounding box
folium.GeoJson(box(*bbox), style_function=lambda x: {'color': 'blue', 'fill': False}).add_to(m)
for _, row in bbox_veg_grenser.iterrows():
    folium.GeoJson(row.geometry, style_function = lambda x: {'color': 'red'}).add_to(m)
m
#%%
import folium
from shapely.geometry import box
bbox = [10.4081, 63.4305, 10.4101, 63.4325]
#bbox = [270202.737422,7041627.464458, 270250.921554,7041681.901292 ] # _25833
veg_omrader_4326 = veg_omrader.to_crs( 'EPSG:4326')
bbox_veg_omrader =veg_omrader_4326.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

# plot in simple folium map
m = folium.Map(location=[63.4305, 10.3951], zoom_start=13)
# Add a rectangle for the bounding box
folium.GeoJson(box(*bbox), style_function=lambda x: {'color': 'blue', 'fill': False}).add_to(m)
for _, row in bbox_veg_omrader.iterrows():
    folium.GeoJson(row.geometry, style_function = lambda x: {'color': 'red'}).add_to(m)
m
# %%
import fiona

# Assuming FKB_bygning_path is already defined as in your excerpt
FKB_bygning_path = os.path.join(root_dir, 'data/raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb')

# Specify the layer you are interested in
layer_name = 'fkb_bygning_omrade'

# Open the GDB file and access the specified layer
with fiona.open(FKB_bygning_path, layer=layer_name) as layer:
    # Access and print the metadata for the layer
    metadata = layer.meta
    print("Metadata for layer '{}':".format(layer_name))
    print(metadata)