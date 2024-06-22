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
bygg_omrader = gpd.read_file(FKB_bygning_path, layer='fkb_bygning_omrade')

#%%
