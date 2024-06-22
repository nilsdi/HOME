'''
Plotting the road density/which tiles we actually predict on in Norway.
'''
#%%
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

root_dir   = Path(__file__).resolve().parents[4]
#print(root_dir)
# %%
# import the road map of Norway (as the prediction mask - check, it should be dialated
# by 1, meaning all tiles neighboring a tile with a road are also considered to have a road)
path = root_dir / "data/ML_prediction/prediction_mask/prediction_mask.csv"
#print(path)
# read in the csv as a pandas dataframe
df = pd.read_csv(path, index_col=0)
# %%
plt.figure(figsize=(20, 20))
#array = df.astype(int).values
# try your own colormap here - e.g., 
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gist_heat']
current_cmap = cmaps[-1]
plt.imshow(df, cmap = current_cmap, interpolation='bilinear')
plt.axis('off')
plt.savefig(root_dir/f'data/figures/ML_prediction/tiling/Norway_road_map_{current_cmap}.png',  
                dpi=150, bbox_inches='tight')
plt.show()
# %%
#print(df.head())
# %%
df.sum().sum()
