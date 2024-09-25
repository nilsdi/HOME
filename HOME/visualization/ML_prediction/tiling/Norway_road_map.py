"""
Plotting the road density/which tiles we actually predict on in Norway.
"""

# %%
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from HOME.get_data_path import get_data_path
import scipy.sparse

root_dir = Path(__file__).resolve().parents[4]
data_dir = get_data_path(root_dir)
# print(root_dir)
# %%
# import the road map of Norway (as the prediction mask - check, it should be dialated
# by 1, meaning all tiles neighboring a tile with a road are also considered to have a road)
path = root_dir / "data/ML_prediction/prediction_mask/roads/prediction_mask_0.3_512.csv"
# print(path)
# read in the csv as a pandas dataframe
df = pd.read_csv(path, index_col=0)
# %%
plt.figure(figsize=(20, 20))
# array = df.astype(int).values
# try your own colormap here - e.g.,
cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "gist_heat"]
current_cmap = cmaps[-1]
plt.imshow(df, cmap=current_cmap, interpolation="bilinear")
plt.axis("off")
plt.savefig(
    root_dir / f"data/figures/ML_prediction/tiling/Norway_road_map_{current_cmap}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
# %%
# print(df.head())
# %%
df.sum().sum()


# %% Building map
path = (
    data_dir
    / "ML_prediction/prediction_mask/buildings/masksparse_0.3_512_-494_41990.npz"
)
parts = str(path).split("_")
min_x = int(parts[-2])
min_y = int(parts[-1].split(".")[0])

building_mask = scipy.sparse.load_npz(path)
# %%
plt.figure(figsize=(20, 20))
current_cmap = cmaps[-1]
plt.imshow(
    building_mask.toarray(), cmap=current_cmap, interpolation="bilinear", origin="lower"
)
plt.axis("off")
plt.show()

# %% Plot building map for trondheim
trondheim_bounds = ([10.27, 63.3205, 10.5201, 63.4525],)
