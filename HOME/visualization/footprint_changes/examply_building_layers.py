# %%
from HOME.visualization.footprint_changes.plot_building_layers import (
    plot_building_layers,
    bshape_from_tile_coords,
)

# %%
# %% first example

project_list = [
    "trondheim_1991",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    "trondheim_2016",
    "trondheim_kommune_2022",
]
bshape = bshape_from_tile_coords(3696, 45796)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(10, 50))
# %% second example
bshape = bshape_from_tile_coords(3754, 45755)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %% third example
bshape = bshape_from_tile_coords(3695, 45788)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %%
bshape = bshape_from_tile_coords(3696, 45796)
plot_building_layers(project_list, bshape, layer_overlap=0.1)
# %%
bshape = bshape_from_tile_coords(3695, 45798)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(20, 20))

# %%
project_list1 = [
    "trondheim_1969",
    "trondheim_1977",
    "trondheim_1983",
    "trondheim_1988",
    # "trondheim_1991",
    "trondheim_1994",
    "trondheim_1999",
    "trondheim_2006",
    "trondheim_2011",
    # "trondheim_2016",
    "trondheim_2017",
    "trondheim_kommune_2022",
]
bshape = bshape_from_tile_coords(3715, 45798)
plot_building_layers(
    project_list1, bshape, layer_overlap=0.1, figsize=(10, 40), cmap="tab20"
)
# %%

bshape = bshape_from_tile_coords(3725, 45798)
plot_building_layers(project_list, bshape, layer_overlap=0.1, figsize=(20, 20))
