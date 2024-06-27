# %%
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
import contextily as ctx
from pathlib import Path
from random import choice
import os
from shapely.geometry import box


# Load the image
root_dir = Path(__file__).parents[3]
img_dir = "data/ML_prediction/topredict/image/res_0.2/trondheim_mof_2023/i_lzw_25"

image_paths = [
    os.path.join(root_dir, img_dir, img)
    for img in os.listdir(os.path.join(root_dir, img_dir))
]

trondheim = gpd.read_file(
    root_dir
    / "/scratch/mueller_andco/orthophoto/HOME/data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb",
    layer="fkb_bygning_omrade",
).to_crs(epsg=25833)

# %%
# Pick random image from the images list

tile_size_px = 512
res = 0.2
grid_size = tile_size_px * res

image_path = choice(image_paths)

# %%
parts = image_path.split("/")[-1].split(".")[0].split("_")
xmin = int(parts[-2]) * grid_size
ymax = int(parts[-1]) * grid_size

xmax = xmin + grid_size
ymin = ymax - grid_size

img = Image.open(image_path)

# Define the coordinates (xmin, ymin, xmax, ymax)
# These should be the real world coordinates for your image
coords = (xmin, ymin, xmax, ymax)

fig, ax = plt.subplots(figsize=(10, 10))

# Create a GeoDataFrame with the image bounding box
# Add trondheim polygons
zoom_factor = 0.1
bbox = box(
    coords[0] - zoom_factor * grid_size,
    coords[1] - zoom_factor * grid_size,
    coords[2] + zoom_factor * grid_size,
    coords[3] + zoom_factor * grid_size,
)

trondheim.cx[bbox.bounds[0] : bbox.bounds[2], bbox.bounds[1] : bbox.bounds[3]].plot(
    ax=ax, color="red"
)

# Add basemap from contextily
ctx.add_basemap(ax, crs="EPSG:25833", source=ctx.providers.Esri.WorldImagery)


# Plot the image
extent = [coords[0], coords[2], coords[1], coords[3]]
ax.imshow(img, extent=extent, origin="upper", alpha=0.8)


plt.show()


# %%
