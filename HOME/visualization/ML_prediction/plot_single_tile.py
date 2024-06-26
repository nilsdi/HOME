# %%
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
import contextily as ctx

# Load the image
img_path = "path_to_your_image.tiff"
img = Image.open(img_path)

# Define the coordinates (xmin, ymin, xmax, ymax)
# These should be the real world coordinates for your image
coords = (xmin, ymin, xmax, ymax)

fig, ax = plt.subplots(figsize=(10, 10))

# Create a GeoDataFrame with the image bounding box
gdf = gpd.GeoDataFrame({"geometry": [gpd.box(*coords)]}, crs="EPSG:4326")

# Plot the basemap
gdf.plot(ax=ax, alpha=0)

# Add basemap from contextily
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)

# Plot the image
extent = [coords[0], coords[2], coords[1], coords[3]]
ax.imshow(img, extent=extent, origin="upper", alpha=0.6)

# Set the extent of the plot to match the image coordinates
ax.set_xlim(coords[0], coords[2])
ax.set_ylim(coords[1], coords[3])

plt.show()
