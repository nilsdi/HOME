# %% imports
from step_01_reassembling_tiles import (
    extract_tile_numbers,
    get_max_min_extend,
    get_large_tiles,
)

# %% test get_large_tiles
extend_tile_coords = [2, 42, 2, 52]
n_tiles_edge = 8
n_overlap = 1
large_tiles = get_large_tiles(extend_tile_coords, n_tiles_edge, n_overlap)
print(large_tiles)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()

# Plot the initial rectangle
initial_rect = patches.Rectangle(
    (extend_tile_coords[0], extend_tile_coords[2] - 1),
    extend_tile_coords[1] + 1 - extend_tile_coords[0],
    extend_tile_coords[3] - extend_tile_coords[2] + 1,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(initial_rect)

# Plot rectangles for each entry in large_tiles
for key, value in large_tiles.items():
    lower_left = value[0]
    upper_right = value[1]
    width = upper_right[0] - lower_left[0]
    height = upper_right[1] - lower_left[1]
    rect = patches.Rectangle(
        (lower_left[0], lower_left[1]),
        width,
        height,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    # Calculate center of the rectangle for text placement
    center_x = lower_left[0] + width / 2
    center_y = lower_left[1] + height / 2
    ax.text(center_x, center_y, key, ha="center", va="center", fontsize=6)

# Set limits and show plot
ax.set_xlim(
    [extend_tile_coords[0] - n_tiles_edge, extend_tile_coords[1] + 2 * n_tiles_edge]
)
ax.set_ylim(
    [extend_tile_coords[2] - n_tiles_edge, extend_tile_coords[3] + 2 * n_tiles_edge]
)
# aspect equal
ax.set_aspect("equal", "box")
# ax.set_ylim([min([v[0][1] for v in large_tiles.values()]), max([v[1][1] for v in large_tiles.values()])])
plt.show()

# %% plot reassembled tiles in reference map
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
data_path = root_dir / "data"

reassambled_tiles = data_path / "temp/test_assembly"
# all files in the directory
files = [f for f in reassambled_tiles.glob("*")]

tile = files[3]
# Path to your GeoTIFF file
geotiff_path = tile

# %% plot
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Path to your GeoTIFF tile
tile_path = tile
print(tile_path)
# Open the GeoTIFF file
with rasterio.open(tile_path) as src:
    # Plot the GeoTIFF
    plt.figure(figsize=(6, 6))
    show(src)
    plt.show()

    # Print the bounding box coordinates
    bbox = src.bounds
    print(f"Bounding Box: {bbox}")
# %%
import folium
import rasterio
from rasterio.warp import transform_bounds

# Path to your GeoTIFF tile
tile_path = tile
print(tile_path)

# Open the GeoTIFF file
with rasterio.open(tile_path) as src:
    # Transform the bounding box to EPSG:4326
    print(src.bounds)
    # flip longitude and latitude of the bounds
    flipped_bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
    bbox_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)

    # Calculate the center of the transformed bounding box
    center = [(bbox_4326[1] + bbox_4326[3]) / 2, (bbox_4326[0] + bbox_4326[2]) / 2]

# Create a Folium map centered at the transformed bounding box center
m = folium.Map(location=center, zoom_start=12)

# Draw the transformed bounding box as a rectangle on the map
folium.Rectangle(
    bounds=[[bbox_4326[1], bbox_4326[0]], [bbox_4326[3], bbox_4326[2]]],
    color="blue",
    fill=True,
    fill_color="blue",
    fill_opacity=0.2,
).add_to(m)

# Display the map
m

# %%
import folium
import rasterio
from rasterio.warp import transform_bounds
from folium.raster_layers import ImageOverlay


# Open the GeoTIFF file
with rasterio.open(tile_path) as src:
    # Transform the bounding box to EPSG:4326
    bbox_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)

    # Calculate the center of the transformed bounding box
    center = [(bbox_4326[1] + bbox_4326[3]) / 2, (bbox_4326[0] + bbox_4326[2]) / 2]

# Create a Folium map centered at the transformed bounding box center
m = folium.Map(location=center, zoom_start=12)

# Add the GeoTIFF as an overlay
ImageOverlay(
    name="GeoTIFF Overlay",
    image=str(tile_path),
    bounds=[[bbox_4326[1], bbox_4326[0]], [bbox_4326[3], bbox_4326[2]]],
    opacity=0.6,
    interactive=True,
    cross_origin=False,
    zindex=1,
).add_to(m)

# Add layer control and display the map
folium.LayerControl().add_to(m)
m

# %%
