# %%
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from pathlib import Path
import math


def pixel_to_geographic_coordinates(dataset, x, y):
    # Get the GeoTransform
    geotransform = dataset.GetGeoTransform()
    if geotransform is None:
        return None

    # Transform pixel coordinates to geographic coordinates
    x_geo = geotransform[0] + (x * geotransform[1]) + (y * geotransform[2])
    y_geo = geotransform[3] + (x * geotransform[4]) + (y * geotransform[5])

    return x_geo, y_geo


def display_rgb_geotiff_subset(file_path, x_start, y_start, width, height):
    # Open the GeoTIFF file
    dataset = gdal.Open(file_path)
    if dataset is None:
        print("Error: Unable to open the GeoTIFF file.")
        return

    # Read the subset of the RGB bands
    subset_rgb = dataset.ReadAsArray(x_start, y_start, width, height)

    # Plot the RGB subset using Matplotlib
    # Transpose the array to match Matplotlib's expectations for RGB images
    plt.imshow(np.transpose(subset_rgb, (1, 2, 0)))
    plt.colorbar(label='Pixel Intensity')
    plt.title("Subset of RGB GeoTIFF")

    # Set x-axis and y-axis ticks to show coordinates in degrees
    x_ticks = np.linspace(0, width, 5)
    y_ticks = np.linspace(0, height, 5)
    x_tick_labels = []
    y_tick_labels = []
    for x, y in zip(x_ticks, y_ticks):
        x_geo, y_geo = pixel_to_geographic_coordinates(dataset, x_start + x,
                                                       y_start + y)
        x_tick_labels.append('{:.4f}'.format(x_geo))
        y_tick_labels.append('{:.4f}'.format(y_geo))
    plt.xticks(x_ticks, labels=x_tick_labels, rotation=45)
    plt.yticks(y_ticks, labels=y_tick_labels)

    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")

    plt.show()


root_dir = str(Path(__file__).parents[2])
file_path = (root_dir + "/data/raw/orthophoto/res_0.3/trondheim_2019/" +
             "Eksport-nib_4326.tif")
display_rgb_geotiff_subset(file_path, 77200, 9200, 1000, 1000)

# %% display size and resolution of the image

dataset = gdal.Open(file_path)
print(f"Size of the image: {dataset.RasterXSize} x {dataset.RasterYSize}")
geotransform = dataset.GetGeoTransform()
pixel_width = geotransform[1]
pixel_height = geotransform[5]

# Constants
EARTH_RADIUS = 6371 * 1000  # Earth's radius in meters
LATITUDE = math.radians(63.43)  # Latitude of Trondheim, Norway in radians

# Pixel resolution in degrees
pixel_width_deg = abs(pixel_width)
pixel_height_deg = abs(pixel_height)

# Convert to radians
pixel_width_rad = math.radians(pixel_width_deg)
pixel_height_rad = math.radians(pixel_height_deg)

# Convert to meters
pixel_width_m = pixel_width_rad * EARTH_RADIUS * math.cos(LATITUDE)
pixel_height_m = pixel_height_rad * EARTH_RADIUS

print(f"Pixel resolution: {pixel_width_m:.4f} x {pixel_height_m:.4f} meters")
# %%
