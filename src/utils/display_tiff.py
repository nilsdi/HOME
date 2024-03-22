# %%
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from pathlib import Path
import math
from pyproj import Transformer
import rasterio
import os
import random

root_dir = str(Path(__file__).resolve().parents[2])
# %%


def pixel_to_geographic_coordinates(dataset, x, y):
    # Get the GeoTransform
    geotransform = dataset.GetGeoTransform()
    if geotransform is None:
        return None

    # Transform pixel coordinates to geographic coordinates
    x_geo = geotransform[0] + (x * geotransform[1]) + (y * geotransform[2])
    y_geo = geotransform[3] + (x * geotransform[4]) + (y * geotransform[5])

    # Get the projection from the dataset
    projection = dataset.GetProjection()
    srs = osr.SpatialReference(wkt=projection)
    unit = srs.GetLinearUnitsName()

    # If the projection is in meters, convert to degrees
    if unit == 'metre':
        transformer = Transformer.from_crs(srs.ExportToProj4(), 'EPSG:4326',
                                           always_xy=True)
        x_geo, y_geo = transformer.transform(x_geo, y_geo)

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
    if subset_rgb.shape[0] == 3:
        subset_rgb = np.transpose(subset_rgb, (1, 2, 0))
    plt.imshow(subset_rgb)
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


file_path = (root_dir + "/data/raw/orthophoto/res_0.3/trondheim_2019/" +
             "i_lzw_25/Eksport-nib.tif")
display_rgb_geotiff_subset(file_path, 77200, 9200, 1000, 1000)

# %% display size and resolution of the image

dataset = gdal.Open(file_path)
projection = dataset.GetProjection()
authority = projection.split('"')[1]
print(f"Projection: {authority}")
print(f"Size of the image: {dataset.RasterXSize} x {dataset.RasterYSize}")
geotransform = dataset.GetGeoTransform()
pixel_width = abs(geotransform[1])
pixel_height = abs(geotransform[5])

# Check units of the projection
projection = dataset.GetProjection()
srs = osr.SpatialReference(wkt=projection)
unit = srs.GetLinearUnitsName()

if unit != 'metre':
    # Constants
    EARTH_RADIUS = 6371 * 1000  # Earth's radius in meters
    LATITUDE = math.radians(63.43)  # Latitude of Trondheim, Norway in radians

    # Pixel resolution in degrees
    pixel_width_deg = pixel_width
    pixel_height_deg = pixel_height

    # Convert to radians
    pixel_width_rad = math.radians(pixel_width_deg)
    pixel_height_rad = math.radians(pixel_height_deg)

    # Convert to meters
    pixel_width = pixel_width_rad * EARTH_RADIUS * math.cos(LATITUDE)
    pixel_height = pixel_height_rad * EARTH_RADIUS

print(f"Pixel resolution: {pixel_width:.4f} x {pixel_height:.4f} meters")
# %%

file_path = (root_dir + "/data/temp/pretrain/labels/" +
             "trondheim_2019_rect.tif")
display_rgb_geotiff_subset(file_path, 0, 0, 28000, 17000)

# %%
file_path = (root_dir + "/data/temp/pretrain/images/" +
             "trondheim_2019_rect.tif")
display_rgb_geotiff_subset(file_path, 0, 0, 28000, 17000)


# %% display image, label and prediction side to side

def display_images_side_by_side(name, prediction: bool = False):
    if name == 'random':
        if prediction:
            prediction_folder = root_dir + "/data/model/predictions/"
            files_in_folder = [f for f in os.listdir(prediction_folder)]
            name = files_in_folder[random.randint(0, len(files_in_folder))]
            image_path = (root_dir + f"/data/model/train/image/{name}")
            label_path = (root_dir + f"/data/model/train/label/{name}")
            prediction_path = (root_dir + f"/data/model/predictions/{name}")
        else:
            image_folder = root_dir + "/data/model/train/image/"
            files_in_folder = [f for f in os.listdir(image_folder)]
            name = files_in_folder[random.randint(0, len(files_in_folder))]
            image_path = (root_dir + f"/data/model/train/image/{name}")
            label_path = (root_dir + f"/data/model/train/label/{name}")
    else:
        image_path = (root_dir + f"/data/model/train/image/{name}.tif")
        label_path = (root_dir + f"/data/model/train/label/{name}.tif")
        if prediction:
            prediction_path = (
                root_dir + f"/data/model/predictions/{name}.tif")

    # Open the files
    if prediction:
        with (rasterio.open(image_path) as image,
              rasterio.open(label_path) as label,
              rasterio.open(prediction_path) as prediction):
            # Read the data
            image_data = image.read([1, 2, 3])
            label_data = label.read(1)
            prediction_data = prediction.read(1)

            # Create a figure with three subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Display the image
            axs[0].imshow(image_data.transpose((1, 2, 0)))
            axs[0].set_title('Image')
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            # Display the label
            axs[1].imshow(label_data, cmap='gray')
            axs[1].set_title('Ground Truth')
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            # Display the prediction
            axs[2].imshow(prediction_data, cmap='gray')
            axs[2].set_title('Prediction')
            axs[2].set_xticks([])
            axs[2].set_yticks([])

            # Show the figure
            plt.show()
    else:
        with (rasterio.open(image_path) as image,
              rasterio.open(label_path) as label):
            # Read the data
            image_data = image.read([1, 2, 3])
            label_data = label.read(1)

            # Create a figure with two subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Display the image
            axs[0].imshow(image_data.transpose((1, 2, 0)))
            axs[0].set_title('Image')
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            # Display the label
            axs[1].imshow(label_data, cmap='gray')
            axs[1].set_title('Ground Truth')
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            # Show the figure
            plt.show()
    return (fig)


fig = display_images_side_by_side('oslo_0_latest_14_7', prediction=False)

# %%

for i in range(8):
    fig = display_images_side_by_side('random', prediction=False)
    fig.savefig(root_dir + f"/figures/figure_{i}.png")


# %% display image, label and prediction for Inria, WHU and Mass side to side
filename = 'oslo_0_latest_14_7'
datasets = ['Inria', 'WHU', 'Mass']

image_path = (root_dir + f"/data/model/train/image/{filename}.tif")
label_path = (root_dir + f"/data/model/train/label/{filename}.tif")

# Open the files
with (rasterio.open(image_path) as image,
      rasterio.open(label_path) as label):
    # Read the data
    image_data = image.read([1, 2, 3])
    label_data = label.read(1)

    # Create a figure with four subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Display the image
    axs[0].imshow(image_data.transpose((1, 2, 0)))
    axs[0].set_title('Image')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Display the label
    axs[1].imshow(label_data, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    for i, dataset in enumerate(datasets):
        prediction_path = (
            root_dir + f"/data/model/predictions/{dataset}/{filename}.tif")

        # Open the files
        with rasterio.open(prediction_path) as prediction:
            # Read the data
            prediction_data = prediction.read(1)
            axs[i + 2].imshow(prediction_data, cmap='gray')
            axs[i + 2].set_title(f'{datasets[i]} Weights Prediction')
            axs[i + 2].set_xticks([])
            axs[i + 2].set_yticks([])

    # Show the figure
    plt.show()
    fig.savefig(root_dir + "/figures/pretrain_comparison.png")
