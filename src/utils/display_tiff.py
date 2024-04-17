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


def display_rgb_geotiff_subset(file_path, x_start, y_start, width=None,
                               height=None, dpi=None):

    if dpi is None:
        dpi = 100

    # Open the GeoTIFF file
    dataset = gdal.Open(file_path)
    if dataset is None:
        print("Error: Unable to open the GeoTIFF file.")
        return

    # IF width and height not specific, then entire image
    if width is None:
        width = dataset.RasterXSize
        height = dataset.RasterYSize

    # Read the subset of the RGB bands
    subset_rgb = dataset.ReadAsArray(x_start, y_start, width, height)

    # Plot the RGB subset using Matplotlib
    # Transpose the array to match Matplotlib's expectations for RGB images
    if subset_rgb.shape[0] == 3:
        subset_rgb = np.transpose(subset_rgb, (1, 2, 0))
        cmap = None
    else:
        cmap = 'gray'
    plt.figure(dpi=dpi)
    plt.imshow(subset_rgb, cmap=cmap)
    # plt.colorbar(label='Pixel Intensity')
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


file_path = ("data/raw/orthophoto/res_0.3/trondheim_strinda_1937/i_lzw_25/" +
             "Eksport-nib.tif")
display_rgb_geotiff_subset(file_path, 0, 0, 10000, 10000, dpi=300)

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

def display_images_side_by_side(
        name, prediction: bool = False, label: bool = True,
        prediction_folder=root_dir + "/data/model/original/predictions/",
        image_folder=root_dir + "/data/model/original/train/image/",
        label_folder=root_dir + "/data/model/original/train/label/"):

    assert prediction or label, (
        "At least one of prediction or label must be True")
    if name == 'random':
        if prediction:
            files_in_folder = [f for f in os.listdir(prediction_folder)]
            name = files_in_folder[random.randint(0, len(files_in_folder))]
            image_path = image_folder + name
            label_path = label_folder + name
            prediction_path = prediction_folder + name
        else:
            files_in_folder = [f for f in os.listdir(image_folder)]
            name = files_in_folder[random.randint(0, len(files_in_folder))]
            image_path = image_folder + name
            label_path = label_folder + name
        print('name:', name)
    else:
        image_path = image_folder + name
        label_path = label_folder + name
        if prediction:
            prediction_path = prediction_folder + name

    # Open the files
    if prediction:
        with (rasterio.open(image_path) as image,
              rasterio.open(prediction_path) as prediction):
            # Read the data
            image_data = image.read([1, 2, 3])
            prediction_data = prediction.read(1)

            if label:
                with rasterio.open(label_path) as label:
                    label_data = label.read(1)

            # Create a figure with three subplots
            fig, axs = plt.subplots(1, 3 if label else 2, figsize=(15, 5))

            # Display the image
            axs[0].imshow(image_data.transpose((1, 2, 0)))
            axs[0].set_title('Image')
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            if label:
                # Display the label
                axs[1].imshow(label_data, cmap='gray')
                axs[1].set_title('Ground Truth')
                axs[1].set_xticks([])
                axs[1].set_yticks([])

            # Display the prediction
            axs[-1].imshow(prediction_data, cmap='gray')
            axs[-1].set_title('Prediction')
            axs[-1].set_xticks([])
            axs[-1].set_yticks([])

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


image_folder = root_dir + "/data/model/topredict/train/image/"
prediction_folder = root_dir + "/data/model/topredict/predictions/BW_1937/"

fig = display_images_side_by_side('random', prediction=True,
                                  prediction_folder=prediction_folder,
                                  image_folder=image_folder, label=False)

# %%

for i in range(8):
    image_folder = root_dir + "/data/model/train/image/"
    files_in_folder = [f for f in os.listdir(image_folder)]
    name = files_in_folder[random.randint(0, len(files_in_folder))]
    image_path = (root_dir + f"/data/model/train/image/{name}")
    label_path = (root_dir + f"/data/model/train/label/{name}")

    # Copy the files to the test folder
    os.system(f"cp {image_path} {root_dir}/figures/images/{name}")
    os.system(f"cp {label_path} {root_dir}/figures/labels/{name}")


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
    fig.savefig(root_dir + "/figures/pretrain_comparison.png",
                bbox_inches='tight', pad_inches=0)


# %% display image, label and prediction for trondheim in 3 years: 1937, 1999
# and 2023. 3*3 images with years on the left and image, label and prediction
# on the right

folder_images = root_dir + "/data/model/topredict/train/image/"
folder_labels = root_dir + "/data/model/topredict/train/label/"
folder_predictions = root_dir + "/data/model/topredict/predictions/"

years = ['1937', '1999', '2023']
filenames = ['trondheim_0.3_1937_1_1_',
             'trondheim_0.3_2019_1_0_',
             'trondheim_0.3_2023_1_0_']


def display_several_years(folder_images, folder_labels, folder_predictions,
                          years, filenames, suffix=None):
    if suffix is None:
        files_in_folder = [f for f in os.listdir(folder_images) if
                           f.startswith(filenames[0])]
        name = files_in_folder[random.randint(0, len(files_in_folder))]
        suffix = name.split('_')[-2] + '_' + name.split('_')[-1]

    label_path = folder_labels + (filenames[-1] + suffix)
    label_data = rasterio.open(label_path).read(1)

    # Create a figure with nine subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, year in enumerate(years):
        if year == '2023':
            alpha = 1
        else:
            alpha = 0.5
        image_path = folder_images + (filenames[i] + suffix)
        prediction_path = folder_predictions + (
            filenames[i] + suffix)
        # Open the files
        with (rasterio.open(image_path) as image,
              rasterio.open(prediction_path) as prediction):
            # Read the data
            image_data = image.read([1, 2, 3])
            prediction_data = prediction.read(1)

            # Display the image
            axs[i, 0].imshow(image_data.transpose((1, 2, 0)))
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])

            # Display the prediction
            axs[i, 1].imshow(prediction_data, cmap='gray')
            axs[i, 1].set_xticks([])
            axs[i, 1].set_yticks([])

            # Display the label
            axs[i, 2].imshow(label_data, cmap='gray', alpha=alpha)
            axs[i, 2].set_xticks([])
            axs[i, 2].set_yticks([])

        # Add year to the left
        axs[i, 0].text(-0.1, 0.5, year, fontsize=12, ha='center',
                       va='center', transform=axs[i, 0].transAxes)

    # Add image prediction label to the top
    axs[0, 0].text(0.5, 1.05, 'Image', fontsize=12, ha='center',
                   va='center', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, 1.05, 'Prediction', fontsize=12, ha='center',
                   va='center', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.5, 1.05, 'Ground Truth (2023)', fontsize=12, ha='center',
                   va='center', transform=axs[0, 2].transAxes)

    # Add filename as title
    fig.suptitle(suffix, fontsize=16)

    # Show the figure
    plt.show()
    return fig


fig = display_several_years(folder_images, folder_labels, folder_predictions,
                            years, filenames)

# %% Function to plot side by side image and different predictions


def display_predictions(image_folder, prediction_folders: list,
                        prediction_names: list,
                        label_folder=None, name='random'):
    if name == 'random':
        files_in_folder = [f for f in os.listdir(prediction_folders[-1])]
        name = files_in_folder[random.randint(0, len(files_in_folder))]
        image_path = image_folder + name
        if label_folder is not None:
            label_path = label_folder + name
    # Open the files
    with rasterio.open(image_path) as image:
        # Read the data
        image_data = image.read([1, 2, 3])

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, len(prediction_folders) + 2 if label_folder
                                is not None else len(prediction_folders) + 1,
                                figsize=(5 * len(prediction_folders), 5))

        # Display the image
        axs[0].imshow(image_data.transpose((1, 2, 0)))
        axs[0].set_title('Image')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        k = 1
        if label_folder is not None:
            with rasterio.open(label_path) as label:
                # Read the data
                label_data = label.read(1)
                axs[1].imshow(label_data, cmap='gray')
                axs[1].set_title('Ground Truth')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
            k = 2

        for i, prediction_folder in enumerate(prediction_folders):
            prediction_path = prediction_folder + name
            with rasterio.open(prediction_path) as prediction:
                # Read the data
                prediction_data = prediction.read(1)
                axs[i + k].imshow(prediction_data, cmap='gray')
                axs[i + k].set_title(f'Prediction {prediction_names[i]}')
                axs[i + k].set_xticks([])
                axs[i + k].set_yticks([])

        # display name of the image
        fig.suptitle(name, fontsize=16)

        # Show the figure
        plt.show()


image_folder = root_dir + "/data/model/topredict/train/image/"
prediction_folders = [root_dir + "/data/model/topredict/predictions/",
                      root_dir + "/data/model/topredict/predictions/BW_1937/"]
prediction_names = ['Original training', 'B&W training']

display_predictions(image_folder, prediction_folders, prediction_names)
# %%
