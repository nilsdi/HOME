# %%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse

# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa

root_dir = str(Path(__file__).parents[3])

# %%

parser = argparse.ArgumentParser(description="Tile generation")
parser.add_argument(
    "--res", default=0.3, type=float, help="resolution of the tiles in meters"
)
args = parser.parse_args()
res = args.res


def partition_and_crop_images(
    input_dir_images,
    input_dir_labels,
    output_dir_images,
    output_dir_labels,
    tile_size=512,
    overlap_rate=0.01,
    image_size=None,
    imbalance_threshold=(0.005, 0.9),
    res=0.3,
):
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # Create archive directories if they don't exist
    archive_dir_images = os.path.join(input_dir_images, "archive")
    archive_dir_labels = os.path.join(input_dir_labels, "archive")
    os.makedirs(archive_dir_images, exist_ok=True)
    os.makedirs(archive_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir_images) if f.endswith(".tif")]

    # Filter to keep only images of the good resolution
    image_files = [f for f in image_files if str(res) in f]

    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the image size if not given
    total_iterations = 0
    if image_size is None:
        for file in image_files:
            label_path = os.path.join(input_dir_labels, file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape

            num_tiles_x = int(np.ceil((width - tile_size) / (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) / (effective_tile_size))) + 1
            total_iterations += num_tiles_x * num_tiles_y
    else:
        height, width = image_size, image_size

    empty_but_kept = {}
    for image_file in image_files:
        empty_but_kept[image_file] = []

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_file in image_files:
            # Load the label
            label_path = os.path.join(input_dir_labels, image_file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape
            num_tiles_x = int(np.ceil((width - tile_size) / (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) / (effective_tile_size))) + 1

            # Load the image
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)

            # Calculate the required padding
            padding_x = (num_tiles_x - 1) * effective_tile_size + tile_size - width
            padding_y = (num_tiles_y - 1) * effective_tile_size + tile_size - height

            # Pad the image and label
            image = np.pad(image, ((0, int(padding_y)), (0, int(padding_x)), (0, 0)))
            label = np.pad(label, ((0, int(padding_y)), (0, int(padding_x))))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the label
                    label_tile = label[y : y + tile_size, x : x + tile_size]

                    # Crop the tile from the image
                    image_tile = image[y : y + tile_size, x : x + tile_size]

                    # Skip this image if it's black (out of bounds)
                    if np.sum(image_tile) > 0:

                        # Save the image tile to the output directory
                        image_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                        image_tile_path = os.path.join(
                            output_dir_images, image_tile_filename
                        )
                        cv2.imwrite(image_tile_path, image_tile)

                        # Save the label tile to the output directory
                        label_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                        label_tile_path = os.path.join(
                            output_dir_labels, label_tile_filename
                        )
                        cv2.imwrite(label_tile_path, label_tile)
                    pbar.update(1)

            # Move the processed image and label to the archive directory
            shutil.move(
                os.path.join(input_dir_images, image_file),
                os.path.join(archive_dir_images, image_file),
            )
            shutil.move(
                os.path.join(input_dir_labels, image_file),
                os.path.join(archive_dir_labels, image_file),
            )


input_dir_images = root_dir + "/data/temp/pretune/images/"
input_dir_labels = root_dir + "/data/temp/pretune/labels"
output_dir_images = root_dir + "/data/ML_training/tune/image/"
output_dir_labels = root_dir + "/data/ML_training/tune/label/"

print("Partitioning and cropping images with labels")
partition_and_crop_images(
    input_dir_images,
    input_dir_labels,
    output_dir_images,
    output_dir_labels,
    tile_size=512,
    overlap_rate=0.01,
    res=res,
)

# %%
