import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def partition_and_crop_images(input_dir_images, input_dir_labels,
                              output_dir_images, output_dir_labels,
                              tile_size=512, overlap_rate=0.01,
                              image_size=None,
                              imbalance_threshold=(0.005, 0.9)):
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(
        input_dir_images) if f.endswith('.tif')]

    # Calculate the image size if not given
    if image_size is None:
        label_path = os.path.join(input_dir_labels, image_files[0])
        label = cv2.imread(label_path)
        height, width, _ = label.shape
    else:
        height, width = image_size, image_size

    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the number of tiles in each dimension
    num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
    num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1

    total_iterations = len(image_files) * num_tiles_x * num_tiles_y
    skipped = 0

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_file in image_files:
            # Load the label
            label_path = os.path.join(input_dir_labels, image_file)
            label = cv2.imread(label_path)

            # Calculate the ratio of positive to negative pixels
            num_positive_pixels = np.sum(label > 0)
            num_negative_pixels = np.sum(label == 0)
            ratio = num_positive_pixels / num_negative_pixels

            # Skip this image if the ratio is above or below the threshold
            if (ratio < imbalance_threshold[0] or
                    ratio > imbalance_threshold[1]):
                skipped += num_tiles_x * num_tiles_y
                continue

            # Load the image
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)

            # Calculate the required padding
            padding_x = ((num_tiles_x-1) * effective_tile_size +
                         tile_size - width)
            padding_y = ((num_tiles_y-1) * effective_tile_size +
                         tile_size - height)

            # Pad the image and label
            image = np.pad(image, ((0, int(padding_y)),
                                   (0, int(padding_x)), (0, 0)))
            label = np.pad(label, ((0, int(padding_y)),
                                   (0, int(padding_x)), (0, 0)))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the label
                    label_tile = label[y:y+tile_size, x:x+tile_size]

                    # Calculate the ratio of positive to negative pixels
                    num_positive_pixels = np.sum(label_tile > 0)
                    num_negative_pixels = np.sum(label_tile == 0)
                    ratio = num_positive_pixels / num_negative_pixels

                    # Skip this image if the ratio is above or below the
                    # threshold
                    if ratio < imbalance_threshold[0] or (
                            ratio > imbalance_threshold[1]):
                        skipped += 1
                        continue

                    # Crop the tile from the image
                    image_tile = image[y:y+tile_size, x:x+tile_size]

                    # Save the image tile to the output directory
                    image_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    image_tile_path = os.path.join(
                        output_dir_images, image_tile_filename)
                    cv2.imwrite(image_tile_path, image_tile)

                    # Save the label tile to the output directory
                    label_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    label_tile_path = os.path.join(
                        output_dir_labels, label_tile_filename)
                    cv2.imwrite(label_tile_path, label_tile)
                    pbar.update(1)

    print(f"Skipped {skipped} tiles due to class imbalance")


root_dir = str(Path(__file__).parents[2])
input_dir_images = root_dir + '/data/temp/pretrain/images/'
input_dir_labels = root_dir + '/data/temp/pretrain/labels'
output_dir_images = root_dir + '/data/train/image/'
output_dir_labels = root_dir + '/data/train/label/'

partition_and_crop_images(input_dir_images, input_dir_labels,
                          output_dir_images, output_dir_labels,
                          tile_size=512, overlap_rate=0.01)
