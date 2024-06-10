
# %%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa

root_dir = str(Path(__file__).parents[3])

#%%
# didnt manage to import the function so I copy pasted (ugly)


input_dir_images = root_dir + '/data/temp/test_zoe/images'
#input_dir_labels = root_dir + '/data/temp/test_zoe/labels'
output_dir_images = root_dir + '/data/model/trondheim_1979/tiles/images'
#output_dir_labels = root_dir + '/data/model/test_full_pic/tiles/labels'


#%%
def tile_images_no_labels(input_dir_images,
                          output_dir_images,
                          tile_size=512, overlap_rate=0.01,
                          image_size=None):
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)

    # Create archive directories if they don't exist
    archive_dir_images = os.path.join(input_dir_images, 'archive')
    os.makedirs(archive_dir_images, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(
        input_dir_images) if f.endswith('.tif')]
    print(image_files)
    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the image size if not given
    total_iterations = 0
    if image_size is None:
        for file in image_files:
            image_path = os.path.join(input_dir_images, file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1
            total_iterations += num_tiles_x * num_tiles_y
    else:
        height, width = image_size, image_size

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_file in image_files:
            # Load the image
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1

            # Calculate the required padding
            padding_x = ((num_tiles_x-1) * effective_tile_size +
                         tile_size - width)
            padding_y = ((num_tiles_y-1) * effective_tile_size +
                         tile_size - height)

            # Pad the image and label
            image = np.pad(image, ((0, int(padding_y)),
                                   (0, int(padding_x)), (0, 0)))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the image
                    image_tile = image[y:y+tile_size, x:x+tile_size]

                    # Save the image tile to the output directory
                    image_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    image_tile_path = os.path.join(
                        output_dir_images, image_tile_filename)
                    cv2.imwrite(image_tile_path, image_tile)

                    pbar.update(1)

            # Move the processed image to the archive directory
            shutil.move(os.path.join(input_dir_images, image_file),
                        os.path.join(archive_dir_images, image_file))


# %%
print("Partitioning and cropping images without labels")
tile_images_no_labels(input_dir_images,
                      output_dir_images,
                      tile_size=512,
                      overlap_rate=0.01)