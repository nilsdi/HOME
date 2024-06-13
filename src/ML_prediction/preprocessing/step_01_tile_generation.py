# %%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse
from osgeo import gdal
import pandas as pd

# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa

root_dir = str(Path(__file__).parents[3])


# %%
def tile_images_no_labels(
    input_dir_images,
    output_dir_images,
    tile_size=512,
    overlap_rate=0.00,
    move_to_archive=False,
    project_name=None,
    res=0.3,
    show_progress=False,
    prediction_mask=None,
):
    if prediction_mask is None:
        prediction_mask = pd.read_csv(
            root_dir + "/data/ML_prediction/prediction_mask/prediction_mask.csv",
            index_col=0,
        )
        prediction_mask.columns = prediction_mask.columns.astype(int)
        prediction_mask.index = prediction_mask.index.astype(int)

    skipped_tiles = 0
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)

    # Create archive directories if they don't exist
    if move_to_archive:
        archive_dir_images = os.path.join(input_dir_images, "archive")
        os.makedirs(archive_dir_images, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir_images) if f.endswith(".tif")]

    effective_tile_size = tile_size * (1 - overlap_rate)

    total_tiles = 0

    print(f"Processing {len(image_files)} images")
    for image_file in image_files:
        image_path = os.path.join(input_dir_images, image_file)
        # Load the image
        dataset = gdal.Open(image_path)
        geotransform = dataset.GetGeoTransform()

        # Calculate the coordinates of the top left corner
        top_left_x = geotransform[0]
        top_left_y = geotransform[3]

        # Calculate the offset and the coordinates of the top left corner of the first tile
        pixel_size = res * effective_tile_size

        offset_x = abs(top_left_x) % (pixel_size)
        offset_y = pixel_size - abs(top_left_y) % (pixel_size)

        coord_top_left_x = int(np.floor(top_left_x / pixel_size))
        coord_top_left_y = int(np.ceil(top_left_y / pixel_size))

        # Pad the image to ensure that the top right point lies on the grid
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(
            image,
            int(np.ceil(offset_y)),
            0,
            int(np.ceil(offset_x)),
            0,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        height, width, _ = image.shape

        num_tiles_x = int(np.ceil((width - tile_size) / (effective_tile_size))) + 1
        num_tiles_y = int(np.ceil((height - tile_size) / (effective_tile_size))) + 1

        # Calculate the required padding
        padding_x = (num_tiles_x - 1) * effective_tile_size + tile_size - width
        padding_y = (num_tiles_y - 1) * effective_tile_size + tile_size - height

        # Pad the image to make sure it contains an integer number of tiles
        image = np.pad(image, ((0, int(padding_y)), (0, int(padding_x)), (0, 0)))

        # Iterate over each tile
        total_iterations = num_tiles_x * num_tiles_y
        total_tiles += total_iterations
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates within Norway
                    grid_x = coord_top_left_x + i
                    grid_y = coord_top_left_y - j

                    # Only keep that tile if it's in the prediction mask
                    if prediction_mask.loc[grid_y, grid_x]:

                        # Calculate the tile coordinates within the image
                        x = int(i * effective_tile_size)
                        y = int(j * effective_tile_size)

                        # Crop the tile from the image
                        image_tile = image[y : y + tile_size, x : x + tile_size]

                        if image_tile.sum() != 0:

                            # Save the image tile to the output directory
                            if project_name:
                                image_tile_filename = f"{project_name}_{image_file[-5:-4]}_{coord_top_left_x + i}_{coord_top_left_y - j}.tif"
                            else:
                                image_tile_filename = f"{image_file[:-4]}_{coord_top_left_x + i}_{coord_top_left_y - j}.tif"
                            image_tile_path = os.path.join(
                                output_dir_images, image_tile_filename
                            )
                            cv2.imwrite(image_tile_path, image_tile)

                        else:
                            skipped_tiles += 1

                    else:
                        skipped_tiles += 1
                    pbar.update(1)

            # Move the processed image to the archive directory
            if project_name:
                shutil.move(
                    os.path.join(input_dir_images, image_file),
                    os.path.join(input_dir_images, f"{project_name}_{image_file[-5:]}"),
                )
            elif move_to_archive:
                shutil.move(
                    os.path.join(input_dir_images, image_file),
                    os.path.join(archive_dir_images, image_file),
                )
    print(f"Skipped {skipped_tiles} out of {total_tiles} tiles with no information")

    return


# %% Similar functions but for labels without images
# Probably shouldn't be used anymore

# def tile_labels_no_images(
#     input_dir_labels,
#     output_dir_labels,
#     tile_size=512,
#     overlap_rate=0.00,
#     image_size=None,
# ):
#     # Create output directories if they don't exist
#     os.makedirs(output_dir_labels, exist_ok=True)

#     # Get list of all image files in the input directory
#     label_files = [f for f in os.listdir(input_dir_labels) if f.endswith(".tif")]

#     effective_tile_size = tile_size * (1 - overlap_rate)

#     # Calculate the image size if not given
#     total_iterations = 0
#     if image_size is None:
#         for file in label_files:
#             label_path = os.path.join(input_dir_labels, file)
#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#             height, width = label.shape

#             num_tiles_x = int(np.ceil((width - tile_size) / (effective_tile_size))) + 1
#             num_tiles_y = int(np.ceil((height - tile_size) / (effective_tile_size))) + 1
#             total_iterations += num_tiles_x * num_tiles_y
#     else:
#         height, width = image_size, image_size

#     with tqdm(total=total_iterations, desc="Processing") as pbar:
#         for label_file in label_files:
#             # Load the label
#             label_path = os.path.join(input_dir_labels, label_file)
#             label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#             height, width = label.shape

#             num_tiles_x = int(np.ceil((width - tile_size) / (effective_tile_size))) + 1
#             num_tiles_y = int(np.ceil((height - tile_size) / (effective_tile_size))) + 1

#             # Calculate the required padding
#             padding_x = (num_tiles_x - 1) * effective_tile_size + tile_size - width
#             padding_y = (num_tiles_y - 1) * effective_tile_size + tile_size - height

#             # Pad the image and label
#             label = np.pad(label, ((0, int(padding_y)), (0, int(padding_x))))

#             # Iterate over each tile
#             for i in range(num_tiles_x):
#                 for j in range(num_tiles_y):
#                     # Calculate the tile coordinates
#                     x = int(i * effective_tile_size)
#                     y = int(j * effective_tile_size)

#                     # Crop the tile from the label
#                     label_tile = label[y : y + tile_size, x : x + tile_size]

#                     # Save the label tile to the output directory
#                     label_tile_filename = f"{label_file[:-4]}_{i}_{j}.tif"
#                     label_tile_path = os.path.join(
#                         output_dir_labels, label_tile_filename
#                     )
#                     cv2.imwrite(label_tile_path, label_tile)

#                     pbar.update(1)


# input_dir_labels = root_dir + "/data/temp/prepred/labels/"
# output_dir_labels = root_dir + "/data/model/topredict/train/label/"


# %%
def tile_generation(
    project_name, res=0.3, compression="i_lzw_25", prediction_mask=None
):

    input_dir_images = (
        root_dir + f"/data/raw/orthophoto/res_{res}/{project_name}/{compression}/"
    )
    output_dir_images = (
        root_dir
        + f"/data/ML_prediction/topredict/image/res_{res}/{project_name}/{compression}/"
    )

    print(
        f"Tiling images, project {project_name}, resolution {res}, "
        + f"compression {compression}"
    )

    tile_images_no_labels(
        input_dir_images,
        output_dir_images,
        tile_size=512,
        overlap_rate=0.00,
        project_name=project_name,
        res=res,
        prediction_mask=prediction_mask,
    )
    return


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile raw orthophotos for prediction with ML model"
    )
    parser.add_argument("--project_name", required=True, type=str)
    parser.add_argument("--res", required=False, type=float, default=0.3)
    parser.add_argument("--compression", required=False, type=str, default="i_lzw_25")
    args = parser.parse_args()
    tile_generation(args.project_name, args.res, args.compression)
