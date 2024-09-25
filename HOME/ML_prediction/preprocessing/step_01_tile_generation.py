"""
Preparing any large geotiff images for prediction with the ML model by tiling them into
smaller images. The tiling is done in a grid that starts at 0,0 in EPSG:25833 and extends
towards north and east, each output tile fits into this grid.
"""

# %%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse
from osgeo import gdal
import pandas as pd
import geopandas as gpd
import scipy.sparse
from HOME.ML_training.preprocessing.get_label_data.get_labels import get_labels
import pickle

# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa
from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)


# %%
def tile_images_no_labels(
    input_dir_images,
    output_dir_images,
    tile_size,
    res,
    overlap_rate=0,
    move_to_archive=False,
    project_name=None,
    prediction_mask=None,
    prediction_type="buildings",
):
    """
    Creates tiles in tilesize from images in input_dir_images and saves them in
    output_dir_images. The tiles are named according to their position relative in the
    grid that would be started at absolute 0,0 in CRS EPSG:25833 and extends towards
    north and east. The tiles are only created if the corresponding grid cell is in the
    prediction_mask.
    """
    # Load the prediction mask we have premade if no other is provided
    if prediction_mask is None:
        folderpath = data_path / f"ML_prediction/prediction_mask/{prediction_type}"
        filepath = [
            f
            for f in os.listdir(folderpath)
            if (str(res) in f)
            and (str(tile_size) in f)
            and (str(overlap_rate)) in f
            and f.endswith(".npz")
        ][0]
        prediction_mask = scipy.sparse.load_npz(folderpath / filepath)
        parts = filepath.split("_")
        min_x = int(parts[-2])
        min_y = int(parts[-1].split(".")[0])

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
        grid_size_m = res * effective_tile_size

        # the logical grid starts at 0,0 in EPSG:25833, the coords of the new top left are:
        coordgrid_top_left_x = int(np.floor(top_left_x / grid_size_m))
        coordgrid_top_left_y = int(np.ceil(top_left_y / grid_size_m))

        offset_x_px = (top_left_x - coordgrid_top_left_x * grid_size_m) / res
        offset_y_px = (coordgrid_top_left_y * grid_size_m - top_left_y) / res

        # Pad the image to ensure that the top right point lies on the grid
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(
            image,
            int(np.round(offset_y_px)),
            0,
            int(np.round(offset_x_px)),
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
        image = cv2.copyMakeBorder(
            image,
            0,
            int(padding_y),
            0,
            int(padding_x),
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        # Iterate over each tile
        total_iterations = num_tiles_x * num_tiles_y
        total_tiles += total_iterations
        with tqdm(total=total_iterations, desc="Processing") as pbar:
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates within Norway
                    grid_x = coordgrid_top_left_x + i
                    grid_y = coordgrid_top_left_y - j

                    # Only keep that tile if it's in the prediction mask
                    if prediction_mask[grid_y - min_y, grid_x - min_x]:

                        # Calculate the tile coordinates within the image
                        x = int(i * effective_tile_size)
                        y = int(j * effective_tile_size)

                        # Crop the tile from the image
                        image_tile = image[y : y + tile_size, x : x + tile_size]

                        if image_tile.sum() != 0:  # no need to write a black tile
                            # Save the image tile to the output directory
                            if project_name:
                                image_tile_filename = f"{project_name}_{image_file[-5:-4]}_{grid_x}_{grid_y}.tif"
                            else:
                                image_tile_filename = (
                                    f"{image_file[:-4]}_{grid_x}_{grid_y}.tif"
                                )
                            image_tile_path = os.path.join(
                                output_dir_images, image_tile_filename
                            )
                            cv2.imwrite(image_tile_path, image_tile)

                        else:
                            skipped_tiles += 1

                    else:  # no need to write a tile outside the prediction mask
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


def tile_labels(
    project_name,
    res=0.3,
    compression="i_lzw_25",
    tile_size=512,
    overlap_rate=0,
    output_dir_images=None,
    output_dir_labels=None,
    image_size=None,
    gdf_omrade=None,
):

    year = int(project_name.split("_")[-1])

    # Create output directories if they don't exist
    os.makedirs(output_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    image_tiles = [f for f in os.listdir(output_dir_images) if f.endswith(".tif")]

    # Initialize min and max coordinates
    min_grid_x = min_grid_y = float("inf")
    max_grid_x = max_grid_y = float("-inf")

    for image_tile in image_tiles:
        # Split the filename on underscore
        parts = image_tile.split(".")[0].split("_")

        # Extract grid_x and grid_y
        grid_x = int(parts[-2])
        grid_y = int(parts[-1])

        # Update min and max coordinates
        min_grid_x = min(min_grid_x, grid_x)
        min_grid_y = min(min_grid_y, grid_y)
        max_grid_x = max(max_grid_x, grid_x)
        max_grid_y = max(max_grid_y, grid_y)

    effective_tile_size = int(tile_size * (1 - overlap_rate))
    grid_size_m = res * effective_tile_size

    if gdf_omrade is None:
        path_label = (
            data_path / "raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
        )
        with open(path_label, "rb") as f:
            gdf_omrade = pickle.load(f)
        buildings_year = pd.read_csv(
            data_path / "raw/FKB_bygning/buildings.csv", index_col=0
        )
        gdf_omrade = gdf_omrade.merge(
            buildings_year, left_on="bygningsnummer", right_index=True, how="left"
        )

    # filter by year
    gdf_omrade_subset = gdf_omrade[gdf_omrade["Building Year"] <= year]

    bbox = (
        np.array([min_grid_x, min_grid_y - 1, max_grid_x + 1, max_grid_y]) * grid_size_m
    )

    label, _ = get_labels(gdf_omrade_subset, bbox, res, in_degree=False)
    label = (
        cv2.copyMakeBorder(
            label,
            0,
            int(
                (
                    np.ceil(label.shape[0] / effective_tile_size)
                    - label.shape[0] / effective_tile_size
                )
                * effective_tile_size
            ),
            0,
            int(
                (
                    np.ceil(label.shape[1] / effective_tile_size)
                    - label.shape[1] / effective_tile_size
                )
                * effective_tile_size
            ),
            cv2.BORDER_CONSTANT,
            value=[0, 0],
        )
        * 255
    )

    # Calculate the image size if not given
    total_iterations = len(image_tiles)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_tile in image_tiles:

            # Split the filename on underscore
            parts = image_tile.split(".")[0].split("_")

            # Extract grid_x and grid_y
            grid_x = int(parts[-2])
            grid_y = int(parts[-1])

            x = (grid_x - min_grid_x) * effective_tile_size
            y = (max_grid_y - grid_y) * effective_tile_size

            label_tile = label[y : y + tile_size, x : x + tile_size]

            # Save the label tile to the output directory
            label_tile_filename = f"{project_name}_{parts[-3]}_{grid_x}_{grid_y}.tif"
            label_tile_path = os.path.join(output_dir_labels, label_tile_filename)

            cv2.imwrite(label_tile_path, label_tile)

            pbar.update(1)

    return


# %%
def tile_generation(
    project_name,
    res,
    compression,
    prediction_mask=None,
    prediction_type="buildings",
    tile_size=512,
    overlap_rate=0,
    labels=False,
    gdf_omrade=None,
):

    input_dir_images = (
        data_path / f"raw/orthophoto/res_{res}/{project_name}/{compression}/"
    )
    output_dir_images = (
        data_path
        / f"ML_prediction/topredict/image/res_{res}/{project_name}/{compression}/"
    )

    print(
        f"Tiling images, project {project_name}, resolution {res}, "
        + f"compression {compression}"
    )

    tile_images_no_labels(
        input_dir_images,
        output_dir_images,
        tile_size=tile_size,
        overlap_rate=overlap_rate,
        project_name=project_name,
        res=res,
        prediction_mask=prediction_mask,
        prediction_type=prediction_type,
    )

    if labels:
        output_dir_labels = (
            data_path
            / f"ML_prediction/topredict/label/res_{res}/{project_name}/{compression}/"
        )
        print(
            f"Tiling labels, project {project_name}, resolution {res}, compression {compression}"
        )
        tile_labels(
            project_name,
            res=res,
            compression=compression,
            tile_size=tile_size,
            overlap_rate=overlap_rate,
            output_dir_images=output_dir_images,
            output_dir_labels=output_dir_labels,
            gdf_omrade=gdf_omrade,
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
    parser.add_argument(
        "--prediction_type", required=False, type=str, default="buildings"
    )
    parser.add_argument("-l", "--labels", required=False, type=bool, default=False)
    args = parser.parse_args()
    tile_generation(
        args.project_name,
        args.res,
        args.compression,
        prediction_type=args.prediction_type,
        labels=args.labels,
    )
