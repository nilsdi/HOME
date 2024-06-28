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
from HOME.ML_training.preprocessing.get_label_data.get_labels import get_labels

# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa

root_dir = str(Path(__file__).parents[3])


# %%
def tile_images_no_labels(
    input_dir_images,
    output_dir_images,
    tile_size,
    res,
    overlap_rate=0.00,
    move_to_archive=False,
    project_name=None,
    prediction_mask=None,
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
        prediction_mask = pd.read_csv(
            root_dir + f"/data/ML_prediction/prediction_mask/prediction_mask_{res}.csv",
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
                    if prediction_mask.loc[grid_y, grid_x]:

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
    res=0.2,
    compression="i_lzw_25",
    tile_size=512,
    overlap_rate=0.00,
    image_size=None,
):

    year = int(project_name.split("_")[-1])
    year_dt_utc = pd.to_datetime(year, format="%Y").tz_localize("UTC")

    ### Makes label for a given project
    dir_images = (
        root_dir
        + f"/data/ML_prediction/topredict/image/res_{res}/{project_name}/{compression}/"
    )

    output_dir_labels = (
        root_dir
        + f"/data/ML_prediction/topredict/label/res_{res}/{project_name}/{compression}/"
    )

    # Create output directories if they don't exist
    os.makedirs(output_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    image_tiles = [f for f in os.listdir(dir_images) if f.endswith(".tif")]

    # Initialize min and max coordinates
    min_coord_x = min_coord_y = float("inf")
    max_coord_x = max_coord_y = float("-inf")

    for image_tile in image_tiles:
        # Split the filename on underscore
        parts = image_tile.split(".")[0].split("_")

        # Extract coord_x and coord_y
        coord_x = int(parts[-2])
        coord_y = int(parts[-1])

        # Update min and max coordinates
        min_coord_x = min(min_coord_x, coord_x)
        min_coord_y = min(min_coord_y, coord_y)
        max_coord_x = max(max_coord_x, coord_x)
        max_coord_y = max(max_coord_y, coord_y)

    effective_tile_size = int(tile_size * (1 - overlap_rate))
    grid_size_m = res * effective_tile_size

    path_label = (
        root_dir
        + "/data/raw/FKB_bygning"
        + "/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb"
    )
    gdf_omrade = gpd.read_file(path_label, driver="FileGDB", layer="fkb_bygning_omrade")
    gdf_omrade_subset = gdf_omrade[gdf_omrade["datafangstdato"] <= year_dt_utc]

    bbox = (
        np.array([min_coord_x, min_coord_y, max_coord_x + 1, max_coord_y + 1])
        * grid_size_m
    )

    label, _ = get_labels(gdf_omrade_subset, bbox, res, in_degree=False)

    # Calculate the image size if not given
    total_iterations = len(image_tiles)

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_tile in image_tiles:

            # Split the filename on underscore
            parts = image_tile.split(".")[0].split("_")

            # Extract coord_x and coord_y
            coord_x = int(parts[-2])
            coord_y = int(parts[-1])

            x = (coord_x - min_coord_x) * effective_tile_size
            y = (coord_y - min_coord_y) * effective_tile_size

            label_tile = label[y : y + effective_tile_size, x : x + effective_tile_size]

            # Save the label tile to the output directory
            label_tile_filename = f"{project_name}_{parts[2]}_{coord_x}_{coord_y}.tif"
            label_tile_path = os.path.join(output_dir_labels, label_tile_filename)

            cv2.imwrite(label_tile_path, label_tile)

            pbar.update(1)

    return


# %%
def tile_generation(project_name, res, compression, prediction_mask=None):

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
    parser.add_argument("--res", required=False, type=float, default=0.2)
    parser.add_argument("--compression", required=False, type=str, default="i_lzw_25")
    args = parser.parse_args()
    tile_generation(args.project_name, args.res, args.compression)


# # %% Some tests

# padding_left = int(np.round(offset_x_px))
# padding_right = int(padding_x)
# padding_top = int(np.round(offset_y_px))
# padding_bottom = int(offset_y)

# src_band = dataset.GetRasterBand(1)

# # %% Calculate new dimensions with padding
# new_xsize = dataset.RasterXSize + padding_left + padding_right
# new_ysize = dataset.RasterYSize + padding_top + padding_bottom

# # %% Create new dataset
# driver = gdal.GetDriverByName("GTiff")
# dst_ds = driver.Create(
#     os.path.join(input_dir_images, f"padded_{image_file}"),
#     new_xsize,
#     new_ysize,
#     1,
#     src_band.DataType,
# )
# dst_band = dst_ds.GetRasterBand(1)

# # %% Adjust geotransform for the new dataset to account for padding
# gt = list(dataset.GetGeoTransform())
# gt[0] -= padding_left * gt[1]  # Adjust origin X
# gt[3] -= (
#     padding_top * gt[5]
# )  # Adjust origin Y (note: gt[5] is negative for north-up images)
# dst_ds.SetGeoTransform(gt)

# # Set projection
# dst_ds.SetProjection(dataset.GetProjection())

# # Initialize the padded area with no data value or a specific value
# nodata_value = src_band.GetNoDataValue()
# if nodata_value is not None:
#     dst_band.SetNoDataValue(nodata_value)
#     dst_band.Fill(nodata_value)
# else:
#     dst_band.Fill(0)  # Or any other value you wish to use for padding

# # Read data from the original dataset
# src_data = src_band.ReadAsArray()

# # %% Write data to the new dataset with padding offsets
# dst_band.WriteArray(src_data, padding_left, padding_top)

# # %% Close datasets
# dataset = None
# dst_ds = None
# # %%
