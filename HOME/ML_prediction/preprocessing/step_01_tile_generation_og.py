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
import pandas as pd
import scipy.sparse
import pickle
from shapely.geometry import box
from HOME.ML_training.preprocessing.get_label_data.get_labels import get_labels
from HOME.utils.project_coverage_area import project_coverage_area
import geopandas as gpd


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


def coords_from_sos(sos_file, grid_size_m):
    with open(sos_file, "r") as f:
        meta_text = f.read()
        pos_min_no = meta_text.find("...MIN-NØ")
        space_pos = meta_text.find(" ", pos_min_no + 10)
        end_pos = meta_text.find("\n", space_pos)
        image_y_min = float(meta_text[pos_min_no + 10 : space_pos])
        image_x_min = float(meta_text[space_pos + 1 : end_pos])

        pos_max_no = meta_text.find("...MAX-NØ")
        space_pos = meta_text.find(" ", pos_max_no + 10)
        end_pos = meta_text.find("\n", space_pos)
        image_y_max = float(meta_text[pos_max_no + 10 : space_pos])
        image_x_max = float(meta_text[space_pos + 1 : end_pos])

        crs_id = meta_text[
            meta_text.find("...KOORDSYS")
            + 12 : meta_text.find("\n", meta_text.find("...KOORDSYS"))
        ]

        assert crs_id in ["22", "23"], "CRS not supported"

        crs = 25832 if crs_id == "22" else 25833
        (
            image_x_min,
            image_y_min,
            image_x_max,
            image_y_max,
        ) = (
            gpd.GeoSeries(
                box(image_x_min, image_y_min, image_x_max, image_y_max),
                crs=crs,
            )
            .to_crs(25833)[0]
            .bounds
        )

        image_coords_grid = (
            int(np.floor(image_x_min / grid_size_m)),
            int(np.ceil(image_y_min / grid_size_m)),
            int(np.ceil(image_x_max / grid_size_m)),
            int(np.floor(image_y_max / grid_size_m)),
        )

    return image_coords_grid, (image_x_min, image_y_min, image_x_max, image_y_max)


# %%
def tile_images_no_labels(
    project_name,
    tile_size,
    res,
    overlap_rate=0,
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
    output_dir_images = (
        data_path / f"ML_prediction/topredict/image/res_{res}/{project_name}/"
    )
    os.makedirs(output_dir_images, exist_ok=True)

    # Create archive directories if they don't exist
    input_dir_images = os.path.join(
        data_path, f"raw/orthophoto/originals/{project_name}/"
    )

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir_images) if f.endswith(".tif")]
    metadata_files = [f for f in os.listdir(input_dir_images) if f.endswith(".sos")]

    with open(os.path.join(input_dir_images, metadata_files[0]), "r") as f:
        meta_text = f.read()
        res_original = float(
            meta_text[
                meta_text.find("..PIXELSTØRRELSE")
                + 17 : meta_text.find("\n", meta_text.find("..PIXELSTØRRELSE"))
            ]
        )

    effective_tile_size = tile_size * (1 - overlap_rate)
    grid_size_m = res * effective_tile_size

    tile_coverage = project_coverage_area(project_name, res, tile_size, overlap_rate)

    tile_pixels = {(x, y): [] for (x, y) in tile_coverage.index}

    for image_file in tqdm(image_files):
        metadata_path = os.path.join(input_dir_images, image_file[:-4] + ".sos")
        image_coords_grid, image_coords_m = coords_from_sos(metadata_path, grid_size_m)

        tiles_in_image = tile_coverage.loc[
            (
                slice(image_coords_grid[0], image_coords_grid[2] - 1),
                slice(image_coords_grid[1], image_coords_grid[3] - 1),
            ),
            :,
        ]
        if len(tiles_in_image) > 0:
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)

            for x_grid, y_grid in tiles_in_image.index:
                tile_coords_m = (
                    x_grid * grid_size_m,
                    (y_grid - 1) * grid_size_m,
                    (x_grid + 1) * grid_size_m,
                    y_grid * grid_size_m,
                )

                tile_coords_px = (
                    int((tile_coords_m[0] - image_coords_m[0]) / res_original),
                    int((image_coords_m[3] - tile_coords_m[1]) / res_original),
                    int((tile_coords_m[2] - image_coords_m[0]) / res_original),
                    int((image_coords_m[3] - tile_coords_m[3]) / res_original),
                )

                tile = image[
                    max(0, tile_coords_px[1]) : min(image.shape[0], tile_coords_px[3]),
                    max(0, tile_coords_px[0]) : min(image.shape[1], tile_coords_px[2]),
                ]

                if tile.sum() > 0:
                    padding = (
                        max(0, -tile_coords_px[0]),
                        max(0, tile_coords_px[1] - image.shape[0]),
                        max(0, tile_coords_px[2] - image.shape[1]),
                        max(0, -tile_coords_px[3]),
                    )

                    if padding != (0, 0, 0, 0):
                        tile = cv2.copyMakeBorder(
                            tile,
                            padding[3],
                            padding[1],
                            padding[0],
                            padding[2],
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0],
                        )

                    tile_pixels[(x_grid, y_grid)].append(tile)

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

    print(f"Skipped {skipped_tiles} out of {total_tiles} tiles with no information")

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
