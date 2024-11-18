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
import argparse
from HOME.utils.project_coverage_area import project_coverage_area
import json
from datetime import datetime
import pickle
import pandas as pd

# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa
from HOME.get_data_path import get_data_path
from HOME.ML_training.preprocessing.get_label_data.get_labels import get_labels

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

        image_coords_grid = (
            int(np.floor(image_x_min / grid_size_m)),
            int(np.ceil(image_y_min / grid_size_m)),
            int(np.ceil(image_x_max / grid_size_m)),
            int(np.floor(image_y_max / grid_size_m)),
        )

    return image_coords_grid, (image_x_min, image_y_min, image_x_max, image_y_max)


# %%
def tile_images(
    project_name,
    tile_size,
    grid_size_m,
    input_dir_images,
    output_dir_images,
    tile_coverage,
    res_original,
):
    """
    Creates tiles in tilesize from images in input_dir_images and saves them in
    output_dir_images. The tiles are named according to their position relative in the
    grid that would be started at absolute 0,0 in CRS EPSG:25833 and extends towards
    north and east. The tiles are only created if the corresponding grid cell is in the
    prediction_mask.
    """
    image_files = [f for f in os.listdir(input_dir_images) if f.endswith(".tif")]
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
                    int(
                        np.round((tile_coords_m[0] - image_coords_m[0]) / res_original)
                    ),
                    int(
                        np.round((image_coords_m[3] - tile_coords_m[1]) / res_original)
                    ),
                    int(
                        np.round((tile_coords_m[2] - image_coords_m[0]) / res_original)
                    ),
                    int(
                        np.round((image_coords_m[3] - tile_coords_m[3]) / res_original)
                    ),
                )

                tile = image[
                    max(0, tile_coords_px[3]) : min(image.shape[0], tile_coords_px[1]),
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

                        if (
                            np.array(tile_pixels[(x_grid, y_grid)]).sum(axis=0) == 0
                        ).any():
                            tile_is_complete = False
                        else:
                            tile = np.array(tile_pixels[(x_grid, y_grid)]).sum(axis=0)
                            tile_pixels[(x_grid, y_grid)] = []
                            tile_is_complete = True
                    else:
                        tile_is_complete = True

                    if tile_is_complete:
                        tile_in_res = cv2.resize(
                            tile.astype(np.uint8),
                            None,
                            fx=tile_size / tile.shape[1],
                            fy=tile_size / tile.shape[0],
                            interpolation=cv2.INTER_AREA,
                        )
                        tile_filename = f"{project_name}_{x_grid}_{y_grid}.tif"
                        tile_path = os.path.join(output_dir_images, tile_filename)
                        cv2.imwrite(tile_path, tile_in_res)

    # Some tiles might still contain black pixels (edges)
    for x_grid, y_grid in tqdm(tile_coverage.index):
        if tile_pixels[(x_grid, y_grid)] != []:
            tile = np.array(tile_pixels[(x_grid, y_grid)]).sum(axis=0)
            tile_in_res = cv2.resize(
                tile.astype(np.uint8),
                None,
                fx=tile_size / tile.shape[1],
                fy=tile_size / tile.shape[0],
                interpolation=cv2.INTER_AREA,
            )
            tile_filename = f"{project_name}_{x_grid}_{y_grid}.tif"
            tile_path = os.path.join(output_dir_images, tile_filename)
            cv2.imwrite(tile_path, tile_in_res)

    return


# %% To validate the prediction with the recall, we need to tile the labels as well.


def tile_labels(
    project_name,
    res=0.3,
    tile_size=512,
    overlap_rate=0,
    output_dir_images=None,
    output_dir_labels=None,
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
    tile_size=512,
    res=0.3,
    overlap_rate=0,
    labels=False,
    gdf_omrade=None,
):

    # Add project metadata to the log
    with open(data_path / "metadata_log/tiled_projects.json", "r") as file:
        tiled_projects_log = json.load(file)
    highest_tile_key = int(max([int(key) for key in tiled_projects_log.keys()]))
    tile_key = highest_tile_key + 1

    # Create output directories if they don't exist
    output_dir_images = (
        data_path / f"ML_prediction/topredict/image/{project_name}/tiles_{tile_key}"
    )
    os.makedirs(output_dir_images, exist_ok=True)

    # Create archive directories if they don't exist
    input_dir_images = os.path.join(
        data_path, f"raw/orthophoto/originals/{project_name}/"
    )

    # Get list of all image files in the input directory
    metadata_files = [f for f in os.listdir(input_dir_images) if f.endswith(".sos")]

    with open(os.path.join(input_dir_images, metadata_files[0]), "r") as f:
        meta_text = f.read()
        res_original = float(
            meta_text[
                meta_text.find("...PIXEL-STØRR")
                + 15 : meta_text.find(" ", meta_text.find("...PIXEL-STØRR") + 15)
            ]
        )

        crs_id = meta_text[
            meta_text.find("...KOORDSYS")
            + 12 : meta_text.find("\n", meta_text.find("...KOORDSYS"))
        ]

        assert crs_id in ["22", "23"], "CRS not supported"
        crs = 25832 if crs_id == "22" else 25833

    tiled_projects_log[tile_key] = {
        "project_name": project_name,
        "tile_size": tile_size,
        "res": res,
        "overlap_rate": overlap_rate,
        "crs": crs,
        "tile_directory": str(output_dir_images),
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_labels": labels,
    }

    with open(data_path / "metadata_log/tiled_projects.json", "w") as file:
        json.dump(tiled_projects_log, file, indent=4)

    effective_tile_size = tile_size * (1 - overlap_rate)
    grid_size_m = res * effective_tile_size

    tile_coverage = project_coverage_area(
        project_name, res, tile_size, overlap_rate, crs
    )

    tile_images(
        project_name=project_name,
        tile_size=tile_size,
        grid_size_m=grid_size_m,
        input_dir_images=input_dir_images,
        output_dir_images=output_dir_images,
        tile_coverage=tile_coverage,
        res_original=res_original,
    )

    if labels:
        output_dir_labels = (
            data_path / f"ML_prediction/topredict/label/{project_name}/tiles_{tile_key}"
        )
        tile_labels(
            project_name=project_name,
            tile_size=tile_size,
            res=res,
            overlap_rate=overlap_rate,
            output_dir_images=output_dir_images,
            output_dir_labels=output_dir_labels,
            gdf_omrade=gdf_omrade,
        )
    return tile_key


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile raw orthophotos for prediction with ML model"
    )
    parser.add_argument("--project_name", required=True, type=str)
    parser.add_argument("--res", required=False, type=float, default=0.3)
    parser.add_argument("--overlap_rate", required=False, type=float, default=0)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument(
        "--prediction_type", required=False, type=str, default="buildings"
    )
    args = parser.parse_args()
    tile_generation(
        project_name=args.project_name,
        res=args.res,
        tile_size=args.tile_size,
        overlap_rate=args.overlap_rate,
    )
