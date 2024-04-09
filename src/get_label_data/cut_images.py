import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path
import sys
parent_dir = Path(__file__).parent
sys.path.append(str(parent_dir))
from utils.bbox_to_meters import convert_bbox_to_meters  # noqa


def cut_geotiff(geotiff_path, bbox: list, pixel_size: float) -> np.array:
    """
    Function that reads a geotiff and returns a subset of the image

    Arguments:
    geotiff_path : str : path to the geotiff file
    bbox : list : [left, bottom, right, top] coordinates of the bounding box
    resolution : int : resolution specified by size of edges of pixels in m

    Returns:
    np.array : subset of the image
    """
    target_crs = 'EPSG:25833'
    # Define the bounding box and the resolution
    [left, bottom, right, top] = convert_bbox_to_meters(bbox)

    width = int((right - left) / pixel_size)
    height = int((top - bottom) / pixel_size)

    # Read the GeoTIFF file
    with rasterio.open(geotiff_path) as src:

        # if the crs of the tiff is not the target crs, warn us
        assert src.crs == target_crs, \
            f"The crs of the geotiff is not ETRS89, but {src.crs}"
        # Convert the bounding box to pixel coordinates
        left_col, top_row = src.index(left, top)
        right_col, bottom_row = src.index(right, bottom)
        # print(left_col, top_row, right_col, bottom_row)

        # Make a window from the bounding box
        window = Window(top_row, left_col, width, height)

        # Calculate the transform for the subset
        subset_transform = src.window_transform(window)

        # Read a subset of the GeoTIFF data
        subset = src.read([1, 2, 3], window=window)

        # Rearrange the dimensions of the array
        # subset = np.transpose(subset, (1, 2, 0))
    return subset, subset_transform


def save_cut_geotiff(data: np.ndarray, file_name: str,
                     transform: np.ndarray,
                     save_folder: str = "data/temp/pretrain/images/") -> None:
    """
    Function to save the subset of the geotiff to a new file

    Arguments:
    data : np.ndarray : subset of the geotiff
    file_name : str : the path to save the file

    Returns:
    None
    """
    # Set up the metadata
    meta = {
        'driver': 'GTiff',
        'dtype': rasterio.uint8,
        'count': 3,
        'width': data.shape[2],
        'height': data.shape[1],
        'crs': 'EPSG:25833',
        'transform': transform
        # 'transform': transform # lets pray it doens't need a transform
    }
    root_dir = Path(__file__).parents[2]
    file_path = root_dir / save_folder / f"{file_name}.tif"
    # Write the data to a new GeoTIFF file
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data)

    return
