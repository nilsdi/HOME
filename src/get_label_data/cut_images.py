import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path


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
    target_crs = 'EPSG:4326'  # WGS84
    # Define the bounding box and the resolution
    [left, bottom, right, top] = bbox

    # Read the GeoTIFF file
    with rasterio.open(geotiff_path) as src:

        # if the crs of the tiff is not the target crs, warn us
        assert src.crs == target_crs, \
            f"The crs of the geotiff is not WGS84, but {src.crs}"
        # Convert the bounding box to pixel coordinates
        left_col, top_row = src.index(left, top)
        right_col, bottom_row = src.index(right, bottom)
        print(left_col, top_row, right_col, bottom_row)

        # Make a window from the bounding box
        window = Window(top_row, left_col, bottom_row -
                        top_row, right_col - left_col)

        # Read a subset of the GeoTIFF data
        subset = src.read([1, 2, 3], window=window)

        # Rearrange the dimensions of the array
        subset = np.transpose(subset, (1, 2, 0))
    return subset


def save_cut_geotiff(data: np.ndarray, file_name: str) -> None:
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
        'width': data.shape[1],
        'height': data.shape[0],
        'crs': 'EPSG:4326',  # always WGS84
        # 'transform': transform # lets pray it doens't need a transform
    }
    root_dir = Path(file_name).parents[2]
    file_path = root_dir + f"/data/temp/pretrain/images/{file_name}.tif"
    # Write the data to a new GeoTIFF file
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data)

    return
