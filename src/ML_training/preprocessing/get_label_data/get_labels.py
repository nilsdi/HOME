import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import numpy as np
from pathlib import Path
from src.utils.bbox_to_meters import convert_bbox_to_meters  # noqa


def get_labels(fkb_omrade_gdf, bbox: list, pixel_size: float) -> tuple[np.ndarray]:
    """
    Function to get the labels of the buildings in the area of interest

    Arguments:
    fkb_omrade_gdf: GeoDataFrame containing ALL the building footprints for
    the area
    bbox: list, the bounding box of the area of interest - coordinates in the
              form [left, bottom, right, top] in WGS84 (EPSG:4326)
    pixel_size: float, the size of the pixel in meters

    Returns:
    data: np.ndarray, labels for buildings in the area (1 for building, 0 for
    no building)
    transform: np.ndarray, the transformation matrix for the GeoTIFF
    """
    target_crs = "EPSG:25833"
    # Define the bounding box and the resolution
    [left, bottom, right, top] = convert_bbox_to_meters(bbox)
    # Calculate the scaling factors
    # scaling factor in the y direction
    # Calculate the dimensions of the new GeoTIFF
    width = int((right - left) / pixel_size)
    height = int((top - bottom) / pixel_size)
    transform = from_bounds(left, bottom, right, top, width, height)

    # Create an empty array of the same size as the GeoTIFF
    data = np.zeros((height, width), dtype=rasterio.uint8)

    fkb_omrade_gdf = fkb_omrade_gdf.to_crs(target_crs)
    # filter the gdf for the selection:
    fkb_omrade_gdf_filtered = fkb_omrade_gdf.cx[left:right, bottom:top]

    geometries = fkb_omrade_gdf_filtered.geometry.to_list()
    mask = geometry_mask(
        geometries, transform=transform, out_shape=(height, width), invert=True
    )

    # Set the corresponding elements of the array to 1
    data[mask] = 1

    return data, transform


def save_labels(
    data: np.ndarray,
    file_name: str,
    transform: np.ndarray,
    data_scale: int = 255,
    save_folder: str = "data/temp/pretrain/labels",
) -> None:
    """
    Function to save the labels to a GeoTIFF file

    Arguments:
    data: np.ndarray, labels for buildings in the area (1 for building, 0 for
        no building)
    file_path: str, the path to save the file
    transform: np.ndarray, the transformation matrix for the GeoTIFF
    data_scale: int, the scale to use for the data (default is 255, but 1 is
        also reasonable)

    Returns:
    None
    """
    # Scale up the values in the data
    data_scaled = data * data_scale
    # Set up the metadata
    meta = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "count": 1,
        "width": data.shape[1],
        "height": data.shape[0],
        "crs": "EPSG:25833",  # always WGS84
        "transform": transform,
    }
    # get the path
    root_dir = Path(__file__).parents[4]
    file_path = root_dir / save_folder / f"{file_name}.tif"

    # Write the data to the file
    with rasterio.open(file_path, "w", **meta) as dst:
        dst.write(data_scaled, 1)
    return
