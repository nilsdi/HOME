import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import numpy as np
import os


def get_labels(fkb_omrade_gdf, bbox : list, pixel_size : float) -> tuple[np.ndarray]:
    '''
    Function to get the labels of the buildings in the area of interest

    Arguments:
    fkb_omrade_gdf: GeoDataFrame containing ALL the building footprints for the area
    bbox: list, the bounding box of the area of interest - coordinates in the 
              form [left, bottom, right, top] in WGS84 (EPSG:4326)
    pixel_size: float, the size of the pixel in meters

    Returns:
    data: np.ndarray, labels for buildings in the area (1 for building, 0 for no building)
    transform: np.ndarray, the transformation matrix for the GeoTIFF
    '''
    target_crs = 'EPSG:4326'  # WGS84
    # Define the bounding box and the resolution
    [left, bottom, right, top] = bbox
    # Calculate the scaling factors
    scaling_factor_y = 1 / (pixel_size / 111320)  # scaling factor in the y direction
    scaling_factor_x = scaling_factor_y * np.cos(np.radians((bottom + top) / 2))  # scaling factor in the x direction

    # Calculate the dimensions and the transform of the new GeoTIFF
    width = int((right - left) * scaling_factor_x)
    height = int((top - bottom) * scaling_factor_y)
    transform = from_bounds(left, bottom, right, top, width, height)

    # Create an empty array of the same size as the GeoTIFF
    data = np.zeros((height, width), dtype=rasterio.uint8)

    fkb_omrade_gdf = fkb_omrade_gdf.to_crs(target_crs)
    # filter the gdf for the selection:
    fkb_omrade_gdf_filtered = fkb_omrade_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # For each building
    for _, building in fkb_omrade_gdf_filtered.iterrows():
        # Calculate the pixel coordinates of the building
        building_geom = building.geometry  # assuming building is a GeoDataFrame
        mask = geometry_mask([building_geom], transform=transform, out_shape=(height, width), invert=True)

        # Set the corresponding elements of the array to 1
        data[mask] = 1
    
    return data, transform

def save_labels(data:np.ndarray, file_name:str, transform:np.ndarray, data_scale:int = 255) -> None:
    '''
    Function to save the labels to a GeoTIFF file

    Arguments:
    data: np.ndarray, labels for buildings in the area (1 for building, 0 for no building)
    file_path: str, the path to save the file
    transform: np.ndarray, the transformation matrix for the GeoTIFF
    data_scale: int, the scale to use for the data (default is 255, but 1 is also reasonable)

    Returns:
    None
    '''
    # Scale up the values in the data
    data_scaled = data * data_scale
    # Set up the metadata
    meta = {
        'driver': 'GTiff',
        'dtype': rasterio.uint8,
        'count': 1,
        'width': data.shape[1],
        'height': data.shape[0],
        'crs': 'EPSG:4326' , # always WGS84
        'transform': transform
    }
    # get the path
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    file_path = parent_dir + f"/data/temp/pretrain/labels/{file_name}.tif"

    # Write the data to the file
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data_scaled, 1)
    return