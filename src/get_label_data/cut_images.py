from rasterio import warp
from affine import Affine
import rasterio
from rasterio.windows import Window
import numpy as np

def get_image_from_geotiff(geotiff_path, bbox:list, pixel_size:float)->np.array:
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
        print(f'the size of the image is {src.width}x{src.height} pixels')
        # if the crs of the tiff is not the target crs, reproject the bbox
        # if the crs of the tiff is not the target crs, reproject the bbox
        # if the crs of the tiff is not the target crs, reproject the bbox
        if src.crs != target_crs:
            left, bottom = warp.transform(target_crs, src.crs, [left], [bottom])
            right, top = warp.transform(target_crs, src.crs, [right], [top])
            left, bottom = left[0], bottom[0]
            right, top = right[0], top[0]

        # Convert the bounding box to pixel coordinates
        left_col, top_row = src.index(left, top)
        right_col, bottom_row = src.index(right, bottom)
        print(left_col, top_row, right_col, bottom_row)

        # Make a window from the bounding box
        window = Window(top_row, left_col, bottom_row - top_row, right_col - left_col)
        print(f' the window is {window}')

        # Read a subset of the GeoTIFF data
        subset = src.read([1,2,3], window=window)

        # before overwriting the top, left etc in the src crs, lets make the transform of the subset.
        # Calculate the transform for the subset for the later reprojecting


        # Make a window from the bounding box
        #window = Window(left_col, top_row, right_col - left_col, bottom_row - top_row)

        # Calculate the transform for the window
        subset_transform = src.window_transform(window)

        # if the crs of the tiff is not the target crs, reproject the subset
        if src.crs != target_crs:
            # Define the bounding box and the resolution for the target crs
            left, bottom, right, top = bbox
            # Calculate the scaling factors
            scaling_factor_y = 1 / (pixel_size / 111320)  # scaling factor in the y direction
            scaling_factor_x = scaling_factor_y * np.cos(np.radians((bottom + top) / 2))  # scaling factor in the x direction

            # Calculate the dimensions and the transform of the new GeoTIFF
            width = int((right - left) * scaling_factor_x)
            height = int((top - bottom) * scaling_factor_y)

            # Create an empty array for the reprojected data
            reprojected_subset = np.empty((subset.shape[0], height, width))

            # Calculate the new transform
            new_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
            print(f' the new_transform is {new_transform}')
            print(f' the subset_transform is {subset_transform}')
            print(f' the src transform is {src.transform}')
            # Calculate the transform for the subset
            # Calculate the transform for the window
            #subset_transform = src.window_transform(window)
            print(f' the subset_transform is {subset_transform}')
            # Reproject the subset
            warp.reproject(
                source=subset,
                destination=reprojected_subset,
                src_transform=subset_transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=target_crs,
                resampling=warp.Resampling.nearest)

            subset = reprojected_subset

        # Rearrange the dimensions of the array
        subset = np.transpose(subset, (1, 2, 0))
    return subset

def save_cut_geotiff(data, )