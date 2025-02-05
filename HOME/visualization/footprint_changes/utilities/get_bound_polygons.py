# %%
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Polygon
from pathlib import Path


def get_bound_polygons(polygon_gdfs: list[str or Path], bshape: Polygon):
    """
    Get all polygons from the given gdfs that lie within the bshape

    Args:
        polygon_gdfs: list of str or Path, the paths to the polygon files
        bshape: Polygon, the bounding shape for the polygons

    Returns:
        bound_polygons: GeoDataFrame, the polygons that lie within the bshape
    """
    all_polygons_gdf = gpd.GeoDataFrame()
    for polygon_gdf in polygon_gdfs:
        gdf = gpd.read_file(polygon_gdf).to_crs(25832)
        all_polygons_gdf = gpd.GeoDataFrame(
            pd.concat([all_polygons_gdf, gdf], ignore_index=True)
        )
        # print(f" we have a gdf with {len(gdf)} polygons")
        # print(f"so our all_polygons_gdf has {len(all_polygons_gdf)} polygons")
    bound_polygons = all_polygons_gdf[all_polygons_gdf.within(bshape)]
    bound_polygons_unique = bound_polygons.drop_duplicates(subset="geometry")
    return bound_polygons_unique


def get_bound_tifs(tifs: list[str or Path], bshape: Polygon, BW: bool = False):
    """
    Get the image from the given tifs that lie within the bshape

    Args:
        tifs: list of str or Path, the paths to the tif files
        bshape: Polygon, the bounding shape for the image
        BW: bool, whether the image is black and white or not

    Returns:
        img: np.array, the image that lies within the bshape

    Raises:
        NotImplementedError: if more than one tif is given
    """
    if len(tifs) > 1:
        raise NotImplementedError("Currently only one tif is supported")
    elif len(tifs) == 1:
        with rasterio.open(tifs[0]) as src:
            if BW:
                img = src.read(1)
            else:
                img = np.dstack([src.read(i) for i in (1, 2, 3)])
    return img


if __name__ == "__main__":
    from HOME.visualization.footprint_changes.utilities.get_bounding_shape import (
        get_bounding_shape,
    )
    from matplotlib import pyplot as plt

    # Test the function
    root_dir = Path(__file__).parents[4]
    data_path = root_dir / "data"

    selected_large_tiles = [
        "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/polygons/trondheim_kommune_2022/tiles_10003/prediction_20003/assembly_30069/polygons_40067/polygons_trondheim_kommune_2022_resolution0.3_3645_45819.fgb"
    ]

    x_tile = 3653  # 3696
    y_tile = 45811  # 45796
    bshape = get_bounding_shape(x_tile, y_tile)

    bound_polygons = get_bound_polygons(selected_large_tiles, bshape)
    print(bound_polygons)
    bound_polygons.plot()

    selected_small_tiles = [
        "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/topredict/image/trondheim_kommune_2022/tiles_10003/trondheim_kommune_2022_3653_45811.tif"
    ]
    bound_tifs = get_bound_tifs(selected_small_tiles, bshape)
    plt.imshow(bound_tifs)
# %%
