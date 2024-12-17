# %%
import geopandas as gpd
import rasterio
import numpy as np
import warnings
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.ops import unary_union

from HOME.visualization.footprint_changes.utils import bshape_from_tile_coords

tile_size = 512
res = 0.3
overlap = 0
grid_size = tile_size * res * (1 - overlap)


def combine_geometries(
    geometry_fgbs: list[str], polygon_directory: str, bshape: Polygon = None
):
    """
    Combine the geometries of the given geometry_gdbs that are within the bshape
    """
    combined_geometries = []
    for geometry_fgb in geometry_fgbs:
        gdf = gpd.read_file(f"{polygon_directory}/{geometry_fgb}").to_crs(25832)
        if bshape:
            gdf = gdf[gdf.within(bshape)]
        # gdf = gdf.loc[gdf.area.sort_values(ascending=False)[1:].index]
        # remove the biggest one if it is as big as a tile
        gdf = gdf.loc[gdf.area != grid_size**2]  # TODO: internalize grid_size
        # if there are shapes left in the gdf, we append it to the combined geometries
        if not gdf.empty:
            combined_geometries.append(gdf)
    # Concatenate all GeoDataFrames
    if len(combined_geometries) > 1:
        combined_gdf = gpd.GeoDataFrame(
            pd.concat(combined_geometries, ignore_index=True)
        )
    elif len(combined_geometries) == 1:
        combined_gdf = combined_geometries[0]
    else:
        combined_gdf = gpd.GeoDataFrame()
    # Remove duplicate geometries
    combined_gdf = combined_gdf.drop_duplicates(subset="geometry")

    return combined_gdf


def reassemble_and_cut_small_tiles(
    small_tiles: list[str], small_tile_directory: str, bshape: Polygon, BW: bool = False
):
    """
    Reassemble the small tiles that are within the bshape and cut them to the bshape.
    Should return a single geotiff.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=rasterio.errors.NotGeoreferencedWarning
        )

        if len(small_tiles) >= 1:
            # open the last tile
            with rasterio.open(f"{small_tile_directory}/{small_tiles[-1]}") as src:
                if BW:
                    return src.read(1)
                else:
                    return np.dstack([src.read(i) for i in (1, 2, 3)])
    return None


if __name__ == "__main__":
    x_tile = 3754
    y_tile = 45755
    bshape = bshape_from_tile_coords(x_tile, y_tile)
    selected_large_tiles = [
        "polygons_trondheim_kommune_2022_resolution0.3_3753_45756.fgb"
    ]
    dir_40003 = "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/polygons/trondheim_kommune_2022/tiles_10003/prediction_20003/assembly_30003/polygons_40003"
    combined_geometries = combine_geometries(selected_large_tiles, dir_40003, bshape)
    print(combined_geometries)
    print(combined_geometries.ID)

    selected_small_tiles = ["trondheim_kommune_2022_3754_45755.tif"]
    dir_10003 = "/scratch/mueller_andco/orthophoto/HOME/data/ML_prediction/topredict/image/trondheim_kommune_2022/tiles_10003"
    img = reassemble_and_cut_small_tiles(selected_small_tiles, dir_10003, bshape)
    plt.imshow(img)

# %%
