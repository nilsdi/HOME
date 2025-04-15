# %%
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import MultiLineString
from shapely.geometry import shape
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)


def convert_multilinestring_to_mesh(
    multilinestring_gdf: gpd.GeoDataFrame, grid_size: tuple[int] = (1000, 1000)
):
    """
    Convert a GeoDataFrame containing MultiLineString objects into a mesh grid.

    Args:
        multilinestring_gdf: gpd.GeoDataFrame - the GeoDataFrame containing MultiLineString objects
        grid_size: tuple of ints - the size of the grid

    Returns:
        grid_x: np.array (shape: (grid_size, grid_size)) - the x coordinates of the grid
        grid_y: np.array (shape: (grid_size, grid_size)) - the y coordinates of the grid
        grid_z: np.array (shape: (grid_size, grid_size)) - the z coordinates of the grid
    """
    points = []
    values = []
    # print(len(multilinestring_gdf))
    for geom, value in zip(multilinestring_gdf.geometry, multilinestring_gdf["hoyde"]):
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                points.extend(line.coords)
                values.extend([value] * len(line.coords))
        else:
            raise ValueError("Not a MultiLineString")

    points = np.array(points)
    print(f"points shape: {points.shape}")
    values = np.array(values)

    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size[0]),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size[1]),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
    min_value = values.min()
    max_value = values.max()
    return grid_x, grid_y, grid_z, min_value, max_value


def get_shading(elevation: np.array, azimuth: float, altitude: float) -> np.array:
    """
    Get the shading of the elevation data.
    core code taken from: https://www.geophysique.be/2014/02/25/shaded-relief-map-in-python/

    Args:
        elevation: np.array (shape: (n, m)) - the elevation data
        azimuth: float - the azimuth angle in degrees
        altitude: float - the altitude angle in degrees

    Returns:
        shaded: np.array (shape: (n, m)) - the shaded elevation data
    """
    azimuth = np.deg2rad(azimuth)
    altitude = np.deg2rad(altitude)
    x, y = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))

    # -x here because of pixel orders in the SRTM tile
    aspect = np.arctan2(-x, y)

    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
        slope
    ) * np.cos((azimuth - np.pi / 2.0) - aspect)

    return shaded


def get_topography(bbox: tuple, root_dir: Path, grid_size: tuple[int] = (1000, 1000)):
    N50_path = (
        root_dir
        / "HOME/data/raw/maps/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
    )
    topographic_layer = "N50_Høyde_senterlinje"
    topographic_gdf = gpd.read_file(N50_path, layer=topographic_layer, bbox=bbox)

    grid_x, grid_y, grid_z, min_value, max_value = convert_multilinestring_to_mesh(
        topographic_gdf, grid_size=grid_size
    )

    return grid_x, grid_y, grid_z, min_value, max_value


# %%
if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).parents[4]
    print(root_dir)
    city = "trondheim"
    city_boundaries = shape(get_municipal_boundaries(city))
    city_boundaries = gpd.GeoDataFrame(geometry=[city_boundaries], crs="EPSG:4326")
    city_boundaries = city_boundaries.to_crs("EPSG:25833")
    city_bbox = city_boundaries.total_bounds
    print(city_bbox)
    # print(set(city_bbox.values.flatten()))
    # extend the bounds by 5km

    extension = 5000
    city_bbox_extended = (
        city_bbox[0] - extension,
        city_bbox[1] - extension,
        city_bbox[2] + extension,
        city_bbox[3] + extension,
    )

    grid_size = (1500, 1500)
    grid_x, grid_y, grid_z, min_value, max_value = get_topography(
        city_bbox_extended, root_dir, grid_size=grid_size
    )
    shaded = get_shading(grid_z, azimuth=45, altitude=315)
    # cut the shaded area to the original bounding box

    extension_pixels_x = grid_size[0] * (
        extension / (city_bbox_extended[2] - city_bbox_extended[0])
    )
    extension_pixels_y = grid_size[1] * (
        extension / (city_bbox_extended[3] - city_bbox_extended[1])
    )
    reduced_shaded = shaded[
        int(extension_pixels_x) : grid_size[0] - int(extension_pixels_x),
        int(extension_pixels_y) : grid_size[1] - int(extension_pixels_y),
    ]
    print(reduced_shaded.shape)
    extend = [city_bbox[0], city_bbox[2], city_bbox[1], city_bbox[3]]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        shaded,  # reduced_shaded,
        cmap="Greys",
        vmin=0,
        vmax=1,
        # extent=extend,
        origin="lower",
        alpha=1,
    )
    N50_path = (
        root_dir
        / "HOME/data/raw/maps/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
    )
    cover_layer = "N50_Arealdekke_omrade"
    cover_gdf = gpd.read_file(N50_path, layer=cover_layer, bbox=tuple(city_bbox))
    for objtype, group in cover_gdf.groupby("objtype"):
        if objtype in ["Havflate", "Innsjø", "InnsjøRegulert"]:
            group.plot(ax=ax, facecolor="white", edgecolor="white")
    # plt.axis("off")

# %%
