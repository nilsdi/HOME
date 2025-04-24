# %%
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import MultiLineString
from shapely.geometry import shape, Polygon
from pathlib import Path
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)


def get_cover_gdf(bbox: tuple, root_dir: Path, crs: str = "EPSG:25833"):

    # convert the bbox to a GeoDataFrame
    bbox_polygon = Polygon(
        [
            (bbox[0], bbox[1]),  # Bottom-left
            (bbox[0], bbox[3]),  # Top-left
            (bbox[2], bbox[3]),  # Top-right
            (bbox[2], bbox[1]),  # Bottom-right
            (bbox[0], bbox[1]),  # Close the polygon
        ]
    )
    bbox = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=crs)
    # convert the bbox to the crs of the N50 data
    bbox = bbox.to_crs("EPSG:25833")
    # print(f"bbox bounds: {bbox.total_bounds}")
    N50_path = (
        root_dir / "data/raw/maps/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
    )
    cover_layer = "N50_Arealdekke_omrade"
    cover_gdf = gpd.read_file(N50_path, layer=cover_layer, bbox=bbox)
    cover_gdf = cover_gdf.to_crs(crs)
    return cover_gdf


def convert_multilinestring_to_mesh(
    multilinestring_gdf: gpd.GeoDataFrame, grid_size: tuple[int] = (1000, 1000)
):
    """
    Convert a GeoDataFrame containing MultiLineString objects into a mesh grid (topography).

    Args:
        multilinestring_gdf: gpd.GeoDataFrame - the GeoDataFrame containing MultiLineString objects
        grid_size: tuple of ints - number of points in the x and y directions

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
    # print(f"points shape: {points.shape}")
    # get the bbox of the points
    min_x, min_y = points[:, 0].min(), points[:, 1].min()
    max_x, max_y = points[:, 0].max(), points[:, 1].max()
    bbox = (
        min_x,
        min_y,
        max_x,
        max_y,
    )
    # print the bbox
    # print(f"bbox: {bbox}")
    values = np.array(values)

    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size[0]),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size[1]),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
    min_value = values.min()
    max_value = values.max()
    return grid_x, grid_y, grid_z, min_value, max_value, bbox


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


def get_topography(
    bbox: tuple,
    root_dir: Path,
    grid_size: tuple[int] = (1000, 1000),
    crs: str = "EPSG:25833",
):
    N50_path = (
        root_dir / "data/raw/maps/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
    )
    topographic_layer = "N50_Høyde_senterlinje"
    # convert the bbox to a GeoDataFrame
    bbox_polygon = Polygon(
        [
            (bbox[0], bbox[1]),  # Bottom-left
            (bbox[0], bbox[3]),  # Top-left
            (bbox[2], bbox[3]),  # Top-right
            (bbox[2], bbox[1]),  # Bottom-right
            (bbox[0], bbox[1]),  # Close the polygon
        ]
    )
    # print(
    #     f"converting the bbox to a GeoDataFrame, (shape: {bbox_polygon}) specifying the crs: {crs}"
    # )
    bbox = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=crs)
    # convert the bbox to the crs of the N50 data
    bbox = bbox.to_crs("EPSG:25833")
    # print(f"bbox for topo readin: {bbox}")
    # print(f"bbox bounds: {bbox.total_bounds}")
    topographic_gdf = gpd.read_file(N50_path, layer=topographic_layer, bbox=bbox)
    topographic_gdf = topographic_gdf.to_crs(crs)
    # print(f"length of topographic_gdf: {len(topographic_gdf)}")

    grid_x, grid_y, grid_z, min_value, max_value, bbox = (
        convert_multilinestring_to_mesh(topographic_gdf, grid_size=grid_size)
    )
    return grid_x, grid_y, grid_z, min_value, max_value, bbox


def plot_topography_around_city(
    ax: plt.Axes,
    city_bbox: tuple,
    root_dir: Path,
    crs: str = "EPSG:25833",
    buffer: float = 5000,
    grid_size: tuple = (1000, 1000),
    topo_cmap: str = "terrain",
    topo_alpha: float = 0.1,
    shaded_cmap: str = "Greys",
    shaded_cmap_alpha: float = 1,
) -> plt.Axes:
    """
    Plot the topography around a city.

    Args:
        ...

    Returns:
        ax: plt.Axes - the axes with the topography plotted
    """
    # extend the bounds by 5km
    city_bbox_extended = (
        city_bbox[0] - buffer,
        city_bbox[1] - buffer,
        city_bbox[2] + buffer,
        city_bbox[3] + buffer,
    )
    grid_x, grid_y, grid_z, min_value, max_value, topo_bbox = get_topography(
        city_bbox_extended, root_dir, grid_size=grid_size, crs=crs
    )
    shaded = get_shading(grid_z, azimuth=45, altitude=315)
    topo_extent = [
        topo_bbox[0],
        topo_bbox[2],
        topo_bbox[1],
        topo_bbox[3],
    ]
    ax.imshow(
        shaded,
        cmap=shaded_cmap,
        vmin=-0.4,
        vmax=0.8,
        extent=topo_extent,
        origin="lower",
        alpha=shaded_cmap_alpha,
    )
    ax.imshow(
        grid_z,
        cmap=topo_cmap,
        vmin=-400,
        vmax=1000,
        alpha=topo_alpha,
        extent=topo_extent,
        origin="lower",
    )
    # impose white color on the water bodies
    cover_gdf = get_cover_gdf(city_bbox_extended, root_dir, crs=crs)
    for objtype, group in cover_gdf.groupby("objtype"):
        if objtype in ["Havflate", "Innsjø", "InnsjøRegulert", "FerskvannTørrfall"]:
            group.plot(ax=ax, facecolor="white", edgecolor="white", alpha=1)

    return ax


# %%
if __name__ == "__main__":
    root_dir = Path(__file__).parents[3]
    print(root_dir)
    city = "trondheim"
    city_boundaries = shape(get_municipal_boundaries(city))
    city_boundaries = gpd.GeoDataFrame(geometry=[city_boundaries], crs="EPSG:4326")
    final_crs = "EPSG:4326"  # "EPSG:25833"
    city_boundaries = city_boundaries.to_crs(final_crs)
    city_bbox = city_boundaries.total_bounds
    print(city_bbox)
    # testi
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plot_topography_around_city(
        ax,
        root_dir=root_dir,
        city_bbox=city_bbox,
        buffer=0.05,  # 5000,
        grid_size=(1000, 1000),
        crs=final_crs,
    )
    ax.set_title("Topography around Trondheim")
    ax.set_xlim(city_bbox[0], city_bbox[2])
    ax.set_ylim(city_bbox[1], city_bbox[3])
    # plt.axis("off")
# %%
