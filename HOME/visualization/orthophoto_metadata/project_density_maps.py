# %% imports and definition of class
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import json


from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.plot import show

from shapely.geometry import shape

import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from datetime import datetime


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pathlib import Path


class ProjectDensityGrid:
    """
    Handles the project density grid for an area.
    Key methods:
    - plot_density_grid: plots the density grid
    - save_density_grid: saves the density grid
    """

    def __init__(
        self,
        metadata_all_projects: list,
        region_shape_file: str,
        resolution: float = 500,  # resolution in meters per pixel
        region: str = "Norway",
    ) -> None:
        """
        Args:
        metadata_all_projects (list): metadata for all projects, needs to work with ___
        region_shape (str): path to a geopanda readable shape of the region
        """
        self.metadata_all_projects = metadata_all_projects
        gdf = gpd.read_file(region_shape_file)
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.is_valid]
        # print(f'the crs of the shape file is {gdf.crs}')
        gdf.geometry = gdf.geometry.simplify(tolerance=0.001, preserve_topology=True)
        gdf = gdf.to_crs("EPSG:32633")
        self.region_shape_utm = gdf
        self.region_shape_utm = gpd.read_file(region_shape_file).to_crs("EPSG:32633")

        self.resolution = resolution
        self.region = region
        (
            self.region_grid,
            self.bounds,
            self.transform,
            self.raster_width,
            self.raster_height,
        ) = self.create_region_grid()

        self.density_grid = self.get_density_grid()
        self.oldest_grid = self.get_oldest_project_grid()
        # self._display_raster(density_grid, self.transform, cmap='gist_heat_r',
        #                                plot_boundaries=True, plot_colorbar=True,
        #                                color_bar_label='Project coverage - # of projects')

    def create_region_grid(self):
        """
        Creates a grid of the region that we can use to fill it up.
        """
        bounds, transform = self._get_region_transform()
        width = int(np.ceil((bounds[2] - bounds[0]) / self.resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.resolution))

        # creaty empty grid
        raster = np.zeros((height, width), dtype=np.uint16)
        return raster, bounds, transform, width, height

    def _get_region_transform(self):
        """
        transform to match other geodata onto the region grid
        """
        oj_bounds = self.region_shape_utm.total_bounds
        # print(oj_bounds)
        buffer = 10**5  # 100 km buffer
        bounds = [
            oj_bounds[0] - buffer,
            oj_bounds[1] - buffer,
            oj_bounds[2] + buffer,
            oj_bounds[3] + buffer,
        ]
        # print(bounds)
        # calculate size in pixels for the given dimensions
        width = int(np.ceil((bounds[2] - bounds[0]) / self.resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.resolution))
        transform = from_origin(bounds[0], bounds[3], self.resolution, self.resolution)
        return bounds, transform

    def _get_project_grid(
        self,
        project_geometry: dict,
        transform,
        width: int,
        height: int,
        area_fill: float = 1,
        empty_fill: float = 0,
    ):
        """
        Args:
        project_geometry (dict): geometry of the project in crs ('EPSG:32633')
        area_fill (float): fill value for the project area
        empty_fill (float): fill value for the empty area
        """
        project_raster = rasterize(
            project_geometry,
            out_shape=(height, width),
            fill=empty_fill,
            default_value=area_fill,
            transform=transform,
            dtype=np.uint16,
        )
        return project_raster

    def _get_project_geometry(self, project_metadata: dict):
        """
        Get the geometry of the project - may need to be adjusted for different metadata formats
        """
        # for Norway:
        if self.region.lower() == "norway":
            project_geometry = gpd.GeoSeries(shape(project_metadata["geometry"]))
            project_geometry.crs = "EPSG:25833"
            project_geometry = project_geometry.to_crs("EPSG:32633")
            return project_geometry
        else:
            raise NotImplementedError(
                "fetching the geometry of metadata for the Region not implemented"
            )

    def _get_project_year(self, project_metadata: dict):
        """
        Get the year of the project - may need to be adjusted for different metadata formats
        """
        # for Norway:
        if self.region.lower() == "norway":
            project_year = int(project_metadata["properties"]["aar"])
            return project_year
        else:
            raise NotImplementedError(
                "fetching the year of metadata for the Region not implemented"
            )

    def get_density_grid(self):
        """
        Get the density grid for the region
        """
        density_grid = np.zeros(
            (self.raster_height, self.raster_width), dtype=np.uint16
        )
        for project_metadata in self.metadata_all_projects:
            project_geometry = self._get_project_geometry(project_metadata)
            project_raster = self._get_project_grid(
                project_geometry, self.transform, self.raster_width, self.raster_height
            )
            density_grid += project_raster
        return density_grid

    def get_oldest_project_grid(self):
        """
        get the grid of the oldest project for each year
        """
        coverage_grid = self.density_grid.copy()
        coverage_grid[coverage_grid > 0] = 1
        coverage_grid = coverage_grid.astype("int32")
        coverage_grid = coverage_grid * 5000
        for project_metadata in self.metadata_all_projects:
            project_geometry = self._get_project_geometry(project_metadata)
            project_raster = self._get_project_grid(
                project_geometry, self.transform, self.raster_width, self.raster_height
            )
            project_raster = project_raster.astype("int32")
            project_year = self._get_project_year(project_metadata)

            # we want to make sure only the pixels with the actual project coverage get the year
            # the remaining pixels should get a year that is much higher.
            project_raster = project_raster * -project_year
            project_raster = (
                project_raster + np.ones(np.shape(project_raster)) * project_year * 2
            )  # effectively sets all not covered cells to double the actual year.
            coverage_grid = np.minimum(coverage_grid, project_raster)
        return coverage_grid

    def _display_raster(
        self,
        raster,
        transform,
        cmap="viridis",
        plot_boundaries: bool = True,
        plot_colorbar: bool = True,
        color_bar_label: str = None,
        ignore_zero: bool = False,
        axis_off: bool = True,
        show_plot: bool = True,
        save_as: str = None,
        fig: plt.figure = None,
        ax: plt.Axes = None,
        figsize: tuple = None,
    ) -> None:
        """
        Display any raster
        """
        if fig and ax:
            pass
        elif figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=(15, 15))
        # calculate the extend of the raster so we can pass it to imshow
        if ignore_zero:
            # Create a colormap from the string input
            if type(cmap) == str:
                cmap = plt.get_cmap(cmap)
            # Mask the zero values in the raster
            masked = np.ma.masked_where(raster == 0, raster)
            # Use imshow with the masked raster
            im = ax.imshow(
                masked,
                cmap=cmap,
                extent=[self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]],
                aspect="equal",
            )
            # print(f'the extent that we pass to the plot is {self.bounds}')
        else:
            # show(raster, transform=transform, ax=ax, cmap=cmap)
            im = ax.imshow(
                raster,
                cmap=cmap,
                extent=[self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]],
                aspect="equal",
            )  # doesnt work with the boundaries

        if plot_boundaries:
            max_geom_extent = [10**10, 10**10, -(10**10), -(10**10)]

            def extend_max_extend(x, y, max_extent):
                max_extent[0] = min(max_extent[0], min(x))
                max_extent[1] = min(max_extent[1], min(y))
                max_extent[2] = max(max_extent[2], max(x))
                max_extent[3] = max(max_extent[3], max(y))
                return max_extent

            for geom in self.region_shape_utm.geometry:
                if geom.geom_type == "Polygon":
                    px, py = geom.exterior.xy
                    # Apply the Affine transformation to each coordinate pair
                    # px, py = zip(*[transform * (x_, y_) for x_, y_ in zip(x, y)])
                    ax.plot(px, py, color="darkgrey", linewidth=0.5)
                    max_geom_extent = extend_max_extend(px, py, max_geom_extent)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        px, py = poly.exterior.xy
                        ax.plot(px, py, color="darkgrey", linewidth=0.5)
                        max_geom_extent = extend_max_extend(px, py, max_geom_extent)
            # print(f'the extent of the geometry is {max_geom_extent}')

        if plot_colorbar:
            if self.region.lower() == "norway":
                axins = inset_axes(
                    ax,
                    width="5%",
                    height="50%",
                    loc="lower right",
                    bbox_to_anchor=(0.05, 0.2, 0.65, 0.8),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
            else:
                print(
                    f"no specific location for colorbar for country {self.region} defined yet."
                )
                axins = inset_axes(
                    ax,
                    width="5%",
                    height="50%",
                    loc="upper right",
                    bbox_to_anchor=(0.05, 0.0, 0.8, 0.8),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
            if ignore_zero:
                norm = colors.Normalize(
                    vmin=np.min(raster[raster > 0]), vmax=np.max(raster)
                )
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                cb_ticks = [
                    int(n)
                    for n in np.linspace(np.min(raster[raster > 0]), np.max(raster), 5)
                ]
            else:
                max_value = np.max(raster)
                min_value = np.min(raster)
                norm = plt.Normalize(vmin=min_value, vmax=max_value)
                # Create a ScalarMappable instance
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                cb_ticks = [
                    int(n) for n in np.linspace(np.min(raster), np.max(raster), 5)
                ]

            # Create a colorbar
            cbar = plt.colorbar(
                sm,
                cax=axins,
                orientation="vertical",
                pad=0.02,
                fraction=0.02,
                ticks=cb_ticks,
            )
            cbar.ax.tick_params(labelsize=12)
            if color_bar_label:
                cbar.set_label(
                    color_bar_label, fontsize=16, labelpad=10
                )  # 'Project coverage - # of projects'
        if axis_off:
            ax.axis("off")
        if save_as:
            plt.savefig(f"project_density_maps/{save_as}", dpi=300)
        if show_plot:
            plt.show()
        # plt.close()
        # plt.clf()
        return fig, ax


# %% actually plot
if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = Path(__file__).resolve().parents[3]
    # add path to Norway shapes in data to path
    path_to_shape = (
        root_dir / "data" / "raw" / "maps" / "Norway_boundaries" / "NOR_adm0.shp"
    )

    path_to_data = root_dir / "data" / "raw" / "orthophoto"
    # list all files (not directories) in the path
    metadata_files = [
        f
        for f in os.listdir(path_to_data)
        if os.path.isfile(os.path.join(path_to_data, f))
    ]

    # the last digits in the file name is the date and time of the metadata, we want the latest
    # Function to extract datetime from filename
    def extract_datetime(filename):
        # Assuming the date is at the end of the filename and is in a specific format
        # Adjust the slicing as per your filename format
        date_str = filename.split("_")[-1].split(".")[
            0
        ]  # Adjust based on your filename format
        # print(date_str)
        return datetime.strptime(
            date_str, "%Y%m%d%H%M%S"
        )  # Adjust the format as per your filename

    # Sort files by datetime
    sorted_files = sorted(metadata_files, key=extract_datetime, reverse=True)

    # The newest file
    newest_file = sorted_files[0]
    print("Newest file:", newest_file)
    # print(path_to_data / newest_file)
    with open(path_to_data / newest_file, "r") as f:
        metadata_all_projects = json.load(f)

    # create the density grid
    resolution = 5000
    figsize = (10, 10)
    Norway_density = ProjectDensityGrid(
        metadata_all_projects["ProjectMetadata"],
        region_shape_file=path_to_shape,
        resolution=resolution,
        region="Norway",
    )
    fig, axs = plt.subplots(1, 2, figsize=(14, 9))
    plt.subplots_adjust(wspace=0)
    # plot the density grid
    density_cmap = "gist_heat_r"
    fig, ax = Norway_density._display_raster(
        Norway_density.density_grid,
        Norway_density.transform,
        cmap=density_cmap,
        plot_boundaries=False,
        plot_colorbar=True,
        # save_as=f'density_project_map_res{resolution}_cmap{density_cmap}.png',
        color_bar_label="# of projects",
        fig=fig,
        ax=axs[0],
        show_plot=False,
    )
    axs[0].text(
        0.04,
        0.96,
        "a)",
        transform=axs[0].transAxes,
        fontsize=16,
        verticalalignment="top",
    )

    # plot the oldest project grid
    cmap_colors = [
        (1, 1, 1),
        (0.12137254901960784, 0.58823529411764706, 0.7196078431372549),
        (0.06137254901960784, 0.38823529411764706, 0.5196078431372549),
        (0.04137254901960784, 0.26823529411764706, 0.4396078431372549),
        (0.04137254901960784, 0.20823529411764706, 0.3996078431372549),
        (0.09, 0.09, 0.09),
    ]
    cmap_colors.reverse()
    # Create a custom colormap from the defined colors
    CustomBlues = mcolors.LinearSegmentedColormap.from_list("CustomBlues", cmap_colors)
    oldest_cmaps = ["copper", "Blues_r", CustomBlues]  #'pink'#'copper'
    oldest_cmap = oldest_cmaps[-1]
    fig, ax = Norway_density._display_raster(
        Norway_density.oldest_grid,
        Norway_density.transform,
        cmap=oldest_cmap,
        plot_boundaries=False,
        plot_colorbar=True,
        ignore_zero=True,
        # save_as=f'oldest_project_map_res{resolution}_cmap{oldest_cmap}.png',
        color_bar_label="Oldest project",
        fig=fig,
        ax=axs[1],
        show_plot=False,
    )
    axs[1].text(
        0.04,
        0.96,
        "b)",
        transform=axs[1].transAxes,
        fontsize=14,
        verticalalignment="top",
    )
    if type(oldest_cmap) != str:
        oldest_cmap = oldest_cmap.name
    plt.savefig(
        root_dir
        / f"data/figures/orthophoto_metadata/Norway_coverage_map_res{resolution}_cmaps_{density_cmap}_{oldest_cmap}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%
