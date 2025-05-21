# %%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import Point, MultiPolygon, Polygon, shape
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from HOME.footprint_analysis.matrikkel_comparison.matrikkel_data.prepare_municipality_data import (
    get_city_data,
    make_buildings,
    get_time_series,
    BuildingMatrikkel,
    BuildingFKB,
)
from HOME.footprint_analysis.matrikkel_comparison.HOME_data.prepare_municipality_footprints import (
    get_total_footprint_data,
)
from HOME.footprint_analysis.matrikkel_comparison.municipality_boundaries import (
    get_municipal_boundaries,
)
from HOME.visualization.utils.shade_relief_maps import plot_topography_around_city


root_dir = Path(__file__).parents[3]
print(root_dir)
# %%
matrikkel_data_path = (
    root_dir
    / "data/matrikkel_comparison/matrikkel_municipal_data/trondheim_2025-05-04_22-21-29_buildings.pkl"
)
with open(matrikkel_data_path, "rb") as f:
    matrikkel_buildings = pickle.load(f)

HOME_data_path = (
    root_dir
    / "data/matrikkel_comparison/HOME_municipal_data/trondheim_2025-05-04_13-10-42_project_time_period_coverage.pkl"
)
with open(HOME_data_path, "rb") as f:
    HOME_time_period_coverage = pickle.load(f)

# %% the HOME data
time_line, lower_bound_values, upper_bound_values = get_total_footprint_data(
    HOME_time_period_coverage
)
print(f"sample of all last 10 entries:\n")
print(f"{time_line[-10:]}")
print(f"{lower_bound_values[-10:]}")
print(f"{upper_bound_values[-10:]}")


# %%
def get_rolling_average_mean_percentiles(
    upper_bounds: list[float], lower_bounds: list[float], window_size: int
):
    rolling_side = window_size // 2
    if window_size % 2 != 0:
        raise ValueError(f"Window size must be an even int, not {window_size}.")
    length = len(upper_bounds)
    mean = np.zeros(length)
    lower_percentile = np.zeros(length)
    upper_percentile = np.zeros(length)
    lower_percentile_from_std = np.zeros(length)
    upper_percentile_from_std = np.zeros(length)
    for i in range(length):  # rolling through timeseries
        lower_i = i - rolling_side
        upper_i = i + rolling_side
        if lower_i < 0:
            lower_i = 0
        if upper_i > length - 1:
            upper_i = length - 1
        data_in_window = list(upper_bounds[lower_i:upper_i]) + list(
            lower_bounds[lower_i:upper_i]
        )
        lower_percentile[i] = np.percentile(data_in_window, 25)
        upper_percentile[i] = np.percentile(data_in_window, 75)
        mean[i] = np.average(data_in_window)
        std = (
            np.std(data_in_window) / 2.36
        )  # adjusting std based on the sampling from 95%
        lower_percentile_from_std[i] = mean[i] - 2 * std
        upper_percentile_from_std[i] = mean[i] + 2 * std

    return (
        mean,
        lower_percentile,
        upper_percentile,
        lower_percentile_from_std,
        upper_percentile_from_std,
    )


(
    mean,
    lower_percentile,
    upper_percentile,
    lower_percentile_from_std,
    upper_percentile_from_std,
) = get_rolling_average_mean_percentiles(
    lower_bound_values, upper_bound_values, window_size=120
)
print(f"sample of all last 10 entries:\n")
print(f"{mean[-10:]}")
print(f"{lower_percentile[-10:]}")
print(f"{upper_percentile[-10:]}")
print(f"{lower_percentile_from_std[-10:]}")
print(f"{upper_percentile_from_std[-10:]}")


# %%
def plot_rolling_average_mean_percentiles(
    time, mean, lower_percentile, upper_percentile, lower_values, upper_values, ax=None
):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
        set_xlim = True
    else:
        set_xlim = False

    ax.plot(
        time, lower_values, "*", label="lower bound data", alpha=0.5, color="darkred"
    )
    ax.plot(
        time, upper_values, "*", label="upper bound data", alpha=0.5, color="darkorange"
    )
    # ax.fill_between(
    #     time_line,
    #     lower_percentile,
    #     upper_percentile,
    #     alpha=0.4,
    #     label="Rolling average percentiles",
    # )
    ax.fill_between(
        time_line,
        lower_percentile_from_std,
        upper_percentile_from_std,
        color="crimson",
        alpha=0.7,
        label="Rolling average 95 CI",
    )
    ax.plot(time, mean, label="Rolling average", lw=4.5, color="crimson")
    if set_xlim:
        ax.set_xlim(datetime(1985, 1, 1), datetime(2018, 1, 1))
        ax.set_ylim(0.5e7, 1.21e7)
        ax.legend(loc="upper left")

    return ax


plot_rolling_average_mean_percentiles(
    time_line,
    mean,
    lower_percentile,
    upper_percentile,
    lower_bound_values,
    upper_bound_values,
)


# %%
def get_matrikkel_age_map_data(
    HOME_time_period_coverage: dict,
    city_boundaries: gpd.GeoDataFrame,
    resolution_long_edge: int = 512,
    time_stamps: list[int] = None,
):
    ## make the grid
    bounds = city_boundaries.bounds
    x_min = bounds.minx[0]
    x_max = bounds.maxx[0]
    y_min = bounds.miny[0]
    y_max = bounds.maxy[0]
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_extent = max(x_range, y_range)
    grid_size = max_extent / resolution_long_edge
    x_grid = int(x_range / grid_size)
    y_grid = int(y_range / grid_size)
    # make an array to store the boundaries of the grid cells
    x_grid_bounds = [x_max + i * grid_size for i in range(x_grid + 1)]
    y_grid_bounds = [y_min + i * grid_size for i in range(y_grid + 1)]

    # we need to set up arrays as lists of lists (to insert whatever we want)
    # for each time period, and in the grid.
    time_periods = [[t0, t1] for t0, t1 in zip(time_stamps, time_stamps[1:])]

    # how can we get this:
    # we can, for each cell, find the buildings that are in it (upper and lower bounds).
    # we can then for this cell do a little rolling average of the total population,
    # which will also yield the increases over time.
    # we can use the increases for each year to define when in the grid cell
    # the buildings were mostly built - simply weighing the increase /years together.abs

    # we should establish a timeline (we can use the months again).
    time_line = pd.date_range(
        datetime(time_stamps[0], 1, 1), datetime(time_stamps[-1], 1, 1), freq="M"
    )
    lower_bound_values_xyt = np.zeros((y_grid, xgrid, len(time_line)))
    upper_bound_values_xyt = np.zeros((y_grid, xgrid, len(time_line)))
    for patch_id, patch_data in HOME_time_period_coverage.items():
        start_date = patch_data["start_date"]
        end_date = patch_data["end_date"]
        if not start_date:
            start_date = datetime(time_stamps[0], 1, 1)
        if not end_date:
            end_date = datetime(time_stamps[-1], 1, 1)
        start_index = time_line.get_loc(start_date, method="nearest")
        end_index = time_line.get_loc(end_date, method="nearest")
        if "lower_bound_polygons" in patch_data.keys():
            lower_bound_polygons = patch_data["lower_bound_polygons"]
        else:
            lower_bound_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:25833")
        if "upper_bound_polygons" in patch_data.keys():
            upper_bound_polygons = patch_data["upper_bound_polygons"]
        else:
            upper_bound_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:25833")
        lower_bound_polygons = lower_bound_polygons.to_crs("EPSG:4326")
        upper_bound_polygons = upper_bound_polygons.to_crs("EPSG:4326")
        for x_grid_bound in range(x_grid):
            for y_grid_bound in range(y_grid):
                # get the grid cell that the building is in
                x_index = int((x_grid_bound - x_min) / grid_size)
                y_index = int((y_grid_bound - y_min) / grid_size)
                if (
                    x_index >= 0
                    and x_index < x_grid
                    and y_index >= 0
                    and y_index < y_grid
                ):
                    # get the buildings in this grid cell
                    cell_bounds = Polygon(
                        x_grid_bound,
                        y_grid_bound,
                        x_grid_bound + grid_size,
                        y_grid_bound + grid_size,
                    )
                    lower_bound_polygons_in_cell = lower
                    upper_bound_polygons_in_cell = upper_bound_polygons.cx[
                        x_grid_bounds[x_index], y_grid_bounds[y_index]
                    ]
                    if len(lower_bound_polygons_in_cell) > 0:
                        lower_bound_values_xyt[y_index][x_index][
                            start_index:end_index
                        ] += lower_bound_polygons_in_cell.to_crs(
                            "EPSG:25833"
                        ).area.sum()
                    if len(upper_bound_polygons_in_cell) > 0:
                        upper_bound_values_xyt[y_index][x_index][
                            start_index:end_index
                        ] += upper_bound_polygons_in_cell["footprint_area"].values

    ages_array_txya = [
        [[[0] for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]  # time, x, y, ages
    areas_array_txya = [
        [[[0] for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]  # time, x, y, areas
    average_age_array_txy = [
        [[0 for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]
    total_area_array_txy = [
        [[0 for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]

    transformer_from_5122 = Transformer.from_crs(
        "EPSG:25832", "EPSG:4326", always_xy=True  # building.location_crs
    ).transform
    for building in matrikkel_buildings:
        if building.location:
            if building.location_crs == "EPSG:5122":
                location = transform(transformer_from_5122, Point(building.location))
            else:
                location = Point(building.location)
        x = location.x
        y = location.y
        # get the grid cell that the building is in
        x_index = int((x - x_min) / grid_size)
        y_index = int((y - y_min) / grid_size)
        age = building.cohort
        # print(f"building age: {age}, type {type(age)}")

        area = building.footprint_area
        for i, [t0, t1] in enumerate(time_periods):
            if age < t0:
                time_index = -1
                break
            elif age > t0 and age < t1:
                time_index = i
                break
            else:
                time_index = -1
        if x_index >= 0 and x_index < x_grid and y_index >= 0 and y_index < y_grid:
            if time_index >= 0:
                ages_array_txya[time_index][x_index][y_index].append(age)
                areas_array_txya[time_index][x_index][y_index].append(area)
        else:
            # pass  # raise Exception
            print(f"can't place building in grid: {x_index=}, {y_index=}")
    for t in range(len(time_periods)):
        for x in range(x_grid):
            for y in range(y_grid):
                average_age_entries = [
                    age for age in ages_array_txya[t][x][y] if age != -1
                ]
                areas_entries = [
                    area
                    for area, age in zip(
                        areas_array_txya[t][x][y], ages_array_txya[t][x][y]
                    )
                    if age != -1
                ]
                if len(average_age_entries) == 0:
                    average_age_array_txy[t][x][y] = -1
                    continue
                total_area = sum(areas_entries)
                if (
                    total_area == 0
                ):  # if all available buildings sum to 0, we count each the same
                    areas_entries = [1 for _ in range(len(areas_entries))]
                average_age_array_txy[t][x][y] = np.average(
                    average_age_entries, weights=areas_entries
                )
                total_area_array_txy[t][x][y] = total_area
    return average_age_array_txy, total_area_array_txy, time_periods


# %% the matrikkel data
(
    time,
    n_buildings,
    build_area_upper,
    build_area_stock_upper,
    build_area_stock_lower,
) = get_time_series(matrikkel_buildings)
print(f"sample of all last 10 entries:\n")
print(f"{time[-10:]}")
print(f"{n_buildings[-10:]}")
print(f"{build_area_upper[-10:]}")
print(f"{build_area_stock_upper[-10:]}")
print(f"{build_area_stock_lower[-10:]}")

# %%


# %%
def plot_matrikkel_range(
    time, build_area_stock_lower, building_area_stock_upper, ax=None
):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
        set_xlim = True
    else:
        set_xlim = False
    time = [datetime(y, 1, 1) for y in time]
    # ax.plot(time, build_area_stock_lower, "*", label="lower bound data", alpha=0.5)
    # ax.plot(time, building_area_stock_upper, "*", label="upper bound data", alpha=0.5)
    ax.fill_between(
        time,
        build_area_stock_lower,
        building_area_stock_upper,
        alpha=0.6,
        color="gold",
        label="Cadaster range - external validation",
    )
    if set_xlim:
        ax.set_xlim(datetime(1985, 1, 1), datetime(2018, 1, 1))
        ax.set_ylim(0.5e7, 1.21e7)

    return ax


plot_matrikkel_range(time, build_area_stock_lower, build_area_stock_upper, ax=None)


# %%
def get_matrikkel_age_map_data(
    matrikkel_buildings: list[BuildingMatrikkel, BuildingFKB],
    city_boundaries: gpd.GeoDataFrame,
    resolution_long_edge: int = 512,
    time_stamps: list[int] = None,
):
    ## make the grid
    bounds = city_boundaries.bounds
    x_min = bounds.minx[0]
    x_max = bounds.maxx[0]
    y_min = bounds.miny[0]
    y_max = bounds.maxy[0]
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_extent = max(x_range, y_range)
    grid_size = max_extent / resolution_long_edge
    x_grid = int(x_range / grid_size)
    y_grid = int(y_range / grid_size)
    # make an array to store the boundaries of the grid cells
    x_grid_bounds = [x_max + i * grid_size for i in range(x_grid + 1)]
    y_grid_bounds = [y_min + i * grid_size for i in range(y_grid + 1)]

    # we need to set up arrays as lists of lists (to insert whatever we want)
    # for each time period, and in the grid.
    time_periods = [[t0, t1] for t0, t1 in zip(time_stamps, time_stamps[1:])]
    ages_array_txya = [
        [[[0] for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]  # time, x, y, ages
    areas_array_txya = [
        [[[0] for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]  # time, x, y, areas
    average_age_array_txy = [
        [[0 for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]
    total_area_array_txy = [
        [[0 for _ in range(y_grid)] for i in range(x_grid)] for _ in time_periods
    ]

    transformer_from_5122 = Transformer.from_crs(
        "EPSG:25832", "EPSG:4326", always_xy=True  # building.location_crs
    ).transform
    for building in matrikkel_buildings:
        if building.location:
            if building.location_crs == "EPSG:5122":
                location = transform(transformer_from_5122, Point(building.location))
            else:
                location = Point(building.location)
        x = location.x
        y = location.y
        # get the grid cell that the building is in
        x_index = int((x - x_min) / grid_size)
        y_index = int((y - y_min) / grid_size)
        age = building.cohort
        # print(f"building age: {age}, type {type(age)}")

        area = building.footprint_area
        for i, [t0, t1] in enumerate(time_periods):
            if age < t0:
                time_index = -1
                break
            elif age > t0 and age < t1:
                time_index = i
                break
            else:
                time_index = -1
        if x_index >= 0 and x_index < x_grid and y_index >= 0 and y_index < y_grid:
            if time_index >= 0:
                ages_array_txya[time_index][x_index][y_index].append(age)
                areas_array_txya[time_index][x_index][y_index].append(area)
        else:
            # pass  # raise Exception
            print(f"can't place building in grid: {x_index=}, {y_index=}")
    for t in range(len(time_periods)):
        for x in range(x_grid):
            for y in range(y_grid):
                average_age_entries = [
                    age for age in ages_array_txya[t][x][y] if age != -1
                ]
                areas_entries = [
                    area
                    for area, age in zip(
                        areas_array_txya[t][x][y], ages_array_txya[t][x][y]
                    )
                    if age != -1
                ]
                if len(average_age_entries) == 0:
                    average_age_array_txy[t][x][y] = -1
                    continue
                total_area = sum(areas_entries)
                if (
                    total_area == 0
                ):  # if all available buildings sum to 0, we count each the same
                    areas_entries = [1 for _ in range(len(areas_entries))]
                average_age_array_txy[t][x][y] = np.average(
                    average_age_entries, weights=areas_entries
                )
                total_area_array_txy[t][x][y] = total_area
    return average_age_array_txy, total_area_array_txy, time_periods


def plot_age_map(
    average_age_array_txy, area_added, time_periods, city_boundaries, axes=None
):
    len_time_grid = len(average_age_array_txy)
    len_x_grid = len(average_age_array_txy[0])
    len_y_grid = len(average_age_array_txy[0][0])
    colors_txyc = [
        [[[0, 0, 0, 1] for y in range(len_y_grid)] for x in range(len_x_grid)]
        for t in range(len_time_grid)
    ]
    if not axes:
        fig, axes = plt.subplots(1, len_time_grid)

    max_area_added = max(
        max(
            [max(area_added[t][i]) for i in range(len_x_grid)]
            for t in range(len_time_grid)
        )
    )
    print(f"max area added: {max_area_added}")
    norm = mcolors.Normalize(vmin=time_periods[0][0], vmax=time_periods[-1][1])
    cmap = matplotlib.colormaps.get_cmap("gist_heat")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for t in range(len_time_grid):
        for x in range(len_x_grid):
            for y in range(len_y_grid):
                if average_age_array_txy[t][x][y] == -1:
                    colors_txyc[t][x][y] = (1, 1, 1, 1)  # make white
                    continue
                color = cmap(norm(average_age_array_txy[t][x][y]))
                color_rgba = mcolors.to_rgba(color)
                color_rgba = (
                    color_rgba[0],
                    color_rgba[1],
                    color_rgba[2],
                    color_rgba[3]
                    * min([area_added[t][x][y] / (max_area_added * 0.3), 1]) ** 0.25,
                )
                colors_txyc[t][x][y] = color_rgba
    city_bounds = city_boundaries.bounds
    for ax, colors_xyc in zip(axes, colors_txyc):
        ax = plot_topography_around_city(
            ax,
            (
                city_bounds.minx[0],
                city_bounds.miny[0],
                city_bounds.maxx[0],
                city_bounds.maxy[0],
            ),
            root_dir,
            buffer=0.05,
            crs="EPSG:4326",
            grid_size=(5000, 5000),
            shaded_cmap="Greys",
            shaded_cmap_alpha=0.3,
            topo_cmap="terrain",
            topo_alpha=0.16,
        )
        colors_array_xyc = np.array(colors_xyc)
        colors_array_xyc = np.rot90(colors_array_xyc)

        ax.imshow(
            colors_array_xyc,
            extent=(
                city_bounds.minx[0],
                city_bounds.maxx[0],
                city_bounds.miny[0],
                city_bounds.maxy[0],
            ),
            interpolation="bilinear",
            aspect="auto",
        )
        # add a colorbar
        sm.set_array([])
        inset_axis_cbar = inset_axes(
            ax,
            width=0.1,
            height="25%",
            loc="lower right",
            bbox_to_anchor=(0, 0.05, 0.8, 0.8),
            bbox_transform=ax.transAxes,
            borderpad=0.1,
        )
        cbar = plt.colorbar(
            sm, cax=inset_axis_cbar, orientation="vertical", pad=0.1, shrink=0.2
        )
        # cbar.set_label("Average construction year")
        # fig.colorbar(sm, ax=ax, label="Average building age")
        # add the city boundaries
        # city_gdf = gpd.GeoSeries(city_boundaries, crs=4326)
        city_boundaries.plot(
            ax=ax, edgecolor="darkgrey", linewidth=0.5, facecolor="none"
        )
        ax.set_xlim(10.27, 10.55)
        ax.set_ylim(63.32, 63.49)
        # add a title
        # ax.set_title(f"Building age map for {city_name}")
        ax.axis("off")

    return


# %%

municipality = "trondheim"
municipality_boundaries = get_municipal_boundaries(municipality).to_crs("EPSG:4326")

average_age_array_txy, total_area_array_txy, time_periods = get_matrikkel_age_map_data(
    matrikkel_buildings,
    municipality_boundaries,
    time_stamps=[1981, 1988, 1995, 2002, 2009, 2016],
)
# print a little bit of the data
print(f"the time periods are:\n {time_periods}")
print(f"the average age array is of size  {np.shape(np.array(average_age_array_txy))}")
print(
    f"the center of the array looks like this: {np.array(average_age_array_txy)[2, 120:342, 120:242]}"
)
print(
    f"sum across the entire average age array: {np.sum(np.sum(average_age_array_txy))}"
)
# %%
print(f"the total area array is of size  {np.shape(np.array(total_area_array_txy))}")
print(
    f"the center of the array looks like this: {np.array(total_area_array_txy)[2, 120:342, 120:242]}"
)

# %%
plot_age_map(
    average_age_array_txy,
    total_area_array_txy,
    time_periods,
    municipality_boundaries,
)

# %%# %%


# %% the actual plot (requires all the above)
fig = plt.figure(figsize=(15, 15))  # Adjust the figure size as needed
gs = gridspec.GridSpec(5, 5, figure=fig)

# Add the large plot spanning the top 3 rows and all columns
ax_large = fig.add_subplot(gs[:3, :])
plot_matrikkel_range(time, build_area_stock_lower, build_area_stock_upper, ax=ax_large)
plot_rolling_average_mean_percentiles(
    time_line,
    mean,
    lower_percentile,
    upper_percentile,
    lower_bound_values,
    upper_bound_values,
    ax_large,
)
ax_large.set_xlim(datetime(1981, 1, 1), datetime(2016, 1, 1))
ax_large.set_ylim(0.5e7, 1.21e7)
ax_large.legend(loc="upper left")
ax_large.set_ylabel("Built-up area (m2)")

# ax_large.set_title("Large Plot (Top 3 Rows)")
# ax_large.fill_between(
#     time, build_area_stock_lower, build_area_stock_upper, color="navy", alpha=0.3
# )
# ax_large.set_xlim(1985, 2025)
# ax_large.set_ylabel("Built-up area (m2)")

# Add the individual tiles for the lower 2 rows and 5 columns

axes_matrikkel = []
row = 3
for col in range(5):  # Columns 0 to 4
    ax = fig.add_subplot(gs[row, col])
    # ax.set_title(f"Tile {row-2},{col}")
    ax.axis("off")  # Hide the axes for the tiles
    axes_matrikkel.append(ax)
axes_matrikkel[0].text(
    -0.1,
    0.5,
    "Cadaster construction maps",
    fontsize=12,
    rotation=90,
    va="center",
    ha="center",
    transform=axes_matrikkel[0].transAxes,
)
plot_age_map(
    average_age_array_txy,
    total_area_array_txy,
    time_periods,
    municipality_boundaries,
    axes=axes_matrikkel,
)
axes_HOME = []
row = 4
for col in range(5):  # Columns 0 to 4
    ax = fig.add_subplot(gs[row, col])
    # ax.set_title(f"Tile {row-2},{col}")
    ax.axis("off")  # Hide the axes for the tiles
    axes_HOME.append(ax)
# axes_HOME[0].text(
#     -0.1,
#     0.5,
#     "Orthophoto construction maps",
#     fontsize=12,
#     rotation=90,
#     va="center",
#     ha="center",
#     transform=axes_HOME[0].transAxes,
# )

# Adjust layout
plt.tight_layout()
plt.show()
# %%
fig.savefig(
    root_dir / "data/figures/matrikkel_comparison/trondheim_building_stock.png",
    dpi=500,
    bbox_inches="tight",
)

# %%
