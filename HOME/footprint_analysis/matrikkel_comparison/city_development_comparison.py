"""
Hosting the making of the large graphic we want to talk about
"""

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from HOME.footprint_analysis.matrikkel_comparison.matrikkel_data.prepare_municipality_data import (
    get_city_data,
    make_buildings,
    get_time_series,
)


# %%
def prepare_city_data(city: str):
    """
    Prepare the city data for plotting.
    """
    city_buildings_matrikkel, FKB_bygning_city = get_city_data(city)
    city_building_objects, fkb_matches = make_buildings(
        city_buildings_matrikkel, FKB_bygning_city
    )
    (
        time,
        n_buildings,
        build_area_upper,
        build_area_stock_upper,
        build_area_stock_lower,
    ) = get_time_series(city_building_objects)
    return (
        city_building_objects,
        time,
        build_area_upper,
        build_area_stock_upper,
        build_area_stock_lower,
    )


def make_age_map(
    city_building_objects: list,
    city_boundaries: dict,
    resolution_long_edge: int = 512,
    city_name: str = "Trondheim",
    root_dir: Path = root_dir,
    xlim: tuple = None,
    ylim: tuple = None,
    color_bar_location: str = "lower left",
    show_axis: bool = False,
):
    """
    Make a map of the buildings in the city, colored by their age.
    We therefore build a grid of the city and assign the buildings to the grid cells.
    the cells are colored by the age of the buildings in the cell.

    Args:
        city_building_objects (list): The list of building objects to plot.
        city_boundaries (dict): The city boundaries to plot the buildings in - epsg:25832
        resolution_long_edge (int): The resolution of the grid cells - default is 512

    Returns:
        fig, ax: The figure and axis of the plot.
    """
    # make a grid of the city boundaries
    bounds = city_boundaries.bounds
    x_min = bounds[0]
    x_max = bounds[2]
    y_min = bounds[1]
    y_max = bounds[3]
    x_range = x_max - x_min
    y_range = y_max - y_min

    max_extended = max(x_range, y_range)
    grid_size = max_extended / resolution_long_edge
    x_grid = int(x_range / grid_size)
    y_grid = int(y_range / grid_size)
    # make an array to store the boundaries of the grid cells
    x_grid_bounds = [x_max + i * grid_size for i in range(x_grid + 1)]
    y_grid_bounds = [y_min + i * grid_size for i in range(y_grid + 1)]
    # make an array to store the age of the buildings in the grid cells
    ages_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    areas_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    average_age_array = [[0 for _ in range(y_grid)] for i in range(x_grid)]
    total_area_array = [[0 for _ in range(y_grid)] for i in range(x_grid)]
    color_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    # print(ages_array)
    transformer_from_5122 = Transformer.from_crs(
        "EPSG:25832", "EPSG:4326", always_xy=True  # building.location_crs
    ).transform
    for building in city_building_objects:
        if building.location:
            if building.location_crs == "EPSG:5122":
                location = transform(transformer_from_5122, Point(building.location))
            else:
                location = Point(building.location)
                # print(building.location_crs)
            # get the coordinates of the building
            x = location.x
            y = location.y
            # get the grid cell that the building is in
            x_index = int((x - x_min) / grid_size)
            y_index = int((y - y_min) / grid_size)
            # print(
            #     f"building {building.bygningsnummer} is in grid cell {x_index}, {y_index}"
            # )
            # check if the building is in the grid cell
            if x_index >= 0 and x_index < x_grid and y_index >= 0 and y_index < y_grid:
                # add the age of the building to the grid cell
                ages_array[x_index][y_index].append(building.cohort)
                areas_array[x_index][y_index].append(building.footprint_area)

    # insert the average age (weighed by the area) of the buildings in the grid cells
    for i in range(x_grid):
        for j in range(y_grid):
            # exclude all age entries that are -1
            average_age_entries = [age for age in ages_array[i][j] if age != -1]
            areas_entries = [
                area
                for area, age in zip(areas_array[i][j], ages_array[i][j])
                if age != -1
            ]
            if len(average_age_entries) == 0:
                average_age_array[i][j] = -1
                color_array[i][j] = -1
                continue
            total_area = sum(areas_entries)
            if (
                total_area == 0
            ):  # if all available buildings sum to 0, we count each the same
                areas_entries = [1 for _ in range(len(areas_entries))]
            average_age_array[i][j] = np.average(
                average_age_entries, weights=areas_entries
            )
            total_area_array[i][j] = total_area

    max_total_area = max([max(total_area_array[i]) for i in range(x_grid)])
    # create a colormap for the ages - 1900-2025
    norm = mcolors.Normalize(vmin=1900, vmax=2025)
    cmap = plt.cm.viridis
    cmap = plt.cm.get_cmap("viridis", 256)
    cmap = matplotlib.colormaps.get_cmap("gist_heat")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for i in range(x_grid):
        for j in range(y_grid):
            if average_age_array[i][j] == -1:
                color_array[i][j] = (1, 1, 1, 1)  # make white
                continue

            color = cmap(norm(average_age_array[i][j]))
            color_rgba = mcolors.to_rgba(color)
            color_rgba = (
                color_rgba[0],
                color_rgba[1],
                color_rgba[2],
                color_rgba[3]
                * min([total_area_array[i][j] / (max_total_area * 0.3), 1]) ** 0.35,
            )
            color_array[i][j] = color_rgba

    print(ages_array[294][161])
    print(areas_array[294][161])
    print(average_age_array[294][161])

    # make a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    # put the topography in the background
    ax = plot_topography_around_city(  # TODO need to adjust crs between the two functions
        ax,
        city_boundaries.bounds,
        root_dir=root_dir,
        buffer=0.05,
        crs="EPSG:4326",
        grid_size=(5000, 5000),
        shaded_cmap="Greys",
        shaded_cmap_alpha=0.5,
        topo_cmap="terrain",
        topo_alpha=0.20,
    )
    color_array = np.array(color_array)
    color_array = np.rot90(color_array)
    # display the color array as an image
    ax.imshow(
        color_array,
        extent=(x_min, x_max, y_min, y_max),
        interpolation="bilinear",
        aspect="auto",
    )
    # add a colorbar
    sm.set_array([])
    inset_axis_cbar = inset_axes(
        ax,
        width=0.2,
        height="40%",
        loc=color_bar_location,
        bbox_to_anchor=(0.1, 0, 0.8, 0.8),
        bbox_transform=ax.transAxes,
        borderpad=0.1,
    )
    cbar = plt.colorbar(
        sm, cax=inset_axis_cbar, orientation="vertical", pad=0.1, shrink=0.5
    )
    cbar.set_label("Average building age")
    # fig.colorbar(sm, ax=ax, label="Average building age")
    # add the city boundaries
    city_gdf = gpd.GeoSeries(city_boundaries, crs=4326)
    city_gdf.plot(ax=ax, edgecolor="darkgrey", linewidth=0.5, facecolor="none")
    if xlim:
        x_min = xlim[0]
        x_max = xlim[1]
    if ylim:
        y_min = ylim[0]
        y_max = ylim[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # add a title
    ax.set_title(f"Building age map for {city_name}")
    if not show_axis:
        ax.axis("off")

    return


# %%
if __name__ == "__main__":
    # Create the figure and gridspec

    # ax_large.plot([0, 1, 2], [0, 1, 0])  # Example plot
    # city1: Trondheim
    for city in ["trondheim"]:
        (
            city_building_objects,
            time,
            build_area_upper,
            build_area_stock_upper,
            build_area_stock_lower,
        ) = prepare_city_data(city)
    # %%
    fig = plt.figure(figsize=(12, 12))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(5, 5, figure=fig)

    # Add the large plot spanning the top 3 rows and all columns
    ax_large = fig.add_subplot(gs[:3, :])
    # ax_large.set_title("Large Plot (Top 3 Rows)")
    ax_large.fill_between(
        time, build_area_stock_lower, build_area_stock_upper, color="navy", alpha=0.3
    )
    ax_large.set_xlim(1985, 2025)
    ax_large.set_ylabel("Built-up area (m2)")
    # Add the individual tiles for the lower 2 rows and 5 columns
    axes = []
    for row in range(3, 5):  # Rows 3 and 4 (0-indexed)
        for col in range(5):  # Columns 0 to 4
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"Tile {row-2},{col}")
            # ax.axis("off")  # Hide the axes for the tiles
            axes.append(ax)

    # Adjust layout
    plt.tight_layout()
    plt.show()
# %%
