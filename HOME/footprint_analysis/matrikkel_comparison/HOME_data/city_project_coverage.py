"""
Checking the coverage of the projects we used for each city vs the municipal boundaries
"""

# %%
from pathlib import Path
import json
from shapely.geometry import shape, box
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from HOME.utils.get_project_metadata import (
    get_project_metadata,
    get_project_geometry,
    _get_newest_metadata,
)
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)

root_dir = Path(__file__).parents[4]

# %%
# list the projects by city:
if __name__ == "__main__":
    cities = [
        "trondheim",
        "oslo",
        "bergen",
        "stavanger",
        "bærum",
        "kristiansand",
        "drammen",
        "asker",
        "lillestrøm",
        "fredrikstad",
        "sandnes",
        "tromsø",
        "skien",
        "ålesund",
        "bodø",
    ]

    # %%
    # get the boundaries for the cities
    municipality_boundaries = {city: get_municipal_boundaries(city) for city in cities}
    # also check which projects would be interesting from a coverage perspective
    metadata_all_projects = _get_newest_metadata()
    all_projects = metadata_all_projects["ProjectList"][:]
    geometries = get_project_geometry(all_projects)
    geometries = [geo.to_crs(epsg=4326) for geo in geometries]
    # %%

    # Convert city boundaries to shapely geometries
    city_boundaries_shapely = {
        city: shape(boundary) for city, boundary in municipality_boundaries.items()
    }
    city_boundaries_area = {
        city: city_boundary.area
        for city, city_boundary in city_boundaries_shapely.items()
    }

    def get_project_date(capture_date: str):
        try:
            # Try to convert the input as a Unix timestamp in milliseconds
            timestamp_ms = int(capture_date)
            timestamp_s = timestamp_ms / 1000
            return datetime.fromtimestamp(timestamp_s)
        except ValueError:
            # If conversion to int fails, assume the input is a date string
            try:
                return datetime.strptime(capture_date, "%Y-%m-%d")
            except ValueError:
                # If conversion to date fails, return None
                return capture_date

    City_overlaps = {city: [] for city in cities}
    City_full_coverage = {city: [] for city in cities}
    City_coverage_rel_area = {city: {} for city in cities}
    City_overlaps_shapes = {city: {} for city in cities}
    # Check for overlaps and full coverage
    for project, metadata, geometry_series in zip(
        all_projects, metadata_all_projects["ProjectMetadata"], geometries
    ):
        # check first that the resolution is at least 0.3m:
        if float(metadata["properties"]["pixelstorrelse"]) > 0.3:
            continue
        for city, city_boundary in city_boundaries_shapely.items():
            # print(f'Checking project {project} in city {city}')
            intersects = False
            completly_covers = False
            total_overlap_area = 0
            for geometry in geometry_series:
                if geometry.intersects(city_boundary):
                    intersects = True
                    try:
                        intersects_area = geometry.intersection(city_boundary).area
                        total_overlap_area += intersects_area
                    except:
                        print(f"Error in project {project} in city {city}")
                if geometry.contains(city_boundary):
                    completly_covers = True
            if intersects:
                City_overlaps[city].append(project)
                City_coverage_rel_area[city][project] = {
                    "relative_overlap": total_overlap_area / city_boundaries_area[city],
                    "time": get_project_date(metadata["properties"]["fotodato_date"]),
                }
                City_overlaps_shapes[city][project] = {
                    "relative_overlap": total_overlap_area / city_boundaries_area[city],
                    "time": get_project_date(metadata["properties"]["fotodato_date"]),
                    "geometry": geometry_series,
                }
            if completly_covers:
                City_full_coverage[city].append(project)
            # print(f'Intersects: {intersects}, completly covers: {completly_covers}')
    print(City_overlaps)
    print(City_full_coverage)
    print(City_coverage_rel_area)

    # save city coverage data to a json
    def datetime_converter(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    with open(
        root_dir
        / "HOME/footprint_analysis/matrikkel_comparison/HOME_data/city_project_coverage.json",
        "w",
    ) as f:
        json.dump(City_coverage_rel_area, f, default=datetime_converter)

    for city, coverage in City_coverage_rel_area.items():
        print(f"City: {city}")
        fig, ax = plt.subplots()
        for project, coverage_data in coverage.items():
            ax.plot(
                coverage_data["time"],
                coverage_data["relative_overlap"],
                "o",
                label=project,
            )
        ax.set_title(f"Relative coverage of {city}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Relative coverage")
        fig.show()

    # %%
    def get_city_best_coverage(city, coverage_data, n_projects_decade=10):
        """
        Get the best coverage for a city, by selecting the n_projects_decade
        best (highest relative coverage withtin the municipality boundaries)
        projects for each decade
        """
        best_candidates = {}
        decades = []
        for project, data in coverage_data.items():
            project_year = data["time"].year
            project_decade = project_year - project_year % 10
            if project_decade not in decades:
                decades.append(project_decade)
                best_candidates[project_decade] = {"candidates": {}}
            best_candidates[project_decade]["candidates"][project] = data
        for decade, data in best_candidates.items():
            data["candidates"] = dict(
                sorted(
                    data["candidates"].items(),
                    key=lambda item: item[1]["relative_overlap"],
                    reverse=True,
                )
            )
        selected_candidates = {}
        for decade, data in best_candidates.items():
            selected_candidates[decade] = dict(
                list(data["candidates"].items())[:n_projects_decade]
            )
        return selected_candidates

    trondheim_best_coverage = get_city_best_coverage(
        "trondheim", City_overlaps_shapes["trondheim"]
    )
    # trondheim_best_coverage == trondheim_coverage_geo_plot
    print(trondheim_best_coverage)


# %% bar plot of covered area
def bar_plot_coverage(coverage_data, city):
    decades = list(coverage_data.keys())
    print(decades)
    fig, ax = plt.subplots(len(decades), 1, figsize=(10, 16))
    for i, (decade, data) in enumerate(coverage_data.items()):
        ax[i].bar(data.keys(), [i["relative_overlap"] for i in data.values()])
        plt.setp(ax[i].get_xticklabels(), fontsize=5)
        ax[i].set_title(f"{decade}")
        ax[i].set_ylim(0, 1)
    plt.tight_layout()
    fig.show()


bar_plot_coverage(trondheim_best_coverage, "trondheim")


# %%
def plot_city_coverage(city, coverage_data):
    # Load the natural earth low resolution dataset
    world = gpd.read_file(
        root_dir / "data/raw/maps/world_high_res/ne_10m_land.shp", crs=4326
    )
    # # Check the initial CRS of the world dataset
    # print("Initial CRS:", world.crs)
    # # Inspect the geometries before and after the transformation
    # print("Geometries before transformation:", world.geometry.head())
    world = world.to_crs(
        epsg=25832
    )  # for some reason 25833 does not work at all (check world plot)
    # # Check the CRS after transformation
    # print("Transformed CRS:", world.crs)
    # print("Geometries after transformation:", world.geometry.head())

    # # Plot the transformed geometries
    # fig, ax = plt.subplots( figsize=(10, 10))
    # world.plot(ax = ax)
    # plt.xlim(-1e7, 1e7)
    # plt.ylim(-0e7, 2e7)
    # plt.show()
    # plt.close()

    decades = list(coverage_data.keys())
    ncols = 3
    nrows = len(coverage_data.keys()) // ncols + 1

    city_gdf = gpd.GeoSeries(city_boundaries_shapely[city], crs=4326).to_crs(epsg=25832)
    city_extend = city_gdf.total_bounds
    city_heigh_width_ratio = (city_extend[3] - city_extend[1]) / (
        city_extend[2] - city_extend[0]
    )
    print(city_heigh_width_ratio)
    desired_aspect_ratio = 1.5  # ratio between height and width of the plot
    closest_ratio_difference = 10 * 10
    for n_cols in range(1, len(decades) + 1):
        n_rows = len(decades) // n_cols + 1
        if n_cols * (n_rows - 1) == len(decades):
            n_rows -= 1
        current_aspect_ratio = city_heigh_width_ratio * n_rows / n_cols
        ratio_difference = abs(current_aspect_ratio - desired_aspect_ratio)
        if ratio_difference < closest_ratio_difference:
            closest_ratio_difference = ratio_difference
            best_n_cols = n_cols
            best_n_rows = n_rows
    ncols = best_n_cols
    nrows = best_n_rows
    print(
        f"best_n_cols: {best_n_cols}, best_n_rows: {best_n_rows} for a total of {len(decades)} plots"
    )
    fig_width = 25
    fig_height = fig_width * desired_aspect_ratio

    # fig_height = 30
    # fig_width = fig_height /city_heigh_width_ratio

    print(f"fig_width: {fig_width}, fig_height: {fig_height}")
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    for r in range(nrows):
        for c in range(ncols):
            axs[r, c].axis("off")
    gs = GridSpec(nrows, ncols, figure=fig)
    # make ten colors from the tab10 colormap
    tab10 = cm.get_cmap("tab10")
    colors = [tab10(i) for i in range(10)]

    world_clipped = world.cx[
        city_extend[0] : city_extend[2], city_extend[1] : city_extend[3]
    ]
    # Intersect the world data with the bounding box of the city
    # world_clipped = gpd.overlay(world, city_bbox_gdf, how='intersection')
    hatch_patch = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="//", label="land", alpha=0.2
    )
    city_border_patch = Line2D(
        [0], [0], color="crimson", linewidth=1.5, label=f"{city} borders"
    )
    for i, (decade, data) in tqdm(enumerate(coverage_data.items()), total=len(decades)):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        city_gdf.plot(
            ax=ax,
            alpha=1,
            edgecolor="crimson",
            linewidth=1.5,
            facecolor="none",
        )
        handles, labels = ax.get_legend_handles_labels()
        for j, (project, pdata) in enumerate(data.items()):
            gdf = gpd.GeoSeries(pdata["geometry"], crs=4326).to_crs(epsg=25832)
            gdf.plot(
                ax=ax,
                alpha=0.3,
                edgecolor=colors[j],
                facecolor=colors[j],
            )
            project_patch = mpatches.Patch(
                color=colors[j],
                label=f"{project} (t ={pdata['time'].year},%={pdata['relative_overlap']:.2f})",
            )
            handles.append(project_patch)
        # print(gdf)

        world_clipped.plot(
            ax=ax, color="none", hatch="//", edgecolor="black", alpha=0.2
        )
        # Add the custom legend entry to the existing legend handles
        handles.append(city_border_patch)
        handles.append(hatch_patch)
        ax.legend(handles=handles, fontsize=8)

        # world_clipped.plot(ax=ax[row, col], color="none", edgecolor="black", alpha=0.2)
        ax.set_title(f"projects from the {decade}s", fontsize=20)
        # automatically set the limits
        ax.set_xlim(city_extend[0] - 0.1, city_extend[2] + 0.1)
        ax.set_ylim(city_extend[1] - 0.1, city_extend[3] + 0.1)
        # ax[row, col].legend(fontsize=7)
        # aspect ratio: equal
        # ax[row, col].set_aspect("equal")
        ax.set_axis_off()
        # break

    # deactivate the plot for the remaning tiles:
    for i in range(row * ncols + col + 1, nrows * ncols):
        fig.add_subplot(gs[i // ncols, i % ncols]).axis("off")
    # set title for entire plot
    fig.suptitle(f"Coverage of {city}", fontsize=36, y=1 + 0.2 / fig_height)
    plt.tight_layout()
    # fig.subplots_adjust(top=0.3, hspace=0.1, wspace=0.1)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.show()
    return fig, ax


plot_city_coverage("trondheim", trondheim_best_coverage)
# %%
for city in cities:
    best_coverage = get_city_best_coverage(city, City_overlaps_shapes[city])
    fig, ax = plot_city_coverage(city, best_coverage)
    # save each plot just here
    fig.savefig(
        root_dir
        / f"data/figures/matrikkel_comparison/data_selection_cities/{city}_coverage_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
