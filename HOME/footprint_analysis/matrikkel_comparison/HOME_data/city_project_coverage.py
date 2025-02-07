"""
Checking the coverage of the projects we used for each city vs the municipal boundaries
"""

# %%
from pathlib import Path
import json
from shapely.geometry import shape
from datetime import datetime
import matplotlib.pyplot as plt


from HOME.utils.get_project_metadata import (
    get_project_metadata,
    get_project_geometry,
    _get_newest_metadata,
)
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)

root_dir = Path(__file__).parents[3]

# %%
# list the projects by city:
if __name__ == "__main__":
    cities = ["trondheim", "oslo", "bergen", "stavanger", "tromsø"]

    # %%
    # get the boundaries for the cities
    municipality_boundaries = {city: get_municipal_boundaries(city) for city in cities}
    # also check which projects would be interesting from a coverage perspective
    metadata_all_projects = _get_newest_metadata()
    print(metadata_all_projects.keys())
    # %%
    all_projects = metadata_all_projects["ProjectList"][:]
    # exclude some with bad crs:
    # bad_projects = []
    # for project, metadata in zip(
    #     all_projects, metadata_all_projects["ProjectMetadata"]
    # ):
    #     if metadata["properties"]["opprinneligbildesys"] not in ["22", "23"]:
    #         print(f"Project {project} has bad crs: {metadata['properties']['opprinneligbildesys']}")
    #         bad_projects.append(project)
    # bad_projects = ["Vest-Finnmark midlertidig ortofoto 2023", "Hammerfest sentrum 2022", "Nordvest Finnmark 2022"]
    # for project in bad_projects:
    #     all_projects.remove(project)
    # all_projects.remove("Vest-Finnmark midlertidig ortofoto 2023")
    geometries = get_project_geometry(all_projects)
    # %%
    # geometries = [geo.to_crs(crs = 25833) for geo in geometries]
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
        / "footprint_analysis/matrikkel_comparison/HOME_data/city_project_coverage.json",
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
    def get_city_best_coverage(city, coverage_data):
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
            selected_candidates[decade] = dict(list(data["candidates"].items())[:10])
        return selected_candidates

    trondheim_best_coverage = get_city_best_coverage(
        "trondheim", City_overlaps_shapes["trondheim"]
    )
    trondheim_best_coverage == trondheim_coverage_geo_plot
    print(trondheim_best_coverage)
    # %%
    trondheim_coverage_geo_plot_candidates = {}
    decades = []
    for project, data in City_overlaps_shapes["trondheim"].items():
        project_year = data["time"].year
        project_decade = project_year - project_year % 10
        # print(f"Project {project} from {project_year} in decade {project_decade}")
        if project_decade not in decades:
            decades.append(project_decade)
            trondheim_coverage_geo_plot_candidates[project_decade] = {"candidates": {}}
        trondheim_coverage_geo_plot_candidates[project_decade]["candidates"][
            project
        ] = {
            "overlap": data["relative_overlap"],
            "geometry": data["geometry"],
            "time": data["time"],
        }

    # sort the projects within each decade by relative overlap
    for decade, data in trondheim_coverage_geo_plot_candidates.items():
        data["candidates"] = dict(
            sorted(
                data["candidates"].items(),
                key=lambda item: item[1]["overlap"],
                reverse=True,
            )
        )

    print(trondheim_coverage_geo_plot)

    # make the plot data with the first 10 candidates only
    trondheim_coverage_geo_plot = {}
    for decade, data in trondheim_coverage_geo_plot_candidates.items():
        trondheim_coverage_geo_plot[decade] = dict(
            list(data["candidates"].items())[:10]
        )
    print(trondheim_coverage_geo_plot)


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

fig, ax = plt.subplots(len(decades), 1, figsize=(10, 16))
for i, (decade, data) in enumerate(trondheim_coverage_geo_plot.items()):
    ax[i].bar(data.keys(), [i["overlap"] for i in data.values()])
    # set the text size of the x-ticks to 5
    plt.setp(ax[i].get_xticklabels(), fontsize=5)
    ax[i].set_title(f"{decade}")
    ax[i].set_ylim(0, 1)
# ax[i].set_xlabel("Project", rotation=45)
# ax[i].set_ylabel("Relative coverage")
# %% area plot with overlapping projects
ncols = 3
nrows = len(decades) // ncols + 1
fig, ax = plt.subplots(nrows, ncols, figsize=(20, 20))
import geopandas as gpd
from tqdm import tqdm
from matplotlib import cm

# Make ten colors from the tab10 colormap
tab10 = cm.get_cmap("tab10")
colors = [tab10(i) for i in range(10)]

city_gdf = gpd.GeoSeries(city_boundaries_shapely["trondheim"])
citty_extend = city_gdf.total_bounds
print(citty_extend)

for i, (decade, data) in tqdm(
    enumerate(trondheim_coverage_geo_plot.items()), total=len(decades)
):
    for j, (project, pdata) in enumerate(data.items()):
        # print(f"Plotting {project} from {decade}, data: {pdata}")
        row = i // ncols
        col = i % ncols
        gdf = gpd.GeoSeries(pdata["geometry"])
        gdf.plot(ax=ax[row, col], alpha=0.3, edgecolor="k", facecolor=colors[j])
        # add to legend:
        ax[row, col].plot(
            [],
            [],
            color=colors[j],
            label=f"{project} ({pdata['time'].year}, {pdata['overlap']:.2f})",
        )
    # print the trondheim shade

    city_gdf.plot(
        ax=ax[row, col], alpha=1, edgecolor="crimson", linewidth=2, facecolor="none"
    )
    ax[row, col].set_title(f"decade:{decade}")
    # automatically set the limits
    ax[row, col].set_xlim(citty_extend[0] - 0.1, citty_extend[2] + 0.1)
    ax[row, col].set_ylim(citty_extend[1] - 0.1, citty_extend[3] + 0.1)
    # ax[row, col].set_xlim(9.9, 10.8)
    ax[row, col].set_ylim(63.1, 63.6)
    ax[row, col].legend(fontsize=4)
    # aspect ratio: equal
    # ax[row, col].set_aspect("equal")
    ax[row, col].axis("off")

# deactivate the plot for the remaning tiles:
for i in range(row * ncols + col + 1, nrows * ncols):
    ax[i // ncols, i % ncols].axis("off")
plt.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.1)


# %%
def plot_city_coverage(city, coverage_data):
    ncols = 3
    nrows = len(coverage_data.keys()) // ncols + 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(40, 40))

    # make ten colors from the tab10 colormap
    tab10 = cm.get_cmap("tab10")
    colors = [tab10(i) for i in range(10)]

    city_gdf = gpd.GeoSeries(city_boundaries_shapely[city])
    city_extend = city_gdf.total_bounds
    for i, (decade, data) in tqdm(enumerate(coverage_data.items()), total=len(decades)):
        for j, (project, pdata) in enumerate(data.items()):
            row = i // ncols
            col = i % ncols
            gdf = gpd.GeoSeries(pdata["geometry"])
            gdf.plot(ax=ax[row, col], alpha=0.3, edgecolor="k", facecolor=colors[j])
            ax[row, col].plot(
                [],
                [],
                color=colors[j],
                label=f"{project} (t ={pdata['time'].year},%={pdata['relative_overlap']:.2f})",
            )
        # print(gdf)
        city_gdf.plot(
            ax=ax[row, col], alpha=1, edgecolor="crimson", linewidth=2, facecolor="none"
        )
        ax[row, col].set_title(f"decade:{decade}", fontsize=16)
        # automatically set the limits
        ax[row, col].set_xlim(city_extend[0] - 0.1, city_extend[2] + 0.1)
        ax[row, col].set_ylim(city_extend[1] - 0.1, city_extend[3] + 0.1)
        ax[row, col].legend(fontsize=7)
        # aspect ratio: equal
        # ax[row, col].set_aspect("equal")
        ax[row, col].axis("off")

    # deactivate the plot for the remaning tiles:
    for i in range(row * ncols + col + 1, nrows * ncols):
        ax[i // ncols, i % ncols].axis("off")
    # set title for entire plot
    fig.suptitle(f"Coverage of {city}", fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
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
        / f"footprint_analysis/matrikkel_comparison/HOME_data/{city}_coverage_plot.png"
    )
