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
bad_projects = []
for project, metadata in zip(all_projects, metadata_all_projects["ProjectMetadata"]):
    if metadata["properties"]["opprinneligbildesys"] not in ["22", "23"]:
        print(f"Project {project} has bad crs")
        bad_projects.append(project)
# bad_projects = ["Vest-Finnmark midlertidig ortofoto 2023", "Hammerfest sentrum 2022", "Nordvest Finnmark 2022"]
for project in bad_projects:
    all_projects.remove(project)
# all_projects.remove("Vest-Finnmark midlertidig ortofoto 2023")

geometries = get_project_geometry(all_projects)
# %%
geometries = [geo.to_crs(epsg=4326) for geo in geometries]
# %%

# Convert city boundaries to shapely geometries
city_boundaries_shapely = {
    city: shape(boundary) for city, boundary in municipality_boundaries.items()
}
city_boundaries_area = {
    city: city_boundary.area for city, city_boundary in city_boundaries_shapely.items()
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
# Check for overlaps and full coverage
for project, metadata, geometry_series in zip(
    all_projects, metadata_all_projects["ProjectMetadata"], geometries
):
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
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


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
            coverage_data["time"], coverage_data["relative_overlap"], "o", label=project
        )
    ax.set_title(f"Relative coverage of {city}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative coverage")
    fig.show()

# %%
