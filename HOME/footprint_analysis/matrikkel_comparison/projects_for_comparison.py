# %%
from pathlib import Path
import json
from shapely.geometry import shape


from HOME.utils.get_project_metadata import (
    get_project_metadata,
    get_project_geometry,
    _get_newest_metadata,
)
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)

root_dir = Path(__file__).parents[3]
# print(root_dir)

# check all project_names in the metadata log

polygonization_log_path = root_dir / "data/metadata_log/predictions_log.json"
with open(polygonization_log_path) as f:
    polygonization_log = json.load(f)

project_names = []
for ID, log in polygonization_log.items():
    project_names.append(log["project_name"])

project_names = list(set(project_names))
print(project_names)

# %%
# list the projects by city:
cities = ["trondheim", "oslo", "bergen", "stavanger", "tromsø"]

for city in cities:
    print(f"Projects for {city}:")
    for project_name in project_names:
        if city in project_name.lower():
            print(project_name)
    print("\n")

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

City_overlaps = {city: [] for city in cities}
City_full_coverage = {city: [] for city in cities}
# Check for overlaps and full coverage
for project, geometry_series in zip(all_projects, geometries):
    for city, city_boundary in city_boundaries_shapely.items():
        # print(f'Checking project {project} in city {city}')
        intersects = False
        completly_covers = False
        for geometry in geometry_series:
            if geometry.intersects(city_boundary):
                intersects = True

            if geometry.contains(city_boundary):
                completly_covers = True
        if intersects:
            City_overlaps[city].append(project)
        if completly_covers:
            City_full_coverage[city].append(project)
        # print(f'Intersects: {intersects}, completly covers: {completly_covers}')
print(City_overlaps)
print(City_full_coverage)
# %%
