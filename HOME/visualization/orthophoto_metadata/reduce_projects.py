# %% imports
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]

path_to_data = root_dir / "data" / "raw" / "orthophoto"
# list all files (not directories) in the path
metadata_files = [
    f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))
]


# the last digits in the file name is the date and time of the metadata, we want the latest
# Function to extract datetime from filename
def extract_datetime(filename):
    # Assuming the date is at the end of the filename and is in a specific format
    # Adjust the slicing as per your filename format
    date_str = filename.split("_")[-1].split(".")[
        0
    ]  # Adjust based on your filename format
    print(date_str)
    return datetime.strptime(
        date_str, "%Y%m%d%H%M%S"
    )  # Adjust the format as per your filename


# Sort files by datetime
sorted_files = sorted(metadata_files, key=extract_datetime, reverse=True)

# The newest file
newest_file = sorted_files[0]
print("Newest file:", newest_file)
print(path_to_data / newest_file)
with open(path_to_data / newest_file, "r") as f:
    metadata_all_projects = json.load(f)

# %% check the keys
print(f"we have the following keys in the metadata: {metadata_all_projects.keys()}")
print(
    f'we have the following keys in the ProjectMetadata: {metadata_all_projects["ProjectMetadata"][0].keys()}'
)
print(
    f'we have the following keys in the ProjectMetadata properties: {metadata_all_projects["ProjectMetadata"][0]["properties"].keys()}'
)
# get times and resolutions of all projects
time_list = [
    pd.to_datetime(m["properties"]["aar"])
    for m in metadata_all_projects["ProjectMetadata"]
]
time_list = pd.Series(time_list)
resolution_list = [
    float(m["properties"]["pixelstorrelse"])
    for m in metadata_all_projects["ProjectMetadata"]
]
areas = np.array(
    [
        float(m["properties"]["st_area(shape)"])
        for m in metadata_all_projects["ProjectMetadata"]
    ]
)
# %% shape
metadata_all_projects["ProjectMetadata"][0]["geometry"]
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union

geometries = []
for project in metadata_all_projects["ProjectMetadata"]:
    geometries.append(shape(project["geometry"]))

# %%
projects_by_year = {y: [] for y in range(1935, 2024)}
for project, year, res, area in zip(
    metadata_all_projects["ProjectList"], time_list, resolution_list, areas
):
    if res <= 0.5:
        projects_by_year[year.year].append(project)
print(projects_by_year)
for year, projects in projects_by_year.items():
    shapes = []
    for project in projects:
        pass
# %%
