"""
For adding information to the project details after the fact.
"""

# %% imports
import requests
import os
import zipfile
from pathlib import Path
from osgeo import gdal, osr
from tqdm import tqdm
import json
from datetime import datetime

root_dir = Path(__file__).parents[3]
print(f"root_dir: {root_dir}")
# load in metadata:
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

# %% open the file and change it
with open(
    root_dir / "data/ML_prediction/project_log/project_details.json", "r"
) as file:
    project_details = json.load(file)

picture_types = {
    "1": "IR",
    "2": "BW",
    "3": "RGB",
    "4": "RGBI",
}
# alter the list so it fits our naming scheme.
MD_project_list = [
    s.lower().replace(" ", "_") for s in metadata_all_projects["ProjectList"]
]
for project_name in project_details.keys():
    # we have to al
    meta_data_index = MD_project_list.index(project_name)
    properties = metadata_all_projects["ProjectMetadata"][meta_data_index]["properties"]
    colour_data = properties["bildekategori"]
    colour_data = picture_types[colour_data]
    project_details[project_name]["channels"] = colour_data

print(project_details)
# %% save the file
with open(
    root_dir / "data/ML_prediction/project_log/project_details.json", "w"
) as file:
    json.dump(project_details, file, indent=4)
# %%
