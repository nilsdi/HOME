"""Adding information to the project details after the download

Contains a main block only. meant as a temporary file, eventually all this should be done
automatically by the functions that download the data and use it.
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
from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)
orthophoto_dir = data_path / "raw" / "orthophoto"
# %% load the metadata
if __name__ == "__main__":
    metadata_files = [
        f
        for f in os.listdir(orthophoto_dir)
        if os.path.isfile(os.path.join(orthophoto_dir, f))
    ]

    # the last digits in the file name is the date and time of the metadata, we want the latest
    # Function to extract datetime from filename
    def extract_datetime(filename: str):
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
    print(orthophoto_dir / newest_file)
    with open(orthophoto_dir / newest_file, "r") as f:
        metadata_all_projects = json.load(f)

    # %% open the file and change it
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "r"
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
        properties = metadata_all_projects["ProjectMetadata"][meta_data_index][
            "properties"
        ]
        colour_data = properties["bildekategori"]
        colour_data = picture_types[colour_data]
        project_details[project_name]["channels"] = colour_data

    print(project_details)
    # %% save the file
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "w"
    ) as file:
        json.dump(project_details, file, indent=4)
