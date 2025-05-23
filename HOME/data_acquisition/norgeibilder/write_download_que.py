""" Prepare the jsons that say what we want to download

Contains a main block only. 
We specify some details for download and cross check with the metadata.
we then save the details in the download_que folder.
"""

# %% imports
from pathlib import Path
import json
import os
from datetime import datetime
import time
from HOME.utils.get_project_metadata import get_project_details

root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)

# %% import metadata for the projects to check availabilitly
if __name__ == "__main__":
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

    # specify the download details
    resolution = 0.3
    compression_method = 5
    compression_value = 25
    mosaic = 3

    project_names = [
        "Oslo vår 2023",
        "Bergen 2022",
        "Moss 2022",
        "Stavanger 2023",
        "Trondheim MOF 2023",
        "Tromsø midlertidig ortofoto 2023",
    ]
    project_names = [
        "Trondheim 1971",
        "Trondheim 1977",
        "Trondheim 1979",
        "Trondheim 1982",
        "Trondheim 1984",
        "Trondheim 1985",
        "Trondheim 1991",
        "Trondheim 1992",
        "Trondheim 1994",
        "Trondheim 1999",
        "Trondheim 2017",
        "Trondheim 2019",
        "Trondheim kommune 2020",
        "Trondheim kommune 2021",
        "Trondheim kommune 2022",
        "Trondheim-Meldal 1964",
        "Skaun 1957",
        "Trondheim-Gauldal 1947",
    ]

    project_names = [
        "Trondheim 2013",
        "Trondheim 2014",
        # "Trondheim 2015",
        # "Trondheim 2016",
        # "Trondheim 2017",
        # "Trondheim kommune rektifisert 2018",
        "Trondheim 2019",
        "Trondheim kommune 2020",
        "Trondheim kommune 2021",
        "Trondheim kommune 2022",
        # "Trondheim kommune 2023 MOF",
    ]
    project_names = [
        "Trondheim 2023",
    ]
    que_path = root_dir / "data/temp/norgeibilder/download_que/"
    # current time for the file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # read in the json for project management:
    #  create the jsons
    for project in project_names:
        details = get_project_details(project)
        resolution_available = details["original_resolution"]
        if resolution_available > resolution:
            print(
                f"{project} is not available in the resolution {resolution}, but only in {resolution_available}"
            )
            continue

        download_details = {
            "project": project,
            "resolution": resolution,
            "compression_method": compression_method,
            "compression_value": compression_value,
            "mosaic": mosaic,
            "mapsheet_size": 2000,
            "crs": 25833,
        }

        with open(
            que_path
            / f'project_{project.lower().replace(" ", "_")}_time_{current_time}.json',
            "w",
        ) as f:
            json.dump(download_details, f)

# %%
