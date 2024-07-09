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
    resolution = 0.2
    compression_method = 5
    compression_value = 25
    mosaic = False

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

    project_names = []
    que_path = root_dir / "data/temp/norgeibilder/download_que/"

    # read in the json for project management:
    with open(
        root_dir / "data/ML_prediction/project_log/project_details.json", "r"
    ) as file:
        project_details = json.load(file)
    #  create the jsons
    for project in project_names:
        # check if the project is already in the que
        if (que_path / f'{project.lower().replace(" ", "_")}.json').exists():
            print(f"{project} is already in the que")
            continue

        # check if the project is available in the resolution (from metadata)
        meta_data_index = metadata_all_projects["ProjectList"].index(project)
        properties = metadata_all_projects["ProjectMetadata"][meta_data_index][
            "properties"
        ]
        resolution_available = float(properties["pixelstorrelse"])
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
        }
        project_details = {
            "status": "queued",
            "channels": None,
            "resolution": resolution,
            "compression_name": compression_method,
            "compression_value": compression_value,
        }
        project_details[project.lower().replace(" ", "_")] = project_details

        with open(que_path / f'{project.lower().replace(" ", "_")}.json', "w") as f:
            json.dump(download_details, f)

    #  save the updated project details
    with open(
        root_dir / "data/ML_prediction/project_log/project_details.json", "w"
    ) as file:
        json.dump(project_details, file, indent=4)

# %%
