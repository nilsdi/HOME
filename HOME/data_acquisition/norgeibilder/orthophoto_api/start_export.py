"""Manages the requests of orthophotos from the Norge i bilder API.

Contains two functions:
- start_export: Request an export of the orthophoto project specified. (causes an email!)
- save_export_job: Save the export job details to a file for later reference.
"""

import requests
from pathlib import Path
import json
import os
import time
from requests.auth import HTTPBasicAuth
from HOME.get_data_path import get_data_path

root_dir = Path(__file__).parents[4]
# print(root_dir)


def start_export(
    project: str,
    resolution: float,
    compression_method: int = 5,
    compression_value: float = 25,
    mosaic: int = 3,
    mapsheet_size: int = 2000,
    crs: int = 25832,
) -> int:
    """
    Request an export of the orthophoto project specified.
    The export exportID returned can be used to fetch the status of the export.
    User and password are taken from the geonorge_login.json file.
    Details for the settings can be found in the documentation - link below.
    https://backend-api.klienter-prod-k8s2.norgeibilder.no/swagger/index.html

    Args:
    - project (str): The project ID of the orthophoto to be exported.
    - resolution (float): The resolution of the orthophoto to be exported in meters.
    - compression_method (int): The compression method to be used for the export (see doc).
    - compression_value (float): The compression value to be used for the export (see doc).
    - mosaic (int): type of mosaic - see documentation for details.
    - mapsheet_size (float): The size of the map sheet (not sure which unit...)
    - crs (int): The coordinate reference system of the orthophoto to be exported.

    Returns:
    - int: The exportID of the export request.
    """
    export_url = "https://backend-api.klienter-prod-k8s2.norgeibilder.no/export/start"

    # Get the directory of this file
    current_dir = Path(__file__).resolve().parents[0]
    # Construct the path to the JSON file
    json_file_path = os.path.join(current_dir, "geonorge_login.json")
    # Open the JSON file
    with open(json_file_path, "r") as file:
        # Load the JSON data
        login = json.load(file)
    # create auth object
    auth_basic = HTTPBasicAuth(login["Username"], login["Password"])
    export_payload = {
        "copyEmail": "nils.dittrich@ntnu.no",
        "comment": "low scale export for HOME project",
        "cutNationalBorder": True,
        "format": 4,
        "resolution": 0.3,
        "outputWkid": crs,
        "fillColor": 0,
        "fillImage": False,
        "projects": ["Trondheim 2019"],
        "compressionMethod": compression_method,
        "compressionValue": compression_value,
        "imagemosaic": 3,
        "mapsheetSize": mapsheet_size,
        "exportFilename": f"{project.lower().replace(' ', '_') }.zip",
    }

    export_response = requests.post(export_url, auth=auth_basic, json=export_payload)

    if export_response.status_code != 200:
        raise Exception(
            f"Export request failed with status code {export_response.status_code}."
        )
    else:
        print(
            f"Export request successful with status code {export_response.status_code}."
        )
        response_json = export_response.json()
        print(response_json)
        exportID = export_response.json()["exportId"]

    return int(exportID)


# remove whitespace from name
# change coordinate system WSG84
# proof the path stuff (for servers?)


def save_export_job(
    exportID: int,
    project: str,
    resolution: float,
    compression_method: int = 5,
    compression_value: float = 25,
    mosaic: int = 3,
    mapsheet_size: int = 2000,
    crs: int = 25832,
    data_path: Path = None,
) -> None:
    """
    Save the export job details to a file for later reference.

    Args:
    - exportID (int): The exportID  from NiB.
    - project (str): The project ID of the orthophoto to be exported.
    - resolution (float): The resolution of the orthophoto to be exported in meters.
    - compression_method (int): The compression method to be used for the export (see doc).
    - compression_value (float): The compression value to be used for the export (see doc).
    - mosaic (int): type of mosaic - see documentation for details.
    - mapsheet_size (float): The size of the map sheet (not sure which unit...)
    - crs (int): The coordinate reference system of the orthophoto to be exported.
    - data_path (Path): The path to the data folder.

    Returns:
        None
    """
    if data_path is None:
        data_path = root_dir / "data"

    # current time for the file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"Export_{project.lower()}_{current_time}.json"
    file_path = os.path.join(data_path, f"temp/norgeibilder/jobids/{file_name}")

    if compression_method != 5:
        raise Exception(
            "Only LZW compression (type 5) is supported in saving the job at the moment."
        )
    export_job = {
        "exportID": exportID,
        "project": project,
        "resolution": resolution,
        "compression_method": compression_method,
        "compression_value": compression_value,
        "mosaic": mosaic,
        "mapsheet_size": mapsheet_size,
        "crs": crs,
        "date": current_time,
    }

    with open(file_path, "w") as file:
        json.dump(export_job, file)

    return
