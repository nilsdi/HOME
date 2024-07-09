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
from HOME.get_data_path import get_data_path

root_dir = Path(__file__).parents[4]
# print(root_dir)
data_path = get_data_path(root_dir)
# print(data_path)


def start_export(
    project: str,
    resolution: float,
    format: int = 4,
    compression_method: int = 5,
    compression_value: float = 50,
    mosaic: bool = False,
) -> int:
    """
    Request an export of the orthophoto project specified.
    The export JobID returned can be used to fetch the status of the export.
    User and password are taken from the geonorge_login.json file.

    Args:
    - project (str): The project ID of the orthophoto to be exported.
    - resolution (float): The resolution of the orthophoto to be exported in meters.
    - format (int): The format of the orthophoto to be exported (see documentation for details).
    - compression_method (int): The compression method to be used for the export (see doc).
    - compression_value (float): The compression value to be used for the export (see doc).
    - mosaic (bool): Whether to export the orthophoto as a mosaic or not - not yet implemented.

    Returns:
    - int: The JobID of the export request.
    """
    rest_export_url = "https://tjenester.norgeibilder.no/rest/startExport.ashx"

    # Get the directory of this file
    current_dir = Path(__file__).resolve().parents[0]

    # Construct the path to the JSON file
    json_file_path = os.path.join(current_dir, "geonorge_login.json")

    # Open the JSON file
    with open(json_file_path, "r") as file:
        # Load the JSON data
        login = json.load(file)

    export_payload = {
        "Username": login["Username"],
        "Password": login["Password"],
        "CopyEmail": "nils.dittrich@ntnu.no",  # so both Daniel and I get an email!
        "Format": format,
        "Resolution": resolution,
        "CompressionMethod": str(compression_method),
        "CompressionValue": str(compression_value),
        "Projects": project,
        "Imagemosaic": 2,  # 2 means no mosaic
        "support_files": 1,  # medata or not - we choose yes
    }
    # we need to send the payload as a json in a request calling it the request
    export_payload_json = json.dumps(export_payload)
    export_query = {"request": export_payload_json}
    export_response = requests.get(rest_export_url, params=export_query)

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
        JobID = export_response.json()["JobID"]

    return JobID


# remove whitespace from name
# change coordinate system WSG84
# proof the path stuff (for servers?)


def save_export_job(
    JobID: int,
    project: str,
    resolution: float,
    compression_method: int,
    compression_value: float,
    mosaic: bool,
) -> None:
    """
    Save the export job details to a file for later reference.

    Args:
    - JobID (int): The JobID of the export request.
    - project (str): The name of the orthophoto project to be exported.
    - resolution (float): The resolution of the orthophoto to be exported in meters.
    - compression_method (int): The compression method to be used for the export.
    - compression_value (float): The compression value to be used for the export.
    - mosaic (bool): Whether to export the orthophoto as a mosaic or not.

    Returns:
    - None
    """
    # greatgrandparent_dir = Path(__file__).resolve().parents[4]

    # current time for the file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"Export_{project.lower()}_{current_time}.json"
    file_path = os.path.join(data_path, f"temp/norgeibilder/jobids/{file_name}")

    if compression_method != 5:
        raise Exception(
            "Only LZW compression (type 5) is supported in saving the job at the moment."
        )

    export_job = {
        "JobID": JobID,
        "project": project,
        "resolution": resolution,
        "compression_method": compression_method,
        "compression_value": compression_value,
        "mosaic": mosaic,
    }

    with open(file_path, "w") as file:
        json.dump(export_job, file)

    return
