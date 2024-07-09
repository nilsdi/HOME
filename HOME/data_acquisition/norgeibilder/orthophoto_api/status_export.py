"""Manages the checking on production of requested orthophotos from the Norgeibilder

Contains 2 functions:
- status_export: Request the status of an export job
- save_download_url: Manage the url and job id json
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Tuple, Optional
from HOME.get_data_path import get_data_path

root_dir = Path(__file__).parents[4]
# print(root_dir)
data_path = get_data_path(root_dir)
# print(data_path)


def status_export(JobID: int) -> Tuple[bool, Optional[str]]:
    """
    Request the status of an export job specified by the JobID.
    The status returned can be used to check if the export is complete.

    Args:
    - JobID: The JobID of the export request.

    Returns:
    - The status of the export request. If true, the string is the url for
        the download.

    Raises:
    - Exception: If the request failed (for any reason)
    """
    rest_status_url = "https://tjenester.norgeibilder.no/rest/" + "exportStatus.ashx"
    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the JSON file
    json_file_path = os.path.join(script_dir, "geonorge_login.json")

    # Open the JSON file
    with open(json_file_path, "r") as file:
        # Load the JSON data
        login = json.load(file)

    status_payload = {
        "Username": login["Username"],
        "Password": login["Password"],
        "JobID": JobID,
    }
    status_payload_json = json.dumps(status_payload)
    status_query = {"request": status_payload_json}
    status_response = requests.get(rest_status_url, params=status_query)

    if status_response.status_code != 200:
        raise Exception(
            "Status request failed with status" + f"code {status_response.status_code}."
        )
    else:
        status = status_response.json()["Status"]
        if status == "complete":
            return True, status_response.json()["Url"]
        else:
            return False, ""


def save_download_url(
    download_url: str,
    project: str,
    resolution: float,
    compression_method: int,
    compression_value: float,
    mosaic: bool,
) -> None:
    """
    Save the download url to a file for later reference.

    Args:
    - download_url (str): The url to the download.
    - project (str): The name of the project (regular name including spaces)
    - resolution (float): The resolution of the orthophoto in m/px
    - compression_method (int): The compression method used encoded
    - compression_value (float): The compression value used (0-100)
    - mosaic (bool): Whether the orthophoto is a mosaic or not

    Returns:
    - None

    Raises:
    - Exception: If the compression method is not supported yet
    """
    greatgrandparent_dir = Path(__file__).resolve().parents[4]

    # current time for the file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"Download_{project.lower()}_{current_time}.json"
    file_path = os.path.join(data_path, f"temp/norgeibilder/urls/{file_name}")

    # this entire block should be changed - we should have a different
    # variable for the compresssion name.
    if compression_method == 5:
        compression = "lzw"
    elif compression_method == "lzw":
        compression = "lzw"
    else:
        raise Exception(
            "Only LZW compression (type 5) is supported in saving the job"
            + "at the moment."
        )

    export_job = {
        "download_url": download_url,
        "project": project,
        "resolution": resolution,
        "compression_name": compression,
        "compression_value": compression_value,
        "mosaic": mosaic,
    }

    with open(file_path, "w") as file:
        json.dump(export_job, file)

    return
