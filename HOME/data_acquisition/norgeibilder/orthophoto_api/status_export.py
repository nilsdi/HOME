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
from requests.auth import HTTPBasicAuth

root_dir = Path(__file__).parents[4]
# print(root_dir)


def status_export(exportID: int) -> Tuple[bool, Optional[list[str]]]:
    """
    Request the status of an export job specified by the JobID.
    The status returned can be used to check if the export is complete.

    Args:
        exportID: The ID of the export request.

    Returns:
        The status of the export request. If true, the string is the url for
        the download.

    Raises:
        Exception: If the request failed (for any reason)
    """
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

    status_url = (
        f"https://backend-api.klienter-prod-k8s2.norgeibilder.no/export/{exportID}"
    )
    status_response = requests.get(status_url, auth=auth_basic)

    if status_response.status_code != 200:
        raise Exception(
            "Status request failed with status" + f"code {status_response.status_code}."
        )
    else:
        status = status_response.json()["Status"]
        if status == "complete":
            return True, status_response.json()["Urls"]
        else:
            return False, ""


def save_download_urls(
    download_urls: list[str],
    project: str,
    resolution: float,
    compression_method: int,
    compression_value: float,
    mosaic: int = 3,
    mapsheet_size: int = 2000,
    crs: int = 25832,
    date: str = None,
    data_path: Path = None,
) -> None:
    """
    Save the download url to a file for later reference.

    Args:
    - download_url (list[str]): The url to the download.
    - project (str): The name of the project (regular name including spaces)
    - resolution (float): The resolution of the orthophoto in m/px
    - compression_method (int): The compression method used encoded
    - compression_value (float): The compression value used (0-100)
    - mosaic (int): type of mosaic - see documentation for details.
    - mapsheet_size (float): The size of the map sheet (not sure which unit...)
    - crs (int): The coordinate reference system of the orthophoto to be exported.
    - date (str): The date of the export request, default is None (=> current time)
    - data_path (Path): The path to the data folder, default is None (=> HOME/data)

    Returns:
    - None

    Raises:
    - Exception: If the compression method is not supported yet
    """
    greatgrandparent_dir = Path(__file__).resolve().parents[4]

    if data_path is None:
        data_path = root_dir / "data"
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
    if not date:
        date = current_time
    export_job = {
        "download_urls": download_urls,
        "project": project,
        "resolution": resolution,
        "compression_name": compression,
        "compression_value": compression_value,
        "mosaic": mosaic,
        "mapsheet_size": mapsheet_size,
        "crs": crs,
        "date": date,
    }

    with open(file_path, "w") as file:
        json.dump(export_job, file)

    return
