import requests
import json
import time
from pathlib import Path


def status_export(JobID: int) -> tuple[bool, str]:
    '''
    Request the status of an export job specified by the JobID.
    The status returned can be used to check if the export is complete.

    Arguments:
    - JobID: The JobID of the export request.

    Returns:
    - The status of the export request. If true, the string is the url for
        the download.
    '''
    rest_status_url = 'https://tjenester.norgeibilder.no/rest/' + \
        'exportStatus.ashx'
    status_payload = {
        "Username": "UNTNU_MULDAN",
        "Password": "GeoNorge2024",
        "JobID": JobID
    }
    status_payload_json = json.dumps(status_payload)
    status_query = {"request": status_payload_json}
    status_response = requests.get(rest_status_url, params=status_query)

    if status_response.status_code != 200:
        raise Exception(
            "Status request failed with status" +
            f"code {status_response.status_code}.")
    else:
        status = status_response.json()['Status']
        if status == "complete":
            return True, status_response.json()["Url"]
        else:
            return False, ''


def save_download_url(download_url: str, project: str, resolution: float,
                      compression_method: int, compression_value: float,
                      mosaic: bool) -> None:
    '''
    Save the download url to a file for later reference.
    '''
    grandparent_dir = Path(__file__).resolve().parents[2]

    # current time for the file name
    current_time = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"Download_{project.lower()}_{current_time}.json"
    file_path = grandparent_dir / "data/temp/urls/" / file_name

    # this entire block should be changed - we should have a different
    # variable for the compresssion name.
    if compression_method == 5:
        compression = "lzw"
    elif compression_method == "lzw":
        compression = "lzw"
    else:
        raise Exception(
            "Only LZW compression (type 5) is supported in saving the job" +
            "at the moment.")

    export_job = {
        "download_url": download_url,
        "project": project,
        "resolution": resolution,
        "compression_name": compression,
        "compression_value": compression_value,
        "mosaic": mosaic
    }

    with open(file_path, 'w') as file:
        json.dump(export_job, file)

    return
