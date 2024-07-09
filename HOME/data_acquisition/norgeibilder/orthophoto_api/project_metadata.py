"""Managing the requesting of metadata from the norgeibilder API

This module contains 2 functions of metadata requests from the norgeibilder API:
- get_all_projects: Get all orthophoto projects available for export.
- get_project_metadata: Get all the metadata of the orthophoto project(s) specified.
"""

import requests
import json


def get_all_projects() -> list[str]:
    """
    Get all orthophoto projects available for export.

    Returns:
    - A list of all orthophoto projects available for export.
    """
    rest_metatdata_url = "http://tjenester.norgeibilder.no/rest/projectMetadata.ashx"
    metadata_payload = {}
    metadata_payload_json = json.dumps(metadata_payload)
    metadata_query = {"json": metadata_payload_json}
    meta_data_response = requests.get(rest_metatdata_url, params=metadata_query)

    if meta_data_response.status_code != 200:
        raise Exception(
            f"Project request failed with status code {meta_data_response.status_code}."
        )
    else:
        projects = meta_data_response.json()["ProjectList"]

    return projects


def get_project_metadata(projects: list[str], geometry: bool = False) -> dict:
    """
    Get the metadata of the orthophoto project specified.
    Seems to not work as of now (05.03.2024) - not clear if the purpose of this service
    is to get medata back for specific projects, or just to give a list of projects fitting the
    search criteria. Note that the coordinates come in the default system of EPSG25833.

    Args:
    - projects: a list of project IDs of the orthophoto to get metadata from.
    - geometry: whether to include the geometry of the orthophoto project in the metadata.

    Returns:
    - A dictionary containing the metadata of the orthophoto project.

    Raises:
    - ValueError: If the number of projects is greater than 100 (limit for the API request)
    - Exception: If the request failed (for any reason)
    """
    # Base URL
    base_url = "https://tjenester.norgeibilder.no/rest/projectMetadata.ashx"

    if len(projects) > 100:
        raise ValueError("Maximum number of projects is 100")
    projects_str = ",".join(projects)
    if geometry:
        params = {
            "request": "{Projects:'%s',ReturnMetadata:true,ReturnGeometry:true}"
            % projects_str
        }
    else:
        params = {"request": "{Projects:'%s',ReturnMetadata:true}" % projects_str}

    # Send the request
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    return response.json()
