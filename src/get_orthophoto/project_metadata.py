import requests
import json

def get_all_projects()->list[str]:
    '''
    Get all orthophoto projects available for export.

    Returns:
    - A list of all orthophoto projects available for export.
    '''
    rest_metatdata_url = "http://tjenester.norgeibilder.no/rest/projectMetadata.ashx" 
    metadata_payload = {}
    metadata_payload_json = json.dumps(metadata_payload)
    metadata_query = {"json": metadata_payload_json}
    meta_data_response = requests.get(rest_metatdata_url, params = metadata_query)

    if meta_data_response.status_code != 200:
        raise Exception(f"Project request failed with status code {meta_data_response.status_code}.")
    else:
        projects = meta_data_response.json()["ProjectList"]

    return projects

def get_project_metadata(project:str)->dict:
    '''
    Get the metadata of the orthophoto project specified.
    Seems to not work as of now (05.03.2024) - not clear if the purpose of this service
    is to get medata back for specific projects, or just to give a list of projects fitting the
    search criteria.

    Arguments:
    - project: The project ID of the orthophoto to get metadata from.

    Returns:
    - A dictionary containing the metadata of the orthophoto project.
    '''
    rest_metatdata_url = "http://tjenester.norgeibilder.no/rest/projectMetadata.ashx" 
    metadata_payload = {
        "Project": project,
        "ReturnMedata":'True'
        }
    metadata_payload_json = json.dumps(metadata_payload)
    metadata_query = {"json": metadata_payload_json}
    meta_data_response = requests.get(rest_metatdata_url, params = metadata_query)

    if meta_data_response.status_code != 200:
        raise Exception(f"Metadata request failed with status code {meta_data_response.status_code}.")
    else:
        metadata = meta_data_response.json()

    return metadata