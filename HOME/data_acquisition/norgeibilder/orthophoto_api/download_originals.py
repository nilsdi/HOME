# %%
import requests
import os
import json
from requests.auth import HTTPBasicAuth
from pathlib import Path
from tqdm import tqdm

root_dir = Path(__file__).parents[4]


def request_download(project_id: int, purpose: str = "testing") -> str:
    original_export_url = f"https://backend-api.klienter-prod-k8s2.norgeibilder.no/export/originalExport/{project_id}"
    original_export_payload = {
        "copyEmail": "nils.dittrich@ntnu.no",
        "comment": "testing swagger API",
        "sendEmail": False,
    }

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

    original_export_response = requests.post(
        original_export_url, auth=auth_basic, json=original_export_payload
    )
    if original_export_response.status_code != 200:
        raise Exception(
            f"Export reported an unexpected status code {original_export_response.status_code}."
        )
    response_dict = json.loads(original_export_response.text)

    if not response_dict["success"]:
        raise Exception(
            f"Export request failed with status code {original_export_response.status_code}."
        )
    return response_dict["files"]


def download_original_NIB(
    download_urls: list[dict[str]], project_name: str, data_path: str = None
):
    if data_path is None:
        data_path = root_dir / "data"

    save_dir = data_path / "raw/orthophoto/originals" / project_name
    os.makedirs(save_dir, exist_ok=True)

    for details in tqdm(download_urls):
        filename = details["name"].lower()
        # Check that not already downloaded
        if not (save_dir / filename).exists():
            save_path = save_dir / filename
            url = details["url"]
            response = requests.get(url)
            with open(save_path, "wb") as file:
                file.write(response.content)


# %% test runs

if __name__ == "__main__":
    project_id = 4251
    download_urls = request_download(project_id)
    download_original_NIB(download_urls, "trondheim_2023")
    print("Download complete.")
