# %%
from pathlib import Path
import json
import shutil
import sys


from HOME.data_acquisition.orthophoto_api.download_project import (
    download_project,
)  # noqa

root_directory = Path(__file__).parents[2]
possible_downloads = root_directory.glob("data/temp/norgeibilder/urls/*")
possible_downloads = [p for p in possible_downloads if "." in str(p)]
print(f"current urls: {possible_downloads}.")

# %%
for current_download in possible_downloads:
    # read in json:
    job_path = root_directory / "data/temp/norgeibilder/urls/"
    with open(current_download, "r") as f:
        job_details = json.load(f)
    for key in job_details:
        print(f"{key}: {job_details[key]}")

    # download the project
    download_project(**job_details)
    # move the job to archive
    shutil.move(current_download, job_path / "used_urls" / current_download.name)

# %%
