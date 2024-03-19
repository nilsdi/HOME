# %%
from pathlib import Path
import json
import shutil

from get_orthophoto.download_project import download_project

root_directory = Path(__file__).parents[2]
possible_downloads = root_directory.glob("data/temp/urls/*")
possible_downloads = [p for p in possible_downloads if "." in str(p)]
print(f'current urls: {possible_downloads}.')
if len(possible_downloads) > 0:
    current_download = possible_downloads[0]
    # read in json:
    job_path = root_directory / "data/temp/urls/"
    with open(job_path / current_download, "r") as f:
        job_details = json.load(f)
    for key in job_details:
        print(f'{key}: {job_details[key]}')

    # download the project
    download_project(**job_details)
    # move the job to archive
    shutil.move(job_path / current_download, job_path / "used_urls/" /
                current_download)

# %%
