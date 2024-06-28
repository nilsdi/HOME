"""
downloads all orthophotos which were requested and finished processing.

We check the status of all processing jobs using the `status_export` function and the
JobID stored in the JSON files in the `data/temp/norgeibilder/jobids/` directory.
If the job is complete, it saves the download URL using the `save_download_url` function,
and moves the job file to the `used_jobids` directory.
The files are then downloaded  to the `data/raw/orthophotos/` directory.

This script continues to check the status of all jobs in a loop until all jobs are complete.
"""

# %% imports
from HOME.data_acquisition.norgeibilder.orthophoto_api.status_export import (
    status_export,
    save_download_url,
)
from HOME.data_acquisition.norgeibilder.orthophoto_api.download_project import (
    download_project,
)
from pathlib import Path
import os
import json
import shutil
import time
from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)
# %% check job status


def check_all_jobs() -> bool:
    jobids_dir = data_path / "temp/norgeibilder/jobids/"
    jobs = [j for j in os.listdir(jobids_dir) if "." in j]
    print(f"available jobs: {jobs}.")
    all_jobs_complete = True
    for current_job in jobs:
        job_path = jobids_dir / current_job
        with open(job_path, "r") as f:
            job_details = json.load(f)
        complete, url = status_export(job_details["JobID"])
        print(
            f'The status of job {job_details["JobID"]} is {complete}, the url is {url}.'
        )
        if complete:
            print("Export complete, JobID moved to archive")
            job_details.pop("JobID")
            save_download_url(url, **job_details)
            # move the job to archive
            shutil.move(job_path, jobids_dir / "used_jobids/" / current_job)
        else:
            all_jobs_complete = False
    return all_jobs_complete


# check once manually
all_jobs_complete = check_all_jobs()


# immediatly download the finished jobs
def download_all_possible():
    urls_dir = data_path / "temp/norgeibilder/urls/"
    possible_downloads = [
        d for d in os.listdir(urls_dir) if "." in d
    ]  # jobs = [j for j in os.listdir(jobids_dir) if "." in j]
    # possible_downloads_files = root_dir.glob(data_path / "temp/norgeibilder/urls/*")
    # possible_downloads = [p for p in possible_downloads_files if "." in str(p)]
    print(f"current urls: {possible_downloads}.")
    for current_download in possible_downloads:
        # read in json:
        job_path = data_path / "temp/norgeibilder/urls/"
        with open(job_path / current_download, "r") as f:
            job_details = json.load(f)
        for key in job_details:
            print(f"{key}: {job_details[key]}")
        # download the project
        download_project(**job_details)
        # move the job to archive
        shutil.move(
            job_path / current_download, job_path / "used_urls" / current_download
        )
    return


# download once manually
download_all_possible()


while not all_jobs_complete:
    print(
        "Not all jobs are complete. Waiting for 60 minutes before" + " checking again."
    )
    time.sleep(3600)  # Wait for 60 minutes
    all_jobs_complete = check_all_jobs()
    download_all_possible()


# %%
