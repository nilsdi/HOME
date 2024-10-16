"""Checking available projects and actually download the data

Contains a two functions and a main block:
- check_all_jobs: Checks the status to see if some project is ready for download
- download_all_possible: Downloads the projects that are ready for download
- main block: running the above functions in a loop until all projects are downloaded:
check the status of all processing jobs using the `status_export` function and the JobID 
stored in the JSON files in the `data/temp/norgeibilder/jobids/` directory. If the job is 
complete, it saves the download URL using the `save_download_url` function, 
and moves the job file to the `used_jobids` directory.
The files are then downloaded  to the data/ 'raw/orthophotos/` directory.
This script continues to check the status of all jobs in a loop until all jobs are complete.
"""

# %% imports
from HOME.data_acquisition.norgeibilder.orthophoto_api.status_export import (
    status_export,
    save_download_urls,
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
# data_path = get_data_path(root_dir)


# print(data_path)
# %% check job status
def check_all_jobs(data_path: Path = None) -> bool:
    """
    Goes through the jobid jsons in the temp folder in data to check status

    Args:
        data_path: Path to the data folder, default is None (=> HOME/data)

    Returns:
        all_jobs_complete: True if all jobs are complete, False otherwise
    """
    if data_path is None:
        data_path = root_dir / "data"
    jobids_dir = data_path / "temp/norgeibilder/exportIDs/"
    jobs = [j for j in os.listdir(jobids_dir) if "." in j]
    print(f"available jobs: {jobs}.")
    all_jobs_complete = True
    for current_job in jobs:
        job_path = jobids_dir / current_job
        with open(job_path, "r") as f:
            job_details = json.load(f)
        complete, urls = status_export(job_details["exportID"])
        print(
            f'The status of job {job_details["exportID"]} is {complete}, the urls are {urls}.'
        )
        if complete:
            print("Export complete, exportID moved to archive")
            job_details.pop("exportID")
            save_download_urls(urls, **job_details)
            # print(f"saved the following: {urls} ({job_details}).")
            # move the job to archive
            shutil.move(job_path, jobids_dir / "used_exportIDs/" / current_job)
        else:
            all_jobs_complete = False
    return all_jobs_complete


def download_all_possible(data_path: Path = None):
    """
    Goes through the list of urls in the data/temp folder and downloads the projects

    Returns:
    - None
    """
    if data_path is None:
        data_path = root_dir / "data"
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


if __name__ == "__main__":
    # check once manually
    all_jobs_complete = check_all_jobs()
    # immediatly download the finished jobs (once manually)
    download_all_possible()
    # afterwards we check regularly
    while not all_jobs_complete:
        print(
            "Not all jobs are complete. Waiting for 60 minutes before checking again."
        )
        time.sleep(3600)  # Wait for 60 minutes
        all_jobs_complete = check_all_jobs()
        download_all_possible()


# %%
