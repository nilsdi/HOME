'''
downloads all orthophotos which were requested and finished processing.

We check the status of all processing jobs using the `status_export` function and the
JobID stored in the JSON files in the `data/temp/norgeibilder/jobids/` directory. 
If the job is complete, it saves the download URL using the `save_download_url` function, 
and moves the job file to the `used_jobids` directory.
The files are then downloaded  to the `data/raw/orthophotos/` directory.

This script continues to check the status of all jobs in a loop until all jobs are complete.
'''
#%% imports
from src.data_acquisition.orthophoto_api.status_export import status_export, save_download_url
from src.data_acquisition.orthophoto_api.download_project import download_project
from pathlib import Path
import os
import json
import shutil
import time

root_dir = Path(__file__).resolve().parents[2]
#%% check job status



def check_all_jobs() -> bool:
    jobids_dir = root_dir / "data/temp/norgeibilder/jobids/"
    jobs = [j for j in os.listdir(jobids_dir) if "." in j]
    print(f'available jobs: {jobs}.')
    all_jobs_complete = True
    for current_job in jobs:
        job_path = jobids_dir / current_job
        with open(job_path, "r") as f:
            job_details = json.load(f)
        complete, url = status_export(job_details["JobID"])
        print(f'The status of job {job_details["JobID"]} is {complete}, the url is {url}.')
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
    possible_downloads_files = root_dir.glob("data/temp/norgeibilder/urls/*")
    possible_downloads = [p for p in possible_downloads_files if "." in str(p)]
    print(f"current urls: {possible_downloads}.")
    for current_download in possible_downloads:
        # read in json:
        job_path = root_dir / "data/temp/norgeibilder/urls/"
        with open(current_download, "r") as f:
            job_details = json.load(f)
        for key in job_details:
            print(f"{key}: {job_details[key]}")
        # download the project
        download_project(**job_details)
        # move the job to archive
        shutil.move(current_download, job_path / "used_urls" / current_download.name)
    return
download_all_possible()


while not all_jobs_complete:
    print("Not all jobs are complete. Waiting for 20 minutes before" +
              " checking again.")
    time.sleep(1200)  # Wait for 20 minutes
    all_jobs_complete = check_all_jobs()
    download_all_possible()
        


# %%
