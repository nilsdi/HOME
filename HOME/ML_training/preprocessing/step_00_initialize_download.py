"""
This script is responsible for initializing the download process for orthophotos

It takes in donwload details from downloads at `data/temp/norgeibilder/download_que/'
The downloads that are specified there are then started using the `start_export` function.
The JobID of the download is then saved at `data/temp/norgeibilder/jobids/`
We then check the status of all processing jobs using the `status_export` function.
If the job is complete, it saves the download URL using the `save_download_url` function,
and moves the job file to the `used_jobids` directory.

This script continues to check the status of all jobs in a loop until all jobs are complete.

Dependencies: (if optional files are not provided, the script doesn't do anything, but works)
- Functions from `status_export` module: `status_export`, `save_download_url`
- JSON file with download details in `data/temp/norgeibilder/download_que/` dir (optional)
- JSON files with job details in `data/temp/norgeibilder/jobids/` directory (optional)
"""

from HOME.data_acquisition.orthophoto_api.start_export import (
    start_export,
    save_export_job,
)
from HOME.data_acquisition.orthophoto_api.status_export import (
    status_export,
    save_download_url,
)
from pathlib import Path
import os
import json
import shutil
import time

root_dir = Path(__file__).resolve().parents[2]
export_que_dir = root_dir / "data/temp/norgeibilder/exports/"
exports = [
    d for d in os.listdir(export_que_dir) if ".json" in d
]  # get all plain json files
for current_export in exports:
    export_path = export_que_dir / current_export
    with open(export_path, "r") as f:
        export_details = json.load(f)
    JobID = start_export(**export_details)
    save_export_job(JobID, **export_details)
    # move the job to jobids directory
    shutil.move(export_path, export_que_dir / "used_exports/" / current_export)


jobids_dir = root_dir / "data/temp/norgeibilder/jobids/"
jobs = [j for j in os.listdir(jobids_dir) if "." in j]
print(f"available jobs: {jobs}.")
all_jobs_complete = True
for current_job in jobs:
    job_path = jobids_dir / current_job
    with open(job_path, "r") as f:
        job_details = json.load(f)
    complete, url = status_export(job_details["JobID"])
    print(f"The job is done: {complete}, the url is {url}.")
    if complete:
        print("Export complete, JobID moved to archive")
        job_details.pop("JobID")
        save_download_url(url, **job_details)
        # move the job to archive
        shutil.move(job_path, jobids_dir / "used_jobids/" / current_job)
    else:
        all_jobs_complete = False


# %%

while not all_jobs_complete:
    all_jobs_complete = True
    for current_job in jobs:
        job_path = jobids_dir / current_job
        with open(job_path, "r") as f:
            job_details = json.load(f)
        complete, url = status_export(job_details["JobID"])
        print(f"The job is done: {complete}, the url is {url}.")
        if complete:
            print("Export complete, JobID moved to archive")
            job_details.pop("JobID")
            save_download_url(url, **job_details)
            # move the job to archive
            shutil.move(job_path, jobids_dir / "used_jobids/" / current_job)
        else:
            all_jobs_complete = False
    else:
        print(
            "Not all jobs are complete. Waiting for 20 minutes before"
            + " checking again."
        )
        time.sleep(1200)  # Wait for 20 minutes
