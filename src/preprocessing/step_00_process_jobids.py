from src.get_orthophoto.status_export import status_export
from src.get_orthophoto.status_export import save_download_url
from pathlib import Path
import os
import json
import shutil
import time

root_dir = Path(__file__).resolve().parents[2]
jobids_dir = root_dir / "data/temp/jobids/"
jobs = [j for j in os.listdir(jobids_dir) if "." in j]
print(f'available jobs: {jobs}.')

for current_job in jobs:
    job_path = jobids_dir / current_job
    with open(job_path, "r") as f:
        job_details = json.load(f)
    complete, url = status_export(job_details["JobID"])
    print(f'The job is done: {complete}, the url is {url}.')
    if complete:
        print("Export complete, JobID moved to archive")
        job_details.pop("JobID")
        save_download_url(url, **job_details)
        # move the job to archive
        shutil.move(job_path, jobids_dir / "used_jobids/" / current_job)


# %%
all_jobs_complete = False
while not all_jobs_complete:
    all_jobs_complete = True
    for current_job in jobs:
        job_path = jobids_dir / current_job
        with open(job_path, "r") as f:
            job_details = json.load(f)
        complete, url = status_export(job_details["JobID"])
        print(f'The job is done: {complete}, the url is {url}.')
        if complete:
            print("Export complete, JobID moved to archive")
            job_details.pop("JobID")
            save_download_url(url, **job_details)
            # move the job to archive
            shutil.move(job_path, jobids_dir / "used_jobids/" / current_job)
        else:
            all_jobs_complete = False
    else:
        print("Not all jobs are complete. Waiting for 20 minutes before" +
              " checking again.")
        time.sleep(1200)  # Wait for 10 minutes
