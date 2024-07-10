"""
Sets up all the gitignored directories needed for the orthophoto project.
"""

# %% imports
import os
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parents[1]
print(root_dir)

# %% make the directories

# make data
data_dir = root_dir / "data_eptx"
os.makedirs(data_dir, exist_ok=True)

# make doc, figures, ML_model, ML_training, ML prediction, raw and temp subfolders
doc_dir = data_dir / "doc"
os.makedirs(doc_dir, exist_ok=True)
figures_dir = data_dir / "figures"
os.makedirs(figures_dir, exist_ok=True)
ML_model_dir = data_dir / "ML_model"
os.makedirs(ML_model_dir, exist_ok=True)
ML_training_dir = data_dir / "ML_training"
os.makedirs(ML_training_dir, exist_ok=True)
ML_prediction_dir = data_dir / "ML_prediction"
os.makedirs(ML_prediction_dir, exist_ok=True)
raw_dir = data_dir / "raw"
os.makedirs(raw_dir, exist_ok=True)
temp_dir = data_dir / "temp"
os.makedirs(temp_dir, exist_ok=True)

# make the subfolders for the raw data - FKB_bygning, FKB_veg, maps, matrikkel, orthophotos
FKB_bygning_dir = raw_dir / "FKB_bygning"
os.makedirs(FKB_bygning_dir, exist_ok=True)  # subfolders er gdbs
FKB_veg_dir = raw_dir / "FKB_veg"
os.makedirs(FKB_veg_dir, exist_ok=True)  # subfolders er gdbs
maps_dir = raw_dir / "maps"
os.makedirs(maps_dir, exist_ok=True)
Norway_boundaries_dir = raw_dir / "Norway_boundaries"
os.makedirs(Norway_boundaries_dir, exist_ok=True)
matrikkel_dir = raw_dir / "matrikkel"
os.makedirs(matrikkel_dir, exist_ok=True)
municipality_pickles_dir = matrikkel_dir / "municipality_pickles"
os.makedirs(municipality_pickles_dir, exist_ok=True)
orthophoto_dir = raw_dir / "orthophoto"
os.makedirs(orthophoto_dir, exist_ok=True)  # rest is made while downloading.


# make the subfolders for the temp data - norgeibilder: jobids, urls, download_que
norgeibilder_dir = temp_dir / "norgeibilder"
os.makedirs(norgeibilder_dir, exist_ok=True)
jobids_dir = norgeibilder_dir / "jobids"
os.makedirs(jobids_dir, exist_ok=True)
used_jobids_dir = jobids_dir / "used_jobids"
os.makedirs(used_jobids_dir, exist_ok=True)
urls_dir = norgeibilder_dir / "urls"
os.makedirs(urls_dir, exist_ok=True)
used_urls_dir = urls_dir / "used_urls"
os.makedirs(used_urls_dir, exist_ok=True)
download_que_dir = norgeibilder_dir / "download_que"
os.makedirs(download_que_dir, exist_ok=True)
old_download_que_dir = download_que_dir / "old_download_que"
os.makedirs(old_download_que_dir, exist_ok=True)

# subfolders in the ML_prediction folder:
# dataset, prediction_mask, predictions, project_log, topredict, validation
dataset_dir = ML_prediction_dir / "dataset"
os.makedirs(dataset_dir, exist_ok=True)
prediction_mask_dir = ML_prediction_dir / "prediction_mask"
os.makedirs(prediction_mask_dir, exist_ok=True)
predictions_dir = ML_prediction_dir / "predictions"
os.makedirs(predictions_dir, exist_ok=True)
project_log_dir = ML_prediction_dir / "project_log"
os.makedirs(project_log_dir, exist_ok=True)
topredict_dir = ML_prediction_dir / "topredict"
os.makedirs(topredict_dir, exist_ok=True)
validation_dir = ML_prediction_dir / "validation"
os.makedirs(validation_dir, exist_ok=True)

# %% initialize some files
# project_details.json
project_details = {}
# check if file is already there
if not (project_log_dir / "project_details.json").exists():
    with open(project_log_dir / "project_details.json", "w") as f:
        json.dump(project_details, f)

# %%
print(
    f"very long messssssssssssssssssssssssssssssss"
    + "sssssssssssssssssssssssssssssssssssssssssssssssssssage"
)
print(f"Very long message" + "dkjsdkj      ")

testi = {"testi": "testi", "grape": "grape", "stuff": "stuff"}
print(testi)
