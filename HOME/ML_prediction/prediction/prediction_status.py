"""
Tracks he status based on the number of tiles in to_predict and predictions folders.
"""

# %%
import os
from pathlib import Path
import shutil
import numpy as np
import json

root_dir = Path(__file__).parents[3]


# %%
def prediction_status(project_name):
    # Assuming root_dir is already defined and is a Path object
    project_details_path = (
        root_dir / "data/ML_prediction/project_log/project_details.json"
    )

    # Open the file and load its contents into a variable
    with open(project_details_path, "r") as file:
        project_details = json.load(file)
    res, compression_name, compression_value = (
        project_details[project_name]["resolution"],
        project_details[project_name]["compression_name"],
        project_details[project_name]["compression_value"],
    )
    # to predict and prediction folder
    to_predict_folder = (
        root_dir
        / f"data/ML_prediction/topredict/image/res_{res}/{project_name}/i_{compression_name}_{compression_value}"
    )
    prediction_folder = (
        root_dir
        / f"data/ML_prediction/predictions/res_{res}/{project_name}/i_{compression_name}_{compression_value}"
    )

    # count the number of tiles in to_predict and predictions folders
    to_predict_tiles = len(os.listdir(to_predict_folder))
    predicted_tiles = len(os.listdir(prediction_folder))
    print(
        f"Number of tiles to predict: {to_predict_tiles}, Number of tiles predicted: {predicted_tiles}, meaning {np.round(predicted_tiles/to_predict_tiles*100,1)}% are done"
    )


if __name__ == "__main__":
    prediction_status("trondheim_kommune_2021")
# %%
