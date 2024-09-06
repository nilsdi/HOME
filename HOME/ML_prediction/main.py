# %%
import json
import pandas as pd
from pathlib import Path
import logging
import torch

from HOME.ML_prediction.preprocessing import (
    step_01_tile_generation,
    step_02_make_text_file,
)
from HOME.ML_prediction.prediction import predict

from HOME.visualization.ML_prediction.visual_inspection.plot_prediction_input import (
    plot_prediction_input,
)

# from src.ML_prediction.postprocessing import (
#     step_01_reassembling_tiles,
#     step_02_regularization,
# )

from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


def main(list_of_projects: list):
    """
    Main function to run the prediction pipeline

    Arguments:
    list_of_projects: list, list of project names to run the prediction pipeline
    """

    # root_dir = Path(__file__).parents[2]

    with open(
        data_path / "ML_prediction/project_log/project_details.json", "r"
    ) as file:
        project_details = json.load(file)

    if list_of_projects == ["all"]:
        list_of_projects = list(project_details.keys())

    projects_to_run = []
    pred_res = 0  # the resolution for which we open the prediction mask
    for project_name in list_of_projects:
        if project_details[project_name]["status"] == "downloaded":
            projects_to_run.append(project_name)
        pred_res = project_details[project_name]["resolution"]

    # load prediction mask
    prediction_mask = pd.read_csv(
        data_path / f"ML_prediction/prediction_mask/prediction_mask_{pred_res}_512.csv",
        index_col=0,
    )
    prediction_mask.columns = prediction_mask.columns.astype(int)
    prediction_mask.index = prediction_mask.index.astype(int)

    for project_name in projects_to_run:
        res, compression_name, compression_value, channels = (
            project_details[project_name]["resolution"],
            project_details[project_name]["compression_name"],
            project_details[project_name]["compression_value"],
            project_details[project_name]["channels"],
        )
        if res != pred_res:
            raise ValueError(
                "The resolution of the project does not match across projects."
            )

        compression = f"i_{compression_name}_{compression_value}"

        print(f"Starting prediction for {project_name}")

        # Step 1: Generate tiles
        step_01_tile_generation.tile_generation(
            project_name=project_name,
            res=res,
            compression=compression,
            prediction_mask=prediction_mask,
        )

        # Step 2: Make text file
        step_02_make_text_file.make_text_file(
            project_name=project_name, res=res, compression=compression
        )

        # Step 3: Predict
        year = int(project_name.split("_")[-1])
        BW = channels == "BW"
        predict.predict(
            project_name=project_name, res=res, compression=compression, BW=BW
        )
        # Step 4: Reassemble tiles
        # step_01_reassembling_tiles(project_name)

        # Step 5: Regularize
        # step_02_regularization(project_name)

        project_details[project_name]["status"] = "predicted"
        with open(
            data_path / "ML_prediction/project_log/project_details.json", "w"
        ) as file:
            json.dump(project_details, file)

        # Step 4: (Optional) Visualize a few tiles.
        # plot_prediction_input(project_name, n_tiles=4, save=True, show=True)


# %%
if __name__ == "__main__":
    list_of_projects = [
        "trondheim_kommune_2021",
        "trondheim_kommune_2022",
    ]
    main(list_of_projects=list_of_projects)
    # print("did something")

# %%
