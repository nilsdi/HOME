import json
import pandas as pd
from pathlib import Path
import logging
import torch

from src.ML_prediction.preprocessing import (
    step_01_tile_generation,
    step_02_make_text_file,
)
from src.ML_prediction.prediction import predict

# from src.ML_prediction.postprocessing import (
#     step_01_reassembling_tiles,
#     step_02_regularization,
# )


def main(list_of_projects:list):
    """
    Main function to run the prediction pipeline

    Arguments:
    list_of_projects: list, list of project names to run the prediction pipeline
    """

    root_dir = Path(__file__).parents[2]
    prediction_mask = pd.read_csv(
        root_dir / "data/ML_prediction/prediction_mask/prediction_mask.csv", index_col=0
    )
    prediction_mask.columns = prediction_mask.columns.astype(int)
    prediction_mask.index = prediction_mask.index.astype(int)

    with open(
        root_dir / "data/ML_prediction/project_log/project_details.json", "r"
    ) as file:
        project_details = json.load(file)

    list_of_projects = list(project_details.keys())

    projects_to_run = []
    for project_name in list_of_projects:
        if project_details[project_name]["status"] == "downloaded":
            projects_to_run.append(project_name)

    for project_name in projects_to_run:
        res, compression_name, compression_value = (
            project_details[project_name]["resolution"],
            project_details[project_name]["compression_name"],
            project_details[project_name]["compression_value"],
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
        BW = year < 1995
        predict.predict(
            project_name=project_name, res=res, compression=compression, BW=BW
        )
        # Step 4: (Optional) Visualize a few tiles.

        # Step 4: Reassemble tiles
        # step_01_reassembling_tiles(project_name)

        # Step 5: Regularize
        # step_02_regularization(project_name)

        project_details[project_name]["status"] = "predicted"
        with open(
            root_dir / "data/ML_prediction/project_log/project_details.json", "w"
        ) as file:
            json.dump(project_details, file)


if __name__ == "__main__":
    list_of_projects = ['trondheim_kommune_2020']
    main(list_of_projects=list_of_projects)
