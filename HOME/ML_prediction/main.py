# %%
import json
import pandas as pd
from pathlib import Path
import logging
import torch
import pickle

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


def main(list_of_projects: list, labels: bool = False):
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

    projects_to_run = []  # list of IDs of projects to run (integers) in the future,
    # names of projects right now
    pred_res = 0  # the resolution for which we open the prediction mask
    for project_name in list_of_projects:
        if project_details[project_name]["status"] == "downloaded":
            projects_to_run.append(project_name)
        pred_res = project_details[project_name]["resolution"]

    if labels:
        path_label = (
            data_path / "raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
        )
        with open(path_label, "rb") as f:
            gdf_omrade = pickle.load(f)
        buildings_year = pd.read_csv(
            data_path / "raw/FKB_bygning/buildings.csv", index_col=0
        )
        gdf_omrade = gdf_omrade.merge(
            buildings_year, left_on="bygningsnummer", right_index=True, how="left"
        )
    else:
        gdf_omrade = None

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
        tile_key = step_01_tile_generation.tile_generation(
            project_name=project_name,
            res=res,
            compression=compression,
            prediction_type="buildings",
            tile_size=512,
            overlap_rate=0,
            labels=labels,
            gdf_omrade=gdf_omrade,
        )

        # Step 2: Make text file
        step_02_make_text_file.make_text_file(
            project_name=project_name, res=res, compression=compression
        )

        # Step 3: Predict
        BW = channels == "BW"
        predict.predict_and_eval(
            project_name=project_name,
            key=tile_key,
            res=res,
            compression=compression,
            BW=BW,
            evaluate=labels,
            batchsize=8,
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
        # "trondheim_1999",
        # "trondheim_kommune_2022",
        "trondheim_mof_2023"
    ]
    main(list_of_projects=list_of_projects, labels=True)
    # print("did something")

# %%
