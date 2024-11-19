# %%
import json
import pandas as pd
from pathlib import Path
import pickle
import shutil

from HOME.ML_prediction.preprocessing import (
    step_01_tile_generation,
    step_02_make_text_file,
)
from HOME.ML_prediction.prediction import predict

from HOME.visualization.ML_prediction.visual_inspection.plot_prediction_input import (
    plot_prediction_input,
)

from HOME.ML_prediction.postprocessing import (
    step_01_reassembling_tiles,
    step_02_regularization_clean,
)

from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


def run_project(
    project_name, project_details, tile_size=512, res=0.3, labels=False, gdf_omrade=None
):
    """
    Run the prediction pipeline for a single project

    Arguments:
    project_name: str, name of the project
    project_details: dict, details of the project
    tile_size: int, size of the tiles
    res: float, resolution of the orthophotos
    labels: bool, whether to use labels or not
    gdf_omrade: geopandas dataframe, labels for the buildings
    """

    channels = project_details[project_name]["channels"]

    # compression = f"i_{compression_name}_{compression_value}"

    print(f"Starting prediction for {project_name}")

    # Step 1: Generate tiles
    tile_key = step_01_tile_generation.tile_generation(
        project_name,
        tile_size,
        res,
        overlap_rate=0,
        labels=labels,
        gdf_omrade=gdf_omrade,
    )

    # Step 2: Make text file
    step_02_make_text_file.make_text_file(project_name=project_name, tile_key=tile_key)

    # Step 3: Predict
    BW = channels == "BW"
    prediction_key = predict.predict_and_eval(
        project_name, tile_key, BW=BW, evaluate=labels
    )

    # Step 4: Reassemble tiles
    n_tiles_edge = 10
    n_overlap = 1
    assembly_key, geotiff_extends = step_01_reassembling_tiles.reassemble_tiles(
        project_name, prediction_key, n_tiles_edge, n_overlap
    )

    # Step 5: Regularize
    polygon_id = step_02_regularization_clean.regularize(project_name, assembly_key)

    project_details[project_name]["status"] = "assembled"
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "w"
    ) as file:
        json.dump(project_details, file)

    # Step 4: (Optional) Visualize a few tiles.
    # plot_prediction_input(project_name, n_tiles=4, save=True, show=True)


def main(list_of_projects: list, labels: bool = False, tile_size=512, res=0.3):
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

    for project_name in list_of_projects:
        if project_details[project_name]["status"] == "pending":
            projects_to_run.append(project_name)

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
        run_project(
            project_name,
            project_details=project_details,
            tile_size=tile_size,
            res=res,
            labels=labels,
            gdf_omrade=gdf_omrade,
        )


def clean_all(project_name, tile_id, labels=False):
    tiling_path = (
        data_path / f"ML_prediction/topredict/image/{project_name}/tiles_{tile_id}"
    )
    prediction_path = (
        data_path / f"ML_prediction/predictions/image/{project_name}/tiles_{tile_id}"
    )
    assembly_path = data_path / f"ML_prediction/large_tiles/tiles_{tile_id}/"
    regularization_path = data_path / f"ML_prediction/polygons/tiles_{tile_id}/"

    # remove the folders
    shutil.rmtree(tiling_path)
    shutil.rmtree(prediction_path)
    shutil.rmtree(assembly_path)
    shutil.rmtree(regularization_path)

    if labels:
        label_path = (
            data_path / f"ML_prediction/topredict/label/{project_name}/tiles_{tile_id}"
        )
        shutil.rmtree(label_path)


# %%
if __name__ == "__main__":
    list_of_projects = ["trondheim_2023"]
    main(list_of_projects=list_of_projects, labels=True)
    # print("did something")

# %%
