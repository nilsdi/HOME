# %%
import pandas as pd
from pathlib import Path
import pickle
import shutil
import argparse
import contextlib
import os
import sys

from HOME.ML_prediction.preprocessing import (
    step_01_tile_generation,
    step_02_make_text_file,
)
from HOME.ML_prediction.prediction import predict
from HOME.ML_prediction.postprocessing import (
    step_01_reassembling_tiles,
    step_02_regularization,
)
from HOME.data_acquisition.norgeibilder.orthophoto_api.download_originals import (
    request_download,
    download_original_NIB,
)
from HOME.get_data_path import get_data_path
from HOME.utils.project_paths import (
    save_project_details,
    load_project_details,
    get_assembling_details,
)
from HOME.data_acquisition.norgeibilder.add_project_details import add_project_details
from HOME.utils.check_project_stage import check_list_stage

# %%
# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


# %%
@contextlib.contextmanager
def suppress_output(filepath=None):
    """Suppress all stdout and stderr output temporarily."""
    if filepath is None:
        filepath = os.devnull
    with open(filepath, "a") as logfile:  # Append mode to keep logs
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = logfile
            sys.stderr = logfile
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_project(
    project_name,
    project_details,
    tile_size=512,
    res=0.3,
    labels=False,
    gdf_omrade=None,
    remove_download=False,
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
    tile_key, labels = step_01_tile_generation.tile_generation(
        project_name,
        tile_size,
        res,
        overlap_rate=0,
        labels=labels,
        gdf_omrade=gdf_omrade,
    )

    if remove_download:
        shutil.rmtree(data_path / f"raw/orthophoto/originals/{project_name}")

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
    polygon_id = step_02_regularization.regularize(
        project_name, assembly_key, geotiff_extends
    )

    # Step 4: (Optional) Visualize a few tiles.
    # plot_prediction_input(project_name, n_tiles=4, save=True, show=True)
    return


def download(project_name, project_details):
    downloaded = False
    if project_details[project_name]["status"] == "pending":
        project_id = project_details[project_name]["id"]
        download_urls = request_download(project_id)
        download_original_NIB(download_urls, project_name)
        downloaded = True
    return downloaded


def process(
    list_of_projects: list,
    labels: bool = False,
    tile_size=512,
    res=0.3,
    remove_download=False,
):
    """
    Main function to run the prediction pipeline

    Arguments:
    list_of_projects: list, list of project names to run the prediction pipeline
    """

    # root_dir = Path(__file__).parents[2]
    with suppress_output():
        project_details = add_project_details(list_of_projects)

    projects_to_run = []  # list of IDs of projects to run (integers) in the future,
    # names of projects right now

    for project_name in list_of_projects:
        if project_details[project_name]["status"] != "processed":
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

    log_folder = data_path / "metadata_log/execution_log"

    for project_name in projects_to_run:
        if len(os.listdir(data_path / f"raw/orthophoto/originals/")) < 5:
            try:
                print(f"Downloading {project_name}")
                with suppress_output(log_folder / f"{project_name}_download.log"):
                    downloaded = download(project_name, project_details)
                if downloaded:
                    project_details = load_project_details(data_path)
                    project_details[project_name]["status"] = "downloaded"
                    save_project_details(project_details, data_path)
                try:
                    print(f"Processing {project_name}")
                    with suppress_output(log_folder / f"{project_name}_process.log"):
                        run_project(
                            project_name,
                            project_details=project_details,
                            tile_size=tile_size,
                            res=res,
                            labels=labels,
                            gdf_omrade=gdf_omrade,
                            remove_download=remove_download,
                        )
                    project_details = load_project_details(data_path)
                    project_details[project_name]["status"] = "processed"
                    save_project_details(project_details, data_path)
                except Exception as e:
                    print(f"Error processing {project_name}: {e}")
            except Exception as e:
                print(f"Error downloading {project_name}: {e}")


def reprocess(
    labels: bool = False,
    tile_size=512,
    res=0.3,
    remove_download=False,
):
    """
    Main function to re-run the prediction pipeline

    Arguments:
    list_of_projects: list, list of project names to run the prediction pipeline
    labels: bool, whether to use labels or not
    tile_size: int, size of the tiles
    res: float, resolution of the orthophotos
    remove_download: bool, whether to remove the downloaded files after tiling
    """
    project_details = load_project_details(data_path)
    list_of_projects = list(project_details.keys())
    log_folder = data_path / "metadata_log/execution_log"

    stages = check_list_stage(list_of_projects)

    if labels:
        totile = False
        for project_name in list_of_projects:
            if stages[project_name]["stage"] in ["downloaded"]:
                totile = True
                break
        if totile:
            path_label = (
                data_path
                / "raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
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
    else:
        gdf_omrade = None

    for project_name in list_of_projects:
        status = stages[project_name]["stage"]

        channels = project_details[project_name]["channels"]
        BW = channels == "BW"

        topredict = False
        if status == "downloaded":
            print(f"{project_name}: Tiling")
            with suppress_output(log_folder / f"{project_name}_process.log"):
                tile_key = step_01_tile_generation.tile_generation(
                    project_name,
                    tile_size,
                    res,
                    overlap_rate=0,
                    labels=labels,
                    gdf_omrade=gdf_omrade,
                )
            topredict = True

            if remove_download:
                shutil.rmtree(data_path / f"raw/orthophoto/originals/{project_name}")

        assemble = False
        if status == "tiled" or topredict:
            if not topredict:
                tile_key = stages[project_name]["ids"][-1]
            print(f"{project_name}: Predicting")
            with suppress_output(log_folder / f"{project_name}_process.log"):
                step_02_make_text_file.make_text_file(
                    project_name=project_name, tile_key=tile_key
                )
                prediction_key = predict.predict_and_eval(
                    project_name, tile_key, BW=BW, evaluate=labels
                )
            assemble = True

        if status == "predicted" or assemble or status == "assembled":
            if not assemble:
                if status == "assembled":
                    # Projects should be reassembled to get the geotiff extends
                    assemble_key = stages[project_name]["ids"][-1]
                    prediction_key = get_assembling_details(assemble_key, data_path)[
                        "prediction_key"
                    ]
                else:
                    prediction_key = stages[project_name]["ids"][-1]

            n_tiles_edge = 10
            n_overlap = 1
            print(f"{project_name}: Reassembling")
            with suppress_output(log_folder / f"{project_name}_process.log"):
                assembly_key, geotiff_extends = (
                    step_01_reassembling_tiles.reassemble_tiles(
                        project_name, prediction_key, n_tiles_edge, n_overlap
                    )
                )
                polygon_id = step_02_regularization.regularize(
                    project_name, assembly_key, geotiff_extends
                )
            print(f"{project_name} processed")
            project_details = load_project_details(data_path)
            project_details[project_name]["status"] = "processed"
            save_project_details(project_details, data_path)

        elif status == "processed":
            print(f"{project_name} already processed")


def clean_all(project_name, tile_id, labels=False):
    tiling_path = (
        data_path / f"ML_prediction/topredict/image/{project_name}/tiles_{tile_id}"
    )
    prediction_path = (
        data_path / f"ML_prediction/predictions/{project_name}/tiles_{tile_id}"
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
    parser = argparse.ArgumentParser(description="Run the ML prediction pipeline")
    parser.add_argument(
        "--projects",
        nargs="+",
        help="List of projects to run the prediction pipeline",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Whether to use labels for the prediction",
        default=False,
    )
    parser.add_argument(
        "--remove_download",
        action="store_true",
        help="Whether to remove the downloaded files after prediction",
        default=False,
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Whether to reprocess the projects",
        default=False,
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Size of the tiles",
    )
    parser.add_argument(
        "--res",
        type=float,
        default=0.3,
        help="Resolution of the tiles",
    )
    args = parser.parse_args()

    if args.reprocess:
        reprocess(
            labels=args.labels,
            tile_size=args.tile_size,
            res=args.res,
            remove_download=args.remove_download,
        )
    else:
        assert (
            args.projects is not None
        ), "Please provide a list of projects with --projects"
        process(
            list_of_projects=args.projects,
            labels=args.labels,
            remove_download=args.remove_download,
            tile_size=args.tile_size,
            res=args.res,
        )

# %%
