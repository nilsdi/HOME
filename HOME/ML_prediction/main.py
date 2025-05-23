# %%
import pandas as pd
from pathlib import Path
import pickle
import shutil
import argparse
import contextlib
import os
import sys
import time
import json
import traceback
import subprocess
import numpy as np
from datetime import datetime

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
from HOME.utils.get_project_metadata import get_project_details
from HOME.utils.project_paths import (
    get_assembling_details,
    get_tile_ids,
    get_prediction_ids,
    get_assemble_ids,
    get_polygon_ids,
)
from HOME.utils.check_project_stage import check_list_stage
from HOME.utils.directory_size import get_directory_size_human_readable

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
        except Exception as e:
            # Log the full traceback to the file
            logfile.write("An exception occurred:\n")
            traceback.print_exc(file=logfile)
            raise  # Re-raise the exception after logging
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_unique_log_filename(log_folder, project_name, base_name="download.log"):
    """
    Generate a unique log file name by appending a number if the file already exists.

    Args:
        log_folder (Path): Path to the folder where the log file will be saved.
        project_name (str): Name of the project.
        base_name (str): Base name of the log file.

    Returns:
        Path: A unique log file path.
    """
    log_file = log_folder / f"{project_name}_{base_name}"
    counter = 1
    while log_file.exists():
        log_file = (
            log_folder / f"{project_name}_{base_name.split('.')[0]}_{counter}.log"
        )
        counter += 1
    return log_file


def get_directory_size(directory):
    """Get size of directory in GB"""
    total = 0
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    return total / (1024 * 1024 * 1024)  # Convert bytes to GB


def run_project(
    project_name,
    project_details,
    tile_size=512,
    res=0.3,
    labels=False,
    gdf_omrade=None,
    remove_download=False,
) -> dict:
    """
    Run the prediction pipeline for a single project

    Args:
        project_name: str, name of the project
        project_details: dict, details of the project from the metadata file
        tile_size: int, size of the tiles
        res: float, resolution of the orthophotos
        labels: bool, whether to use labels or not
        gdf_omrade: geopandas dataframe, labels for the buildings

    Returns:
        dict, runtime of the different steps
    """

    channels = project_details["bandwidth"]

    # compression = f"i_{compression_name}_{compression_value}"

    print(f"Starting prediction for {project_name}")
    processing_times = {}
    # Step 1: Generate tiles
    t0 = time.time()
    tile_key, labels = step_01_tile_generation.tile_generation(
        project_name,
        tile_size,
        res,
        overlap_rate=0,
        labels=labels,
        gdf_omrade=gdf_omrade,
    )
    t1 = time.time()
    processing_times["tile_generation"] = t1 - t0
    if remove_download:
        shutil.rmtree(data_path / f"raw/orthophoto/originals/{project_name}")
        print(
            f"Removed downloaded files for {project_name} in {data_path / f'raw/orthophoto/originals/{project_name}'}"
        )

    # Step 2: Make text file
    t2 = time.time()
    step_02_make_text_file.make_text_file(project_name=project_name, tile_key=tile_key)
    t3 = time.time()
    processing_times["make_text_file"] = t3 - t2
    # Step 3: Predict
    t4 = time.time()
    BW = channels == "BW"
    prediction_key = predict.predict_and_eval(
        project_name, tile_key, BW=BW, evaluate=labels
    )
    t5 = time.time()
    processing_times["prediction"] = t5 - t4

    # Step 4: Reassemble tiles
    n_tiles_edge = 10
    n_overlap = 1
    assembly_key, geotiff_extends = step_01_reassembling_tiles.reassemble_tiles(
        project_name, prediction_key, n_tiles_edge, n_overlap
    )
    t6 = time.time()
    processing_times["reassembling"] = t6 - t5

    # Step 5: Regularize
    polygon_id = step_02_regularization.regularize(
        project_name, assembly_key, geotiff_extends
    )
    t7 = time.time()
    processing_times["regularization"] = t7 - t6

    # Step 4: (Optional) Visualize a few tiles.
    # plot_prediction_input(project_name, n_tiles=4, save=True, show=True)
    return processing_times


def download(project_name, project_details):
    project_id = project_details["id"]
    download_urls = request_download(project_id)
    download_original_NIB(download_urls, project_name)
    return


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

    # check prediction log if project had been predicted before (should in the future also consider settings)
    prediction_log_path = data_path / "metadata_log/predictions_log.json"
    with open(prediction_log_path, "r") as file:
        predictions_log = json.load(file)
    prediction_ids = list(predictions_log.keys())
    predicted_projects = [
        predictions_log[prediction_id]["project_name"]
        for prediction_id in prediction_ids
    ]

    projects_to_run = []  # list of IDs of projects to run (integers) in the future,
    # names of projects right now

    for project_name in list_of_projects:
        if project_name not in predicted_projects:
            projects_to_run.append(project_name)
        else:
            print(
                f"{project_name} has already been predicted with ID {prediction_ids[predicted_projects.index(project_name)]}, we will not run it again."
            )

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
    projects_details = get_project_details(projects_to_run)

    for project_name in projects_to_run:
        with open(data_path / "metadata_log/prediction_main_runtime.json", "r") as file:
            runtimes = json.load(file)
            runtimes[project_name] = {}
        with open(data_path / "metadata_log/download_size.json", "r") as file:
            download_size = json.load(file)
            download_size[project_name] = {}
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Processing project "{project_name} ({current_time})"')
        project_details = projects_details[project_name]
        # check if the project is already downloaded:
        project_downloaded = False
        project_data_folders = os.listdir(data_path / f"raw/orthophoto/originals/")
        du_originals = get_directory_size_human_readable(
            data_path / f"raw/orthophoto/originals/", summary=True, human_readable=False
        )
        print("current disk use of originals folder: ", du_originals)
        if project_name in project_data_folders:
            print(f"Project {project_name} already downloaded")
            project_downloaded = True

        elif (
            int(du_originals) < 10**9
        ):  # if the size of the originals folder is less than 1TB prior to the download, we can download more
            try:
                print(f"Downloading {project_name}")
                # check if there already is a download log file
                log_file = get_unique_log_filename(
                    log_folder, project_name, base_name="download.log"
                )
                # print(f"Log file: {log_file}")
                with suppress_output(log_file):
                    download_start_time = time.time()
                    download(project_name, project_details)
                    project_downloaded = True
                    download_end_time = time.time()
                    runtimes[project_name]["download"] = (
                        download_end_time - download_start_time
                    )
                    download_size[project_name]["size"] = get_directory_size(
                        data_path / f"raw/orthophoto/originals/{project_name}"
                    )
                with open(data_path / "metadata_log/download_size.json", "w") as file:
                    json.dump(download_size, file)
            except Exception as e:
                print(f"Error downloading {project_name}: {e}")
        else:
            raise Exception(
                f"Not enough space to proceed with download of {project_name}. "
                f"Current disk use of {data_path / 'raw/orthophoto/originals/'} is {np.round(int(du_originals)/10**6,0)} GB."
            )
        if project_downloaded:
            try:
                print(f"Processing {project_name}")
                log_file = get_unique_log_filename(
                    log_folder, project_name, base_name="process.log"
                )
                with suppress_output(log_file):
                    processing_times = run_project(
                        project_name,
                        project_details=project_details,
                        tile_size=tile_size,
                        res=res,
                        labels=labels,
                        gdf_omrade=gdf_omrade,
                        remove_download=remove_download,
                    )
                for key, value in processing_times.items():
                    runtimes[project_name][key] = value

                with open(
                    data_path / "metadata_log/prediction_main_runtime.json", "w"
                ) as file:
                    json.dump(runtimes, file)
            except Exception as e:
                print(f"Error processing {project_name}: {e}")


def reprocess(
    list_of_projects: list = None,
    labels: bool = False,
    tile_size=512,
    res=0.3,
    remove_download=False,
    force_stage=None,
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
    projects_details = get_project_details(list_of_projects)
    log_folder = data_path / "metadata_log/execution_log"

    stages = check_list_stage(list_of_projects)

    stage_orders = ["downloaded", "tiled", "predicted", "assembled", "processed"]
    if force_stage is not None:
        force_stage_order = stage_orders.index(force_stage)
        for project in list_of_projects:
            if stage_orders.index(stages[project]["stage"]) > force_stage_order:
                stages[project]["stage"] = force_stage
                if force_stage == "downloaded":
                    stages[project]["ids"] = []
                elif force_stage == "tiled":
                    stages[project]["ids"] = get_tile_ids(project)
                elif force_stage == "predicted":
                    stages[project]["ids"] = get_prediction_ids(project)
                elif force_stage == "assembled":
                    stages[project]["ids"] = get_assemble_ids(project)
                else:
                    print(f"Error: {force_stage} not recognized")

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
        projects_details = projects_details[project_name]

        channels = project_details[project_name]["bandwidth"]
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
        default=None,
    )
    parser.add_argument(
        "--no_labels",
        action="store_false",
        help="Whether to use labels for the prediction",
        default=True,
    )
    parser.add_argument(
        "--remove_download",
        action="store_true",
        help="Whether to remove the downloaded files after prediction",
        default=True,
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
    parser.add_argument(
        "--force_stage",
        type=str,
        default=None,
        help="Force the stage of the project (only with reprocess)",
    )
    args = parser.parse_args()

    if args.reprocess:
        reprocess(
            labels=args.labels,
            tile_size=args.tile_size,
            res=args.res,
            remove_download=args.remove_download,
            force_stage=args.force_stage,
        )
    else:
        assert (
            args.projects is not None
        ), "Please provide a list of projects with --projects"
        process(
            list_of_projects=args.projects,
            labels=args.no_labels,
            remove_download=args.remove_download,
            tile_size=args.tile_size,
            res=args.res,
        )

# %%
