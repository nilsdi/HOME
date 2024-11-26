# %%
import json
import pandas as pd
from pathlib import Path
import pickle
import shutil
import asyncio
from collections import deque
import contextlib
import os
import sys

from HOME.data_acquisition.norgeibilder.add_project_details import add_project_details

from HOME.data_acquisition.norgeibilder.orthophoto_api.download_originals import (
    request_download,
    download_original_NIB,
)

from HOME.ML_prediction.preprocessing import (
    step_01_tile_generation,
    step_02_make_text_file,
)
from HOME.ML_prediction.prediction import predict

from HOME.ML_prediction.postprocessing import (
    step_01_reassembling_tiles,
    step_02_regularization,
)

from HOME.get_data_path import get_data_path

from HOME.utils.project_paths import load_project_details, save_project_details

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


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


async def run_project(
    project_name,
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

    project_details = load_project_details(data_path)
    channels = project_details[project_name]["channels"]

    # compression = f"i_{compression_name}_{compression_value}"

    # Step 1: Generate tiles
    tile_key = step_01_tile_generation.tile_generation(
        project_name,
        tile_size,
        res,
        overlap_rate=0,
        labels=labels,
        gdf_omrade=gdf_omrade,
        project_details=project_details,
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

    return


async def download(project_name):
    project_details = load_project_details(data_path=data_path)
    downloaded = False
    if project_details[project_name]["status"] == "pending":
        project_id = project_details[project_name]["id"]
        download_urls = request_download(project_id)
        download_original_NIB(download_urls, project_name)
        downloaded = True
    return downloaded


async def workflow_manager(project_list, labels=False):
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

    with suppress_output():
        project_details = add_project_details(project_list)

    projects_to_run = [
        name
        for name in project_details
        if project_details[name]["status"] != "processed"
    ]

    log_folder = data_path / "metadata_log/execution_log"

    download_queue = asyncio.Queue()  # Stores downloaded projects
    project_queue = asyncio.Queue()

    # Populate project queue with projects to be processed
    for project in projects_to_run:
        await project_queue.put(project)

    async def download_worker():
        while not project_queue.empty():
            # Enforce download limit
            while download_queue.qsize() >= 2:
                await asyncio.sleep(0)  # Yield to allow processing

            project_name = await project_queue.get()
            print(f"{project_name}: Downloading")
            with suppress_output(log_folder / f"{project_name}_download.log"):
                downloaded = await download(project_name)
            if downloaded:
                project_details = load_project_details(data_path=data_path)
                project_details[project_name]["status"] = "downloaded"
                save_project_details(project_details, data_path=data_path)
            print(f"{project_name}: Downloaded")
            await download_queue.put(project_name)

    async def process_worker():
        while True:
            if project_queue.empty() and download_queue.empty():
                break  # Exit when there are no more tasks

            try:
                project_name = await download_queue.get()
                print(f"{project_name}: Processing")
                with suppress_output(log_folder / f"{project_name}_process.log"):
                    await run_project(
                        project_name,
                        labels=labels,
                        gdf_omrade=gdf_omrade,
                        remove_download=True,
                    )
                print(f"{project_name}: Processed")
                project_details = load_project_details(data_path=data_path)
                project_details[project_name]["status"] = "processed"
                save_project_details(project_details, data_path=data_path)
            except Exception as e:
                print(f"Error processing: {e}")
                continue

    print("Starting workflow")
    # Run workers concurrently
    await asyncio.gather(download_worker(), process_worker())


# %%
if __name__ == "__main__":

    trondheim_projects = [
        "trondheim_kommune_2022",
        "trondheim_kommune_2021",
        "trondheim_kommune_2020",
        "trondheim_2019",
        "trondheim_2017",
        "trondheim_rundt_2016",
        "trondheim_2016",
        "trondheim_2015",
        "trondheim_2014",
        "trondheim_2013",
        "trondheim_sentrum_2013",
        "trondheim_2012",
        "trondheim_2011",
        "trondheim_2010",
        "trondheim_2008",
        "trondheim_2006",
        "trondheim_2005",
        "trondheim_tettbebyggelse_2003",
        "trondheim_1999",
        "byneset-trondheim-stjørdal_1997",
        "trondheim-byneset_1996",
        "trondheim_1994",
        "trondheim_øst_1993",
        "trondheim_1992",
        "trondheim_1991",
    ]
    asyncio.run(workflow_manager(trondheim_projects, labels=True))

# %%
