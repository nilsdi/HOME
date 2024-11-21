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

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


@contextlib.contextmanager
def suppress_output():
    """Suppress all stdout and stderr output temporarily."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


async def run_project(
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

    # Step 1: Generate tiles
    tile_key = step_01_tile_generation.tile_generation(
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

    return project_details


async def download(project_name, project_details):
    project_id = project_details["project_name"]["id"]
    download_urls = request_download(project_id)
    download_original_NIB(download_urls, project_name)


async def workflow_manager(project_list, labels=False):
    project_details = add_project_details(project_list)

    projects_to_run = []  # list of IDs of projects to run in the future,

    for project_name in project_details:
        if project_details[project_name]["status"] == "pending":
            projects_to_run.append(project_name)

    download_queue = deque()  # Stores downloaded projects
    project_queue = asyncio.Queue()

    # Populate project queue with projects to be processed
    for project in projects_to_run:
        await project_queue.put(project)

    async def download_worker():
        while not project_queue.empty():
            project = await project_queue.get()
            while len(download_queue) >= 2:  # Ensure max 2 downloads
                await asyncio.sleep(1)  # Wait until space is available
            print(f"Downloading {project_name}")
            await download(project)
            project_details[project_name]["status"] = "downloaded"
            download_queue.append(project)

    async def process_worker():
        if labels:
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

        while True:
            if download_queue:
                project = download_queue.popleft()
                print(f"Processing {project_name}")
                with suppress_output():
                    await run_project(
                        project,
                        labels=labels,
                        gdf_omrade=gdf_omrade,
                        remove_download=True,
                    )
                project_details[project_name]["status"] = "processed"
            elif project_queue.empty():
                break  # Exit if no more projects to process
            else:
                await asyncio.sleep(1)  # Poll for new downloads

    # Run workers
    await asyncio.gather(download_worker(), process_worker())

    with open(
        data_path / "ML_prediction/project_log/project_details.json", "w"
    ) as file:
        json.dump(project_details, file, indent=4)


# %%
if __name__ == "__main__":

    trondheim_projects = [
        "Trondheim kommune 2022",
        "Trondheim kommune 2021",
        "Trondheim kommune 2020",
        "Trondheim 2019",
        "Trondheim kommune rektifisert 2018",
        "Trondheim 2017",
        "Trondheim rundt 2016",
        "Trondheim 2016",
        "Trondheim 2015",
        "Trondheim 2014",
        "Trondheim 2013",
        "Trondheim sentrum 2013",
        "Trondheim 2012",
        "Trondheim 2011",
        "Trondheim 2010",
        "Trondheim 2008",
        "Trondheim 2006",
        "Trondheim 2005",
        "Trondheim tettbebyggelse 2003",
        "Trondheim 1999",
        "Byneset-Trondheim-Stjørdal 1997",
        "Trondheim-Byneset 1996",
        "Trondheim 1994",
        "Trondheim øst 1993",
        "Trondheim 1992",
        "Trondheim 1991",
    ]
    asyncio.run(workflow_manager(trondheim_projects, labels=True))

# %%
