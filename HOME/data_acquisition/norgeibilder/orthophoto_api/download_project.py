"""Handles download of orthophoto data from Norgeibilder once we have an url.

Contains one function:
- download_project: Download based on an url and project details
"""

# %% imports
import requests
import os
import zipfile
from pathlib import Path
from osgeo import gdal, osr
from tqdm import tqdm
import json
import datetime
from HOME.get_data_path import get_data_path

root_dir = Path(__file__).parents[4]
# print(root_dir)


# %% functions
def download_project(
    download_url: str,
    project: str,
    resolution: float,
    compression_name: str,
    compression_value: float,
    mosaic: bool,
    data_path: Path = None,
) -> None:
    """
    Download zipped orthophoto data from given url, unzips and saves it into
    the correct folder structure. Also checks the CRS of the orthophoto.

    Args:
        download_url (str): url generated via the status API
        project (str): name of the project (for folder structure)
        resolution (float): resolution of the orthophoto (for folder structure)
        compression_name (str): name of the compression used (for folder structure)
        compression_value (float): value of the compression used (for folder structure)
        mosaic (bool): whether the orthophoto is a mosaic or not (for folder structure)
        data_path (Path): path to the data folder, default is None (=> HOME/data)

    Returns:
        None

    Raises:
        Exception: if the download request fails for any reason
        AssertionError: if the CRS of the orthophoto is not EPSG:258
    """
    if data_path is None:
        data_path = root_dir / "data"

    # Retrieve data
    response = requests.get(download_url, allow_redirects=True, stream=True)
    print(f"we get status_code {response.status_code}.")
    if response.status_code != 200:
        raise Exception(
            f"Download request failed with status code {response.status_code}."
        )

    total_size = int(response.headers.get("content-length", 0))

    # set up the path to save the file and unzip it
    if mosaic:
        name_start = "im"
    else:
        name_start = "i"

    file_name = f"{name_start}_{compression_name.lower()}_" + f"{compression_value}.zip"
    extract_path = os.path.join(
        data_path,
        f"raw/orthophoto/res_{resolution}/{project.lower().replace(' ', '_')}/",
    )

    # Create the directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    file_path = os.path.join(extract_path, file_name)

    # write zip file to the specified path
    with open(file_path, "wb") as file:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    pbar.update(len(chunk))

    # Unzip the file
    unzip_folder = os.path.join(extract_path, file_name.split(".")[0])
    os.makedirs(unzip_folder, exist_ok=True)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(unzip_folder)
        # Remove the zip file after extraction
        os.remove(file_path)

    # find all tif files in the folder:
    tif_files = [f for f in os.listdir(unzip_folder) if ".tif" in f]
    for tif_file in tif_files:
        # Check the CRS of the orthophoto
        dataset = gdal.Open(str(os.path.join(unzip_folder, tif_file)))
        srs = osr.SpatialReference(wkt=dataset.GetProjection())

        # Assert CRS is EPSG:25833
        assert srs.GetAuthorityCode(None) == "25833"

    # open the json with all the project details from the project log
    with open(
        data_path / "metadata/ortofoto_downloads.json",
        "r",
    ) as file:
        project_details = json.load(file)
    # increase available lowest key by 1
    lowest_key = min([int(k) for k in project_details.keys()])
    new_key = lowest_key + 1
    project_details[project.lower().replace(" ", "_")] = {
        "project_name": project.lower().replace(" ", "_"),
        "status": "downloaded",
        "resolution": resolution,
        "compression_name": compression_name,
        "compression_value": compression_value,
        "channels": None,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # save the updated json
    with open(
        os.path.join(data_path, "ML_prediction/project_log/project_details.json"), "w"
    ) as file:
        json.dump(project_details, file)
    return
