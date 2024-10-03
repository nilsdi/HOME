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
    download_urls: str,
    project: str,
    resolution: float,
    compression_name: str,
    compression_value: float,
    mosaic: int = 3,
    mapsheet_size: int = 2000,
    crs: int = 25833,
    date: str = None,
    data_path: Path = None,
) -> None:
    """
    Download zipped orthophoto data from given url, unzips and saves it into
    the correct folder structure. Also checks the CRS of the orthophoto.

    Args:
    - download_url (list[str]): The url to the download.
    - project (str): The name of the project (regular name including spaces)
    - resolution (float): The resolution of the orthophoto in m/px
    - compression_method (int): The compression method used encoded
    - compression_value (float): The compression value used (0-100)
    - mosaic (int): type of mosaic - see documentation for details.
    - mapsheet_size (float): The size of the map sheet (not sure which unit...)
    - crs (int): The coordinate reference system of the orthophoto to be exported.
    - date (str): The date of the export request, default is None (=> current time)
    - data_path (Path): The path to the data folder, default is None (=> HOME/data)

    Returns:
    - None

    Raises:
    - Exception: if the download request fails for any reason
    - AssertionError: if the CRS of the orthophoto is not EPSG:258
    """
    if data_path is None:
        data_path = root_dir / "data"

    for i, download_url in enumerate(download_urls):
        response = requests.get(download_url, allow_redirects=True, stream=True)
        print(f"we get status_code {response.status_code}.")
        if response.status_code != 200:
            raise Exception(
                f"Download request failed with status code {response.status_code}."
            )

        total_size = int(response.headers.get("content-length", 0))

        # set up the path to save the file and unzip it
        if mosaic == 3:
            name_start = "im"
        else:
            name_start = "i"

        file_name = (
            f"{name_start}_{compression_name.lower()}_" + f"{compression_value}.{i}.zip"
        )
        extract_path = os.path.join(
            data_path,
            f"raw/orthophoto/res_{resolution}/{project.lower().replace(' ', '_')}/",
        )

        # Create the directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)
        file_path = os.path.join(extract_path, file_name)

        # write zip file to the specified path
        with open(file_path, "wb") as file:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=file_name
            ) as pbar:
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
    tif_files = [f for f in os.listdir(unzip_folder) if f.endswith(".tif")]
    for tif_file in tif_files:
        # Check the CRS of the orthophoto
        dataset = gdal.Open(str(os.path.join(unzip_folder, tif_file)))
        srs = osr.SpatialReference(wkt=dataset.GetProjection())

        # Assert CRS is EPSG:25833
        assert srs.GetAuthorityCode(None) == "25833"

    # open the json with all the project details from the project log
    with open(
        data_path / "metadata_log/ortofoto_downloads.json",
        "r",
    ) as file:
        project_details = json.load(file)
    # increase available lowest key by 1
    highest_key = max([int(k) for k in project_details.keys()])
    new_key = highest_key + 1
    project_details[new_key] = {
        "project_name": project.lower().replace(" ", "_"),
        "resolution": resolution,
        "compression_name": compression_name,
        "compression_value": compression_value,
        "mosaic": mosaic,
        "mapsheet_size": mapsheet_size,
        "crs": crs,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # save the updated json
    with open(data_path / "metadata_log/ortofoto_downloads.json", "w") as file:
        json.dump(project_details, file)
    return


# %% some tests
if __name__ == "__main1__":  # test with EPSG checks
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
    unzip_folder = root_dir / "data/raw/orthophoto/res_0.3/trondheim_2016/i_lzw_25"
    tif_files = [f for f in os.listdir(unzip_folder) if f.endswith(".tif")]
    print(f"found {len(tif_files)} tif files.")
    for tif_file in tif_files:
        # Check the CRS of the orthophoto
        dataset = gdal.Open(str(os.path.join(unzip_folder, tif_file)))
        srs = osr.SpatialReference(wkt=dataset.GetProjection())
        print(f" the srs of the dataset is {srs}.")
        print(f" the authority code is {srs.GetAuthorityCode(None)}.")

        # Assert CRS is EPSG:25833
        assert srs.GetAuthorityCode(None) == "25833"
