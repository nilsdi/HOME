import requests
import os
import zipfile
from pathlib import Path
from osgeo import gdal, osr
from tqdm import tqdm

root_dir = Path(__file__).parents[2]


def download_project(download_url: str, project: str, resolution: float,
                     compression_name: str, compression_value: float,
                     mosaic: bool) -> None:
    # Retrieve data
    response = requests.get(download_url, allow_redirects=True, stream=True)
    print(f'we get status_code {response.status_code}.')
    if response.status_code != 200:
        raise Exception(f"Download request failed with status code {response.status_code}.")
    
    total_size = int(response.headers.get('content-length', 0))

    # set up the path to save the file and unzip it
    if mosaic:
        name_start = "im"
    else:
        name_start = "i"

    file_name = (f"{name_start}_{compression_name.lower()}_" +
                 f"{compression_value}.zip")
    extract_path = (root_dir / "data/raw/orthophoto"
                    / f"res_{resolution}/{project.lower().replace(' ', '_')}/")

    # Create the directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    file_path = extract_path / file_name

    # write zip file to the specified path
    with open(file_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=file_name) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    pbar.update(len(chunk))

    # with open(file_path, 'wb') as file:
    #     file.write(response.content)

    # Unzip the file
    unzip_folder = extract_path / file_name.split(".")[0]
    os.makedirs(unzip_folder, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)
        # Remove the zip file after extraction
        os.remove(file_path)

    # find all tif files in the folder:
    tif_files = [f for f in os.listdir(unzip_folder) if ".tif" in f]
    for tif_file in tif_files:
        # Check the CRS of the orthophoto
        dataset = gdal.Open(str(unzip_folder / tif_file))
        srs = osr.SpatialReference(wkt=dataset.GetProjection())

        # Check if the CRS is already EPSG:25833
        if srs.GetAuthorityCode(None) != "25833":

            output_path = str(unzip_folder / tif_file)
            # Warp the file to EPSG:25833
            warp_options = gdal.WarpOptions(dstSRS='EPSG:25833')
            gdal.Warp(output_path, dataset, options=warp_options)
            # Remove the original file
            # os.remove(unzip_folder / "/Eksport-nib.tif")
    return
