import requests
import os
import zipfile


def download_project(download_url: str, project: str, resolution: float,
                     compression_name: str, compression_value: float,
                     mosaic: bool) -> None:
    response = requests.get(download_url, allow_redirects=True)

    # set up the path to save the file and unzip it
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    if mosaic:
        name_start = "im"
    else:
        name_start = "i"
    file_name = f"{name_start}_{compression_name.lower()}_{compression_value}.zip"
    extract_path = parent_dir + \
        f"/data/raw/orthophoto/res_{resolution}/{project.lower()}/"
    # Create the directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)
    file_path = extract_path + file_name

    # write zip file to the specified path
    with open(file_path, 'wb') as file:
        file.write(response.content)

    # Unzip the file
    unzip_folder = extract_path + file_name.split(".")[0]
    os.makedirs(unzip_folder, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)
        #TODO: remove the zip file after extraction
    return
