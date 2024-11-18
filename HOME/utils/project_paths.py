"""
for convenience throuought the code, whenever data needs to be read or saved.
Function for getting the project path
"""

# %% imports defintions
import numpy as np
import json
from pathlib import Path

root_dir = Path(__file__).parents[2]


def get_download_details(download_id: int, data_path: Path = None) -> dict:
    """
    Get the download details for a specific download id.

    Args:
        download_id (int): The download id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        dict: The download details as in the metadat_log/ortofoto_downloads.json file.
    """
    if data_path is None:
        data_path = root_dir / "data"
    # print(data_path)
    download_log = data_path / "metadata_log/ortofoto_downloads.json"
    with open(download_log, "r") as file:
        download_log = json.load(file)
    download_details = download_log[str(download_id)]
    return download_details


def get_download_str(download_id: int, data_path: Path = None) -> str:
    """
    Get the str for creating the path for a specific download id.

    Args:
        download_id (int): The download id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        str: the path past ..data/raw/orthophoto/...
    """
    if data_path is None:
        data_path = root_dir / "data"

    download_details = get_download_details(download_id, data_path)
    project_name = download_details["project_name"]
    resolution = str(np.round(download_details["resolution"], 1))
    compression_name = download_details["compression_name"]
    compression_value = download_details["compression_value"]
    mosaic = download_details["mosaic"]
    if mosaic == 3:
        mosaic_name = "im"
    else:
        mosaic_name = "i"
    return f"res_{resolution}/{project_name}/{mosaic_name}_{compression_name}_{compression_value}"


def get_tiling_details(tile_id: int, data_path: Path = None) -> dict:
    """
    Get the tiling details for a specific tiling id.

    Args:
        tile_id (int): The tiling id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        dict: The tiling details as in the metadat_log/tiling_log.json file.
    """
    if data_path is None:
        data_path = root_dir / "data"
    # print(data_path)
    tiling_log_path = data_path / "metadata_log/tiled_projects.json"
    with open(tiling_log_path, "r") as file:
        tiling_log = json.load(file)
    tiling_details = tiling_log[str(tile_id)]
    return tiling_details


def get_prediction_details(pred_id: int, data_path: Path = None) -> dict:
    """
    Get the prediction details for a specific prediction id.

    Args:
        pred_id (int): The prediction id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        dict: The prediction details as in the metadat_log/predictions_log.json file.
    """
    if data_path is None:
        data_path = root_dir / "data"
    # print(data_path)
    prediction_log = data_path / "metadata_log/predictions_log.json"
    with open(prediction_log, "r") as file:
        prediction_log = json.load(file)
    prediction_details = prediction_log[str(pred_id)]
    return prediction_details


# %% simple test
if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    download_id = 10001
    print(get_download_details(download_id))
    print(get_download_str(download_id))

    prediction_id = 20001
    print(get_prediction_details(prediction_id))

# %%
