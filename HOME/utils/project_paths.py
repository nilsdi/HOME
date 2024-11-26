"""
for convenience throuought the code, whenever data needs to be read or saved.
Function for getting the project path
"""

# %% imports defintions
import numpy as np
import json
from pathlib import Path

root_dir = Path(__file__).parents[2]


# %%
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


def get_assembling_details(assemble_id: int, data_path: Path = None) -> dict:
    """
    Get the assembling details for a specific assembling id.

    Args:
        assemble_id (int): The assembling id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        dict: The assembling details as in the metadat_log/assembling_log.json file.
    """
    if data_path is None:
        data_path = root_dir / "data"
    assembling_log = data_path / "metadata_log/reassembled_prediction_tiles.json"
    with open(assembling_log, "r") as file:
        assembling_log = json.load(file)
    assembling_details = assembling_log[str(assemble_id)]
    return assembling_details


def get_polygon_details(poly_id: int, data_path: Path = None) -> dict:
    """
    Get the polygon details for a specific polygon id.

    Args:
        poly_id (int): The polygon id
        data_path (Path, optional): The path to the data folder. Defaults to None (root_dir / "data").

    Returns:
        dict: The polygon details as in the metadat_log/polygons_log.json file.
    """
    if data_path is None:
        data_path = root_dir / "data"
    polygons_log = data_path / "metadata_log/polygon_gdfs.json"
    with open(polygons_log, "r") as file:
        polygons_log = json.load(file)
    polygon_details = polygons_log[str(poly_id)]
    return polygon_details


def save_project_details(project_details: int, data_path: Path = None) -> dict:
    if data_path is None:
        data_path = root_dir / "data"
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "w"
    ) as file:
        json.dump(project_details, file, indent=4)


def load_project_details(data_path: Path = None):
    if data_path is None:
        data_path = root_dir / "data"
    with open(
        data_path / "ML_prediction/project_log/project_details.json", "r"
    ) as file:
        project_details = json.load(file)
    return project_details


def get_tile_ids(project_name: str, data_path: Path = None) -> list:
    if data_path is None:
        data_path = root_dir / "data"
    tiling_log_path = data_path / "metadata_log/tiled_projects.json"
    with open(tiling_log_path, "r") as file:
        tiling_log = json.load(file)
    tile_ids = [
        int(tile_id)
        for tile_id in tiling_log.keys()
        if tiling_log[tile_id]["project_name"] == project_name
    ]
    return tile_ids


def get_prediction_ids(project_name: str, data_path: Path = None) -> list:
    if data_path is None:
        data_path = root_dir / "data"
    prediction_log_path = data_path / "metadata_log/predictions_log.json"
    with open(prediction_log_path, "r") as file:
        prediction_log = json.load(file)
    prediction_ids = [
        int(pred_id)
        for pred_id in prediction_log.keys()
        if prediction_log[pred_id]["project_name"] == project_name
    ]
    return prediction_ids


def get_assemble_ids(project_name: str, data_path: Path = None) -> list:
    if data_path is None:
        data_path = root_dir / "data"
    assembling_log_path = data_path / "metadata_log/reassembled_prediction_tiles.json"
    with open(assembling_log_path, "r") as file:
        assembling_log = json.load(file)
    assemble_ids = [
        int(assemble_id)
        for assemble_id in assembling_log.keys()
        if assembling_log[assemble_id]["project_name"] == project_name
    ]
    return assemble_ids


def get_polygon_ids(project_name: str, data_path: Path = None) -> list:
    if data_path is None:
        data_path = root_dir / "data"
    polygon_log_path = data_path / "metadata_log/polygon_gdfs.json"
    with open(polygon_log_path, "r") as file:
        polygon_log = json.load(file)
    polygon_ids = [
        int(poly_id)
        for poly_id in polygon_log.keys()
        if polygon_log[poly_id]["project_name"] == project_name
    ]
    return polygon_ids


# %% simple test
if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]

    prediction_id = 20001
    print(get_prediction_details(prediction_id))

# %%
