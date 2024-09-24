# %%
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
import json


def get_project_metadata(project_name: str) -> dict:
    """
    Get metadata for a project.
    """
    metadata = _get_newest_metadata()
    if project_name not in metadata["ProjectList"]:
        raise NotFoundError(f"Project {project_name} not found in metadata")
    project_index = metadata["ProjectList"].index(project_name)
    metadata_project = metadata["ProjectMetadata"][project_index]
    return metadata_project


def get_project_geometry(project_name: str) -> dict:
    """
    Get geometry for a project from metadata.

    Args:
    - project_name: str, name of the project

    Returns:
    - project_geometry: geopandas.GeoSeries, geometry of the project
    """
    project_metadata = get_project_metadata(project_name)
    project_geometry = gpd.GeoSeries(shape(project_metadata["geometry"]))
    project_geometry.crs = "EPSG:25833"
    return project_geometry


def _get_newest_metadata() -> dict:
    """
    Get the newest metadata file (most recent downloaded in data folder)
    """
    root_dir = Path(__file__).resolve().parents[2]
    # print(root_dir)
    data_path = root_dir / "data"
    orthophoto_data = data_path / "raw" / "orthophoto"
    metadata_files = [
        f
        for f in os.listdir(orthophoto_data)
        if os.path.isfile(os.path.join(orthophoto_data, f))
    ]

    # the last digits in the file name is the date and time of the metadata, we want the latest
    # Function to extract datetime from filename
    def extract_datetime(filename):
        # Assuming the date is at the end of the filename and is in a specific format
        # Adjust the slicing as per your filename format
        date_str = filename.split("_")[-1].split(".")[
            0
        ]  # Adjust based on your filename format
        # print(date_str)
        return datetime.strptime(
            date_str, "%Y%m%d%H%M%S"
        )  # Adjust the format as per your filename

    # Sort files by datetime
    sorted_files = sorted(metadata_files, key=extract_datetime, reverse=True)

    # The newest file
    newest_file = sorted_files[0]
    with open(orthophoto_data / newest_file, "r") as f:
        metadata_all_projects = json.load(f)
    return metadata_all_projects


# %%

if __name__ == "__main__":
    test_metadata = get_project_metadata("Kyken 2023")
    print(test_metadata["properties"])
    test_geometry = test_metadata["geometry"]


# %%
