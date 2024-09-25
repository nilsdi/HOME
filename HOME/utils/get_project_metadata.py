# %%
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
import json


def get_project_metadata(project_name: str) -> dict:
    """
    Get metadata for a project.

    Args:
    - project_name: str, name of the project

    Returns:
    - metadata_project: dict, metadata for the project (type, properties, geometry)

    Raises:
    - NotFoundError: if the project is not found in the list of metadata
    """
    metadata = _get_newest_metadata()
    project_name_adjusted = project_name.lower().replace(" ", "_")
    project_list_adjusted_names = [
        x.lower().replace(" ", "_") for x in metadata["ProjectList"]
    ]
    if project_name_adjusted not in project_list_adjusted_names:
        raise NotFoundError(f"Project {project_name} not found in metadata")
    project_index = project_list_adjusted_names.index(project_name_adjusted)
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


def get_project_details(project_name: str) -> dict:
    """
    Get details for a project from metadata.

    Args:
    - project_name: str, name of the project

    Returns:
    - project_details: dict, details of the project that are relevant
    """
    project_metadata = get_project_metadata(project_name)
    project_properties = project_metadata["properties"]
    # codes copied from the ortophoto specification document
    image_category_codes = {
        1: "IR",
        2: "BW",
        3: "RGB",
        4: "RGBIR",
    }
    ortophoto_type_codes = {
        1: "Orto 10",
        2: "Orto 20",
        3: "Orto 50",
        4: "Orto N50",
        5: "Orto Skog",
        6: "Satellittbilde",
        7: "Infrarødt",
        8: "Rektifiserte flybilder",
        9: "Ortofoto",
        10: "Sant ortofoto",
        11: "3D ortofoto",
        12: "Midlertidig ortofoto",
    }
    capture_method_codes = {
        1: "analogue",
        2: "digital",
    }
    project_details = {
        "capture_date": project_properties["fotodato_date"],
        "original image format": project_properties["opprinneligbildeformat"],
        "bandwidth": image_category_codes[int(project_properties["bildekategori"])],
        "capture method": capture_method_codes[
            int(project_properties["opptaksmetode"])
        ],
        "orthophoto type": ortophoto_type_codes[
            int(project_properties["ortofototype"])
        ],
    }
    return project_details


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
    test_geometry = get_project_geometry("Kyken 2023")
    print(get_project_details("Kyken 2023"))

# %%
