# %%
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
import json
import os


def get_project_metadata(project_names: str or list[str]) -> dict:
    """
    Get metadata for a project or a list of projects.

    Args:
    - project_name: str or list of strings, name of the project

    Returns:
    - metadata_project: dict or list of dicts, metadata for the project (type, properties, geometry)

    Raises:
    - NotFoundError: if the project is not found in the list of metadata
    """
    metadata = _get_newest_metadata()
    if isinstance(project_names, str):
        project_names = [project_names]
    metadata_projects = []
    for project_name in project_names:
        project_name_adjusted = project_name.lower().replace(" ", "_")
        project_list_adjusted_names = [
            x.lower().replace(" ", "_") for x in metadata["ProjectList"]
        ]
        if project_name_adjusted not in project_list_adjusted_names:
            raise Exception(f"Project {project_name} not found in metadata")
        project_index = project_list_adjusted_names.index(project_name_adjusted)
        metadata_project = metadata["ProjectMetadata"][project_index]
        metadata_projects.append(metadata_project)
    if len(metadata_projects) == 1:
        return metadata_projects[0]
    return metadata_projects


def get_project_geometry(project_names: str or list[str]) -> gpd.GeoSeries:
    """
    Get geometry for a project from metadata.

    Args:
    - project_name: str, name of the project

    Returns:
    - project_geometry: geopandas.GeoSeries, geometry of the project
    """
    if isinstance(project_names, str):
        project_names = [project_names]

    project_geometries = []
    project_metadatas = get_project_metadata(project_names)
    if len(project_names) == 1:
        project_metadatas = [project_metadatas]

    for project_name, project_metadata in zip(project_names, project_metadatas):
        project_crs_id = project_metadata["properties"]["opprinneligbildesys"]
        # assert project_crs_id in ["22", "23"]
        # if project_crs_id not in ["22", "23"]:
        #     raise ValueError(
        #         f"Project {project_name} has an unknown crs id: {project_crs_id}"
        #     )
        project_crs = 25832 if project_crs_id == "22" else 25833
        project_geometry = gpd.GeoSeries(shape(project_metadata["geometry"]), crs=25833)
        project_geometries.append(project_geometry)
        # project_geometries.append(project_geometry.to_crs(project_crs))

    if len(project_names) == 1:
        return project_geometries[0]
    return project_geometries


def get_project_details(project_names: str or list[str]) -> dict:
    """
    Get details for a project from metadata.

    Args:
    - project_name: str, name of the project

    Returns:
    - project_details: dict, details of the project that are relevant
    """
    if isinstance(project_names, str):
        project_names = [project_names]
    project_metadatas = get_project_metadata(project_names)
    projects_details = []
    for project_metadata in project_metadatas:
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

        def get_project_date(capture_date: str):
            try:
                # Try to convert the input as a Unix timestamp in milliseconds
                timestamp_ms = int(capture_date)
                timestamp_s = timestamp_ms / 1000
                return datetime.fromtimestamp(timestamp_s)
            except ValueError:
                # If conversion to int fails, assume the input is a date string
                try:
                    return datetime.strptime(capture_date, "%Y-%m-%d")
                except ValueError:
                    # If conversion to date fails, return None
                    return capture_date

        project_details = {}
        try:
            project_details["original_resolution"] = float(
                project_properties["pixelstorrelse"]
            )
        except:
            project_details["original_resolution"] = None
        try:
            project_details["capture_date"] = get_project_date(
                project_properties["fotodato_date"]
            )
        except:
            project_details["capture_date"] = None
        try:
            project_details["original image format"] = project_properties[
                "opprinneligbildeformat"
            ]
        except:
            project_details["original image format"] = None
        try:
            project_details["bandwidth"] = image_category_codes[
                int(project_properties["bildekategori"])
            ]
        except:
            project_details["bandwidth"] = None
        try:
            project_details["capture method"] = capture_method_codes[
                int(project_properties["opptaksmetode"])
            ]
        except:
            project_details["capture method"] = None
        try:
            project_details["orthophoto type"] = ortophoto_type_codes[
                int(project_properties["ortofototype"])
            ]
        except:
            project_details["orthophoto type"] = None
        projects_details.append(project_details)
    return projects_details


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
    print(get_project_details("trondheim_1999"))

# %%
