"""
for convenience throuought the code, whenever data needs to be read or saved.
Function for getting the project path
"""

# %% imports defintions
import numpy as np
import json
from pathlib import Path
from HOME.get_data_path import get_data_path


def get_project_details(root_dir: Path, project_name: str):
    """
    Path (project_log/project_details.json)
    """
    data_path = get_data_path(root_dir)
    # print(data_path)
    project_log = data_path / "ML_prediction/project_log/project_details.json"
    with open(project_log, "r") as file:
        all_project_details = json.load(file)
    project_details = all_project_details[project_name]
    return project_details


def get_project_str_res_name(project_details: dict, project_name: str):
    """
    Path (res/name/compression)
    """
    res = str(np.round(project_details["resolution"], 1))
    return f"res_{res}/{project_name}"


def get_project_str(project_details: dict, project_name: str):
    """
    Path (res/name/compression)
    """
    res = str(np.round(project_details["resolution"], 1))
    compression_name = project_details["compression_name"]
    compression_value = project_details["compression_value"]

    return f"res_{res}/{project_name}/i_{compression_name}_{compression_value}"


# %% simple test
if __name__ == "__main__":
    root_dir = Path(__file__).parents[2]
    project_name = "trondheim_2017"
    project_details = get_project_details(root_dir, project_name)
    print(project_details)
    print(get_project_str_res_name(project_details, project_name))
    print(get_project_str(project_details, project_name))

# %%
