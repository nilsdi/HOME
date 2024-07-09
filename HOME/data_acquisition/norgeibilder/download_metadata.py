""" Downloading all metadata from norgeibilder and save it as a date specific file

Contains a main block, to be run to download the metadata.
"""

# %% imports
import json
import os
from pathlib import Path
from datetime import datetime
from HOME.get_data_path import get_data_path

if __name__ == "__main__":
    # Get the root directory of the project
    root_dir = Path(__file__).resolve().parents[3]
    # print(root_dir)
    # get the data path (might change)
    data_path = get_data_path(root_dir)
    # print(data_path)

    from HOME.data_acquisition.norgeibilder.orthophoto_api.project_metadata import (
        get_all_projects,
        get_project_metadata,
    )

    # %% get projects and metadata
    all_projects = get_all_projects()
    # metadata_all_projects = get_project_metadata(all_projects[0:100], geometry = False)
    metadata_all_projects = {"ProjectList": [], "ProjectMetadata": []}
    n_projects = len(all_projects)
    metadata_counter = 0
    while metadata_counter < n_projects:
        if n_projects - metadata_counter > 100:
            batch = all_projects[metadata_counter : metadata_counter + 100]
        else:
            batch = all_projects[metadata_counter:]
        new_metadata = get_project_metadata(batch, geometry=True)
        # print(new_metadata)
        for key in metadata_all_projects.keys():
            metadata_all_projects[key].extend(new_metadata[key])
        metadata_counter += 100

    # save to json in data folder

    path_to_data = os.path.join(data_path, "raw/orthophoto")
    now = datetime.now()
    metadata_file = f'metadata_all_projects_{now.strftime("%Y%m%d%H%M%S")}.json'
    # make the directory if it does not exist already
    os.makedirs(path_to_data, exist_ok=True)
    with open(os.path.join(path_to_data, metadata_file), "w") as f:
        json.dump(metadata_all_projects, f)
# %%
