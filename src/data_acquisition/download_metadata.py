'''
File is meant to be run in interactive window
'''
#%% imports
import json
import os
import sys
from pathlib import Path
# Get the root directory of the project
root_directory = Path(__file__).resolve().parents[2]

# Add the root directory to sys.path
sys.path.append(str(root_directory))

from src.data_acquisition.orthophoto_api.project_metadata import get_all_projects, get_project_metadata

#%% get projects and metadata
all_projects = get_all_projects()
#metadata_all_projects = get_project_metadata(all_projects[0:100], geometry = False)
metadata_all_projects = {'ProjectList': [], 'ProjectMetadata': []}
n_projects = len(all_projects)
metadata_counter = 0
while metadata_counter < n_projects:
    if n_projects - metadata_counter > 100:
        batch = all_projects[metadata_counter:metadata_counter+100]
    else:
        batch = all_projects[metadata_counter:]
    new_metadata = get_project_metadata(batch, geometry=True)
    #print(new_metadata)
    for key in metadata_all_projects.keys():
        metadata_all_projects[key].extend(new_metadata[key])
    metadata_counter +=100

#%% save to json in data folder
# write the metadata to a file
from datetime import datetime
import json

path_to_data = root_directory / 'data' / 'raw' / 'orthophoto'
now = datetime.now()
metadata_file = f'metadata_all_projects_{now.strftime("%Y%m%d%H%M%S")}.json'
with open(path_to_data / metadata_file, 'w') as f:
    json.dump(metadata_all_projects, f)
# %%
