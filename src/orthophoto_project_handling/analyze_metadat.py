#%% imports
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get the root directory of the project
root_directory = Path(__file__).resolve().parents[2]

# load the metadata
metadata_file = 'metadata_all_projects_20240523000926.json'
path_to_data = root_directory / 'data' / 'raw' / 'orthophoto'
print(path_to_data / metadata_file)
with open(path_to_data / metadata_file, 'r') as f:
    metadata_all_projects = json.load(f)

#%%	plot ortophoto types
ortofoto_type_list = [m['properties']['ortofototype'] for m in metadata_all_projects['ProjectMetadata']]

# bar plot with each of the ortofoto types
# all the unique type of ortofoto
ortofoto_types = list(set(ortofoto_type_list))
plt.figure(figsize=(10,5))
plt.bar(ortofoto_types, [ortofoto_type_list.count(ot) for ot in ortofoto_types])
plt.xticks(rotation=90)
#%% plot resolution histogram
# Convert the resolutions to numbers
resolution_list = [float(m['properties']['pixelstorrelse']) for m in metadata_all_projects['ProjectMetadata']]

plt.figure()
plt.hist(resolution_list, bins=100)
plt.xlabel('Resolution')
plt.ylabel('Number of projects')
plt.title('Resolution histogram')
# tilt xlabels
plt.xticks(rotation=90)
plt.xlim(0, 1)
plt.show()

print(f'we have a total of projects with a higher (worse) resolution that 0.3 m: {sum(np.array(resolution_list) > 0.3)}')
print(f'we have a total of projects with a higher (worse) resolution that 0.5 m: {sum(np.array(resolution_list) > 0.5)}.')
#%% time histogram
# Convert the times to datetime
time_list = [pd.to_datetime(m['properties']['aar']) for m in metadata_all_projects['ProjectMetadata']]
time_list = pd.Series(time_list)
n_bins = 2024 - min(time_list).year
plt.hist(time_list, bins=n_bins)
plt.xlabel('Year')
plt.ylabel('Number of projects')
plt.title('Year histogram')

#%% print a random projects outline on a map of Norway

# %%
# Load the shapefile

