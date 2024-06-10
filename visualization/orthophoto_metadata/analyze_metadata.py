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
ortofoto_types = [int(i) for i in list(set(ortofoto_type_list))]
# sort the types
ortofoto_types.sort()
ortofoto_types = [str(i) for i in ortofoto_types]
# now, we now what each of the types means from the documentation for the labels:
type_definitions = {
    1: 'Orto 10',
    2: 'Orto 20',
    3: 'Orto 50',
    4: 'Orto N50',
    5: 'Orto Skog',
    6: 'Satellittbilde',
    7: 'Infrarödt',
    8: 'Rektifiserte flybilder',
    9: 'Ortofoto',
    10: 'Sant ortofoto',
    11: '3D ortofoto',
    12: 'Midlertidig ortofoto',
}
x_labels = [type_definitions[int(ot)] for ot in ortofoto_types]
plt.figure(figsize=(10,5))
plt.bar(ortofoto_types, [ortofoto_type_list.count(ot) for ot in ortofoto_types])
plt.xticks(ortofoto_types, x_labels, rotation=90)
plt.xticks(rotation=90)
a = 1+1

#%% plot resolution histogram
# Convert the resolutions to numbers
resolution_list = [float(m['properties']['pixelstorrelse']) for m in metadata_all_projects['ProjectMetadata']]

plt.figure()
plt.hist(resolution_list, bins='auto')
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

#%% scatter plot of resolution and time
# we make a scatter plot, but we'll slightly move the individual points randomly
#  so we see each individual point
from matplotlib.patches import Patch
time = [i.year for i in time_list]
resolution = resolution_list
types = [int(t) for t in ortofoto_type_list]

# Assuming you have a dictionary called type_dict that maps integers to strings
type_colors = types  # Since types are already integers

# Create a legend
legend_elements = [Patch(facecolor=plt.cm.Paired(i/max(types)), edgecolor='r', label=type_definitions[i]) for i in set(types)]

# Add some random noise to the data for better visibility in the scatter plot
resolution_jittered = resolution + (0.2 * np.random.rand(len(time)) - 0.1)*resolution
time_jittered = time + 0.05 * np.random.rand(len(time)) - 0.025

plt.figure(figsize=(12, 8))
plt.scatter(time_jittered, resolution_jittered, c=type_colors, cmap='Paired', 
                marker='o', s=15, alpha=0.5)
plt.ylabel('Resolution')
plt.yscale('log')
plt.xlabel('Time')
plt.yticks([0.1, 0.2, 0.5, 1, 2, 5, 10], [0.1, 0.2, 0.5, 1, 2, 5, 10])
plt.title('Resolution vs time - all orthophoto projects Norway')
plt.legend(handles=legend_elements, title='Types')
plt.show()

# %%
