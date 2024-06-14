'''
Prepares the jsons that say what we want to download
'''
#%% imports
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parents[2]
#print(root_dir)

# %% download details
# specify the download details
resolution = 0.2
compression_method = 5
compression_value = 25
mosaic = False

project_names = ['Oslo vår 2023','Bergen 2022', 'Moss 2022','Stavanger 2023', 
                            'Trondheim MOF 2023', 'Tromsø midlertidig ortofoto 2023' ]
que_path = root_dir / 'data/temp/norgeibilder/download_que/'
# %% create the jsons
for project in project_names:
    download_details = {
        'project': project,
        'resolution': resolution,
        'compression_method': compression_method,
        'compression_value': compression_value, 
        'mosaic': mosaic
    }
    with open(que_path/f'{project.lower().replace(" ", "_")}.json', 'w') as f:
        json.dump(download_details, f)




# %%
