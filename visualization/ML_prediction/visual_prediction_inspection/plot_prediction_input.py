'''
Creates a simple plot containing to tiles - input and output of the HD-Net.
'''
# %% imports
import matplotlib.pyplot as plt
from pathlib import Path
import random
import rasterio
import numpy as np
import warnings
from rasterio.errors import NotGeoreferencedWarning
root_dir = Path(__file__).resolve().parents[3]
#print(root_dir)

# %% Function to plot side by side image and prediction

def plot_prediction_input(project_details: dict, n_tiles:int = 1, tiles_per_plot: int = 1, 
                                            tile_coords:str= None, save: bool = False, show: bool = True):
    '''
    project details: dictionary of one project that we use for the project logging (project_deatils.json)
    n_tiles: number of tiles to plot (random selection with fix seed)
    tiles_per_plot: depth of the plot (how many tiles are plotted in one plot)
    tile_coords: specific tile to plot (not implemented yet)
    save: boolean to save the plot
    show: boolean to show the plot
    '''
    # check status of project:
    if project_details['status'] != 'predicted':
        print('Project is not predicted yet.')
        return
    # get overview of all files in the prediction folder:
    prediction_folder = root_dir / 'data/ML_prediction/predictions'
    res = f'res_{project_details["resolution"]}'
    compression = f'i_{project_details["compression_name"]}_{project_details["compression_value"]}'
    prediction_files_folder = prediction_folder/res/project_details["project_name"]/compression
    preds = [f for f in os.listdir(prediction_files_folder) if f.endswith('.tif')]

    input_folder = root_dir / 'data/ML_prediction/topredict/image'
    input_files_folder = input_folder/res/project_details["project_name"]/compression
    inputs = [f for f in os.listdir(input_files_folder) if f.endswith('.tif')]

    
    # can in the future be used to plot a specific tile
    if not tile_coords:
        # set random seed:
        random.seed(42)
        # generate n random numbers in the list of predictions
        preds = random.choices(preds, k=n_tiles)
        inputs = random.choices(inputs, k=n_tiles)
        #print(len(preds))
    else:
        raise NotImplementedError('Functionality to plot specific tile not implemented yet.')
    if save:
        save_path = root_dir / 'visualization/ML_prediction/visual_prediction_inspection'
        save_path = save_path/res/project_details["project_name"]/compression
        save_path.mkdir(parents=True, exist_ok=True)
    for p, i in zip(preds, inputs):
        # Open the files (p, i)
        pred_p = prediction_files_folder / p
        inp_p = input_files_folder / i
        # Suppress NotGeoreferencedWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NotGeoreferencedWarning)
            with rasterio.open(pred_p) as pred_image, rasterio.open(inp_p) as inp_image:
                # Read the data
                pred_data = pred_image.read(1)
                # for the input we need to check which channels to read
                num_channels = inp_image.count
                if num_channels == 1:
                    inp_data = inp_image.read(1)
                    cmap = "gray"
                else:
                    inp_data = inp_image.read([1, 2, 3])
                    inp_data = inp_data.transpose((1, 2, 0))
                    cmap = None
                # plot the data
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs = np.atleast_2d(axs)
                axs[0, 0].imshow(inp_data, cmap=cmap)
                axs[0, 0].set_xlabel("Input")
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(pred_data, cmap='gray')
                axs[0, 1].set_xlabel("Prediction")
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                # display name of the image
                plt.tight_layout()
                plt.suptitle(f'Tile:{i}', fontsize=16, backgroundcolor='white', alpha=0.5, 
                                    bbox=dict(facecolor='lightgrey', alpha=0.95, edgecolor='none'))
                if save:
                    plt.savefig(save_path/f'{i}.png', dpi=300, bbox_inches='tight')
                if show:
                    plt.show()
                plt.close()

if __name__ == '__main__':
    projects = ['troindheim-gauldal_1947', 'trondheim_2019', 'trondheim_1992']
    project_details = {'project_name': projects[2], 'resolution': 0.3, 
                            'compression_name': 'lzw', 'compression_value': 25, 'status': 'predicted'}
    plot_prediction_input(project_details, n_tiles=10, tiles_per_plot=1, tile_coords=None,
                                        save=True, show=True)                
# %%
