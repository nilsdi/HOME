"""
Creates a simple plot containing to tiles - input and output of the HD-Net + added result of the polygonization.
currently I get the polygons bu running the code on the given image, but eventially I want to get the polygons from the pickled geodataframe.

"""

# %% imports
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
import rasterio
import numpy as np
import warnings
from rasterio.errors import NotGeoreferencedWarning
import os
import re
import sys
from HOME.ML_prediction.postprocessing.step_02_regularization import process_project_tiles
root_dir = Path(__file__).parents[4]
print(root_dir)



# %% Function to plot side by side image and prediction


def plot_prediction_input(
    project_name: str,
    n_tiles: int = 1,
    tiles_per_plot: int = 1,
    tile_coords: str = None,
    save: bool = False,
    show: bool = True,
):
    """
    project details: dictionary of one project that we use for the project logging (project_deatils.json)
    n_tiles: number of tiles to plot (random selection with fix seed)
    tiles_per_plot: depth of the plot (how many tiles are plotted in one plot)
    tile_coords: specific tile to plot (not implemented yet)
    save: boolean to save the plot
    show: boolean to show the plot
    """
    # check status of project:
    project_details = pd.read_json(
        root_dir / "data/ML_prediction/project_log/project_details.json"
    )
    project_details = project_details[project_name]
    print(project_details)
    if project_details["status"] != "predicted":
        print("Project is not predicted yet.")
        return
    # get overview of all files in the prediction folder:
    prediction_folder = root_dir / "data/ML_prediction/predictions"
    res = f'res_{project_details["resolution"]:.1f}'
    compression = f'i_{project_details["compression_name"]}_{project_details["compression_value"]}'
    prediction_files_folder = prediction_folder / res / project_name / compression

    preds = [f for f in os.listdir(prediction_files_folder) if f.endswith(".tif")]

    input_folder = root_dir / "data/ML_prediction/topredict/image"
    input_files_folder = input_folder / res / project_name / compression
    inputs = [f for f in os.listdir(input_files_folder) if f.endswith(".tif")]
        
    file_pattern = re.compile(r'^stitched_tif_(?P<project_name>.+)_(?P<col>\d+)_(?P<row>\d+)\.tif$')

    for f in preds:
        match = file_pattern.match(preds[0])
        if match:
            project_name = match.group('project_name')
            row = int(match.group('row'))
            col = int(match.group('col'))
            
            og_file_path = os.path.join(input_files_folder, f)
            file_path = os.path.join(prediction_files_folder, f)
            files_info = {
                'file_path': file_path,
                'original_file_path': og_file_path, 
                'project_name': project_name,
                'row': row,
                'col': col
            }

    # can in the future be used to plot a specific tile
    if not tile_coords:
        # set random seed:
        random.seed(42)
        # generate n random numbers in the list of predictions
        # just generate indices and then use them to get the files
        indices = random.sample(range(len(preds)), n_tiles)
        preds = [preds[i] for i in indices]
        inputs = [inputs[i] for i in indices]
        files_info = [files_info[i] for i in indices]
        gdf_poly = process_project_tiles(files_info)

        # print(len(preds))
    else:
        raise NotImplementedError(
            "Functionality to plot specific tile not implemented yet."
        )
    if save:
        save_path = root_dir / "data/figures//ML_prediction/prediction_inspection"
        save_path = save_path / res / project_name / compression
        save_path.mkdir(parents=True, exist_ok=True)
    for p, i in zip(preds, inputs):
        # Open the files (p, i)
        pred_p = prediction_files_folder / p
        inp_p = input_files_folder / i
        # Access the polygons for file p

        # Suppress NotGeoreferencedWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
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
                axs[0, 1].imshow(pred_data, cmap="gray")
                axs[0, 1].set_xlabel("Prediction")
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                # display name of the image
                plt.tight_layout()
                plt.suptitle(
                    f"Tile:{i}",
                    fontsize=16,
                    backgroundcolor="white",
                    alpha=0.5,
                    bbox=dict(facecolor="lightgrey", alpha=0.95, edgecolor="none"),
                )
                if save:
                    plt.savefig(save_path / f"{i}.png", dpi=300, bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()


if __name__ == "__main__":
    projects = [
        "trondheim-gauldal_1947",
        "trondheim_2019",
        "trondheim_1992",
        "trondheim_kommune_2020",
        "trondheim_kommune_2022",
    ]
    # project_details = {'project_name': projects[-1], 'resolution': 0.3,
    #                        'compression_name': 'lzw', 'compression_value': 25, 'status': 'predicted'}
    plot_prediction_input(
        projects[-1],
        n_tiles=10,
        tiles_per_plot=1,
        tile_coords=None,
        save=True,
        show=True,
    )
# %%
