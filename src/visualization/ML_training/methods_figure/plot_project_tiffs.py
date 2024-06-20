"""
Plots and saves and entire big geotiff from a project within a black frame.
"""

# %% imports
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

root_dir = Path(__file__).resolve().parents[4]


def plot_project_tif(
    project_name: str,
    figsize: tuple = None,
    save_as: str = None,
    downsample_factor: int = 20,
):
    """
    Plots the entire geotiff from a project within a black frame.
    Args:
    project_name: str, name of the project to plot, will be looked up in the project_log
    figsize: tuple, size of the figure
    save_as: str, path to save the figure
    """
    project_details = pd.read_json(
        root_dir / "data/ML_prediction/project_log/project_details.json"
    )
    project_details = project_details[project_name]
    res = f'{project_details["resolution"]:.1f}'
    compression = f'i_{project_details["compression_name"]}_{project_details["compression_value"]}'
    tif_folder = (
        root_dir / f"data/raw/orthophoto/res_{res}/{project_name}/{compression}"
    )
    # find the first tif file in the folder
    tif_file = [f for f in os.listdir(tif_folder) if f.endswith(".tif")][0]
    # open the tif file in a downsampled version
    with rasterio.open(tif_folder / tif_file) as src:
        # Calculate the downsampling factor
        # This example assumes a downsampling to 1/10th of the original size for both dimensions
        out_shape = (
            src.count,
            int(src.height / downsample_factor),
            int(src.width / downsample_factor),
        )
        img = src.read(
            out_shape=out_shape,
            resampling=Resampling.bilinear,  # Use bilinear resampling for downsampling
        )
        # Adjust the transform and metadata for the downsampled image
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-1]), (src.height / img.shape[-2])
        )
    # determine colors and adjust for RGB if necessary
    if project_details["channels"] == "BW":
        cmap = "gray"
        image_to_plot = img[0]  # Assuming the first band for BW
    else:
        cmap = None
        # Transpose the image from [channels, height, width] to [height, width, channels]
        # This is necessary only if the image is in RGB format
        image_to_plot = np.transpose(img, (1, 2, 0))
    # plot the image
    if not figsize:
        figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_to_plot, cmap=cmap)
    ax.axis("off")
    if save_as:
        if save_as == "yes":
            save_as = f"{project_name}.png"
        plt.savefig(
            root_dir / f"data/figures/ML_training/methods_figure/{save_as}", dpi=300
        )
    plt.show()
    print(f"Plotted the entire (first) geotiff from project {project_name}")
    return


# %% example
if __name__ == "__main__":
    plot_project_tif(project_name="trondheim_1971", save_as="yes")
# %%
