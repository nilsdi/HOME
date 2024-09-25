# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import os
import random

root_dir = str(Path(__file__).resolve().parents[3])

# %% Function to plot side by side image and different predictions


def display_any(
    folders: list,
    column_names: list,
    row_names: list = None,
    filenames: list = "random",
):
    if filenames == "random":
        filenames = []
        for i, row in enumerate(folders):
            files_in_folder = [f for f in os.listdir(row[-1])]
            filenames.append(files_in_folder[random.randint(0, len(files_in_folder))])

    # Open the files
    num_rows = len(folders)
    num_cols = len(folders[0])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axs = np.atleast_2d(axs)
    for i, row in enumerate(folders):
        filename = filenames[i]
        for j, folder in enumerate(row):
            path = folder + filename
            with rasterio.open(path) as image:
                # Read the data
                num_channels = image.count
                if num_channels == 1:
                    data = image.read(1)
                    cmap = "gray"
                else:
                    data = image.read([1, 2, 3])
                    data = data.transpose((1, 2, 0))
                    cmap = None
                axs[i, j].imshow(data, cmap=cmap)
                axs[i, j].set_title(f"{column_names[j]}")
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        # display name of the image
        print(filename)

        # display row name if provided
        if row_names is not None:
            axs[i, 0].set_ylabel(row_names[i], fontsize=16)

    # Show the figure
    plt.tight_layout()
    plt.show()

    return fig


# %%
folders = [
    [
        root_dir
        + "/data/ML_prediction/topredict/image/res_0.3/trondheim_1999/i_lzw_25/",
        root_dir + "/data/ML_prediction/predictions/res_0.3/trondheim_1999/i_lzw_25/",
        root_dir
        + "/data/ML_prediction/topredict/label/res_0.3/trondheim_1999/i_lzw_25/",
    ]
]
names = ["Image", "Prediction", "Label"]

fig = display_any(folders, names)


# %%
folders = [
    [
        root_dir
        + "/data/ML_prediction/topredict/image/res_0.3/trondheim_1999/i_lzw_25/",
        root_dir + "/data/ML_prediction/predictions/res_0.3/trondheim_1999/i_lzw_25/",
        root_dir
        + "/data/ML_prediction/topredict/label/res_0.3/trondheim_1999/i_lzw_25/",
    ]
]
names = ["Image", "Prediction", "Label"]

fig = display_any(folders, names)
