"""
Plots and saves some building footprint labels from the training data.
"""

# %% imports
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
from pathlib import Path

root_dir = Path(__file__).resolve().parents[4]
# print(root_dir)
# load the FKB data
FKB_bygning_path = os.path.join(
    root_dir, "data/raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb"
)
FKB_veg_path = os.path.join(
    root_dir, "data/raw/FKB_veg/Basisdata_0000_Norge_5973_FKB-Veg_FGDB.gdb"
)
# list all layers
layers_bygning = fiona.listlayers(FKB_bygning_path)
# print(f' the layers in the FKB bygning data are: {layers_bygning}')

# %% load the data
bygg_omrader = gpd.read_file(FKB_bygning_path, layer="fkb_bygning_omrade")

# %% define plot function


def plot_label_shapes(
    label_shapes: list, n_rows, n_cols: int, figsize: tuple = None, save_as: str = None
):
    """
    Plots the shapes from the label_shapes list in a grid with n_rows and n_cols.
    Args:
    label_shapes: list, list of geopandas dataframes to plot
    """
    if not figsize:
        figsize = (6, 4)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if i < len(label_shapes):
            multipolgyon = label_shapes.iloc[i]["geometry"]
            geoseries = gpd.GeoSeries([multipolgyon])
            geoseries.plot(facecolor="none", edgecolor="crimson", linewidth=3, ax=ax)
        ax.axis("off")
    plt.tight_layout()
    plt.axis("off")
    if save_as:
        plt.savefig(
            root_dir / f"data/figures/ML_training/methods_figure/{save_as}", dpi=300
        )
    plt.show()
    return


# %% example
if __name__ == "__main__":
    seed = 42
    gdf_sample = bygg_omrader.sample(200, random_state=seed)
    gdf_sample.reset_index(drop=True, inplace=True)
    # cols and rows:
    n_cols = 4
    n_rows = 4
    plot_label_shapes(gdf_sample, n_cols, n_rows, save_as="building_labels.png")

# %%
