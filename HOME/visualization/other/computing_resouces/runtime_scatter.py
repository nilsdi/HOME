# %%
import json
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Polygon
from pathlib import Path
from datetime import datetime

from HOME.utils.get_project_metadata import (
    get_project_geometry,
    filter_projects_without_metadata,
)


root_dir = Path(__file__).resolve().parents[4]
data_path = root_dir / "data"
print(root_dir)
runtimes_path = data_path / "metadata_log/prediction_main_runtime.json"
print(runtimes_path)
today = datetime.today().strftime("%Y-%m-%d")
print(f"today: {today}")

# %%
if __name__ == "__main__":
    with open(runtimes_path, "r") as f:
        runtimes = json.load(f)
    projects = list(runtimes.keys())
    selected_projects = filter_projects_without_metadata(projects)
    print(
        f" reduced projects from {len(projects)} to {len(selected_projects)} by filtering out all without metadata"
    )
    geometries = get_project_geometry(selected_projects)
    runtimes_selected = {}
    for project, geometry in zip(selected_projects, geometries):
        # get the area
        geometry.to_crs(epsg=32633)  # convert to UTM33N
        total_area = geometry.area.sum()
        runtimes_selected[project] = runtimes[project]
        runtimes_selected[project]["total_area"] = total_area
    print(runtimes_selected)

    # %%
    def scatter_plot_runtimes(
        entry_name: str,
        runtimes: dict,
        save: bool = True,
        ax=None,
        fig=None,
        color: str = "blue",
    ):
        """
        Create a scatter plot of the runtimes for each project
        """
        if not ax or not fig:
            fig, ax = plt.subplots()
        runtimes_entry = {}
        for runtime in runtimes.values():
            if entry_name in runtime.keys():
                # if the entry_name is not in the runtime, skip it
                # this is to avoid errors when the entry_name is not present
                # in all runtimes
                runtimes_entry[runtime["total_area"]] = runtime[entry_name]
        ax.scatter(
            runtimes_entry.keys(), runtimes_entry.values(), marker="o", color=color
        )
        # make the axes logarithmic
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("area - m2")
        ax.set_ylabel(f"{entry_name} - seconds")
        ax.set_title(f"Runtimes for {entry_name}")
        # plt.show()
        save_path = data_path / "figures/other/computing_resources"
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path / f"{entry_name}_runtimes_scatter_{today}.png",
                dpi=300,
                bbox_inches="tight",
            )
        return fig, ax

    def normalized_histogram_runtimes(
        entry_name: str, runtimes: dict, save: bool = True, ax=None, fig=None
    ):
        """
        Create a histogram of the normalized runtimes for each project
        """
        if not ax or not fig:
            fig, ax = plt.subplots()
        runtimes_entry = {}
        for runtime in runtimes.values():
            if entry_name in runtime.keys():
                # if the entry_name is not in the runtime, skip it
                # this is to avoid errors when the entry_name is not present
                # in all runtimes
                runtimes_entry[runtime["total_area"]] = runtime[entry_name]
        normalized_runtimes = [
            seconds / area * 10**6
            for area, seconds in runtimes_entry.items()
            if area > 0
        ]
        ax.hist(normalized_runtimes, bins="auto")
        ax.set_xlabel(f"normalized {entry_name} - seconds per km2")
        ax.set_title(f"Normalized runtimes for {entry_name}")
        plt.show()
        save_path = data_path / "figures/other/computing_resources"
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path / f"{entry_name}_runtimes_histogram_{today}.png",
                dpi=300,
                bbox_inches="tight",
            )
        return fig, ax

    def visualize_runtimes(
        entry_name: str,
        runtimes: dict,
        save: bool = True,
        ax=None,
        fig=None,
        color: str = "blue",
    ):
        """Visualize the runtimes for a given entry name"""
        if not ax or not fig:
            fig, ax = plt.subplots(figsize=(10, 7))
        scatter_plot_runtimes(
            entry_name, runtimes, save=save, ax=ax, fig=fig, color=color
        )
        plt.show()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig, ax = normalized_histogram_runtimes(
            entry_name, runtimes, save=save, ax=ax, fig=fig
        )
        return

    visualize_runtimes("download", runtimes_selected)
    visualize_runtimes("prediction", runtimes_selected)
    visualize_runtimes("tile_generation", runtimes_selected)
    visualize_runtimes("regularization", runtimes_selected)
    # %%
    fig, ax = plt.subplots(figsize=(10, 7))
    fig, ax = scatter_plot_runtimes(
        "download", runtimes_selected, save=False, ax=ax, fig=fig, color="orange"
    )
    fig, ax = scatter_plot_runtimes(
        "prediction", runtimes_selected, save=False, ax=ax, fig=fig, color="green"
    )
    fig, ax = scatter_plot_runtimes(
        "tile_generation", runtimes_selected, save=False, ax=ax, fig=fig, color="red"
    )
    fig, ax = scatter_plot_runtimes(
        "regularization", runtimes_selected, save=False, ax=ax, fig=fig, color="blue"
    )
    ax.legend(
        ["download", "prediction", "tile_generation", "regularization"],
        loc="upper left",
    )
    ax.set_title("Runtimes for all steps")
    ax.set_xlabel("area - m2")
    ax.set_ylabel("seconds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e6)
    plt.show()
    save_path = data_path / "figures/other/computing_resources"
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path / f"all_runtimes_scatter_{today}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # %%
