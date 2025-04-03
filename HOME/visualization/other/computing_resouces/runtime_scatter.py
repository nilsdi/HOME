# %%
import json
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Polygon
from pathlib import Path
from HOME.utils.get_project_metadata import get_project_geometry


root_dir = Path(__file__).resolve().parents[4]
data_path = root_dir / "data"
print(root_dir)
runtimes_path = data_path / "metadata_log/prediction_main_runtime.json"
print(runtimes_path)

# %%
if __name__ == "__main__":
    with open(runtimes_path, "r") as f:
        runtimes = json.load(f)
    projects = list(runtimes.keys())
    selected_projects = [  # filter out all that contain "midlertidig"
        x for x in projects if "midlertidig" not in x
    ]
    print(
        f' reduced projects from {len(projects)} to {len(selected_projects)} by filtering out "midlertidig"'
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
    def visualize_runtimes(entry_name: str, runtimes: dict, save: bool = True):
        """
        Visualize the runtimes for a given entry name
        """
        runtimes_entry = {}
        for runtime in runtimes.values():
            if entry_name in runtime.keys():
                runtimes_entry[runtime["total_area"]] = runtime[entry_name]
        fig, ax = plt.subplots()
        ax.scatter(
            runtimes_entry.keys(), runtimes_entry.values(), marker="o", color="blue"
        )
        # make the axes logarithmic
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("area - m2")
        ax.set_ylabel(f"{entry_name} - seconds")
        ax.set_title(f"Runtimes for {entry_name}")
        plt.show()
        save_path = data_path / "figures/other/computing_resources"
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_path / f"{entry_name}_runtimes_scatter.png",
                dpi=300,
                bbox_inches="tight",
            )

        # histogram for normalized runtimes
        normalized_runtimes = [
            seconds / area * 10**6
            for area, seconds in runtimes_entry.items()
            if area > 0
        ]
        fig, ax = plt.subplots()
        ax.hist(normalized_runtimes, bins="auto")
        ax.set_xlabel(f"normalized {entry_name} - seconds per km2")
        ax.set_title(f"Normalized runtimes for {entry_name}")
        plt.show()
        print(f"average runtime is {np.mean(normalized_runtimes):.2} seconds per km2")
        mean_time = np.mean(list(runtimes_entry.values()))
        print(
            f"average total runtime for {entry_name} is {mean_time:.0f} seconds or {mean_time/60:.2f} minutes"
        )
        if save:
            fig.savefig(
                save_path / f"{entry_name}_runtimes_histogram.png",
                dpi=300,
                bbox_inches="tight",
            )
        return

    visualize_runtimes("download", runtimes_selected)
    visualize_runtimes("prediction", runtimes_selected)
    visualize_runtimes("tile_generation", runtimes_selected)
    visualize_runtimes("regularization", runtimes_selected)
    # %%
