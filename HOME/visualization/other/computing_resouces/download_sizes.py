# %%
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from shapely.geometry import Polygon
from pathlib import Path
from HOME.utils.get_project_metadata import get_project_geometry, get_project_details


root_dir = Path(__file__).resolve().parents[4]
data_path = root_dir / "data"
print(root_dir)
runtimes_path = data_path / "metadata_log/prediction_main_runtime.json"
download_size_path = data_path / "metadata_log/download_size.json"
print(runtimes_path)

# %%
if __name__ == "__main__":
    # with open(runtimes_path, "r") as f:
    #     runtimes = json.load(f)
    with open(download_size_path, "r") as f:
        download_sizes = json.load(f)
    projects = list(download_sizes.keys())
    selected_projects = [  # filter out all that contain "midlertidig"
        x for x in projects if "midlertidig" not in x
    ]
    print(
        f' reduced projects from {len(projects)} to {len(selected_projects)} by filtering out "midlertidig"'
    )
    geometries = get_project_geometry(selected_projects)
    project_details = get_project_details(selected_projects)
    downloads_selected = {}
    for project, geometry in zip(selected_projects, geometries):
        # get the area
        geometry.to_crs(epsg=32633)  # convert to UTM33N
        total_area = geometry.area.sum()
        downloads_selected[project] = download_sizes[project]
        downloads_selected[project]["total_area"] = total_area
        downloads_selected[project]["original_resolution"] = project_details[project][
            "original_resolution"
        ]

    print(downloads_selected)
    # %%
    fig, ax = plt.subplots()
    disk_uses = []
    areas = []
    resolutions = []
    for value in downloads_selected.values():
        if "size" in value.keys():
            disk_uses.append(value["size"])
            areas.append(value["total_area"])
            resolutions.append(value["original_resolution"])
    # print unique resolutions
    unique_resolutions = set(resolutions)
    print(f"unique resolutions: {unique_resolutions}")
    # normalize the colormap
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min(resolutions), vmax=max(resolutions))
    colors = cmap(norm(resolutions))
    for area, disk_use, resolution, color in zip(areas, disk_uses, resolutions, colors):
        ax.scatter(
            area,
            disk_use,
            marker="o",
            color=color,
            label=f"{resolution} m",
        )
    # ax.scatter(
    #     areas,
    #     disk_uses,
    #     marker="o",
    #     color="blue",
    # )
    # make the axes logarithmic
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("area - m2")
    ax.set_ylabel(f"size - GB")
    # add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("resolution - m")

    ax.set_title(f"Download sizes")
    plt.show()
    save_path = data_path / "figures/other/computing_resources"
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path / "download_sizes_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )

    # %%
    # histogram for normalized runtimes
    normalized_downloads = [
        size / area * 10**6 for area, size in zip(areas, disk_uses) if area > 0
    ]
    fig, ax = plt.subplots()
    ax.hist(
        normalized_downloads,
        bins=35,
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(f"normalized size - GB per km2")
    ax.set_title(f"Normalized download sizes")
    plt.show()
    fig.savefig(
        save_path / "download_sizes_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
