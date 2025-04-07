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
    kicked_out = [x for x in projects if x not in selected_projects]
    print(
        f' reduced projects from {len(projects)} to {len(selected_projects)} by filtering out "midlertidig, removed {kicked_out}"',
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
    project_names = []
    for project, value in downloads_selected.items():
        if "size" in value.keys():
            disk_uses.append(value["size"])
            areas.append(value["total_area"])
            resolutions.append(value["original_resolution"])
            project_names.append(project)
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
    fig, ax = plt.subplots()
    ax.hist(
        disk_uses,
        bins=35,
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel(f"absolute size - GB")
    ax.set_title(f"Absolute download sizes")
    plt.show()
    # %%
    # lets find a better way to visualize the absolute download sizes: we print a box (square) for each project
    # where the area of the box is the download size and the color is the resolution. we sort the projects by download size
    # and plot them in a line from smallest to largest
    fig, ax = plt.subplots(figsize=(20, 3))
    # sort the projects by download size
    size_details = [
        [project, size, resolution]
        for project, size, resolution in zip(project_names, disk_uses, resolutions)
    ]
    size_details.sort(key=lambda x: x[1])

    # print the projects in a line
    x_start = 0
    text_labels = {}  # to store the text labels
    for i, (project_name, size, resolution) in enumerate(size_details):
        # print(size)
        # print(resolution)
        # get the color
        color = cmap(norm(resolution))
        # print(color)
        # create a square with area size
        coords = [
            (x_start, 0),
            (x_start + size**0.5, 0),
            (x_start + size**0.5, size**0.5),
            (x_start, size**0.5),
            (x_start, 0),
        ]
        # print(coords)
        square = Polygon(coords)
        ax.add_patch(plt.Polygon(coords, facecolor=color, edgecolor="black"))
        x_start += size**0.5 * 1.1
        available_x_space = size**0.5
        # add the text label details
        text_labels[i] = {
            "size": size,
            "project_name": project_name,
            "resolution": resolution,
            "available_x_space": available_x_space,
            "label_middle": x_start - available_x_space * 0.6,
        }
    for i, text_label in text_labels.items():
        if text_label["available_x_space"] > x_start / 80:
            ax.text(
                text_label["label_middle"],
                0,
                f"{text_label['project_name']}",
                ha="center",
                va="top",
                fontsize=6,
                rotation=90,
            )
            ax.text(
                text_label["label_middle"],
                text_label["size"] ** 0.5 * 1.15,
                f"{text_label['size']:.0f} GB",
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=90,
            )
    # also plot the total size as a single box in the background
    total_size = sum(disk_uses)
    total_coords = [
        (x_start, 0),
        (x_start, total_size**0.5),
        (x_start + total_size**0.5, total_size**0.5),
        (x_start + total_size**0.5, 0),
        (x_start, 0),
    ]
    ax.add_patch(
        plt.Polygon(total_coords, facecolor="lightgrey", edgecolor="black", alpha=0.5)
    )
    ax.text(
        x_start + total_size**0.5 * 0.5,
        total_size**0.5 * 0.5,
        f"Total size: {total_size:.0f} GB",
        ha="center",
        va="center",
        fontsize=7,
        rotation=0,
    )
    # add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("resolution - m")
    # rescale the col

    ax.set_xlim(0, x_start + total_size**0.5 * 1.05)
    ax.set_ylim(-0.05, total_size**0.5)
    ax.set_aspect("equal")
    plt.axis("off")
    fig.savefig(
        save_path / "download_sizes_boxes.png",
        dpi=300,
        bbox_inches="tight",
    )

# %%
for p in size_details:
    print(p)

# %%
