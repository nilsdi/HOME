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
    get_project_metadata,
    get_project_details,
)


root_dir = Path(__file__).resolve().parents[4]
data_path = root_dir / "data"
print(root_dir)
predictions_log_path = data_path / "metadata_log/predictions_log.json"
print(predictions_log_path)
today = datetime.today().strftime("%Y-%m-%d")
print(f"today: {today}")

# %%
if __name__ == "__main__":
    with open(predictions_log_path, "r") as f:
        predictions = json.load(f)
    projects = [p["project_name"] for p in predictions.values()]
    selected_projects = filter_projects_without_metadata(projects)
    print(
        f" reduced projects from {len(projects)} to {len(selected_projects)} by filtering out all without metadata"
    )
    geometries = get_project_geometry(selected_projects)
    project_details = get_project_details(selected_projects)
    predictions_selected = {}
    for project, geometry in zip(selected_projects, geometries):
        # get the area
        geometry.to_crs(epsg=32633)  # convert to UTM33N
        total_area = geometry.area.sum()
        for prediction_log in predictions.values():
            if prediction_log["project_name"] == project:
                predictions_selected[project] = prediction_log
                break
        predictions_selected[project]["total_area"] = total_area
        predictions_selected[project]["original_resolution"] = project_details[project][
            "original_resolution"
        ]
        predictions_selected[project]["capture_date"] = project_details[project][
            "capture_date"
        ]
        predictions_selected[project]["bandwidth"] = project_details[project][
            "bandwidth"
        ]

    print(predictions_selected)

    # %%
    def scatter_plot_recall(
        predictions_selected: dict, category: str, save: bool = True
    ):
        """
        Create a scatter plot of the recall for each project
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        recalls = []
        category_data = []
        project_dates = []
        colors = []
        marker_sizes = []
        markersize_constant = (
            400  # Adjust this constant to change the size of the markers
        )
        for project, prediction in predictions_selected.items():
            if "recall" in prediction.keys():
                recalls.append(prediction["recall"])
                category_data.append(prediction[category])

                bandwidth = str(prediction["bandwidth"])
                if bandwidth == "BW":
                    colors.append("black")
                elif "RGB" in bandwidth:
                    colors.append("red")
                else:
                    colors.append("blue")
                    print(
                        f"Project {project} has an unknown orthophoto type.: {bandwidth}"
                    )

                original_resolution = prediction["original_resolution"]
                marker_sizes.append(
                    markersize_constant * original_resolution
                )  # Adjust marker size based on resolution
                project_dates.append(prediction["capture_date"])
            else:
                pass
                # print(f"Project {project} has no recall value.")

        ax.scatter(category_data, recalls, color=colors, s=marker_sizes, alpha=0.6)
        ax.set_xlabel(f"{category}")
        ax.set_ylabel("Recall - %")
        ax.set_title(f"Recall vs {category}")
        # add a legend with the colors, and the marker sizes
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="BW - 0.1m",
                markerfacecolor="black",
                markersize=np.sqrt(0.1 * markersize_constant),
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="RGB - 0.3m",
                markerfacecolor="red",
                markersize=np.sqrt(0.3 * markersize_constant),
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Other - 0.5m",
                markerfacecolor="blue",
                markersize=np.sqrt(0.5 * markersize_constant),
            ),
        ]
        ax.legend(handles=legend_elements, title="color/res legend", loc="upper left")

        figure_path = data_path / "figures/ML_prediction/performance"
        figure_path.mkdir(parents=True, exist_ok=True)
        if save:
            fig.savefig(
                figure_path / f"recall_vs_{category}_{today}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"Figure saved as recall_vs_area_{today}.png")

        plt.show()

    scatter_plot_recall(predictions_selected, "capture_date", save=True)

    # %%
    print([p["original_resolution"] for p in predictions_selected.values()])
