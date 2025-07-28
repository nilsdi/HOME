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
    trondheim_projects = [
        "trondheim_2024",
        "trondheim_2023",
        "trondheim_kommune_2022",
        "trondheim_kommune_cir_2022",
        "trondheim_kommune_2021",
        "trondheim_kommune_2020",
        "trondheim_kommune_cir_2020",
        "trøndelag_2022",
        "trøndelag_cir_2022",
        "trøndelag_2021",
        "trøndelag_cir_2021",
        "brungmarka_2023",
        "brungmarka_cir_2023",
        "melhus_2023",
        "melhus_cir_2023",
        "melhus-skaun-orkland-midtre_gauldal_2021",
        "trøndelag_vest_2022",
        "trøndelag_vest_cir_2022",
        "trøndelag_2014",
        "trondheim_2019",
        "trondheim_2017",
        "trondheim_2016",
        "trondheim_2015",
        "trondheim_2010",
        "trondheim_2014",
        "trondheim_2011",
        "trondheim_2013",
        "trondheim_kommune_rektifisert_2018",
        "trondheim_2012",
        "melhus_2019",
        "trondheim_rundt_2016",
        "melhus-skaun_2011",
        "klæbu_bebyggelse_2010",
        "malvik_2010",
        "melhus-skaun_bebyggelse_2011",
        "trondheim_sentrum_2013",
        "selbu-tydal_2017",
        "trøndelag_2010",
        "trøndelag_2009",
        "trondheim_2006",
        "melhus-skaun-klæbu_2005",
        "trondheim_2005",
        "trondheim_tettbebyggelse_2003",
        "trondheim_2008",
        "klæbu_bebyggelse_2005",
        "melhus_skaun_bebyggelse_2006",
        "malvik_2005",
        "selbu_2007",
        "malvik_bebyggelse_2006",
        "midt_norge_2009",
        "selbu_bebyggelse_2007",
        "byneset-trondheim-stjørdal_1997",
        "trondheim_1991",
        "trondheim_1999",
        "byneset_1998",
        "byneset_1996",
        "klæbu_1996",
        "trondheim_øst_1993",
        "trondheim_1992",
        "trondheim-byneset_1996",
        "heimdal-tiller_1991",
        "trondheim_1994",
        "granåsen-sandmoen_1996",
        "melhus_kommune_1998",
        "skjøla-klæbu_1993",
        "orkanger-børsa-buvika_1996",
        "granåsen-lian_1992",
        "klæbu_1984",
        "trondheim_1985",
        "melhus-trondheim_1986",
        "sør-trøndelag_veger_1980",
        "trondheim_1983",
        "trondheim_og_omegn_1981",
        "trondheim_1988",
        "trondheim_1984",
        "trondheim_og_omegn_1980",
        "trondheim_sør_1987",
        "trondheim_1982",
        "ranheim_og_klett_1988",
        "lade-reppe-vikhammar_1981",
        "ilsvika-nyborg-heimdal_1983",
        "reppe-ranheim-rotvoll_1989",
        "skjøla-brøttem_1980",
        "klæbu_1988",
        "orkdal-skaun_1981",
        "klæbu_sentrum_1983",
        "reppe-storesand_1988",
        "trondheim_1977",
        "bymarka_trondheim_1974",
        "bymarka-midtbyen-heimdal_1979",
        "trondheim_1979",
        "trondheim_1971",
        "trondheim_sør_1973",
        "malvik_1976",
        "klæbu-draksten_1979",
        "trondheim_1976",
        "byneset_1979",
        "lund-hyttfossen_1974",
        "trondheim_øst-heimdal-lian_1971",
        "klæbu_1976",
        "nypan-leinstrand_1970",
        "tanem-klæbu_1973",
        "trondheim_øst_1978",
        "børsa-rindal_1974",
        "malvik_1977",
        "være-hommelvik_1976",
        "trondheim-meldal_1964",
        "melhus-midtre_gauldal_1963",
        "klæbu-malvik_1963",
        "trondheim_søre_1969",
        "trondheim_sør_1969",
        "selbu-tydal-holtålen_sør_1962",
        "selbu-tydal-holtålen_nord_1961",
        "heimdal-tiller_1964",
        "strinda_1964",
        "haukvatnet-tiller_1969",
        "tiller-haukvatnet_1969",
        "byåsen-tempe_1969",
        "rye-byneset_1968",
        "trondheim_1969",
        "skaun_1957",
        "melhus_1956",
        "selbusjøen-tya-nea_1952",
        "trondheim-gauldal_1947",
        "ranheim-muruvik_1947",
        "trondheim_strinda_1937",
        "gråkallbanen_1937",
        "trolla_bruk_1937",
        "trondheim_elgeseter_1937",
    ]
    predictions_selected_trondheim = {
        project: prediction
        for project, prediction in predictions_selected.items()
        if project in trondheim_projects
    }
    print(f"Selected {len(predictions_selected_trondheim)} projects for Trondheim")

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
