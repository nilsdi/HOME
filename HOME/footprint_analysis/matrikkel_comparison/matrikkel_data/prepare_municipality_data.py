"""
Fetching and preparing the municipality data for the matrikkel comparison.
"""

# %%
import pickle
import fiona
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import folium

from pathlib import Path
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import Point, MultiPolygon, Polygon, shape
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from matrikkel.analysis.building_attributes.get_cohort import get_cohort_from_building
from matrikkel.analysis.building_attributes.get_status import get_building_statuses
from matrikkel.analysis.building_attributes.get_build_area import get_build_area
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)


root_dir = Path(__file__).parents[4]
# print(root_dir)


# %%
def get_municipality_pickle(municipality: str):
    municipality = municipality.lower()
    municipality_pickle_path = (
        root_dir / "data" / "raw" / "matrikkel" / "municipality_pickles"
    )
    with open(municipality_pickle_path / f"{municipality}.pkl", "rb") as file:
        municipality_buildings, metadata = pickle.load(file)
    return municipality_buildings, metadata


def get_all_buildings_from_pickle(pickle):
    buildings = []
    for entry in pickle:
        entry_object = entry[1]
        for building in entry_object:
            if building["name"] == "Bygning":
                buildings.append(building)
    return buildings


def read_FKB(fkb_path: str):
    fle_path = root_dir / fkb_path
    layers = fiona.listlayers(fle_path)
    if "fkb_bygning_omrade" in layers:
        bygning_omrader = gpd.read_file(fle_path, layer="fkb_bygning_omrade")
    else:
        raise Exception(f"the layer bygning_omrade is not in the file {fle_path}")
    return bygning_omrader


def read_FKB_pickle(fkb_pickle_path: str):
    file_path = root_dir / fkb_pickle_path
    with open(file_path, "rb") as file:
        fkb_data = pickle.load(file)
    return fkb_data


class Building:
    def __init__(self):
        self.cohort = -1

    def __repr__(self):
        return f"Building {self.bygningsnummer} from {self.cohort}"


class BuildingMatrikkel(Building):
    def __init__(self, building_dict):
        self.building_dict = building_dict
        self.bygningsnummer = self.building_dict["bygningsnummer"]
        self.location, self.location_crs = self.get_location()
        self.cohort = self.get_cohort()
        self.fkb_match = False
        self.set_building_statuses()
        self.set_final_status()
        # self.set_footprint(bygning_omrader)
        self.set_footprint_area_matrikkel()

    def get_cohort(self):
        return get_cohort_from_building(self.building_dict)

    def get_location(self):
        # check if the building has a location
        if not "representasjonspunkt" in self.building_dict.keys():
            return None, None
        x = self.building_dict["representasjonspunkt"]["position"]["x"]
        y = self.building_dict["representasjonspunkt"]["position"]["y"]
        koordinatsystemKodeId = self.building_dict["representasjonspunkt"][
            "koordinatsystemKodeId"
        ]["value"]
        if koordinatsystemKodeId == 10:
            location_crs = "EPSG:5122"  # "EPSG:25832" ?
        elif koordinatsystemKodeId == 11:
            location_crs = "EPSG:5123"
        else:
            raise Exception(
                f"Unknown coordinate system {koordinatsystemKodeId} for building {self.bygningsnummer}"
            )
        return (x, y), location_crs

    def set_footprint(self, FKB_row):
        # print(f'assgigning footprint for {self.bygningsnummer}: {FKB_row}')
        # print(f'geometry: {FKB_row["geometry"]}, area: {FKB_row["SHAPE_Area"]}')
        if FKB_row["geometry"].empty:
            self.footprint = None
            self.footprint_area = None
        else:
            self.footprint = FKB_row["geometry"].values[0]
            self.footprint_area = FKB_row["SHAPE_Area"].values[0]
            self.fkb_match = True

    def set_building_statuses(self):
        self.statuses = get_building_statuses(self.building_dict)

    def set_final_status(self):
        if not self.statuses:
            self.final_status = "No status"
        else:
            self.final_status = self.statuses[max(self.statuses.keys())]

    def set_footprint_area_matrikkel(self):
        self.footprint_area_matrikkel = get_build_area(self.building_dict)


class BuildingFKB(Building):
    def __init__(self, fkb_row):
        self.fkb_row = fkb_row
        self.bygningsnummer = fkb_row["bygningsnummer"]
        self.cohort = -1
        self.final_status = fkb_row["bygningsstatus"]
        self.set_footprint()

    def set_footprint(self):
        # print(f'the geometry of the FKB row is: {self.fkb_row["geometry"]}')
        if not self.fkb_row["geometry"]:
            self.footprint = None
            self.footprint_area = None
            self.location = None
        else:
            self.footprint_area = self.fkb_row["SHAPE_Area"]
            self.footprint = self.fkb_row["geometry"]
            self.location = (
                self.footprint.centroid.x,
                self.footprint.centroid.y,
            )
            self.location_crs = "EPSG:25832"
        return


def make_buildings(buildings: list[dict], FKB: gpd.GeoDataFrame):
    """
    Make building objects by combining both FKB and matrikkel data.
    """
    building_objects = []
    FKB_matches = 0
    FKB_rows_used = []
    for building in tqdm(
        buildings, desc="matrikkel building objects", total=len(buildings)
    ):
        new_building = BuildingMatrikkel(building)
        FKB_row_index = FKB[FKB["bygningsnummer"] == new_building.bygningsnummer].index
        if len(FKB_row_index) == 0:
            continue
        FKB_rows_used.append(int(FKB_row_index.values[0]))
        FKB_row = FKB.loc[FKB_row_index]
        new_building.set_footprint(FKB_row)
        building_objects.append(new_building)
        if new_building.fkb_match:
            FKB_matches += 1
    # print(f'length of FKB_rows_used: {len(FKB_rows_used)}')
    for index, row in tqdm(FKB.iterrows(), desc="FKB building objects", total=len(FKB)):
        # print(f'index: {index}, row: {row}')
        if type(index) != int:
            continue
        # print(f'index: {index}, FKB_used: {type(FKB_rows_used)}, first elements: {FKB_rows_used[0:10]}')#, index in FKB_used: {index in FKB_rows_used}')
        if not index in FKB_rows_used:
            new_building = BuildingFKB(row)
            building_objects.append(new_building)
    return building_objects, FKB_matches


def get_time_series(building_objects):
    time = list(range(1900, 2026))
    n_buildings = [0] * len(time)
    build_area_upper = [0] * len(time)
    build_area_lower = [0] * len(time)
    for building in building_objects:
        lower_area = building.footprint_area
        upper_area = building.footprint_area
        if type(building) == BuildingMatrikkel:
            if building.footprint_area_matrikkel:
                lower_area = min(
                    [building.footprint_area_matrikkel, building.footprint_area]
                )
                upper_area = max(
                    [building.footprint_area_matrikkel, building.footprint_area]
                )
        if building.cohort > 1900:
            n_buildings[building.cohort - 1901] += 1
            if building.footprint_area:
                build_area_upper[building.cohort - 1900] += upper_area
                build_area_lower[building.cohort - 1900] += lower_area
        else:  # for the upper bound we include unlabeled buildings in the older cohorts, for lower bound we ignore them
            n_buildings[0] += 1
            if building.footprint_area:
                build_area_upper[0] += upper_area

    build_area_stock_upper = [0] * len(time)
    build_area_stock_lower = [0] * len(time)
    for i in range(len(time)):
        build_area_stock_upper[i] = sum(build_area_upper[: i + 1])
        build_area_stock_lower[i] = sum(build_area_lower[: i + 1])
        # could in theory check final status and take away the demolished buildings, but that's a very very low number

    return (
        time,
        n_buildings,
        build_area_upper,
        build_area_stock_upper,
        build_area_stock_lower,
    )


def preliminary_building_plots(
    time, n_buildings, build_area, build_area_stock_upper, build_area_stock_lower
):
    plt.bar(time, n_buildings)
    plt.title("Number of buildings constructed")
    plt.show()

    # make a bar plot of the building area
    plt.bar(time, build_area)
    plt.title("Building area constructed")
    plt.show()

    plt.plot(time, build_area_stock_lower, ls=":")
    plt.plot(time, build_area_stock_upper, ls=":")
    plt.title("Building area stock lower and upper bound")
    plt.show()
    return


def filter_FKB_to_overlapping(FKB_data: gpd.GeoDataFrame, city_boundaries):
    """
    Filter the FKB data to only include buildings that overlap with the city boundaries.
    The function transforms the FKB data to the same CRS as the city boundaries
    and then filters the data to only include buildings that intersect with the city boundaries.

    Args:
        FKB_data (gpd.GeoDataFrame): The FKB data to filter - any crs is fine
        city_boundaries (shapely.geometry.Polygon): The city boundaries to filter by - crs EPSG:4326

    Returns:
        gpd.GeoDataFrame: The filtered FKB data that overlaps with the city boundaries.
    """
    # get the FKB data in the same crs as the city boundaries
    FKB_25832 = FKB_data.to_crs("epsg:4326")
    filtered_FKB = FKB_25832[FKB_25832.intersects(city_boundaries)]
    return filtered_FKB


def get_city_data(city: str):
    """
    Get the city data from the matrikkel and FKB data, and make building objects.
    """
    city_pickle, city_metadata = get_municipality_pickle(city)
    city_buildings_matrikkel = get_all_buildings_from_pickle(city_pickle)
    print(f"found {len(city_buildings_matrikkel)} matrikkel buildings in {city}")
    FKB_bygning_Norge_pickle_path = (
        "data/raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
    )
    FKB_bygning_Norge = read_FKB_pickle(FKB_bygning_Norge_pickle_path)
    city_boundaries = shape(get_municipal_boundaries(city))
    FKB_bygning_city = filter_FKB_to_overlapping(FKB_bygning_Norge, city_boundaries)
    print(
        f"found {len(FKB_bygning_city)} overlapping buildings out of {len(FKB_bygning_Norge)}"
    )

    return city_buildings_matrikkel, FKB_bygning_city


def make_age_map(
    city_building_objects: list, city_boundaries: dict, resolution_long_edge: int = 512
):
    """
    Make a map of the buildings in the city, colored by their age.
    We therefore build a grid of the city and assign the buildings to the grid cells.
    the cells are colored by the age of the buildings in the cell.

    Args:
        city_building_objects (list): The list of building objects to plot.
        city_boundaries (dict): The city boundaries to plot the buildings in - epsg:25832
        resolution_long_edge (int): The resolution of the grid cells - default is 512

    Returns:
        fig, ax: The figure and axis of the plot.
    """
    # make a grid of the city boundaries
    bounds = city_boundaries.bounds
    x_min = bounds[0]
    x_max = bounds[2]
    y_min = bounds[1]
    y_max = bounds[3]
    x_range = x_max - x_min
    y_range = y_max - y_min

    max_extended = max(x_range, y_range)
    grid_size = max_extended / resolution_long_edge
    x_grid = int(x_range / grid_size)
    y_grid = int(y_range / grid_size)
    # make an array to store the boundaries of the grid cells
    x_grid_bounds = [x_max + i * grid_size for i in range(x_grid + 1)]
    y_grid_bounds = [y_min + i * grid_size for i in range(y_grid + 1)]
    # make an array to store the age of the buildings in the grid cells
    ages_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    areas_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    average_age_array = [[0 for _ in range(y_grid)] for i in range(x_grid)]
    total_area_array = [[0 for _ in range(y_grid)] for i in range(x_grid)]
    color_array = [[[0] for _ in range(y_grid)] for i in range(x_grid)]
    # print(ages_array)
    transformer_from_5122 = Transformer.from_crs(
        "EPSG:25832", "EPSG:4326", always_xy=True  # building.location_crs
    ).transform
    for building in city_building_objects:
        if building.location:
            if building.location_crs == "EPSG:5122":
                location = transform(transformer_from_5122, Point(building.location))
            else:
                location = Point(building.location)
                # print(building.location_crs)
            # get the coordinates of the building
            x = location.x
            y = location.y
            # get the grid cell that the building is in
            x_index = int((x - x_min) / grid_size)
            y_index = int((y - y_min) / grid_size)
            # print(
            #     f"building {building.bygningsnummer} is in grid cell {x_index}, {y_index}"
            # )
            # check if the building is in the grid cell
            if x_index >= 0 and x_index < x_grid and y_index >= 0 and y_index < y_grid:
                # add the age of the building to the grid cell
                ages_array[x_index][y_index].append(building.cohort)
                areas_array[x_index][y_index].append(building.footprint_area)

    # insert the average age (weighed by the area) of the buildings in the grid cells
    for i in range(x_grid):
        for j in range(y_grid):
            # exclude all age entries that are -1
            average_age_entries = [age for age in ages_array[i][j] if age != -1]
            areas_entries = [
                area
                for area, age in zip(areas_array[i][j], ages_array[i][j])
                if age != -1
            ]
            if len(average_age_entries) == 0:
                average_age_array[i][j] = -1
                color_array[i][j] = -1
                continue
            total_area = sum(areas_entries)
            if (
                total_area == 0
            ):  # if all available buildings sum to 0, we count each the same
                areas_entries = [1 for _ in range(len(areas_entries))]
            average_age_array[i][j] = np.average(
                average_age_entries, weights=areas_entries
            )
            total_area_array[i][j] = total_area

    max_total_area = max([max(total_area_array[i]) for i in range(x_grid)])
    # create a colormap for the ages - 1900-2025
    norm = mcolors.Normalize(vmin=1900, vmax=2025)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for i in range(x_grid):
        for j in range(y_grid):
            if average_age_array[i][j] == -1:
                color_array[i][j] = (1, 1, 1, 1)  # make white
                continue

            color = cmap(norm(average_age_array[i][j]))
            color_rgba = mcolors.to_rgba(color)
            color_rgba = (
                color_rgba[0],
                color_rgba[1],
                color_rgba[2],
                color_rgba[3]
                * min([total_area_array[i][j] / (max_total_area * 0.3), 1]) ** 0.35,
            )
            color_array[i][j] = color_rgba

    print(ages_array[294][161])
    print(areas_array[294][161])
    print(average_age_array[294][161])

    # make a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    color_array = np.array(color_array)
    color_array = np.rot90(color_array)
    # display the color array as an image
    ax.imshow(
        color_array,
        extent=(x_min, x_max, y_min, y_max),
        interpolation="bilinear",
        aspect="auto",
    )
    # add a colorbar
    sm.set_array([])
    inset_axis_cbar = inset_axes(
        ax,
        width=0.2,
        height="40%",
        loc="lower left",
        bbox_to_anchor=(0.1, 0, 0.8, 0.8),
        bbox_transform=ax.transAxes,
        borderpad=0.1,
    )
    cbar = plt.colorbar(
        sm, cax=inset_axis_cbar, orientation="vertical", pad=0.1, shrink=0.5
    )
    cbar.set_label("Average building age")
    # fig.colorbar(sm, ax=ax, label="Average building age")
    # add the city boundaries
    city_gdf = gpd.GeoSeries(city_boundaries, crs=4326)
    city_gdf.plot(ax=ax, edgecolor="darkgrey", linewidth=0.5, facecolor="none")
    ax.axis("off")
    return


# %%
if __name__ == "__main__":
    # get the city data for Trondheim
    city = "trondheim"
    city_buildings_matrikkel, FKB_bygning_city = get_city_data(city)
    city_building_objects, fkb_matches = make_buildings(
        city_buildings_matrikkel, FKB_bygning_city
    )
    print(
        f"We started with {len(city_buildings_matrikkel)} buildings from matrikkel and {len(FKB_bygning_city)} buildings from FKB"
    )
    print(
        f"found {fkb_matches} FKB matches, which are buildings that combine FKB and matrikkel data"
    )
    print(
        f" we further have {len(city_buildings_matrikkel) - fkb_matches} buildings that represent only matrikkel data, "
        + f"and {len(FKB_bygning_city) - fkb_matches} buildings that represent only FKB data"
    )
    # %%
    (
        time,
        n_buildings,
        build_area_upper,
        build_area_stock_upper,
        build_area_stock_lower,
    ) = get_time_series(city_building_objects)
    preliminary_building_plots(
        time,
        n_buildings,
        build_area_upper,
        build_area_stock_upper,
        build_area_stock_lower,
    )
    # %%
    city_boundaries = shape(get_municipal_boundaries(city))
    make_age_map(city_building_objects, city_boundaries, resolution_long_edge=500)
    # %%

    # make a small bbox and only plot the buildings in that bbox
    bbox = [
        [63.418, 10.442],  # southwest corner
        [63.428, 10.460],  # northeast corner
    ]
    center_bbox = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]
    m = folium.Map(location=center_bbox, zoom_start=14)
    # draw the bbox
    folium.Rectangle(
        bounds=bbox, color="blue", fill=True, fill_opacity=0.2, weight=1
    ).add_to(m)

    for building in city_building_objects:  # [0:1000]:
        if type(building) == BuildingFKB:
            continue
        if not building.location:
            continue
        # first we convert the crs
        transformer_to_4326 = Transformer.from_crs(
            "epsg:25832", "epsg:4326", always_xy=True  # building.location_crs
        ).transform
        # print(f'building location: {building.location}, crs: {building.location_crs}')
        building_location = transform(transformer_to_4326, Point(building.location))
        building_location = (
            building_location.y,
            building_location.x,
        )  # convert to tuple
        # if the building location point  is overlapping the bbox, we add it to the map
        if building_location[0] > bbox[0][0] and building_location[0] < bbox[1][0]:
            if building_location[1] > bbox[0][1] and building_location[1] < bbox[1][1]:
                folium.Rectangle(
                    bounds=[
                        [
                            building_location[0] - 0.00001,
                            building_location[1] - 0.00002,
                        ],
                        [
                            building_location[0] + 0.00001,
                            building_location[1] + 0.00002,
                        ],
                    ],
                    color="red",
                    fill=True,
                    fill_opacity=0.2,
                    weight=1,
                ).add_to(m)
                if not building.footprint:
                    continue
                # print(f"added building {building.bygningsnummer} to the map")
                # plot the actual footprint:
                footprint = (
                    building.footprint
                )  # transform(transformer_to_4326, building.footprint)
                # add to the map
                folium.GeoJson(
                    footprint,
                    style_function=lambda x: {
                        "color": "green",
                        "weight": 1,
                        "fillOpacity": 0.2,
                    },
                ).add_to(m)

    m

    # %%
    for building in city_building_objects:
        if type(building) == BuildingMatrikkel:
            continue
        if not building.footprint:
            continue
        # print(building.footprint)
        # break
        # first we convert the crs
        transformer_to_4326 = Transformer.from_crs(
            "epsg:25832", "epsg:4326", always_xy=True  # building.location_crs
        ).transform
        footprint = (
            building.footprint
        )  # transform(transformer_to_4326, building.footprint)
        # add to the map
        folium.GeoJson(
            footprint,
            style_function=lambda x: {
                "color": "purple",
                "weight": 1,
                "fillOpacity": 0.2,
            },
        ).add_to(m)
    m

    # make a folium map of all the buildings
