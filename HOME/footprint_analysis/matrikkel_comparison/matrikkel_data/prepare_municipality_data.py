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
import folium

from pathlib import Path
from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry import Point, MultiPolygon, Polygon, shape

from matrikkel.analysis.building_attributes.get_cohort import get_cohort_from_building
from matrikkel.analysis.building_attributes.get_status import get_building_statuses
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
        pass

    def add_to_stock(self, time, stock):
        for i, year in enumerate(time):
            add = False
            if self.cohort <= year:
                add = True
                if self.final_status in ["Bygging avlyst", "Bygningsnummer utgått"]:
                    add = False
                elif self.final_status in ["Bygning revet/brent"]:
                    if max(self.statuses.keys()).year <= year:
                        add = False
            if add and self.footprint_area:
                stock[i] += self.footprint_area
        return stock

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
        else:
            self.footprint_area = self.fkb_row["SHAPE_Area"]
            self.footprint = self.fkb_row["geometry"]
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
        if building.cohort > 1900:
            n_buildings[building.cohort - 1901] += 1
            if building.footprint_area:
                build_area_upper[building.cohort - 1900] += building.footprint_area
                build_area_lower[building.cohort - 1900] += building.footprint_area
        else:  # for the upper bound we include unlabeled buildings in the older cohorts, for lower bound we ignore them
            n_buildings[0] += 1
            if building.footprint_area:
                build_area_upper[0] += building.footprint_area

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
