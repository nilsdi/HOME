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

from pathlib import Path

from matrikkel.analysis.building_attributes.get_cohort import get_cohort_from_building
from matrikkel.analysis.building_attributes.get_status import get_building_statuses
from HOME.footprint_analysis.matrikkel_comparison.city_bounding_boxes import (
    get_municipal_boundaries,
)


root_dir = Path(__file__).parents[4]
print(root_dir)


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

def read_FKB(fkb_path:str):
    fle_path = root_dir / fkb_path
    layers = fiona.listlayers(fle_path)
    if "fkb_bygning_omrade" in layers:
        bygning_omrader = gpd.read_file(fle_path, layer="fkb_bygning_omrade")
    else:
        raise Exception(f"the layer bygning_omrade is not in the file {fle_path}")
    return bygning_omrader

def read_FKB_pickle(fkb_pickle_path:str):
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
        self.cohort = self.get_cohort()
        self.fkb_match = False
        self.set_building_statuses()
        self.set_final_status()
        #self.set_footprint(bygning_omrader)

    def get_cohort(self):
        return get_cohort_from_building(self.building_dict)

    def set_footprint_old(self, bygning_omrader):
        # find the correct row:
        try:
            index = bygning_omrader[
                bygning_omrader["bygningsnummer"] == self.bygningsnummer
            ].index[0]
            footprint = bygning_omrader.loc[index, "geometry"]
            footprint_area = bygning_omrader.loc[index, "SHAPE_Area"]
        except Exception as e:
            # print(f'error in get_footprint: {e}')
            footprint = None
            footprint_area = None
            self.fkb_match = False
        self.footprint, self.footprint_area =  footprint, footprint_area

    def  set_footprint(self, FKB_row):
        #print(f'assgigning footprint for {self.bygningsnummer}: {FKB_row}')
        #print(f'geometry: {FKB_row["geometry"]}, area: {FKB_row["SHAPE_Area"]}')
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
        #print(f'the geometry of the FKB row is: {self.fkb_row["geometry"]}')
        if not self.fkb_row["geometry"]:
            self.footprint = None
            self.footprint_area = None
        else:
            self.footprint_area = self.fkb_row["SHAPE_Area"]
            self.footprint = self.fkb_row["geometry"]
        return


def make_buildings(buildings:list[dict], FKB: gpd.GeoDataFrame):
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
    #print(f'length of FKB_rows_used: {len(FKB_rows_used)}')
    for index, row in tqdm(FKB.iterrows(), desc="FKB building objects", total=len(FKB)):
        #print(f'index: {index}, row: {row}')
        if type(index) != int:
            continue
        #print(f'index: {index}, FKB_used: {type(FKB_rows_used)}, first elements: {FKB_rows_used[0:10]}')#, index in FKB_used: {index in FKB_rows_used}')
        if not index in FKB_rows_used:
            new_building = BuildingFKB(row)
            building_objects.append(new_building)
    return building_objects, FKB_matches

def get_time_series(building_objects):
    time = list(range(1900, 2026))
    n_buildings = [0] * len(time)
    build_area = [0] * len(time)
    for building in building_objects:
        if building.cohort > 1900:
            n_buildings[building.cohort - 1901] += 1
            if building.footprint_area:
                build_area[building.cohort - 1900] += building.footprint_area
        else:
            n_buildings[0] += 1
            if building.footprint_area:
                build_area[0] += building.footprint_area
            
    build_area_stock = [0] * len(time)
    for i in range(len(time)):
        build_area_stock[i] = sum(build_area[: i + 1])
    
    return time, n_buildings, build_area, build_area_stock
def preliminary_building_plots(time, n_buildings, build_area, build_area_stock):
    plt.bar(time, n_buildings)
    plt.show()

    # make a bar plot of the building area
    plt.bar(time, build_area)
    plt.show()

    plt.plot(time, build_area_stock, ls=':')
    plt.show()
    return
#%%
city = "trondheim"
city_pickle, city_metadata = get_municipality_pickle(city)
city_buildings = get_all_buildings_from_pickle(city_pickle)
print(f"found {len(city_buildings)} buildings in {city}")
FKB_bygning_path = "data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb"
FKB_bygning_pickle_path = "data/raw/FKB_bygning/Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.pkl"
FKB_bygning_pickle_path = "data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.pkl"
#FKB_bygning = read_FKB(FKB_bygning_path)
FKB_bygning = read_FKB_pickle(FKB_bygning_pickle_path)
#%%
city_building_objects, fkb_matches = make_buildings(city_buildings, FKB_bygning)
print(f'We started with {len(city_buildings)} buildings from matrikkel and {len(FKB_bygning)} buildings from FKB')
print(f'found {fkb_matches} FKB matches, meaning we have {fkb_matches} many buildings that combine FKB and matrikkel data')
print(f' we further have {len(city_buildings) - fkb_matches} buildings that represent only matrikkel data, and {len(FKB_bygning) - fkb_matches} buildings that represent only FKB data')
time, n_buildings, build_area, build_area_stock = get_time_series(city_building_objects)
#preliminary_building_plots(time, n_buildings, build_area, build_area_stock)

#%%
print(city_building_objects[0])
print(city_building_objects[0].footprint_area)
print(city_building_objects[0].building_dict)
bd_dict = city_building_objects[0].building_dict
bd_dict["bebygdAreal"]

#%%
for building in city_building_objects:
    footprint_mat = building.building_dict["bebygdAreal"]
    footprint_FKB = building.footprint_area
    if footprint_mat != 0:
        if footprint_mat != footprint_FKB:
            print(f'footprint mismatch for building {building.bygningsnummer}: matrikkel: {footprint_mat}, FKB: {footprint_FKB}')
#%%
time = list(range(1900, 2026))

def get_build_area_stock(time, building_objects):
    build_area_stock = [0] * len(time)
    for building in building_objects:
        build_area_stock = building.add_to_stock(time, build_area_stock)
    return build_area_stock
## atributing unknown cohorts to -1
# matrikkel_and FKB only
building_objects_matrikkel_FKB = [b for b in city_building_objects if type(b) == BuildingMatrikkel]
matrikkel_FKB_stock = get_build_area_stock(time, building_objects_matrikkel_FKB)
# both FKB and matrikkel
building_objects_all = [b for b in city_building_objects if type(b) == BuildingFKB]
object_base_stock = get_build_area_stock(time, city_building_objects)
plt.plot(time, object_base_stock, ls=':', c = 'r', label = 'object base')
plt.plot(time, build_area_stock, ls=':', c = 'b', label = 'total matrikkel')
plt.xlim(1970, 2025)
plt.show()
#%%
time = list(range(1900, 2026))

def get_build_area_stock(time, building_objects):
    build_area_stock = [0] * len(time)
    for building in building_objects:
        build_area_stock = building.add_to_stock(time, build_area_stock)
    return build_area_stock

object_base_stock = get_build_area_stock(time, city_building_objects)
plt.plot(time, object_base_stock, ls=':', c = 'r', label = 'object base')
plt.plot(time, build_area_stock, ls=':', c = 'b', label = 'total matrikkel')
plt.xlim(1970, 2025)
plt.show()

#%%
final_statuses = [b.final_status for b in city_building_objects]
# Create the histogram
plt.hist(final_statuses, bins=len(set(final_statuses)), edgecolor='black')

# Set the xtick labels
plt.xticks(rotation=45, ha='right')

# Set labels and title
plt.xlabel('Final Status')
plt.ylabel('Frequency')
plt.title('Histogram of Final Statuses')

# Show the plot
plt.tight_layout()
plt.show()
#%%
testbuildings = city_building_objects[:10]
testbuildings[0].building_dict
# %%
FKB_bygning.iloc[100000:100010]
print(FKB_bygning.columns)
# print the crs of the FKB data
print(FKB_bygning.crs)

# %%
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform
from pyproj import Transformer

def filter_buildings_to_overlapping(city_building_objects, city, FKB_data):
    city_boundary = shape(get_municipal_boundaries(city))
    transformer_to_4326 = Transformer.from_crs(FKB_data.crs,  "epsg:4326", always_xy=True).transform

    def check_overlap(building_geometry, city_boundary):
        return building_geometry.intersects(city_boundary)
    
    overlapping_buildings = []
    for building in city_building_objects:
        if not building.footprint:
            continue
        converted_building = transform(transformer_25832_to_4326, building.footprint)
        overlapping = check_overlap(converted_building, city_boundary)
        if overlapping:
            overlapping_buildings.append(building)
    return overlapping_buildings

overlapping_buildings = filter_buildings_to_overlapping(city_building_objects, city, FKB_bygning)
print(f'found {len(overlapping_buildings)} overlapping buildings out of {len(city_building_objects)}')
#%%
city_boundary = shape(get_municipal_boundaries(city))
transformer_25832_to_4326 = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True).transform

def check_overlap(building_geometry, city_boundary):
    return building_geometry.intersects(city_boundary)

for building in city_building_objects[90000:90100]:
    print(f'type of building.footprint: {type(building.footprint)}')
    print(f'type of city_boundary: {type(city_boundary)}')
    if not building.footprint:
        overlapping = False
        continue
    converted_building = transform(transformer_25832_to_4326, building.footprint)
    overlapping = check_overlap(converted_building, city_boundary)
    print(f'found overlap for building {building}: {overlapping}')
# %%
