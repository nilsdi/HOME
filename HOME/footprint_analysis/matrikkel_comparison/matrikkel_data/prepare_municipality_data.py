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

class BuildingMatrikkel:
    def __init__(self, building_dict):
        self.building_dict = building_dict
        self.bygningsnummer = self.building_dict["bygningsnummer"]
        self.cohort = self.get_cohort()
        self.fkb_match = True
        self.set_building_statuses()
        self.set_final_status()
        #self.set_footprint(bygning_omrader)

    def get_cohort(self):
        return get_cohort_from_building(self.building_dict)

    def set_footprint(self, bygning_omrader):
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
    
    def set_building_statuses(self):
        self.statuses = get_building_statuses(self.building_dict)
    
    def set_final_status(self):
        if not self.statuses:
            self.final_status = "No status"
        else:
            self.final_status = self.statuses[max(self.statuses.keys())]
    
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

class BuildingFKB:

    def __init__(self, fkb_row):
        self.fkb_row = fkb_row
        self.bygningsnummer = fkb_row["bygningsnummer"]
        self.building_area = fkb_row["SHAPE_Area"]
        self.footpringt = fkb_row["geometry"]


def make_buildings(buildings:list[dict], FKB: gpd.GeoDataFrame):
    building_objects = []
    FKB_matches = 0
    for building in tqdm(
        buildings, desc="building objects", total=len(buildings)
    ):
        new_building = BuildingMatrikkel(building)
        new_building.set_footprint(FKB)
        building_objects.append(new_building)
        if new_building.fkb_match:
            FKB_matches += 1
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
city_building_objects, fkb_matches = make_buildings(city_buildings, FKB_bygning)
print(f'found {fkb_matches} FKB matches from {len(city_buildings)} buildings form matrikkel and {len(FKB_bygning)} buildings from FKB')
time, n_buildings, build_area, build_area_stock = get_time_series(city_building_objects)
preliminary_building_plots(time, n_buildings, build_area, build_area_stock)

#%%
object_base_stock = [0] * len(time)
for building in city_building_objects:
    object_base_stock = building.add_to_stock(time, object_base_stock)
plt.plot(time, object_base_stock, ls=':', c = 'r')
plt.plot(time, build_area_stock, ls=':', c = 'b')
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

# %%

