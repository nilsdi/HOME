"""
Fetching and preparing the municipality data for the matrikkel comparison.
"""

# %%
import pickle
from pathlib import Path
import fiona
import pandas as pd
from tqdm import tqdm

import geopandas as gpd

from matrikkel.analysis.building_attributes.get_cohort import get_cohort_from_building

# %%
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


trondheim, trdf_metadata = get_municipality_pickle("trondheim")
print(trdf_metadata)

# %%
print(trdf_metadata)
# print(trondheim[1])
actutual_buildings = []
for entry in trondheim:
    # print(f'each entry is a {type(entry)} of length {len(entry)}')

    # print(f'the 1st entry in the enrty_object is {entry[0]}')
    # print(f'the 2nd entry in the enrty_object is of type {type(entry[1])} and length {len(entry[1])}')
    entry_object = entry[1]
    for building in entry_object:
        # print(f'each building is a {type(building)}')
        # print(building)
        if building["name"] == "Bygning":
            actutual_buildings.append(building)
            # print(building)

print(f"found {len(actutual_buildings)} buildings")

# %%
import fiona

FKB_bygning_path = os.path.join(
    root_dir, "data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb"
)
# list all layers
layers_bygning = fiona.listlayers(FKB_bygning_path)
print(f" the layers in the FKB bygning data are: {layers_bygning}")

bygning_omrader = gpd.read_file(FKB_bygning_path, layer="fkb_bygning_omrade")
print(bygning_omrader.head())
# %%
import pandas as pd

print(bygning_omrader.columns)
print(bygning_omrader.head())
# %%
building_numbers_fkb = bygning_omrader["bygningsnummer"]
# for b in building_numbers_fkb:
for index, building_footprint in bygning_omrader.iterrows():
    bygningsnummer = building_footprint["bygningsnummer"]
    if not pd.isna(bygningsnummer):
        print(f"{index=}, {bygningsnummer=}")


# print(building_numbers_fkb)
# %%
class Building:
    def __init__(self, building_dict):
        self.building_dict = building_dict
        self.bygningsnummer = self.building_dict["bygningsnummer"]
        self.cohort = self.get_cohort()
        self.fkb_match = True
        self.footprint, self.footprint_area = self.get_footprint(bygning_omrader)
        # self.footprint_area = self.get_footprint_area()

    def get_cohort(self):
        return get_cohort_from_building(self.building_dict)

    def get_footprint(self, bygning_omrader):
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
        return footprint, footprint_area


building_objects = []
FKB_matches = 0
for building in tqdm(
    actutual_buildings, desc="building objects", total=len(actutual_buildings)
):
    building_objects.append(Building(building))
    if building_objects[-1].fkb_match:
        FKB_matches += 1

    # print(f'new building with cohort {building_objects[-1].cohort}, footprint {building_objects[-1].footprint} and footprint area {building_objects[-1].footprint_area}')
# %%
print(
    f"found {FKB_matches} FKB matches out of {len(actutual_buildings)} buildings in matrikkel and {len(bygning_omrader)} buildings in FKB"
)
# %%

time = list(range(1900, 2026))
print(time)
n_buildings = [0] * len(time)
build_area = [0] * len(time)
for building in building_objects:
    if building.cohort > 1900:
        n_buildings[building.cohort - 1901] += 1
        if building.footprint_area:
            build_area[building.cohort - 1900] += building.footprint_area

print(n_buildings)
print(build_area)
# make a bar plot of the number of buildings
import matplotlib.pyplot as plt

plt.bar(time, n_buildings)
plt.show()

# make a bar plot of the building area
plt.bar(time, build_area)
plt.show()


# %%
print(actutual_buildings[0])
for bkey, bdata in actutual_buildings[0].items():
    print(f"{bkey} : {bdata}")
# %%
