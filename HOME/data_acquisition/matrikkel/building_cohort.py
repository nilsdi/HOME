"""
Extract building number and building construction year from the matrikkel data.
"""

# %% Imports
import pickle
import datetime
from pathlib import Path
from suds.client import Client
import csv
from typing import Tuple

root_dir = Path(__file__).parents[3]
print(root_dir)


# %% function defintions
def extract_cohorts_from_building(building_object: dict) -> int:
    """
    Returns something that looks like the cohort - not sure actually.
    """
    b_histories = building_object["bygningsstatusHistorikker"]

    estimate_b_cohort = 2050
    for b_history in b_histories["item"]:
        try:
            b_datetime = b_history["dato"]["date"]
            b_cohort1 = b_datetime.year
            if b_cohort1 < estimate_b_cohort:
                estimate_b_cohort = b_cohort1
        except KeyError:  # in case the building doesn't have a dato
            pass
    if estimate_b_cohort == 2050:
        estimate_b_cohort = -1
    return int(estimate_b_cohort)


def municipality_pickle_to_building_cohorts(pickle_path: Path) -> Tuple[list]:
    """
    Load the matrikkel data from a pickle file and return a list of buildings.
    """
    with open(pickle_path, "rb") as f:
        description, buildings = pickle.load(f)
    building_numbers = [b["bygningsnummer"] for b in buildings["item"]]
    building_years = [extract_cohorts_from_building(b) for b in buildings["item"]]
    return building_numbers, building_years


# %% demonstration: create csv file containing building numbers and building years
if __name__ == "__main__":
    Trondheim_pickle = (
        root_dir
        / "data"
        / "raw"
        / "matrikkel"
        / "municipality_pickles"
        / "5001_Trondheim_116103_buildings_fetched_24.2.15.16.33.pkl"
    )
    building_numbers, building_years = municipality_pickle_to_building_cohorts(
        Trondheim_pickle
    )
    # write into a csv:
    building_years_path = (
        root_dir
        / "data"
        / "ML_prediction"
        / "validation"
        / "trondheim_building_years.csv"
    )
    with open(building_years_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Building Number", "Building Year"])
        for i in range(len(building_numbers)):
            if building_years[i] != -1:
                writer.writerow([building_numbers[i], building_years[i]])
# %%
