"""Extract building number and building construction year from the matrikkel data.

Contains two functions and a main block for applying them:
- extract_cohorts_from_building: matrikkel estimated cohort from matrikkel building object
- municipality_pickle_to_building_cohorts: extract building numbers and years from a list of building objects coming from matrikkel via a pickle file.
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
    Takes in a building object from matrikkel (dict from the suds object)
    and returns the date of the first building history, or -1 if none are available.

    Args:
        building_object (dict): a building object from the matrikkel data

    Returns:
        year (int): the year of the first building history (or -1 if none are available)
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
    Load the data from a request of all buildings from a municipality
    via the matrikkel API and returns a list of the building number and
    the building years (years of the first building history).

    Args:
        pickle_path (Path): path to the pickle file containing building objects (!)

    Returns:
        building_numbers (list): list of building numbers
        building_years (list): list of building years (estimated based on matrikkel)
    """
    with open(pickle_path, "rb") as f:
        description, buildings = pickle.load(f)
    building_numbers = [b["bygningsnummer"] for b in buildings["item"]]
    building_years = [extract_cohorts_from_building(b) for b in buildings["item"]]
    return building_numbers, building_years


def create_municipality_csv(municipality, municipality_pickle):
    """
    Create a csv file containing building numbers and building years for a municipality.

    Args:
        municipality (str): the name of the municipality
        municipality_pickles (list): list of paths to the pickles containing the building objects
    """
    pickle_path = (
        root_dir
        / "data"
        / "raw"
        / "matrikkel"
        / "municipality_pickles"
        / municipality_pickle
    )
    building_numbers, building_years = municipality_pickle_to_building_cohorts(
        pickle_path
    )
    # write into a csv:
    building_years_path = (
        root_dir
        / "data"
        / "ML_prediction"
        / "validation"
        / f"{municipality}_building_years.csv"
    )
    with open(building_years_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Building Number", "Building Year"])
        for i in range(len(building_numbers)):
            if building_years[i] != -1:
                writer.writerow([building_numbers[i], building_years[i]])


# %% demonstration: create csv file containing building numbers and building years
if __name__ == "__main__":
    municipality = "Oslo"
    municipality_pickle = "0301_Oslo_202361_buildings_fetched_24.2.6.12.12.pkl"
    create_municipality_csv(municipality, municipality_pickle)
# %%
