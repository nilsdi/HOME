"""
Providing the data (both for maps and a line plot) for the development of built-up area
within the boundaries of a municipality.
"""

# %%
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

# %%
root_dir = Path(__file__).parents[4]
print(root_dir)

# %%

# Pseudo code of what we need and do.

# final result: need the total area of footprints over time, where all locations are updated 
# with the most recent project.
# also need this information with high spatial resolution.

# without even reading the footprints, we just overlap the municipality boundaries with 
# the project coverage to establish what is covered. i
# if we sort the projects by date, we can take the first one, and make a new boundary for the 
# earliest coverage (and how long this boundary is valid). we can then overlap with 
# the next project, reducing the size of the earlier boundaries. In the end, we have a large set
# of boundaries tied to a project, and a time period.
# after that we can go through the projects, and read out the information (gridded or non gridded)
# for each "boundary" of the project. 
# We then go through time and establish the total number of periods with stable coverage.
# We can then go through these periods and sum the entries of the boundaries and hove something we can plot.

# for plotting: we can determine some uncertainty by checking the difference for each 
# covered area (grid cells?) between two updates and treat those as upper and lower bounds.

def get_coverage_boundaries(projects:list[str], municipality:str):
    """
    Get a dictionary (keyed by project name) with a shape of the the coverage the project
    introduces for a municipality, and the time it did so.
    The coverage is a polygon that can be used to mask the footprint data.

    Args:
        projects (list[str]): List of project names to get the coverage for.
        municipality (str): Name of the municipality to get the coverage for.
    
    Returns:
        dict[str, dict]: Dictionary with project coverage and date.
    """
    city_boundaries = []
    city_coverage_boundaries = {}
    for project in projects:
        # get the coverage of the project
        project_coverage = get_project_coverage(project)
        city_project_coverage = intersect(project_coverage, city_boundaries)
        project_date = get_project_date(project)
        city_coverage_boundaries[project] = {
            "coverage": city_project_coverage,
            "date": project_date
        }
    return city_coverage_boundaries

def get_time_period_coverage(city_coverage_boundaries:dict[str, dict], municipality:str, start_date:str = None, end_date:str = None):
    """
    Separate the coverage of the projects into patches which cover a specific time period, 
    meaning they are the most up to date observation of this patch.
    For potential interpolation between observation times, we also add the patches that
    replace the previous ones - meaning each projects gets cut into two sets of patches.

    Args:
        city_coverage_boundaries (dict[str, dict]): Dictionary with project coverage and date.
        start_date (str, optional): Start date of the time period, defaults to 1900-01-01.
        end_date (str, optional): End date of the time period, defaults to now.

    Returns:
        dict[int, ]: dictionary with id of the patch as key, containing the extend of the patch 
                        (polygon), the date of the observation, the name of the project, and (if the 
                        patch is not the most recent), the patch (with its own coverage, date and
                        project name) that replaces it. 
    
    Raises:
        ValueError: If the project coverage date is not within the specified time period.
    """
    if not start_date:
        start_date = datetime(1900, 1, 1)
    if not end_date:
        end_date = datetime.now()
    municipality_boundaries = get_municipality_boundaries(municipality)
    time_period_coverage = {}
    time_patch_id = 1000
    for project in ["None"]+ list(city_coverage_boundaries.keys()):
        if project == "None": # we initialize with the full city, to get the first observation
            # as the upper bound of the entire time period
            project_date = start_date
            project_coverage = municipality_boundaries
        else:
        project_date = project_data["date"]
        project_coverage = project_data["coverage"]
        if not(project_date >= start_date and project_date <= end_date):
            raise ValueError(f"Project {project} is not within the specified time period.")
        for comparison_project, comparison_project_data in city_coverage_boundaries.items():
            if project == comparison_project:
                continue
            comparison_project_date = comparison_project_data["date"]
            if project_date >= comparison_project_date:
                continue
            comparison_project_coverage = comparison_project_data["coverage"]
            # intersect both and get both the intersection and the remaining project coverage
            intersection = intersect(project_coverage, comparison_project_coverage)
            if intersection.area > 0:
                # add the intersection to the time_period_coverage
                time_period_coverage[time_patch_id] = {
                    "coverage": intersection,
                    "start_date": project_date,
                    "end_date": comparison_project_date,
                    "start_project": project,
                    "end_project": comparison_project,
                }
                time_patch_id += 1
                project_coverage = project_coverage.difference(intersection)
        if project_coverage.area > 0:
            # add the remaining project coverage to the time_period_coverage
            time_period_coverage[time_patch_id] = {
                "coverage": project_coverage,
                "start_date": project_date,
                "end_date": end_date,
                "start_project": project,
                "end_project": None,
            }
            time_patch_id += 1
        # after this we have three types of patches: start patches (only end date), end patches (only start date),
        # and middle patches (both start and end date).
    return time_period_coverage

def add_footprint_data():
    return

#%%
project = "trondheim_2023"
