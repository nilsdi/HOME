"""
Providing the data (both for maps and a line plot) for the development of built-up area
within the boundaries of a municipality.
"""

# %%
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import json
import time

from pathlib import Path
from datetime import datetime

from HOME.utils.project_coverage_area import project_coverage_area
from HOME.utils.get_project_metadata import (
    get_project_details,
    get_project_geometry,
    filter_projects_without_metadata,
)
from HOME.footprint_analysis.matrikkel_comparison.municipality_boundaries import (
    get_municipal_boundaries,
)
from HOME.visualization.footprint_changes.utilities.get_bound_tiles import (
    find_polygonisations,
)

root_dir = Path(__file__).parents[4]
# print(root_dir)

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
# %%


def find_projects_for_municipality(municipality: str) -> list[str]:
    """
    Find all projects that have a polygonisation in the log file, and intersect with the
    boundaries of the municipality.
    Args:
        municipality (str): Name of the municipality to find projects for.
    Returns:
        list[str]: List of project names for the municipality.
    """
    polygonisations_log_path = root_dir / "data/metadata_log/polygon_gdfs.json"
    with open(polygonisations_log_path, "r") as f:
        polygonisations_log = json.load(f)
    # get the projects for the municipality
    projects = []
    potential_projects = []
    for polygonisation in polygonisations_log.values():
        project = polygonisation["project_name"]
        if project not in potential_projects:
            potential_projects.append(project)
    count_before_filter = len(potential_projects)
    potential_projects = filter_projects_without_metadata(potential_projects)
    print(
        f"Filtered {count_before_filter - len(potential_projects)} projects without metadata."
    )
    project_geometries = get_project_geometry(potential_projects)

    municipality_boundaries = get_municipal_boundaries(municipality).to_crs(
        "EPSG:25833"
    )
    for project, project_geometry in zip(potential_projects, project_geometries):
        project_geometry = project_geometry.to_crs("EPSG:25833")
        if municipality_boundaries.intersects(project_geometry).any():
            projects.append(project)
    return projects


def get_coverage_boundaries(projects: list[str], municipality: str):
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
    city_boundaries = get_municipal_boundaries(municipality).to_crs("EPSG:25833")
    city_coverage_boundaries = {}
    projects_details = get_project_details(projects)
    for project in projects:
        # get the coverage of the project
        project_coverage = project_coverage_area(project).to_crs("EPSG:25833")
        city_project_coverage = gpd.overlay(
            project_coverage, city_boundaries
        )  # intersect(project_coverage, city_boundaries)
        project_date = projects_details[project]["capture_date"]
        city_coverage_boundaries[project] = {
            "coverage": city_project_coverage,
            "date": project_date,
        }
        # sort the projects by date
        city_coverage_boundaries = dict(
            sorted(
                city_coverage_boundaries.items(),
                key=lambda item: item[1]["date"],
            )
        )
    return city_coverage_boundaries


def get_time_period_coverage(
    city_coverage_boundaries: dict[str, dict],
    municipality: str,
    start_date: str = None,
    end_date: str = None,
    print_times: bool = False,
    progress_bar: bool = False,
):
    """
    Separate the coverage of the projects into patches which cover a specific time period,
    meaning they are the most up to date observation of this patch.
    For potential interpolation between observation times, we also add the patches that
    replace the previous ones - meaning each projects gets cut into two sets of patches.

    Args:
        city_coverage_boundaries (dict[str, dict]): Dictionary with project coverage and date.
        start_date (str, optional): Start date of the time period, defaults to 1900-01-01.
        end_date (str, optional): End date of the time period, defaults to now.
        print_times(bool, optional): time taken for each step, defaults to False.
        progress_bar(bool, optional): Show a progress bar (projects processed), defaults to False.

    Returns:
        dict[int, ]: dictionary with id of the patch as key, containing the extend of the patch
                        (polygon), the date of the observation, the name of the project, and (if the
                        patch is not the most recent), the patch (with its own coverage, date and
                        project name) that replaces it.

    Raises:
        ValueError: If the project coverage date is not within the specified time period.
    """
    overhead_start_time = time.time()
    if not start_date:
        start_date = datetime(1900, 1, 1)
    if not end_date:
        end_date = datetime.now()
    municipality_boundaries = get_municipal_boundaries(municipality)
    time_period_coverage = {}
    time_patch_id = 1000
    print(["None"] + list(city_coverage_boundaries.keys()))
    overhead_stop_time = time.time()
    print(
        f"Time taken to get the overhead time: {overhead_stop_time - overhead_start_time:.2f} seconds"
    )
    for project in ["None"] + list(city_coverage_boundaries.keys()):
        project_overhead_time = time.time()
        if project == "None":
            # we initialize with the full city, to get the first observation
            # as the upper bound of the entire time period
            project_date = start_date
            project_coverage = municipality_boundaries
        else:
            project_data = city_coverage_boundaries[project]
            project_date = project_data["date"]
            project_coverage = project_data["coverage"]
        if not (project_date >= start_date and project_date <= end_date):
            raise ValueError(
                f"Project {project} is not within the specified time period ({project_date}, range {start_date}, {end_date}."
            )
        project_overhead_finish_time = time.time()
        print(
            f"Time taken to get the project ({project}) overhead time: {project_overhead_finish_time - project_overhead_time:.2f} seconds"
        )
        for (
            comparison_project,
            comparison_project_data,
        ) in city_coverage_boundaries.items():
            start_comparison_time = time.time()
            if project == comparison_project:
                continue
            comparison_project_date = comparison_project_data["date"]
            if project_date >= comparison_project_date:
                continue
            comparison_project_coverage = comparison_project_data["coverage"]
            comparision_poject_loading_time = time.time()
            print(
                f"Time taken to get the comparison project ({comparison_project}) loading time: {comparision_poject_loading_time - start_comparison_time:.2f} seconds"
            )
            # intersect both and get both the intersection and the remaining project coverage
            intersection = gpd.overlay(
                project_coverage,
                comparison_project_coverage,
                how="intersection",
                keep_geom_type=True,
            )
            intersection_time = time.time()
            print(
                f"Time taken to get the intersection of the project ({project}) and comparison project ({comparison_project}): {intersection_time - comparision_poject_loading_time:.2f} seconds"
            )
            # print the types of geometries of project_coverage and comparison_project_coverage
            project_coverage_types = project_coverage.geom_type.unique()
            comparison_project_coverage_types = (
                comparison_project_coverage.geom_type.unique()
            )
            print(f"Project ({project}) coverage types: {project_coverage_types}")
            print(
                f"Comparison ({comparison_project}) project coverage types: {comparison_project_coverage_types}"
            )
            type_print_time = time.time()
            print(
                f"Time taken to print the types of geometries: {type_print_time - intersection_time:.2f} seconds"
            )
            if sum(intersection.area) > 0:
                # add the intersection to the time_period_coverage
                time_period_coverage[time_patch_id] = {
                    "coverage": intersection,
                    "start_date": project_date,
                    "end_date": comparison_project_date,
                    "start_project": project,
                    "end_project": comparison_project,
                }
                time_patch_id += 1
                intersection_saving_time = time.time()
                print(
                    f"Time taken to save the intersection of the project ({project}) and comparison project ({comparison_project}): {intersection_saving_time - type_print_time:.2f} seconds"
                )
                project_coverage = gpd.overlay(
                    project_coverage,
                    intersection,
                    how="difference",
                )
                project_coverage_update_time = time.time()
                print(
                    f"Time taken to update the project coverage ({project}) after intersection: {project_coverage_update_time - intersection_saving_time:.2f} seconds"
                )
                if sum(project_coverage.area) == 0:
                    # if the project coverage is empty, we can stop here
                    break
            post_intersection_time = time.time()
            print(
                f"Time taken to get the project comparison({project}, {comparison_project}) coverage after intersection: {post_intersection_time - intersection_time:.2f} seconds"
            )
            for patch_id, patch_data in time_period_coverage.items():
                print(
                    f'Patch {patch_id}: starts {patch_data["start_date"]}, ends {patch_data["end_date"]}'
                )
                print(
                    f'Patch {patch_id}: start project {patch_data["start_project"]}, end project {patch_data["end_project"]}'
                )
        if sum(project_coverage.area) > 0:
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


def add_footprint_data(time_period_coverage: dict[int, dict], projects: list[str]):
    """
    Add the footprint data to the time period coverage.
    This is done by reading the polygonisations for each project and filtering
    the polygons that are within the coverage of each time period.
    This is quite efficient and seems to work stable.

    Args:
        time_period_coverage (dict[int, dict]): Dictionary with time period coverage.
        projects (list[str]): List of project names to get the footprint data for.

    Returns:
        time_period_coverage (dict[int, dict]): Dictionary with time period coverage and footprint data.
    """
    polygonisations_log_path = root_dir / "data/metadata_log/polygon_gdfs.json"
    with open(polygonisations_log_path, "r") as f:
        polygonisations_log = json.load(f)
    polygonisations = find_polygonisations(
        project_list=projects,
    )
    for project in projects:
        # get the polygonisations for the project
        polygonisation = polygonisations[project]
        gdf_directory = polygonisations_log[str(polygonisation)]["gdf_directory"]
        fgb_files = list(Path(gdf_directory).glob("*.fgb"))
        project_polygons = gpd.GeoDataFrame()
        for polygon_gdf in fgb_files:
            gdf = gpd.read_file(polygon_gdf).to_crs("EPSG:25833")
            project_polygons = gpd.GeoDataFrame(
                pd.concat([project_polygons, gdf], ignore_index=True)
            )
        project_polygons = project_polygons.drop_duplicates(subset="geometry")

        # now we go through eah time period coverage and filter the polygons that are within the coverage
        for patch_id, patch_data in time_period_coverage.items():
            patch_coverage = patch_data["coverage"]
            if project == patch_data["start_project"]:
                save_as = "lower_bound_polygons"
            elif project == patch_data["end_project"]:
                save_as = "upper_bound_polygons"
            else:
                continue
            unified_coverage = patch_coverage.geometry.union_all()
            selected_polygons = project_polygons[
                project_polygons.geometry.within(unified_coverage)
            ]
            time_period_coverage[patch_id][save_as] = selected_polygons
            print(
                f"Project {project} has {len(selected_polygons)} polygons within the coverage of patch {patch_id}."
            )
    return time_period_coverage


def get_total_footprint_data(time_period_coverage: dict[int, dict]):
    """
    Get the total footprint data for each time period coverage.
    This is done by reading the polygonisations for each project and filtering
    the polygons that are within the coverage of each time period.

    Args:
        time_period_coverage (dict[int, dict]): Dictionary with time period coverage.

    Returns:
        time_period_coverage (dict[int, dict]): Dictionary with time period coverage and footprint data.
    """
    # fetch min and max date for the time period coverage (in total)
    min_date = min(
        [
            patch_data["start_date"]
            for patch_data in time_period_coverage.values()
            if patch_data["start_date"]
        ]
    )
    max_date = max(
        [
            patch_data["end_date"]
            for patch_data in time_period_coverage.values()
            if patch_data["end_date"]
        ]
    )
    # create a time lline (vector) with montly intervals between the min and max date
    time_line = pd.date_range(min_date, max_date, freq="M")
    print(f"Time line: {time_line}")
    lower_bound_values = np.zeros(len(time_line))
    upper_bound_values = np.zeros(len(time_line))
    for patch_id, patch_data in time_period_coverage.items():
        # get the start and end date of the patch
        start_date = patch_data["start_date"]
        end_date = patch_data["end_date"]
        if not start_date:
            start_date = min_date
        if not end_date:
            end_date = max_date
        # get the index of the start and end date in the time line
        start_index = np.argmin(np.abs(time_line - pd.Timestamp(start_date)))
        end_index = np.argmin(np.abs(time_line - pd.Timestamp(end_date)))
        print(
            f"Patch {patch_id}: start date {start_date}, end date {end_date}, start index {start_index}, end index {end_index}"
        )
        # get the lower and upper bound polygons
        if "lower_bound_polygons" in patch_data.keys():
            lower_bound_polygons = patch_data["lower_bound_polygons"]
        else:
            lower_bound_polygons = gpd.GeoDataFrame(geometry=[])
        if "upper_bound_polygons" in patch_data.keys():
            upper_bound_polygons = patch_data["upper_bound_polygons"]
        else:
            upper_bound_polygons = gpd.GeoDataFrame(geometry=[])
        # get the area of the polygons
        lower_bound_area = lower_bound_polygons.area.sum()
        upper_bound_area = upper_bound_polygons.area.sum()
        print(
            f"Patch {patch_id}: lower bound area {lower_bound_area}, upper bound area {upper_bound_area}"
        )
        # set the values in the time line
        lower_bound_values[start_index:end_index] += lower_bound_area
        upper_bound_values[start_index:end_index] += upper_bound_area

    return time_line, lower_bound_values, upper_bound_values

def rolling_average(time_series:list[float], window_size:int):
    """
    Calculate the rolling average of a time series.
    Args:
        time_series (list[float]): Time series to calculate the rolling average for.
        window_size (int): Size of the rolling window.
    Returns:
        list[float]: Rolling average of the time series.
    """
    return np.convolve(time_series, np.ones(window_size), "valid") / window_size

def set_uncertainty_interval():
    return

# %%
if __name__ == "__main__":
    trondheim_projects = [
        "trondheim_2023",
        "trondheim_2019",
        "trondheim_2013",
        "trondheim_1992",
        "trondheim_1988",
        "trondheim_1971",
    ]
    trondheim_projects_large = find_projects_for_municipality("trondheim")
    # %%
    project_coverage_areas = get_coverage_boundaries(trondheim_projects, "trondheim")
    # test_coverage = project_coverage_areas["trondheim_2023"]["coverage"]
    # print(test_coverage)
    # print(sum(test_coverage.area))
    # %%
    project_time_period_coverage = get_time_period_coverage(
        project_coverage_areas,
        "trondheim",
        start_date=datetime(1900, 1, 1),
    )

    # %%
    add_timer_start = time.time()
    project_time_period_coverage = add_footprint_data(
        project_time_period_coverage,
        trondheim_projects,
    )
    add_timer_end = time.time()
    print(
        f"Time taken to add the footprint data: {add_timer_end - add_timer_start:.2f} seconds"
    )
    # %%
    print(len(project_time_period_coverage))
    # %%
    time_line, lower_bound_values, upper_bound_values = get_total_footprint_data(
        project_time_period_coverage
    )
    plt.plot(
        time_line,
        lower_bound_values,
        label="Lower bound",
    )
    plt.plot(
        time_line,
        upper_bound_values,
        label="Upper bound",
    )
    plt.xlabel("Date")
    # %%

# %%
if __name__ == "__main__1":
    # start timer
    start = time.time()
    project = "trondheim_2023"  # "trondheim_1971"
    project_coverage = project_coverage_area(project)
    comparison_project = "trondheim_1976"
    comparison_project_coverage = project_coverage_area(comparison_project)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    comparison_project_coverage = get_municipal_boundaries("trondheim")

    # %%
    import matplotlib.pyplot as plt

    # both are geodataframes, so we can use the intersect function from geopandas
    intersection = gpd.overlay(
        project_coverage, comparison_project_coverage, how="intersection"
    )
    remaining_project_coverage = gpd.overlay(
        project_coverage, intersection, how="difference"
    )
    # plot the two projects, then the intersection and the remaining project coverage
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    project_coverage.plot(ax=ax[0, 0], color="red", alpha=0.5)
    comparison_project_coverage.plot(ax=ax[0, 1], color="blue", alpha=0.5)
    intersection.plot(ax=ax[1, 0], color="green", alpha=0.5)
    remaining_project_coverage.plot(ax=ax[1, 1], color="yellow", alpha=0.5)
    ax[0, 0].set_title("Project coverage")
    ax[0, 1].set_title("Comparison project coverage")
    ax[1, 0].set_title("Intersection")
    ax[1, 1].set_title("Remaining project coverage")
    plt.show()

    # %%
    # try loading the building shapes for an entire project
    from HOME.visualization.footprint_changes.utilities.get_bound_tiles import (
        find_polygonisations,
    )
    import json
    import pandas as pd

    polygonisations = find_polygonisations(
        project_list=["trondheim_2023"],
    )
    print(polygonisations)
    polygonisations_log_path = root_dir / "data/metadata_log/polygon_gdfs.json"
    # read in the polygonisations
    with open(polygonisations_log_path, "r") as f:
        polygonisations_log = json.load(f)
    print(polygonisations_log["40066"])
    gdf_directory = polygonisations_log["40066"]["gdf_directory"]
    # list all files in the directory
    gdf_directory = Path(gdf_directory)
    print(gdf_directory)
    # list all files in the directory
    fgb_files = list(gdf_directory.glob("*.fgb"))
    print(fgb_files)

    all_polygons_gdf = gpd.GeoDataFrame()
    for polygon_gdf in fgb_files:
        gdf = gpd.read_file(polygon_gdf).to_crs("EPSG:25833")
        all_polygons_gdf = gpd.GeoDataFrame(
            pd.concat([all_polygons_gdf, gdf], ignore_index=True)
        )
        # print(f" we have a gdf with {len(gdf)} polygons")
        # print(f"so our all_polygons_gdf has {len(all_polygons_gdf)} polygons")
    # drop duplicates
    count_with_duplicates = len(all_polygons_gdf)
    all_polygons_gdf = all_polygons_gdf.drop_duplicates(subset="geometry")
    print(
        f"so our all_polygons_gdf has {len(all_polygons_gdf)} polygons after dropping duplicates, meaning we droppen {count_with_duplicates - len(all_polygons_gdf)} duplicates"
    )

    # %%
    # from shapely.ops import unary_union, union_all

    all_polygons_gdf.plot()
    project_time_period_coverage[1008]["coverage"].to_crs("EPSG:25833").plot()
    selected_polygons = all_polygons_gdf[
        all_polygons_gdf.geometry.within(
            project_time_period_coverage[1008]["coverage"]
            .to_crs("EPSG:25833")
            .union_all()
        )
    ]
    print(selected_polygons)
    selected_polygons.plot()
