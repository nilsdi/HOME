# %%
import subprocess
from datetime import datetime


from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
data_path = root_dir / "data"
print(root_dir)
# now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
# print(f"today, now: {now}")

city_projects_overview_path = data_path / "matrikkel_comparison/city_project_names.txt"
# %%
if __name__ == "__main__":
    with open(city_projects_overview_path, "r") as f:
        city_overview = f.readlines()[3:]
    print(city_overview)
    cities = [city.strip().split(":")[0] for city in city_overview if city.strip()]
    print(cities)
    statuses = [
        city.strip().split(":")[1].strip().split(",")[0]
        for city in city_overview
        if city.strip()
    ]
    projects_lists = [
        city.strip().split(":")[1].strip().split(",")[1][1:]
        for city in city_overview
        if city.strip()
    ]
    print(cities)
    print(statuses)
    print(projects_lists)
    # run the script for each city
    for city, status, projects_list in zip(cities, statuses, projects_lists):

        if "[yes]" in status:
            print(f"City {city} is already fully processed: {status}. Skipping...")
            continue
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(f"Processing {city} with {len(projects_list)} projects, starting {now}.")
        # run the script for each city
        subprocess.run(
            [
                "python",
                str(root_dir) + "/HOME/ML_prediction/main.py",
                "--projects",
            ]
            + projects_list.split(),
            check=True,
        )
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(f"Finished processing {city} at {now}.")
        # update the status in the overview file
        with open(city_projects_overview_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if line.strip().startswith(f"{city}:"):
                # Replace the line with the updated status
                lines[idx] = f"{city}: processed [yes] ({now}), {projects_list}\n"

        with open(city_projects_overview_path, "w") as f:
            f.writelines(lines)

# %%
