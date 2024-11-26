# %%

from pathlib import Path

from HOME.utils.project_paths import (
    get_tile_ids,
    get_prediction_ids,
    get_assemble_ids,
    get_polygon_ids,
    load_project_details,
)
from HOME.get_data_path import get_data_path

# %%
root_dir = Path(__file__).parents[2]
data_path = get_data_path(root_dir)

# %%


def check_project_stage(project_name: str):
    tile_ids = get_tile_ids(project_name)
    if len(tile_ids) > 0:
        prediction_ids = get_prediction_ids(project_name)
        if len(prediction_ids) > 0:
            assemble_ids = get_assemble_ids(project_name)
            if len(assemble_ids) > 0:
                polygon_ids = get_polygon_ids(project_name)
                if len(polygon_ids) > 0:
                    return ("processed", polygon_ids)
                else:
                    return ("assembled", assemble_ids)
            else:
                return ("predicted", prediction_ids)
        else:
            return ("tiled", tile_ids)
    else:
        return ("downloaded", None)


def check_list_stage(project_list: list):
    stages = {}
    for project in project_list:
        stage, ids = check_project_stage(project)
        stages[project] = {"stage": stage, "ids": ids}
    return stages


# %%

if __name__ == "__main__":
    stages = check_list_stage(load_project_details().keys())

# %%
