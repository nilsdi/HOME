# %%
from pathlib import Path
import contextlib
import os
import sys


from HOME.ML_prediction.prediction import predict
from HOME.ML_prediction.postprocessing import (
    step_01_reassembling_tiles,
    step_02_regularization,
)
from HOME.data_acquisition.norgeibilder.orthophoto_api.download_originals import (
    request_download,
    download_original_NIB,
)
from HOME.get_data_path import get_data_path
from HOME.utils.project_paths import (
    save_project_details,
    load_project_details,
    get_assembling_details,
)
from HOME.data_acquisition.norgeibilder.add_project_details import add_project_details
from HOME.utils.check_project_stage import check_list_stage

# %%
# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)
# print(data_path)


# %%
@contextlib.contextmanager
def suppress_output(filepath=None):
    """Suppress all stdout and stderr output temporarily."""
    if filepath is None:
        filepath = os.devnull
    with open(filepath, "a") as logfile:  # Append mode to keep logs
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = logfile
            sys.stderr = logfile
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


project_name = "oslo_2011"
tile_key = 10080
log_folder = data_path / "metadata_log/execution_log"

with suppress_output(log_folder / f"{project_name}_process.log"):
    prediction_key = predict.predict_and_eval(
        project_name, tile_key, BW=False, evaluate=True
    )
    n_tiles_edge = 10
    n_overlap = 1
    print(f"{project_name}: Reassembling")
    assembly_key, geotiff_extends = step_01_reassembling_tiles.reassemble_tiles(
        project_name, prediction_key, n_tiles_edge, n_overlap
    )
    polygon_id = step_02_regularization.regularize(
        project_name, assembly_key, geotiff_extends
    )
    project_details = load_project_details(data_path)
    project_details[project_name]["status"] = "processed"
    save_project_details(project_details, data_path)
