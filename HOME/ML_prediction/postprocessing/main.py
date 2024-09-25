""" 
from prediction to shapes of footprints in gdfs.
"""

from HOME.ML_prediction.postprocessing.step_02_regularization_clean import (
    process_project_tiles,
)

if __name__ == "__main__":
    process_project_tiles(
        tile_dir="/mnt/data/orthophoto/HOME/HOME/ML_prediction/postprocessing/step_01_prediction",
        output_dir=Path(
            "/mnt/data/orthophoto/HOME/HOME/ML_prediction/postprocessing/step_02_regularization"
        ),
    )
