# %%
import os
from pathlib import Path
import argparse

# %%
current_dir = Path(__file__).parents[0]
from HOME.get_data_path import get_data_path

# Get the root directory of the project
root_dir = Path(__file__).resolve().parents[3]
# print(root_dir)
# get the data path (might change)
data_path = get_data_path(root_dir)


def make_text_file(project_name, res=0.3, compression="i_lzw_25"):
    dir_images = (
        data_path
        / f"ML_prediction/topredict/image/res_{res}/{project_name}/{compression}/"
    )

    pred_file = open(
        data_path
        / f"ML_prediction/dataset/pred_{project_name}_{res}_{compression}.txt",
        "w",
    )

    tiles = [
        f"res_{res}/{project_name}/{compression}/{os.path.splitext(tile)[0]}"
        for tile in os.listdir(dir_images)
    ]

    # write the test tiles to test.txt
    for tile in tiles:
        pred_file.write(f"{tile}\n")

    pred_file.close()


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile raw orthophotos for prediction with ML model"
    )
    parser.add_argument("--project_name", required=True, type=str)
    parser.add_argument("--res", required=False, type=float, default=0.3)
    parser.add_argument("--compression", required=False, type=str, default="i_lzw_25")
    args = parser.parse_args()
    make_text_file(args.project_name, args.res, args.compression)
