# %%
from pathlib import Path
from HOME.get_data_path import get_data_path
import cv2
from tqdm import tqdm

root_dir = Path(__file__).parents[3]
data_dir = get_data_path(root_dir)

# %%
path_test = data_dir / "ML_training/dataset/test_tune.txt"
label_dir = data_dir / "ML_training/tune/label"

# %% open all labels in the dataset, write the empty ones to test_empty.txt
with open(path_test, "r") as f:
    tiles = f.readlines()

empty_tiles = 0
with open(data_dir / "ML_training/dataset/test_empty.txt", "w") as f:
    for tile in tqdm(tiles):
        tile = tile.strip()
        label_path = label_dir / f"{tile}.tif"
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label.sum() == 0:
            f.write(tile + "\n")
            empty_tiles += 1

print(f"Empty tiles: {empty_tiles}")


# %%
