# %%
import os
from pathlib import Path
import random
import json

# %%

total_share = 0.5

val_share = 0.1
test_share = 0.2

root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Read bbox from bbox.json
with open(current_dir / "bbox_tune.json", "r") as f:
    bbox = json.load(f)
cities = list(bbox.keys())

data_path = root_dir / f"data/ML_training/tune/image"

train_file = open(root_dir / "data/ML_training/dataset/train_tune.txt", "w")
val_file = open(root_dir / "data/ML_training/dataset/val_tune.txt", "w")
test_file = open(root_dir / "data/ML_training/dataset/test_tune.txt", "w")

random.seed(42)

n_train = 0
n_val = 0
n_test = 0

for city in cities:  # exclude Fredrikstad
    tiles = [
        os.path.splitext(tile)[0] for tile in os.listdir(data_path) if city in tile
    ]

    # Shuffle the tiles
    random.shuffle(tiles)

    tiles = tiles[: int(len(tiles) * total_share)]

    # Calculate the indices for splitting
    val_index = int(len(tiles) * val_share)
    test_index = val_index + int(len(tiles) * test_share)

    # Split the tiles into validation, testing and training sets
    val_tiles = tiles[:val_index]
    test_tiles = tiles[val_index:test_index]
    train_tiles = tiles[test_index:]

    # write the validation tiles to val.txt
    for tile in val_tiles:
        val_file.write(f"{tile}\n")

    # write the training tiles to train.txt
    for tile in train_tiles:
        train_file.write(f"{tile}\n")

    # write the test tiles to test.txt
    for tile in test_tiles:
        test_file.write(f"{tile}\n")

    n_train += len(train_tiles)
    n_val += len(val_tiles)
    n_test += len(test_tiles)

print(f"Number of training samples: {n_train}")
print(f"Number of validation samples: {n_val}")
print(f"Number of testing samples: {n_test}")

train_file.close()
val_file.close()
test_file.close()

# %%
