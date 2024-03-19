import os
from pathlib import Path
import random

root_dir = Path(__file__).parents[2]

data_path = root_dir / 'data/train/label'

train_file = open(root_dir / 'data/train/dataset/train.txt', 'w')
val_file = open(root_dir / 'data/train/dataset/val.txt', 'w')
test_file = open(root_dir / 'data/train/dataset/test.txt', 'w')

tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)]

# Shuffle the tiles
random.shuffle(tiles)

# Calculate the indices for splitting
val_index = int(len(tiles) * 0.1)
test_index = val_index + int(len(tiles) * 0.2)

# Split the tiles into validation, testing and training sets
val_tiles = tiles[:val_index]
test_tiles = tiles[val_index:test_index]
train_tiles = tiles[test_index:]

# write the validation tiles to val.txt
for tile in val_tiles:
    val_file.write(f'{tile}\n')

# write the training tiles to train.txt
for tile in train_tiles:
    train_file.write(f'{tile}\n')

# write the test tiles to test.txt
for tile in test_tiles:
    test_file.write(f'{tile}\n')

train_file.close()
val_file.close()
test_file.close()
