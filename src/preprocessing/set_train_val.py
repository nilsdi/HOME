import os
from pathlib import Path

root_dir = Path(__file__).parents[1]

data_path = root_dir / 'data/Inria/train/label/'
cities = ['austin', 'chicago', 'vienna', 'tyrol', 'kitsap']

train_file = open(root_dir / 'data/Inria/dataset/train.txt', 'w')
val_file = open(root_dir / 'data/Inria/dataset/val.txt', 'w')
test_file = open(root_dir / 'data/Inria/dataset/test.txt', 'w')

tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)]
tiles.sort()  # sort the tiles to ensure consistent order

for city in cities:
    # filter tiles for the current city
    city_tiles = [tile for tile in tiles if tile.startswith(city)]

    # split the tiles into validation and training sets
    val_tiles = city_tiles[:200]
    test_tiles = city_tiles[200:500]
    train_tiles = city_tiles[500:]

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
