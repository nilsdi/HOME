# %%
import os
from pathlib import Path
import random
import json

#%%
root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]
test_file = open(root_dir / 'data/model/trondheim_1937/dataset/test.txt', 'w')
data_path = root_dir / 'data/model/trondheim_1937/tiles/images'

def list_files_in_folder(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        print(file_name)
            # Remove file extension (assuming all files have .tif extension)
        file_name_without_extension = os.path.splitext(file_name)[0]
        file_names.append(file_name_without_extension)
    return file_names
file_names = list_files_in_folder(data_path)
print("Files in the folder (without extension):")
"""for file_name in file_names:
    print(file_name)"""
print(data_path)

#%%
def count_files_in_folder(folder_path):
    file_count = 0
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_count += 1
    return file_count

# Example usage
folder_path = root_dir / 'data/model/test_full_pic/tiles/labels'
file_count = count_files_in_folder(folder_path)
print(f"There are {file_count} files in the folder.")
print(len(file_names))
#%%
val_share = 0.1
test_share = 0.2

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

# Read bbox from bbox.json
with open(current_dir / 'bbox.json', 'r') as f:
    bbox = json.load(f)
cities = list(bbox.keys())

data_path = root_dir / 'data/model/trondheim_1937/tiles/images'

train_file = open(root_dir / 'data/model/trondheim_1937/dataset/train.txt', 'w+')
val_file = open(root_dir / 'data/model/trondheim_1937/dataset/val.txt', 'w+')
test_file = open(root_dir / 'data/model/trondheim_1937/dataset/test.txt', 'w')

random.seed(42)
for city in cities[:-1]:   # exclude Fredrikstad
    tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)
             if city in tile]

    # Shuffle the tiles
    random.shuffle(tiles)

    # Calculate the indices for splitting
    val_index = int(len(tiles) * val_share)
    test_index = val_index + int(len(tiles) * test_share)

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

# %%
