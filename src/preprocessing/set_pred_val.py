# %%
import os
from pathlib import Path
import json

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

data_path = root_dir / 'data/model/topredict/train_augmented/image'

pred_file = open(
    root_dir / 'data/model/topredict/dataset/test_augmented.txt', 'w')

city = 'trondheim'
pred_tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)
              if city in tile]
pred_tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)]

# write the test tiles to test.txt
for tile in pred_tiles:
    pred_file.write(f'{tile}\n')

pred_file.close()

# %%
