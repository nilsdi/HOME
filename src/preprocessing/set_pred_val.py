# %%
import os
from pathlib import Path
import json

val_share = 0.1
test_share = 0.2

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

# Read bbox from bbox.json
with open(current_dir / 'bbox.json', 'r') as f:
    bbox = json.load(f)
cities = list(bbox.keys())

data_path = root_dir / 'data/topredict/train/image'

pred_file = open(root_dir / 'data/topredict/dataset/test.txt', 'w')

city = 'trondheim'
pred_tiles = [os.path.splitext(tile)[0] for tile in os.listdir(data_path)
              if city in tile]

# write the test tiles to test.txt
for tile in pred_tiles:
    pred_file.write(f'{tile}\n')

pred_file.close()

# %%
