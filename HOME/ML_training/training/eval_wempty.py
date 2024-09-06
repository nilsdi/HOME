# %% make a txt file with all testing images but removing those present in
# empty_but_kept.pkl

from pathlib import Path
import pickle

root_dir = Path(__file__).parents[3]
path_test = root_dir / "data/ML_training/dataset/test.txt"

with open(path_test, "r") as f:
    test = f.readlines()

with open(root_dir / "data/ML_training/train/empty_but_kept.pkl", "rb") as f:
    empty_but_kept = pickle.load(f)

# %%
removed = 0
for file in empty_but_kept.keys():
    for i, j in empty_but_kept[file]:
        if f"{file[:-4]}_{i}_{j}\n" in test:
            test.remove(f"{file[:-4]}_{i}_{j}\n")
            removed += 1
print(f"Removed {removed} images from test.txt")

# %% Save the new test.txt
new_path = root_dir / "data/ML_training/dataset/test_no_empty.txt"
with open(new_path, "w") as f:
    f.writelines(test)

# %%
