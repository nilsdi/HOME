#%%
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

root_dir   = Path(__file__).resolve().parents[3]
print(root_dir)
# %%
# import the road map of Norway
path = root_dir / "data/ML_prediction/prediction_mask/prediction_mask.csv"
print(path)
# read in the csv as a pandas dataframe
df = pd.read_csv(path, index_col=0)
# %%
plt.figure(figsize=(10, 10))
array = df.astype(int).values
plt.imshow(df, cmap = 'viridis')
plt.show()
# %%
print(df.head())
# %%
df.sum().sum()
