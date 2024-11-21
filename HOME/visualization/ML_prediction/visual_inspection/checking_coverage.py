# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
from HOME.utils.project_coverage_area import (
    project_coverage_area,
)
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]

tile_id = 10002
prediction_id = 20002
assembly_id = 30002
polygon_id = 40002

path_label = "/scratch/mueller_andco/orthophoto/HOME/data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.pkl"
with open(path_label, "rb") as f:
    gdf_omrade_25832 = pickle.load(f)
gdf_omrade = gdf_omrade_25832.to_crs(25833)
project_geometry = project_coverage_area("trondheim_2023")

# %%
name = "3654_45720"
coverage = gpd.read_file(
    root_dir
    / f"data/ML_prediction/polygons/tiles_{tile_id}/prediction_{prediction_id}/assembly_{assembly_id}/polygons_{polygon_id}/coverage/coverage_trondheim_2023_resolution0.3_{name}.fgb"
)
polygons = gpd.read_file(
    root_dir
    / f"data/ML_prediction/polygons/tiles_{tile_id}/prediction_{prediction_id}/assembly_{assembly_id}/polygons_{polygon_id}/polygons_trondheim_2023_resolution0.3_{name}.fgb"
)

fig, ax = plt.subplots(dpi=300)
coverage.plot(ax=ax, alpha=0.1)
polygons.loc[polygons.area.sort_values()[:-1].index].plot(ax=ax)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
gdf_omrade.cx[xmin:xmax, ymin:ymax].plot(ax=ax, alpha=0.5, color="red")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


# %%
coverage_25832 = project_coverage_area("trondheim_2023", crs=25832)

# %%
min_x = 3717
max_y = 45792
fig, ax = plt.subplots(dpi=300)

coverage = gpd.read_file(
    root_dir
    / f"data/ML_prediction/polygons/tiles_{tile_id}/prediction_{prediction_id}/assembly_{assembly_id}/polygons_{polygon_id}/coverage/coverage_trondheim_2023_resolution0.3_{min_x}_{max_y}.fgb"
).to_crs(25832)
polygons = gpd.read_file(
    root_dir
    / f"data/ML_prediction/polygons/tiles_{tile_id}/prediction_{prediction_id}/assembly_{assembly_id}/polygons_{polygon_id}/polygons_trondheim_2023_resolution0.3_{min_x}_{max_y}.fgb"
).to_crs(25832)

coverage_25832.cx[
    512 * 0.3 * min_x + 1 : 512 * 0.3 * (min_x + 10) - 1,
    512 * 0.3 * (max_y - 10) + 1 : 512 * 0.3 * (max_y) - 1,
].plot(ax=ax, alpha=0.1)

coverage.plot(ax=ax, alpha=0.1, color="green")

polygons.loc[polygons.area.sort_values()[:-1].index].plot(ax=ax, color="blue")

gdf_omrade_25832.cx[
    512 * 0.3 * min_x : 512 * 0.3 * (min_x + 10),
    512 * 0.3 * (max_y - 10) : 512 * 0.3 * (max_y),
].plot(ax=ax, color="red", alpha=0.5)
xticks = []
xticks_labels = []
for x in range(min_x, min_x + 11, 2):
    xticks.append(x * 512 * 0.3)
    xticks_labels.append(x)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels)
yticks = []
yticks_labels = []
for y in range(max_y - 11, max_y):
    yticks.append(y * 512 * 0.3)
    yticks_labels.append(y)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)

path_image = (
    root_dir
    / f"data/ML_prediction/topredict/image/trondheim_2023/tiles_10003/trondheim_2023_{3609}_{45781}.tif"
)

# %%
