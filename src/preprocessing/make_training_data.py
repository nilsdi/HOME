# %%
from pathlib import Path
import geopandas as gpd
import sys
import json
grandparent_dir = Path(__file__).parents[1]
sys.path.append(str(grandparent_dir))
from get_label_data.get_labels import get_labels, save_labels  # noqa
from get_label_data.cut_images import cut_geotiff, save_cut_geotiff  # noqa

# %%
root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]

# Read bbox from bbox.json
with open(current_dir / 'bbox.json', 'r') as f:
    bbox = json.load(f)

# get the labels
cities = bbox.keys()
path_label = (root_dir / "data/raw/FKB_bygning" /
              "Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb")
gdf_omrade = gpd.read_file(path_label, driver="FileGDB",
                           layer="fkb_bygning_omrade")

# %%
for city in cities:
    for i in range(len(bbox[city])):
        print(f"Processing {city} {i}")

        filename = f"{city}_{i}_latest"

        data, transform = get_labels(gdf_omrade, bbox[city][i], 0.3)

        save_labels(data, filename, transform)

        subfolders = list(
            (root_dir / "data/raw/orthophoto/res_0.3").glob(f"*{city}*"))
        geotiff_path = subfolders[0] / "i_lzw_25" / "Eksport-nib.tif"
        image, transform = cut_geotiff(geotiff_path, bbox[city][i], 0.3)

        # save the image
        save_cut_geotiff(image, filename, transform)


# path_label = (root_dir / "data/raw/FKB_bygning" /
#               "Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb")
# gdf_omrade = gpd.read_file(path_label, driver="FileGDB",
#                            layer="fkb_bygning_omrade")

# data, transform = get_labels(gdf_omrade, bbox, 0.3)

# save_labels(data, filename, transform)

# # %% get the image
# subfolder = "data/raw/orthophoto/res_0.3/trondheim_2019/i_lzw_25"
# geotiff_path = root_dir / subfolder / "Eksport-nib.tif"
# image, transform = cut_geotiff(geotiff_path, bbox, 0.3)

# # save the image
# save_cut_geotiff(image, filename, transform)

# %%
