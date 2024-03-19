# %%
from get_label_data.get_labels import get_labels, save_labels
from get_label_data.cut_images import cut_geotiff, save_cut_geotiff
from pathlib import Path
import geopandas as gpd

# rectangle in the center in of trondheim
bbox = [10.3281, 63.3805, 10.4901, 63.4325]

# %% get the labels
root_dir = Path(__file__).parents[1]
path_label = (root_dir / "data/raw/FKB_bygning" /
              "Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb")
gdf_omrade = gpd.read_file(path_label, driver="FileGDB",
                           layer="fkb_bygning_omrade")
# %%
data, transform = get_labels(gdf_omrade, bbox, 0.3)

# %% save the labels
save_labels(data, "trondheim_2019_rect_labels", transform)

# get the image
subfolder = "data/raw/orthophoto/res_0.3/trondheim_2019/i_lzw_25"
geotiff_path = root_dir / subfolder / "Eksport-nib_4326.tif"
image = cut_geotiff(geotiff_path, bbox, 0.3)

# save the image
save_cut_geotiff(image, "trondheim_2019_rect_image")
