# %%
from pathlib import Path
import geopandas as gpd
import json
from src.ML_training.preprocessing.get_label_data.get_labels import (
    get_labels,
    save_labels,
)  # noqa
from src.ML_training.preprocessing.get_label_data.cut_images import (
    cut_geotiff,
    save_cut_geotiff,
)  # noqa
import os

# %%
root_dir = Path(__file__).parents[3]
current_dir = Path(__file__).parents[0]

# Read bbox from bbox.json
with open(current_dir / "bbox.json", "r") as f:
    bbox = json.load(f)

# get the labels
cities = bbox.keys()
path_label = (
    root_dir / "data/raw/FKB_bygning" / "Basisdata_0000_Norge_5973_FKB-Bygning_FGDB.gdb"
)
gdf_omrade = gpd.read_file(path_label, driver="FileGDB", layer="fkb_bygning_omrade")

# %%


def make_training_data(city):
    for i in range(len(bbox[city])):
        print(f"Processing {city} {i}")
        # Get the bounding box coordinates
        bbox_coordinates = bbox[city][i]["bbox"]

        # Check if the bounding box has label data
        if bbox[city][i]["training"] == 1:
            year = bbox[city][i]["year"]
            res = bbox[city][i]["res"]
            subfolders = list(
                (root_dir / f"data/raw/orthophoto/res_{str(res)}").glob(
                    f"*{city}*{year}*"
                )
            )

            assert len(subfolders) == 1, (
                f"Found {len(subfolders)} cities" + " matching training data"
            )
            filename = f"{city}_{i}_{res}_{year}"

            if not os.path.exists(
                root_dir / f"data/temp/pretrain/images/{filename}.tif"
            ):
                data, transform = get_labels(gdf_omrade, bbox_coordinates, res)
                if data.size != 0:
                    save_labels(data, filename, transform)

                geotiff_path = subfolders[0] / "i_lzw_25" / "Eksport-nib.tif"
                image, transform = cut_geotiff(geotiff_path, bbox_coordinates, res)
                if image.size != 0:
                    save_cut_geotiff(image, filename, transform)
        else:
            for res in [0.3, 0.5]:
                subfolders = list(
                    (root_dir / f"data/raw/orthophoto/res_{str(res)}").glob(f"*{city}*")
                )

                for subfolder in subfolders:
                    year = subfolder.name.split("_")[-1]
                    # Get all tif files in folder
                    geotiffs = list((subfolder / "i_lzw_25").glob("*.tif"))

                    for j, geotiff_path in enumerate(geotiffs):

                        filename = f"{city}_{res}_{year}_{i}_{j}"

                        if not os.path.exists(
                            root_dir / f"data/temp/prepred/images/{filename}.tif"
                        ):
                            image, transform = cut_geotiff(
                                geotiff_path, bbox_coordinates, res
                            )

                            # save the image
                            if image.size != 0:
                                save_cut_geotiff(
                                    image,
                                    filename,
                                    transform,
                                    save_folder="data/temp/prepred/images/",
                                )

    return None


# %% Make training data for each city

for city in cities:
    make_training_data(city)


# %%
