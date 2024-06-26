"""
printing some roads and buildings on the FKB data
"""

# %% imports
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import fiona
import folium
from shapely.geometry import box
from pathlib import Path

root_dir = Path(__file__).resolve().parents[4]
# print(root_dir)
# load the FKB data
FKB_bygning_path = os.path.join(
    root_dir, "data/raw/FKB_bygning/Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb"
)
# list all layers
layers_bygning = fiona.listlayers(FKB_bygning_path)
print(f" the layers in the FKB bygning data are: {layers_bygning}")
# %% load the data

bygning_omrader = gpd.read_file(FKB_bygning_path, layer="fkb_bygning_omrade")
# %% limit the buildings to everything within a small bounding box

# bbox = [10.4281, 63.3855, 10.4401, 63.3955]
bbox = [10.4081, 63.4305, 10.4101, 63.4325]
# bbox = [270202.737422,7041627.464458, 270250.921554,7041681.901292 ] # _25833
bygning_omrader_4326 = bygning_omrader.to_crs("EPSG:4326")
bbox_bygning_omrader = bygning_omrader_4326.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
bbox_bygning_omrader.reset_index(drop=True, inplace=True)
# %% load the matrikkel data - a csv in data/raw/FKB_bygning
import pandas as pd

# Assuming the CSV file name is known and is 'matrikkel_data.csv'
matrikkel_file = root_dir / "data/raw/FKB_bygning/buildings.csv"
# Load the CSV data into a DataFrame
matrikkel_data = pd.read_csv(matrikkel_file)

# Display the first few rows of the DataFrame
print(matrikkel_data.head())
# %% plot
# plot in simple folium map
m = folium.Map(location=[63.4005, 10.3951], zoom_start=13)
# Add a rectangle for the bounding box
folium.GeoJson(
    box(*bbox), style_function=lambda x: {"color": "blue", "fill": False}
).add_to(m)

m_numbers = [int(m) for m in matrikkel_data["Building Number"]]
for _, row in bbox_bygning_omrader.iterrows():
    b_number = row["bygningsnummer"]
    if not pd.isna(b_number):
        b_number = int(b_number)
        if b_number in m_numbers:
            cohort_index = m_numbers.index(b_number)
            cohort = matrikkel_data.iloc[cohort_index]["Building Year"]
            # print the shape in the map and print the building year in the center of the shape
            folium_geojson = folium.GeoJson(
                row.geometry, style_function=lambda x: {"color": "red"}
            ).add_to(m)

            # Calculate the centroid of the geometry and place a marker with the building year
            centroid = row.geometry.centroid
            folium.Marker(
                [centroid.y, centroid.x],  # folium uses (lat, lon) order
                icon=folium.DivIcon(html=f'<div style="font-size: 8pt">{cohort}</div>'),
            ).add_to(m)
            # folium.GeoJson(row.geometry, style_function=lambda x: {"color": "red"}).add_to(m)
    # folium.GeoJson(row.geometry, style_function=lambda x: {"color": "red"}).add_to(m)
m

# %%
