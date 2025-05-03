# %%
import folium
from shapely.geometry import box
import numpy as np

from pyproj import Transformer
import json

from pathlib import Path
from shapely.geometry import shape
import geopandas as gpd

# %%
root_dir = Path(__file__).parents[3]
# print(root_dir)
data_path = root_dir / "data"

# %%
import json

kommune_boundaries_path = (
    data_path
    / "raw/maps/municipalities"
    / "Basisdata_0000_Norge_3035_Kommuner_GeoJSON.geojson"
)
# %%Initialize the transformer from EPSG:3035 to EPSG:4326


def get_municipal_boundaries(municipality_name: str, crs="EPSG:25833"):
    """
    Reads in the municipal boundaries (Norway wide) and
    returns the boundaries of a municipality that matches the input name.
    The crs of the boundaries is EPSG:3035 originally, but the function
    transforms the boundaries to EPSG:4326 by default
    """
    # Returns the boundaries of a municipality in EPSG:4326
    with open(kommune_boundaries_path) as f:
        data = json.load(f)

    features_data = data["Kommune"]["features"]
    for feature in features_data:
        name = feature["properties"]["kommunenavn"]
        if municipality_name in name.lower():
            municipality_data = feature
            municipality_geometry = feature["geometry"]
            # print(f"found {municipality_name}: {name}")
    geometry_gdf = gpd.GeoDataFrame(
        geometry=[shape(municipality_geometry)],
        crs="EPSG:3035",
    )
    geometry_gdf = geometry_gdf.to_crs(crs)
    return geometry_gdf


# %%
if __name__ == "__main__":
    # Test the function
    city = "trondheim"
    municipality_boundaries = get_municipal_boundaries(city)
    print(municipality_boundaries)
    municipality_boundaries.plot()
