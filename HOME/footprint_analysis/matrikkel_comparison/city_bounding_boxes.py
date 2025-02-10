"""
Playing around with the municipal boundaries of Norway.
Some of the functions might be used in other plots (e.g.get_municipal_boundaries)
"""

# %%
import folium
from shapely.geometry import box
import numpy as np
import webbrowser

from pyproj import Transformer
import json

from pathlib import Path

# %%
root_dir = Path(__file__).parents[3]
print(root_dir)
data_path = root_dir / "data"

# %%
import json

kommune_boundaries_path = (
    data_path
    / "raw/maps/municipalities"
    / "Basisdata_0000_Norge_3035_Kommuner_GeoJSON.geojson"
)


def inspect_kommune_boundaries_json():
    with open(kommune_boundaries_path) as f:
        data = json.load(f)
    print(data.keys())
    print(data["Kommune"].keys())
    type_data = data["Kommune"]["type"]
    print(type_data)
    print(type(type_data))

    crs_data = data["Kommune"]["crs"]
    print(crs_data)

    features_data = data["Kommune"]["features"]
    for feature in features_data:
        name = feature["properties"]["kommunenavn"]
        if "trondheim" in name.lower():
            print(f"found trondheim: {feature}")
            trondheim_borders = feature["geometry"]
            trondheim_crs = feature["crs"]
            print(trondheim_crs)
    print(trondheim_borders)
    # change EPSG for the trondhiem borders from 3035 to 3857
    # trondheim_borders['crs'] = {'type': 'name', 'properties': {'name': 'EPSG:3857'}}

    # print(type(features_data))
    # print(len(features_data))
    # print(features_data[0].keys())
    # print(features_data[0]['properties'])
    # print(features_data[0]['properties']['kommunenavn'])
    # print(features_data[0]['geometry'])
    # print(data['Kommune']['features'][0].keys())
    # for key in data['Kommune'].keys():
    #     print(data['Kommune'][key])
    # for k,v in data['Kommune'][key].items():
    #     print(k,v)
    #     break


# %%
# Initialize the transformer from EPSG:3035 to EPSG:4326
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def transform_coordinates(coordinates, transformer):
    """
    Transform a list of coordinates from EPSG:3035 to EPSG:3857.

    Args:
        coordinates: A list of coordinates to transform.

    Returns:
        A list of transformed coordinates.
    """
    return [transformer.transform(x, y) for x, y in coordinates]


def transform_geometry(geometry, transformer=None):
    """
    Transform the geometry of a MultiPolygon from EPSG:3035 to EPSG:3857.

    Args:
        geometry: The geometry dictionary to transform.

    Returns:
        The transformed geometry dictionary.
    """
    if not transformer:
        transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    if geometry["type"] == "MultiPolygon":
        transformed_coordinates = []
        for polygon in geometry["coordinates"]:
            transformed_polygon = []
            for ring in polygon:
                transformed_ring = transform_coordinates(ring, transformer)
                transformed_polygon.append(transformed_ring)
            transformed_coordinates.append(transformed_polygon)
        return {"type": "MultiPolygon", "coordinates": transformed_coordinates}
    else:
        raise ValueError("Unsupported geometry type")


def get_municipal_boundaries(municipality_name: str):
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
            print(f"found {municipality_name}: {name}")

    # project_borders to 4326
    municipality_geometry_4326 = transform_geometry(municipality_geometry)
    return municipality_geometry_4326


def print_municipality_borders(municipality_name: str):
    municipality_geometry_4326 = get_municipal_boundaries(municipality_name)

    municipality_center = [
        sum(x) / len(x) for x in zip(*municipality_geometry_4326["coordinates"][0][0])
    ]

    m = folium.Map(
        location=[municipality_center[1], municipality_center[0]], zoom_start=9
    )

    # Add a border print
    folium.GeoJson(
        municipality_geometry_4326,
        style_function=lambda x: {"color": "blue", "fill": False},
    ).add_to(m)
    return m


# %%
if __name__ == "__main__":
    m = print_municipality_borders("trondheim")
    m
    # map_path = "trondheim_map.html"
    # m.save(map_path)

    # Open the HTML file in a web browser
    # webbrowser.open(map_path)
    m
    # %%
    m = print_municipality_borders("bergen")
    m
# %%
