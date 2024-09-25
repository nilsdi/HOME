# %% imports
import folium
import rasterio
from rasterio.warp import transform_bounds
from folium.raster_layers import ImageOverlay
from pathlib import Path


def print_geotiff(geotiff_path):
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Transform the bounding box to EPSG:4326
        bbox_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)

        # Calculate the center of the transformed bounding box
        center = [(bbox_4326[1] + bbox_4326[3]) / 2, (bbox_4326[0] + bbox_4326[2]) / 2]

    # Create a Folium map centered at the transformed bounding box center
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    # Add Google Street Map layer
    folium.TileLayer(
        tiles="http://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Street Map",
        max_zoom=20,
        subdomains=["mt0", "mt1", "mt2", "mt3"],
    ).add_to(m)

    # Add the GeoTIFF as an overlay
    ImageOverlay(
        name="GeoTIFF Overlay",
        image=str(geotiff_path),
        bounds=[[bbox_4326[1], bbox_4326[0]], [bbox_4326[3], bbox_4326[2]]],
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Add layer control and display the map
    folium.LayerControl().add_to(m)
    map_path = "GeoTIFF_on_Folium_Map.html"
    m.save(map_path)
    display(IFrame(map_path, width=800, height=600))


# example tiff:

# %% running test
from HOME.get_data_path import get_data_path
from IPython.display import display
from IPython.display import IFrame

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[3]
    data_path = get_data_path(root_dir)
    tile_path = Path(
        data_path / f"temp/test_assembly/trondheim_kommune_2021resolution0.2_41_9.tif"
    )
    # Example usage
    print_geotiff(tile_path)
# %%
