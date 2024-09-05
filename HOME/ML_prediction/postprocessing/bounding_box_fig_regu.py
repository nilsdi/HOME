#%%
#making a bounding box so we run the regularization only in the area of interest
#imports
import folium
from folium.plugins import Draw
import math

#bounding box around the center of Trondheim
# coordinates in UTM33N
# bounding box around the center of Trondheim
# coordinates in 
lat = 63.4228306
lon = 10.3908306
box_size = 3000  # in meters

# calculate the bounding box coordinates
min_lat = lat - (box_size / 2 / 111111)
max_lat = lat + (box_size / 2 / 111111)
min_lon = lon - (box_size / 2 / (111111 * math.cos(math.radians(lat))))
max_lon = lon + (box_size / 2 / (111111 * math.cos(math.radians(lat))))

# print the bounding box coordinates
print(f"Min Latitude: {min_lat}")
print(f"Max Latitude: {max_lat}")
print(f"Min Longitude: {min_lon}")
print(f"Max Longitude: {max_lon}")

# plot on a folium map

m = folium.Map(location=[lat, lon], zoom_start=14)
#draw the box on the map
folium.Rectangle(
    bounds=[[min_lat, min_lon], [max_lat, max_lon]],
    color="red",
    fill=True,
    fill_color="red",
    fill_opacity=0.2,
).add_to(m)
m

#%% 

#make a function so we only select tiles that are at least partially within the bounding box
def check_tile_in_bbox(tile_path, min_lat, max_lat, min_lon, max_lon):
    # Open the GeoTIFF file
    with rasterio.open(tile_path) as src:
        # Transform the bounding box to EPSG:4326
        bbox_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)

    # Check if the bounding box of the tile intersects with the bounding box of interest
    if (
        bbox_4326[0] < max_lon
        and bbox_4326[2] > min_lon
        and bbox_4326[1] < max_lat
        and bbox_4326[3] > min_lat
    ):
        return True
    else:
        return

# make a function that runs across all tiles and returns the ones that are at least partially within the bounding box

def get_tiles_in_bbox(tile_dir, min_lat, max_lat, min_lon, max_lon):
    # Get all the files in the directory
    files = list(tile_dir.glob("*.tif"))
    # Check if the file is within the bounding box
    tiles_in_bbox = [f for f in files if check_tile_in_bbox(f, min_lat, max_lat, min_lon, max_lon)]
    return tiles_in_bbox

