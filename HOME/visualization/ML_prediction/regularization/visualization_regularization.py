#%%
# imports
from pyproj import Transformer
import folium
from shapely.ops import transform
import random
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import rasterio
# %%
# to plot the polygons on top of a folium map 

# Create a transformer to convert from UTM 33N (EPSG:32633) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs('epsg:32633', 'epsg:4326', always_xy=True)

# Function to apply the transformation to a geometry
def apply_transformation(geom):
    return transform(lambda x, y: transformer.transform(x, y), geom)

def plot_gdf_on_map(gdf, polygon_index=0):
    polygon = gdf.iloc[polygon_index].geometry
    minx, miny, maxx, maxy = polygon.bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=14)
    
    for _, row in gdf.iterrows():
        sim_geo = row['geometry']
        folium.GeoJson(sim_geo, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange'}).add_to(m)
    
    return m

#%%
#plotting corresponding polygons on top of any tile (original or processed)

#extract tile number from the file name (contains the coordinates)
def extract_tile_numbers(filename: str) -> list[int, int]:
    """
    Extracts the x/col and y/row (coordinates) from a filename
    of pattern '_x_y'.

    Args:
        filename (str): name of a tile cut from a larger image.

    Returns:
        tuple: row, col number of the tile  in the absolute system of , meaning a tile with the name '_0_0' is the top left tile.
    """
    parts = filename.split("_")
    col = int(parts[-2])  # x_coord
    row = int(parts[-1].split(".")[0])  # y_coord
    return [col, row]

# from the tile number to the coordinates
def tile_number_to_coordinates(tile_number, tile_size, resolution):
    """
    Converts the tile number to the coordinates of the top left corner of the tile.

    Args:
        tile_number (list[int, int]): row, col number of the tile in the absolute system of coordinates.
        tile_size (int): size of the tile in pixels.
        resolution (int): resolution of the tile in meters.

    Returns:
        list[float, float]: x, y coordinates of the top left corner of the tile.
    """
    x = tile_number[0] * resolution * tile_size #col
    y = tile_number[1] * resolution * tile_size #row
    return [x, y]

# function to only select the polygons that are within a given tile
def filter_polygons_by_tile(gdf, tile_number, tile_size, resolution):
    # Get the coordinates of the top left corner of the tile
    x, y = tile_number_to_coordinates(tile_number, tile_size, resolution)
    x2, y2 = x + tile_size * resolution, y - tile_size * resolution

    #transform the coordinates from utm33n to lat lon
    x, y = transformer.transform(x, y)
    x2, y2 =   transformer.transform(x2, y2)

    # Filter the GeoDataFrame to only include polygons within the tile (add a buffer?)
    gdf_filtered = gdf.cx[x:x2, y:y2]
    return gdf_filtered

def plot_polygons_on_tile(img_path, gdf):
    #do everything in one function
    #tile number
    tile_number = extract_tile_numbers(img_path.name)
    #transform into coordinates
    x, y = tile_number_to_coordinates(tile_number, tile_size, resolution)
    #filter the polygons
    gdf_filtered = filter_polygons_by_tile(gdf, tile_number, tile_size, resolution)

    #now plot the polygons on top of the tile
    # Open the GeoTIFF file
    with rasterio.open(img_path) as src:
        # Read the first band of the image
        img_array = src.read()
        # Plot the image
        plt.imshow(img_array[0], cmap='gray', extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        # Plot the polygons on top of the image
        gdf_filtered.plot(ax=plt.gca(), facecolor='none', edgecolor='gold',linewidth=5)

    plt.axis('off')
    plt.title(f"Polygons for {img_path}")
    plt.show()



#%%

if __name__ == "__main__":
    # get the pickled polygons
    project_name = "trondheim_kommune_2021" 
    root_dir = Path(__file__).parents[4]
    project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
    # Open and read the JSON file
    with open(project_dict_path, 'r') as file:
        project_dict = json.load(file)

    # Get the resolutiom and other details
    resolution = project_dict[project_name]['resolution']
    compression_name = project_dict[project_name]['compression_name']
    compression_value = project_dict[project_name]['compression_value']
    tile_size = 512
    # path to the og tiles
    og_tile_dir = root_dir / f"data/ML_prediction//topredict/image/res_{resolution}/{project_name}/i_{compression_name}_{compression_value}"


    #path to the pickled polygons
    pickle_file_path = root_dir / f"data/ML_prediction/polygons/{project_name}_combined_geodata.pkl"
    with open(pickle_file_path, 'rb') as f:
        combined_gdf = pickle.load(f)

    map = plot_gdf_on_map(combined_gdf, polygon_index=0)
    map

    #plotting polygons on og tiles
    # get a random tile
    tile_files = list(og_tile_dir.glob("*.tif"))
    random_tile = random.choice(tile_files)
    tile_path = og_tile_dir / random_tile

    # run on the random tile
    plot_polygons_on_tile(tile_path, combined_gdf)

    #plot the tif file on a folium map
    # Open the GeoTIFF file
    with rasterio.open(tile_path) as src:
        # Get the top left coordinates
        [x, y] = extract_tile_numbers(tile_path.name)

        #bottom left
        bottom_left = [x*tile_size*resolution, (y -1)* tile_size * resolution]
        top_right = [(x+1)*tile_size*resolution, y*tile_size*resolution]

        #print the length and height of the tile
        print(f"Length: {top_right[0] - bottom_left[0]}")
        print(f"Height: {top_right[1] - bottom_left[1]}")

        # Transform the coordinates to the correct projection (EPSG:4326)
        [x_bottom_left, y_bottom_left] = transformer.transform(bottom_left[0], bottom_left[1])
        [x_top_right, y_top_right] = transformer.transform(top_right[0], top_right[1])

        center_new = [(y_bottom_left + y_top_right) / 2, (x_bottom_left + x_top_right) / 2]
        bottom_left_new = [y_bottom_left, x_bottom_left]
        top_right_new = [y_top_right, x_top_right]
        # Create a folium map
        m = folium.Map(location=center_new, zoom_start=14)

        # Add the GeoTIFF file to the map
        folium.raster_layers.ImageOverlay(
            # show a black square
            image=src.read(1),
            bounds=[bottom_left_new, top_right_new],
            colormap=lambda x: (0, x, 0),
        ).add_to(m)
    m




# %%
#%%
#bounding box around the center of Trondheim
# coordinates in lat lon
# bounding box around the center of Trondheim
# coordinates in 
lat = 63.392307
lon = 10.413476
box_size = 700  # in meters

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


# Get the tiles within the bounding box
# tiles_in_bbox = get_tiles_in_bbox(tile_dir, min_lat, max_lat, min_lon, max_lon)
