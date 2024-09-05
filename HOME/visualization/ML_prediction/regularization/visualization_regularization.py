#%%
# imports
from pyproj import Transformer
import folium
from shapely.ops import transform
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
def tile_number_to_coordinates(tile_number: list[int, int], tile_size: int = 512, resolution: int) -> list[float, float]:
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

def get_EPSG25833_coords(
    row, col, tile_size: int, res: float
) -> tuple[list[int], list[int]]:
    """
    Get the coordinates of the top left corner and bottom right corner of a tile in
    EPSG:25833, based on its  row and column in the grid of tiles.
    """
    # get the coordinates of the top left corner of the tile
    x_tl = col * tile_size * res
    y_tl = row * tile_size * res
    x_br = (x_tl + 1) * tile_size * res
    y_br = (y_tl - 1) * tile_size * res
    return [x_tl, y_tl], [x_br, y_br]


def plot_polygons_on_tile(img_path, gdf):
    # Open the GeoTIFF file
    with rasterio.open(img_path) as src:
        # Transform the polygons to the same CRS as the image
        gdf_transformed = gdf.to_crs(src.crs)
# to plot the polygons on top of any corresponding tile
    # Plotting the GeoDataFrame over the original image
    plt.figure(figsize=(10, 10))
    with rasterio.open(processed_img_path) as src:
        # Read the first band of the image
        img_array = src.read()
        # Plot the image
        plt.imshow(img_array[0], cmap='gray', extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
        
        # Plot the polygons on top of the image
    gdf.plot(ax=plt.gca(), facecolor='none', edgecolor='red')
    plt.axis('off')
    plt.title(f"Polygons for {processed_img_path}")
    plt.show()

# to make a bounding box

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

#%%
#run the function
root_dir = Path(__file__).parents[3]

# Parameters
project_name = "trondheim_2019"  # Example project name

project_dict_path = root_dir / "data/ML_prediction/project_log/project_details.json"
# Open and read the JSON file
with open(project_dict_path, 'r') as file:
    project_dict = json.load(file)

# Get the resolutiom and other details
resolution = project_dict[project_name]['resolution']
compression_name = project_dict[project_name]['compression_name']
compression_value = project_dict[project_name]['compression_value']

# path to the reassembled tiles
tile_dir = root_dir / f"data/ML_prediction/large_tiles/res_{resolution}/{project_name}"

# Get the tiles within the bounding box
tiles_in_bbox = get_tiles_in_bbox(tile_dir, min_lat, max_lat, min_lon, max_lon)


#%%

if __name__ == "__main__":
    # get the pickled polygons
    pickle_file_path = root_dir / f"data/ML_prediction/polygons/{project_name}_combined_geodata.pkl"
    with open(pickle_file_path, 'rb') as f:
        combined_gdf = pickle.load(f)
    # plot the polygons on a folium map
    # Apply the transformation to each geometry in the GeoDataFrame
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(apply_transformation)
    map = plot_gdf_on_map(combined_gdf, polygon_index=0)
    map
    folder_path = tile_dir #root_dir / f"data/ML_prediction/predictions/res_{resolution}/{project_name}/reassembled_tiles"
    #original_folder_path = root_dir / f"data/ML_prediction/topredict/image/res_{resolution}/{project_name}/reassembled_tiles"
    # Regular expression to match the file format