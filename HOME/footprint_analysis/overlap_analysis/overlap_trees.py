# %%
import geopandas as gpd

from shapely.geometry import Polygon
from HOME.footprint_analysis.shap_similarity import bounding_box_overlap


def access_shape(shape_loc: str) -> Polygon:
    """
    Access the shape from the given location.

    Args:
        shape_loc (str): The location of the shape. "fgb_file.shapeID"

    Returns:
        Polygon: The shape
    """
    gdf_path = ".".join(shape_loc.split(".")[:-1])
    polygon_id = shape_loc.split(".")[-1]
    gdf = gpd.read_file(gdf_path)
    return Polygon(gdf[gdf["shapeID"] == polygon_id].geometry)


def overlap_tree(
    shape_loc: str,
    layered_shapes: dict[dict[str, str]],
    extension: float = 5,
):
    """
    Create a tree of overlapping shapes rooted in the given shape.

    Args:
        shape (Polygon): ID of the shape to start from. "fgb_file.shapeID"
        sourrounding_shapes_layers (dict[str, str]): A dictionary of layers/gdf with shapes
        extension (float, optional): The extension of the bounding box. Defaults to 5.
    """
    shape = access_shape(shape_loc)
    tree = {}
    for project_name, project_layer in layered_shapes.items():
        capture_date = project_layer["capture_date"]
        project_covers: bool = True
        for other_shape in layer:
            if bounding_box_overlap(shape, other_shape):
                if other_shape not in tree:
                    tree[other_shape] = overlap_tree(
                        other_shape, sourrounding_shapes_layers
                    )
    return tree
