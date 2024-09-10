# %% imports


# %%


def Hausdorff_distance(shape1, shape2):
    """
    Compute the Hausdorff distance between two shapes
    """
    return shape1.hausdorff_distance(shape2)


def calculate_similarity_measures(shape1, shape2):
    """
    Compute the similarity between two shapes.
    Use Hausdorff distance for shape1 to shape2,
    the IoU,
    the Haussdorff distance for each shape to the Intersect,
    the ratio of intersect to each shape area.
    """

    return 1 - Hausdorff_distance(shape1, shape2) / max(shape1.area, shape2.area)
