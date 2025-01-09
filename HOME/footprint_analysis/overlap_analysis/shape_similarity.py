# %% imports


# %%
# the shape similarity should be hierachical - first check if there is any overlap, then check
# in more detail (e.g. Hausdorff distance) how similar they are
def bounding_box_overlap(shape1, shape2, extension: float = 0):
    """
    Check if the bounding boxes of two shapes overlap.
    Optionally includes an extension (fixed amount) to increase the bounding box size
    => even "close" shapes will be considered overlapping

    Args:
        shape1, shape2: two shapes
        extension: amount to extend the bounding box

    Returns:
        bool: True if the bounding boxes overlap, False otherwise
    """
    bbox1 = shape1.bounds
    bbox2 = shape2.bounds

    bbox1_x_min = bbox1[0] - extension
    bbox1_x_max = bbox1[2] + extension
    bbox1_y_min = bbox1[1] - extension
    bbox1_y_max = bbox1[3] + extension

    bbox2_x_min = bbox2[0] - extension
    bbox2_x_max = bbox2[2] + extension
    bbox2_y_min = bbox2[1] - extension
    bbox2_y_max = bbox2[3] + extension

    # for overlap, there must be overlap in both dimensions
    # to check for overlap, we just exclude the cases where there is no overlap:
    # either, box 1 does not even reach box 2 (max b1  < min b2) or box 1 is already past box 2 (min b1 > max b2)
    x_overlap = not (bbox1_x_max < bbox2_x_min or bbox1_x_min > bbox2_x_max)
    y_overlap = not (bbox1_y_max < bbox2_y_min or bbox1_y_min > bbox2_y_max)
    return x_overlap and y_overlap


def IoU(shape1, shape2):
    """
    Compute the Intersection over Union of two shapes
    """
    intersection = shape1.intersection(shape2).area
    if intersection == 0:
        return 0
    else:
        union = shape1.union(shape2).area
        return intersection / union


def Hausdorff_distance(shape1, shape2):
    """
    Compute the Hausdorff distance between two shapes
    """
    return shape1.hausdorff_distance(shape2)


# TODO add the other similarity measures


def calculate_similarity_measures(shape1, shape2):
    """
    Compute the similarity between two shapes.
    Use Hausdorff distance for shape1 to shape2,
    the IoU,
    the Haussdorff distance for each shape to the Intersect,
    the ratio of intersect to each shape area.
    """
    similarity = {}
    remotely_overlapping = bounding_box_overlap(shape1, shape2, extension=10)
    similarity["remotely_overlapping"] = remotely_overlapping
    if remotely_overlapping:
        similarity["IoU"] = IoU(shape1, shape2)
        similarity["Hausdorff_distance"] = Hausdorff_distance(shape1, shape2)
        similarity["Hausdorff_distance_shape1_intersect"] = Hausdorff_distance(
            shape1, shape1.intersection(shape2)
        )
        similarity["Hausdorff_distance_shape2_intersect"] = Hausdorff_distance(
            shape2, shape1.intersection(shape2)
        )
        # similarity["intersect_area_ratio_shape1"] = shape1.intersection(shape2).area / shape1.area
        # similarity["intersect_area_ratio_shape2"] = shape1.intersection(shape2).area / shape2.area

    return 1 - Hausdorff_distance(shape1, shape2) / max(shape1.area, shape2.area)
