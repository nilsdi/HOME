"""
plotting any footprints into a customly skewed coordinate system
"""

# %%imports
import matplotlib.pyplot as plt


def plot_footprint(
    footprint: dict[str, list[list[float]]],
    ax: plt.Axes = None,
    color="k",
    ls="-",
    lw=1,
    fill: bool = False,
    fill_color: str = "gray",
    fill_alpha: float = 0.5,
):
    # split drawing into exterior and all interiors:
    polygons = [footprint["exterior"]] + footprint["interiors"]
    for polygon in polygons:
        # print("polygon: ", polygon)
        if len(polygon) == 0:
            # print("empty polygon: ", polygon)
            continue
        verts = polygon + [polygon[0]]
        x, y = zip(*verts)  # Unpack vertices into x and y
        if not fill:
            fill_color = None
        else:
            raise NotImplementedError("filling not implemented yet")
        ax.plot(x, y, ls=ls, lw=lw, color=color)  # Plot the polygon

    return


def plot_skewed_footprints(
    footprints: list[dict[str, list[list[float]]]],
    ax: plt.Axes = None,
    skew: float = 0.5,
    flatten: float = 0.9,
):
    for footprint in footprints:
        skewed_flattened_footprint = {}
        skewed_flattened_footprint["exterior"] = skew_flatten_verts(
            footprint["exterior"], skew=skew, flatten=flatten
        )
        skewed_flattened_footprint["interiors"] = [
            skew_flatten_verts(interior, skew=skew, flatten=flatten)
            for interior in footprint["interiors"]
        ]
        plot_footprint(skewed_flattened_footprint, ax=ax)
    return


def skew_flatten_verts(
    verts: list[list[float]], skew: float = 0.5, flatten: float = 0.9
):
    return [[x + y * skew, y * flatten] for x, y in verts]


def get_extend_boxes(footprints_l: list[dict[str, list[list[float]]]]):
    """
    Aid function that makes an extend box for the maximum extend
    across all layers of footprints
    """
    x0s = []
    x1s = []
    y0s = []
    y1s = []
    for i, footprints in enumerate(footprints_l):
        # get box coordinates:
        x0 = min(
            [min([v[0] for v in fp]) for fp["exterior"] in footprints if len(fp) > 0]
        )
        x1 = max(
            [max([v[0] for v in fp]) for fp["exterior"] in footprints if len(fp) > 0]
        )
        y0 = min(
            [min([v[1] for v in fp]) for fp["exterior"] in footprints if len(fp) > 0]
        )
        y1 = max(
            [max([v[1] for v in fp]) for fp["exterior"] in footprints if len(fp) > 0]
        )
        x0s.append(x0)
        x1s.append(x1)
        y0s.append(y0)
        y1s.append(y1)
    x0 = min(x0s)
    x1 = max(x1s)
    y0 = min(y0s)
    y1 = max(y1s)
    x_ext = (x1 - x0) * 0.1
    y_ext = (y1 - y0) * 0.1
    x0 -= x_ext
    x1 += x_ext
    y0 -= y_ext
    y1 += y_ext
    boxes = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for _ in range(len(footprints_t))]
    return boxes


def stacked_skewed_footprints(
    footprints_t: list[list[list[float]]],
    t: list[float],
    ax: plt.Axes = None,
    skew: float = 0.5,
    flatten: float = 0.9,
    overlap: float = -0.1,
    plot_connecting_lines: bool = False,
):
    """
    Plot a series of footprints stacked on top of each other in a skewed coordinate system
    Args:
    footprints_t: list of list  of footprints, each list of footprints represent a time, each
                    footprint is a list of vertices, each vertex is a list of x and y coordinates
    t: list of time values, the time values should be in increasing order
    ax: matplotlib axis object, the axis to plot the footprints on
    skew: float, the skew factor to apply to the x coordinates
    flatten: float, the flatten factor to apply to the y coordinates
    overlap: by how much the closest (!) footprints should overlap,
                 negative values mean distance instead of overlap
    """
    # offset the y coordinates of each footprint by the corresponding t value
    min_t_dist = min([t1 - t0 for t0, t1 in zip(t[:-1], t[1:])])
    y_dist = min(
        [
            max([max([v[1] for v in fp]) for fp in footprints])
            - min([min([v[1] for v in fp]) for fp in footprints])
            for footprints in footprints_t
        ]
    )
    t_dist = [tx - t[0] for tx in t]
    # we want to offset enough to make the footprints not overlap and even have some distance
    # so we assign the y offset per y distance in a way that the overlap is as asked for
    y_offset_factor = y_dist / (min_t_dist) * (1 - overlap) * flatten

    # we make a rectangle for each list of footprints that will show the layer:
    boxes = get_extend_boxes(footprints_t)
    # we first skew and flatten all coordinates
    skewed_flattened_footprints = [
        [skew_flatten_verts(fp, skew=skew, flatten=flatten) for fp in footprints]
        for footprints in footprints_t
    ]
    skew_flattened_boxes = [
        skew_flatten_verts(box, skew=skew, flatten=flatten) for box in boxes
    ]
    # we make a copy of the footprint verts of all but the first time step to later make the
    # connecting lines down
    import copy

    connecting_lines_bottom_verts = copy.deepcopy(
        [[fp for fp in footprints] for footprints in skewed_flattened_footprints[1:]]
    )

    # then we offset the y coordinates of boxes and footprints
    for footprints, t_offset in zip(skewed_flattened_footprints, t_dist):
        for fp in footprints:
            for v in fp:
                v[1] += t_offset * y_offset_factor
    for bottom_fps, t_m1_offset in zip(connecting_lines_bottom_verts, t_dist[:-1]):
        for fp in bottom_fps:
            for v in fp:
                v[1] += t_m1_offset * y_offset_factor
    for box, t_offset in zip(skew_flattened_boxes, t_dist):
        for v in box:
            v[1] += t_offset * y_offset_factor

    # then we plot the boxes and  the footprints of each time step
    for i, (footprints, box) in enumerate(
        zip(skewed_flattened_footprints, skew_flattened_boxes)
    ):
        plot_footprint(
            box,
            ax=ax,
            color="gray",
            ls="--",
            lw=1,
            fill=True,
            fill_color="gray",
            fill_alpha=0.1,
        )
        for fp in footprints:
            plot_footprint(fp, ax=ax, color="crimson")
        # then we plot the connecting lines
        if i > 0:
            bottom_verts = connecting_lines_bottom_verts[i - 1]
            for top_fp, bottom_fp in zip(footprints, bottom_verts):
                for v1, v2 in zip(top_fp, bottom_fp):
                    ax.plot(
                        [v1[0], v2[0]], [v1[1], v2[1]], color="gray", ls="--", lw=0.3
                    )

    return ax


# %%testing
if __name__ == "__main__":
    # Define our custom "footprints"
    TC = [271805, 7043536]  # Center trondheim in EPSG 24833
    h1 = [
        TC,
        [TC[0], TC[1] + 20],
        [TC[0] + 20, TC[1] + 20],
        [TC[0] + 20, TC[1]],
    ]
    h2 = [
        [TC[0] + 40, TC[1] + 30],
        [TC[0] + 40, TC[1] + 30 + 20],
        [TC[0] + 40 + 20, TC[1] + 30 + 20],
        [TC[0] + 40 + 20, TC[1] + 30],
    ]
    s1 = {"exterior": h1, "interiors": []}
    s2 = {"exterior": h2, "interiors": []}
    fig, ax = plt.subplots()
    plot_skewed_footprints([s1, s2], ax=ax, skew=0.25, flatten=0.7)
    plt.axis("off")
    ax.set_aspect("equal")
    plt.show()
    # %%
    # new test
    fig, ax = plt.subplots()
    stacked_skewed_footprints(
        [[h1, h2], [h1, h2], [h1, h2], [h1, h2], [h1, h2]],
        [2000 + e for e in [0, 2, 5, 6.8, 8]],
        ax=ax,
        skew=0.4,
        flatten=0.5,
        overlap=0.1,
        plot_connecting_lines=True,
    )
    plt.axis("off")
    ax.set_aspect("equal")
    plt.show()
# %%
