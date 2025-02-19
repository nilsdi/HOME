import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps
from shapely.geometry import Polygon


def plot_footprint(
    footprint: list[list[float]],
    ax: plt.Axes = None,
    color="k",
    ls="-",
    lw=1,
    line_alpha: float = 1.0,
    fill: bool = False,
    fill_color: str = "gray",
    fill_alpha: float = 0.5,
):
    # Ensure the polygon is closed by repeating the first vertex
    verts = footprint + [footprint[0]]
    x, y = zip(*verts)  # Unpack vertices into x and y,
    if not fill:
        fill_color = None
    ax.plot(x, y, ls=ls, lw=lw, color=color, alpha=line_alpha)  # Plot the polygon
    if fill:
        ax.fill(x, y, color=fill_color, alpha=fill_alpha)  # Optionally fill the polygon
    return


def stacked_combined_plot(
    project_names: list[str],
    footprints_t: dict[str, [dict[int, list[float]]]],
    t: dict[str, float],
    coverages_t: dict[str, Polygon],
    bshape: Polygon,
    tifs: dict[str, str] = None,
    skew: float = 0.5,
    flatten: float = 0.9,
    overlap: float = -0.1,
    cmap: str = "tab20",
    colors: list[str] = None,
    figsize: tuple = None,
    line_width: float = 1,
    special_footprint_ids: dict[str, list[int]] = None,  # TODO: print them special
):
    """
    Plot a series of footprints stacked on top of each other in a skewed coordinate system
    Args:
    footprints_t: list of list  of footprints, each dictionary of footprints represent a time which
                    keys the id of a polygon and a list of vertices, each vertex is a list of x and y coordinates
    t: list of time values, the time values should be in increasing order
    b_shape: shapely Polygon, the bounding shape for the footprints
    tifs: dict of tif images (one per layer) to plot on the right hand side of the footprints
    ax: matplotlib axis object, the axis to plot the footprints on
    skew: float, the skew factor to apply to the x coordinates
    flatten: float, the flatten factor to apply to the y coordinates
    overlap: by how much the closest (!) footprints should overlap,
    negative values mean distance instead of overlap
    cmap: str, the colormap to use for the footprints
    figsize: tuple, the size of the figure
    """

    # prepare colors for each layer
    if not colors:
        colormap = colormaps[cmap]
        colors = colormap(np.arange(len(footprints_t) % colormap.N))

    if not figsize:
        figsize = (20, 20)
    fig, ax = plt.subplots(figsize=figsize)

    min_t = min(t.values())
    t_dist = {p: tx - min_t for p, tx in t.items()}
    ordered_t = sorted(t.values())
    min_t_dist = min([t1 - t0 for t0, t1 in zip(ordered_t[:-1], ordered_t[1:])])

    x_min, y_min, x_max, y_max = bshape.bounds
    y_dist = y_max - y_min

    # the y offset factor is calculated from the distance between the closest time steps,
    # the max distance between layers, and the overlap factor.
    y_offset_factor = y_dist * (1 - overlap) / min_t_dist * flatten

    def skew_flatten_verts(
        verts: list[list[float]], skew: float = 0.5, flatten: float = 0.9
    ):
        return [[x + y * skew, y * flatten] for x, y in verts]

    # coverage for each project, plus one that is max extent
    extend_boxes = [
        [[x, y] for x, y in list(bshape.exterior.coords)] for _ in range(len(t) + 1)
    ]

    # we first skew and flatten all coordinates
    skewed_flattened_footprints = {
        project: {
            pid: skew_flatten_verts(fp, skew=skew, flatten=flatten)
            for pid, fp in footprints.items()
        }
        for project, footprints in footprints_t.items()
    }
    extend_boxes_skewed = [
        skew_flatten_verts(box, skew=skew, flatten=flatten) for box in extend_boxes
    ]
    # then we offset the y coordinates of boxes and footprints
    for footprints, t_offset in zip(
        skewed_flattened_footprints.values(), t_dist.values()
    ):
        for fp in footprints.values():
            for v in fp:
                v[1] += t_offset * y_offset_factor
    for box, t_offset in zip(extend_boxes_skewed, t_dist.values()):
        for v in box:
            v[1] += t_offset * y_offset_factor

    # then we plot the boxes and  the footprints of each time step
    for i, (year, project, extend_box) in enumerate(
        zip(t, project_names, extend_boxes_skewed)
    ):
        if project in coverages_t.keys():
            for coverage_box in coverages_t[project]:
                coverage_box = [[x, y] for x, y in list(coverage_box.exterior.coords)]
                coverage_box_skewed = skew_flatten_verts(
                    coverage_box, skew=skew, flatten=flatten
                )
                for v in coverage_box_skewed:
                    v[1] += t_dist[project] * y_offset_factor
                plot_footprint(
                    coverage_box_skewed,
                    ax=ax,
                    color="gray",
                    ls="--",
                    lw=1,
                    fill=True,
                    fill_color="gray",
                    fill_alpha=0.1,
                )
        # print the date for each layer
        ax.text(
            min([v[0] for v in extend_box]),
            np.mean([v[1] for v in extend_box]),
            f"{t[project]}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )
        # if given, print the tifs for each layer on the right hand side
        if tifs:
            y_offset = (0.05 + overlap / 2) * y_dist * flatten
            # put the tif on the right hand side of the max extent box :
            box_max_x = max([v[0] for v in extend_box])
            box_min_y = min([v[1] for v in extend_box])
            box_max_y = max([v[1] for v in extend_box])
            # box_y_extent = box_max_y - box_min_y - _offset

            tile_min_y = box_min_y + y_offset
            tile_max_y = box_max_y - y_offset
            tile_min_x = box_max_x + 0.2 * y_dist * flatten
            tile_max_x = tile_min_x + tile_max_y - tile_min_y

            if project in tifs.keys():
                img = tifs[project]
                if img is not None:
                    # if rgb:
                    if len(img.shape) == 3:
                        ax.imshow(
                            img,
                            extent=[tile_min_x, tile_max_x, tile_min_y, tile_max_y],
                        )
                    elif len(img.shape) == 2:
                        ax.imshow(
                            img,
                            extent=[tile_min_x, tile_max_x, tile_min_y, tile_max_y],
                            cmap="gray",
                        )
        footprints = skewed_flattened_footprints[project]
        for sid, fp in footprints.items():
            if special_footprint_ids:
                if sid in special_footprint_ids[project]:
                    plot_footprint(fp, ax=ax, color="black", lw=line_width + 1)
                    # plot the id of the special footprint in the middle of the footprint
                    x, y = zip(*fp)
                    ax.text(
                        np.mean(x),
                        np.mean(y),
                        f"{sid}",
                        ha="center",
                        va="center",
                        color="crimson",
                        fontsize=6,
                    )
                else:
                    plot_footprint(fp, ax=ax, color=colors[i], lw=line_width)
            else:
                plot_footprint(fp, ax=ax, color=colors[i], lw=line_width)

    plt.axis("off")
    ax.set_aspect("equal")
    return fig, ax
