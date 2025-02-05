# %%
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from pathlib import Path
from shapely.geometry import Polygon

root_dir = Path(__file__).parents[4]


def get_transform(shape1, shape2) -> tuple[list[float, float]]:
    """
    Get the tranform that transforms shape1 into shape2
    """
    # get the min_max extend of both shapes
    min_x1, min_y1, max_x1, max_y1 = shape1.bounds
    min_x2, min_y2, max_x2, max_y2 = shape2.bounds
    # get the center of both shapes
    center1 = shape1.centroid
    center2 = shape2.centroid
    # get the scale
    scale_x = (max_x2 - min_x2) / (max_x1 - min_x1)
    scale_y = (max_y2 - min_y2) / (max_y1 - min_y1)
    # get the translation
    offset_x = min_x2 - min_x1 * scale_x
    offset_y = min_y2 - min_y1 * scale_y
    return [scale_x, offset_x], [scale_y, offset_y]


def transform_shape(shape: Polygon, transform: tuple[list[float, float]]) -> Polygon:
    """
    Transform a shape with a given transform
    """
    coords = list(shape.exterior.coords)
    # print(coords[0])
    transformed_coords = []
    for point in coords:
        [x, y] = point
        transformed_coords.append(
            [
                x * transform[0][0] + transform[0][1],
                y * transform[1][0] + transform[1][1],
            ]
        )
    return Polygon(transformed_coords)


def plot_comparison_line(
    ax, connection_point1, connection_point2, thickness: float, color: str
):
    """
    Plot a curved line between two connection points.

    Args:
        ax: The matplotlib axis to plot on.
        connection_point1: The first connection point (x1, y1).
        connection_point2: The second connection point (x2, y2).
        thickness: The thickness of the line.
    """
    x1, y1 = connection_point1
    x2, y2 = connection_point2

    # Calculate the midpoint and offset it in the y-direction
    mid_x = (x1 + x2) / 2 + abs(y2 - y1) / 2
    mid_y = (y1 + y2) / 2  # + abs(x2 - x1) / 2

    # Generate points for the Bezier curve
    t = np.linspace(0, 1, 100)
    bezier_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mid_x + t**2 * x2
    bezier_y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * mid_y + t**2 * y2

    # Plot the Bezier curve
    ax.plot(bezier_x, bezier_y, linewidth=thickness, color=color, alpha=0.95, zorder=1)
    return


def tree_schematic(tree_data_path, reorder: dict = None, save_path: str = None):
    with open(tree_data_path, "r") as f:
        tree_data = json.load(f)

    tree = tree_data["tree"]
    # max ad min extends of all shapes in the tree
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    # make all shapes in the tree into shapely Polygon objects again
    for project, shapes in tree.items():
        for shape_id, shape in shapes.items():
            pol = Polygon(tree[project][shape_id]["shape"])
            tree[project][shape_id]["shape"] = pol
            pol_bounds = pol.bounds
            # print(pol_bounds)
            min_x = min(min_x, pol_bounds[0])
            min_y = min(min_y, pol_bounds[1])
            max_x = max(max_x, pol_bounds[2])
            max_y = max(max_y, pol_bounds[3])
    extend_shape = Polygon(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    )
    extend_x_size = max_x - min_x
    extend_y_size = max_y - min_y
    # print(f"min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")
    # print(tree)
    # the tree dimensions: n projects high, max shapes/project wide
    n_projects = len(tree_data["layers"])
    max_shapes = 0
    for project, shapes in tree.items():
        max_shapes = max(max_shapes, len(shapes.keys()))
    print(f"n_projects: {n_projects}, max_shapes: {max_shapes}")

    # we need to set the size of the squares for each node, and then
    # fit the max_extent into this square to have a base for each transform
    box_edge = 0.5
    if extend_x_size > extend_y_size:
        new_extend_y_size = extend_y_size / extend_x_size * box_edge
        new_extend_x_size = box_edge
    else:
        new_extend_x_size = extend_x_size / extend_y_size * box_edge
        new_extend_y_size = box_edge

    # normalize the comparison values
    # Normalize IoU values (later we could do the same for Hausdorff distance), and
    # then the normalization would actually do something
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a ScalarMappable object with the colormap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot_r, norm=norm)
    sm.set_array([])
    plot_scaler = 2
    fig, ax = plt.subplots(figsize=(max_shapes * plot_scaler, n_projects * plot_scaler))
    for i, (project, shapes_data) in enumerate(tree.items()):
        # print the time of each project - starting at y = 0 and increasing
        # by 1 for each project
        ax.text(0, i, tree_data["layer_times"][project], ha="right", va="center")

        n_shapes = len(shapes_data.keys())
        spacing = max_shapes / (n_shapes + 1)
        # if this layer is in the reorder dict, reorder the shapes
        if reorder:
            if project in reorder.keys():
                shapes_data = {k: shapes_data[k] for k in reorder[project]}
        for j, (shape_id, shape_data) in enumerate(shapes_data.items()):

            # print a 0.5x0.5 square below the text
            bottom_left_corner = ((j + 1) * spacing - box_edge / 2, i - box_edge / 2)
            ax.text(
                bottom_left_corner[0] + box_edge / 2,
                i + box_edge / 2 + 0.1,
                shape_id,
                ha="center",
                va="center",
            )
            # ax.add_patch(
            #     plt.Rectangle(
            #         bottom_left_corner,
            #         0.5,
            #         0.5,
            #         color="black",
            #         fill=False,
            #     )
            # )
            # now we need transform the shape and min_max extend so
            # that it fits in this exact box.abs
            # if x extend > y extend, scale x to box_edge, scale y to box_edge * y/x
            if extend_x_size > extend_y_size:
                new_min_x = bottom_left_corner[0]
                new_min_y = bottom_left_corner[1] + (box_edge - new_extend_y_size) / 2
                new_max_x = new_min_x + new_extend_x_size
                new_max_y = new_min_y + new_extend_y_size
            else:
                new_min_y = bottom_left_corner[1]
                new_min_x = bottom_left_corner[0] + (box_edge - new_extend_x_size) / 2
                new_max_y = new_min_y + new_extend_y_size
                new_max_x = new_min_x + new_extend_x_size
            # the connection point will be the center of the righ edge, so we save it to the tree:
            connection_point = (new_max_x, (new_min_y + new_max_y) / 2)
            tree[project][shape_id]["connection_point"] = connection_point
            new_extend_shape = Polygon(
                [
                    (new_min_x, new_min_y),
                    (new_max_x, new_min_y),
                    (new_max_x, new_max_y),
                    (new_min_x, new_max_y),
                ]
            )
            plt.plot(*new_extend_shape.exterior.xy, color="black", zorder=2)
            # get the transform
            transform = get_transform(extend_shape, new_extend_shape)
            # transform the shape
            transformed_shape = transform_shape(shape_data["shape"], transform)
            # print the transformed shape
            plt.plot(
                *transformed_shape.exterior.xy,
                color="blue",
                lw=1.2 * plot_scaler,
                zorder=3,
            )

            for comparison_project, comparison_p in shape_data["comparisons"].items():

                for comparison_id, comparison in comparison_p.items():
                    # print(f"comparison_id: {comparison_id}, comparison: {comparison}")
                    if (
                        comparison["remotely_overlapping"]
                        and comparison_project != project
                    ):
                        IoU = comparison["IoU"]
                        line_width = IoU**2 * 10
                        p1 = connection_point
                        p2 = tree[comparison_project][comparison_id]["connection_point"]
                        # the color corresponds to the IoU and is taken from a colormap (hot_r)
                        color = plt.cm.hot_r(norm(IoU))
                        plot_comparison_line(ax, p1, p2, line_width, color)

                # print the comparison shape
                # transformed_comparison = transform_shape(comparison["shape"], transform)
                # plt.plot(*transformed_comparison.exterior.xy, color="green")
                # print the connection line
                # plt.plot(
                #     [tree[project][shape_id]["connection_point"][0], comparison["connection_point"][0]],
                #     [tree[project][shape_id]["connection_point"][1], comparison["connection_point"][1]],
                #     color="black",
    # aspect equal
    # add a colorbar for the IoU at the bottom
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1, shrink=0.5)
    cbar.set_label("IoU")
    ax.set_aspect("equal")
    ax.set_xlim(0, max_shapes * +0.25 * (n_projects - 1))
    ax.set_ylim(-1, n_projects)
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig, ax


# %%
if __name__ == "__main__":
    data_path = root_dir / "data"
    t1 = data_path / "footprint_analysis/overlap_trees/testing2/244_trondheim_1991.json"
    reordering = {"trondheim_2016": ["167", "191", "208"]}
    tree_schematic(t1, reorder=None)

# %%
