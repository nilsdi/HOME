# %%
import json
import matplotlib.pyplot as plt

from pathlib import Path
from shapely.geometry import Polygon

root_dir = Path(__file__).parents[4]


def tree_schematic(tree_data_path, save_path: str = None):
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
    print(f"min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")
    # print(tree)
    # the tree dimensions: n projects high, max shapes/project wide
    n_projects = len(tree_data["layers"])
    max_shapes = 0
    for project, shapes in tree.items():
        max_shapes = max(max_shapes, len(shapes.keys()))
    print(f"n_projects: {n_projects}, max_shapes: {max_shapes}")
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (project, shapes_data) in enumerate(tree.items()):
        # print the time of each project - starting at y = 0 and increasing
        # by 1 for each project
        ax.text(0, i, tree_data["layer_times"][project], ha="right", va="center")

        n_shapes = len(shapes_data.keys())
        spacing = max_shapes / (n_shapes + 1)
        for j, (shape_id, shape_data) in enumerate(shapes_data.items()):
            ax.text((j + 1) * spacing, i, shape_id, ha="center", va="center")
            # print a 0.5x0.5 square below the text
            ax.add_patch(
                plt.Rectangle(
                    ((j + 1) * spacing - 0.25, i - 0.6),
                    0.5,
                    0.5,
                    color="black",
                    fill=False,
                )
            )
            # now we need transform the shape and min_max extend so
            # that it fits in this exact box.abs
            # TODO
    # aspect equal
    ax.set_aspect("equal")
    ax.set_xlim(0, max_shapes)
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
    tree_schematic(t1)

# %%
