"""
To easily toggle between data paths for different machines, we will always use the
function from this file to refer to the current data folder
"""

import os
from pathlib import Path


def get_data_path(root_dir: str):
    # Path to the text file
    config_file = os.path.join(root_dir, "HOME/data_path.txt")
    # print(f"config_file: {config_file}")
    # Read the current data folder from the file
    try:
        with open(config_file, "r") as file:
            mode = file.read().strip()
    except IOError:
        print("cannot accss the dat_path.txt file - falling back to default (data)")
        # Default to 'data' if the file cannot be read
        mode = "data"

    return Path(os.path.join(root_dir, mode))
