"""
Get the size of a directory in a human-readable format (similar to `du -sh`).
This function uses the `du` command to get the size of a directory and its subdirectories.
Useful for checking for filling up disk space while doing large projects.
"""

# %%
import subprocess


def get_directory_size_human_readable(
    directory, summary: str = False, human_readable: bool = True
):
    """
    Get the size of a directory in a human-readable format (similar to `du -sh`).

    Args:
        directory (str): Path to the directory.
        summary (str): If True, return a summary of the directory sizes, else return sizes of all subdirectories.

    Returns:
        str: Size of the directory in human-readable format.
    """
    if summary:
        if human_readable:
            command_keys = "-sh"
        else:
            command_keys = "-s"
    else:
        if human_readable:
            command_keys = "-h"
        else:
            raise ValueError(
                "non-human-readable format is not supported for detailed sizes."
            )
            command_keys = ""
    try:
        if summary:
            result = subprocess.run(
                ["du", command_keys, directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.split()[0]  # Return the size part of the output
        else:
            result = subprocess.run(
                ["du", command_keys, directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            # return a dictionary witht the subdirectory sizes
            sizes = {}
            for line in result.stdout.splitlines():
                size, path = line.split(maxsplit=1)
                # split the path to get the directory name
                path = path.split("/")[-1]
                sizes[path] = size
            return sizes
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None


# %%
if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[2]
    data_path = root_dir / "data"
    # Example usage
    directory = data_path
    size = get_directory_size_human_readable(directory, summary=True)
    print(f"Size of {directory}: {size}")

    size_machine = get_directory_size_human_readable(
        directory, summary=True, human_readable=False
    )
    print(f"Size of in machine readable. {directory}: {size_machine}")
    # %%
    print(type(size_machine))
    print(int(size_machine))

# %%
