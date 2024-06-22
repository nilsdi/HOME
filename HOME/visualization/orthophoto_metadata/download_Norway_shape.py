# creates a folder 'Norway_boundaries' in the 'maps' folder in the 'raw' folder in 
# the 'data' folder in the root directory of the project that is used for some maps
# the data is publicly available at the stanford libraries, see
# https://earthworks.stanford.edu/catalog/stanford-jm135gj5367
# => needs to be run locally once only.
#%% imports and direct download
import requests
import zipfile
import shutil
from pathlib import Path
import tempfile

# URL of the zip file containing the data
url = "https://stacks.stanford.edu/file/druid:jm135gj5367/data.zip"

# Get the root directory of the project
root_directory = Path(__file__).resolve().parents[3]

# The target path for the Norway_boundaries folder
target_path = root_directory / 'data' / 'raw' / 'maps'

# Step 1: Download the ZIP file
response = requests.get(url)
if response.status_code == 200:
    # Use a temporary file to store the downloaded zip
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        zip_path = Path(tmp_file.name)
else:
    print("Failed to download the file.")
    exit()

#%% Extract and move the files
# Step 2: Extract the ZIP file
# Assuming zip_path and target_path are defined earlier in your code
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_ref.extractall(tmp_dir)
        
        # List all files in the temporary directory
        files = [f for f in Path(tmp_dir).iterdir() if f.is_file()]
        
        # Define the final path for 'Norway_boundaries'
        final_path = target_path / 'Norway_boundaries'
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Move all extracted files to 'Norway_boundaries' at the target location
        for file in files:
            shutil.move(str(file), str(final_path))
        
        print(f"All files have been moved to {final_path}")

# Cleanup the downloaded zip file
zip_path.unlink()
# %%
