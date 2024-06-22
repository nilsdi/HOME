# %%
import cv2
from pathlib import Path

# Load the image
root_dir = Path(__file__).parents[2]
path = "data/model/topredict/train/image/trondheim_0.3_1937_1_1_1_7.tif"
image = cv2.imread(str(root_dir / path))

# Get the number of channels
num_channels = image.shape[2]

# Display the number of channels
print("Number of channels:", num_channels)
