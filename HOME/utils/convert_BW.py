# %%
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm


def convert_to_bw(image_path, output_dir):
    image = Image.open(image_path)
    bw_image = image.convert("L")
    # Convert the grayscale image back to RGB (to have 3 channels)
    rgb_image = bw_image.convert("RGB")
    rgb_image.save(output_dir / os.path.basename(image_path))


def convert_folder_to_bw(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".tif"):
            image_path = input_dir / filename
            convert_to_bw(image_path, output_dir)


# %% Specify the folder path here
root_dir = Path(__file__).parents[2]
input_dir = root_dir / "data/ML_training/train/image"
output_dir = root_dir / "data/ML_training/train_BW/image"
convert_folder_to_bw(input_dir, output_dir)

# %%
