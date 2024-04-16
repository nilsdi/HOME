# %% Imports
from PIL import Image, ImageEnhance, ImageStat
from mean_std_calculation import calculate_mean_std
from pathlib import Path
import os
from tqdm import tqdm

# %%


def calculate_contrast_brightness(image):
    # Convert the image to grayscale
    grayscale = image.convert("L")

    # Calculate the standard deviation of the pixel intensities
    stat = ImageStat.Stat(grayscale)
    return stat.stddev[0], stat.mean[0]


def transform_image(image, contrast_factor, brightness_factor):
    # Convert the image to grayscale
    image = image.convert("L")

    # Adjust the contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Adjust the brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Convert back to RGB
    image = image.convert("RGB")

    # Save the transformed image
    return (image)


# %%
root_dir = Path(__file__).parents[2]
path_old_data = root_dir / 'data/model/topredict/train/image'
path_txt_file = root_dir / 'data/model/topredict/dataset/old.txt'
input_dir = root_dir / 'data/model/original/train/image'
output_dir = root_dir / 'data/model/original/train_poor/image'

# create txt file with all files in folder with 1937 in name
with open(path_txt_file, 'w') as f:
    for file in os.listdir(path_old_data):
        if '1937' in file:
            f.write(file.replace('.tif', '') + '\n')

# %%
mean_old, std_old = calculate_mean_std(path_old_data, path_txt_file)
mean_old, std_old = mean_old[0]*255, std_old[0]*255

# %% Transform all images in the input directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(('.tif', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = Image.open(input_path)
        contrast, brightness = calculate_contrast_brightness(image)
        contrast_factor = std_old / contrast
        brightness_factor = mean_old / brightness
        image_poor = transform_image(image, contrast_factor, brightness_factor)
        image_poor.save(output_path)
# %%
