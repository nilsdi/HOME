# %% Imports
from PIL import Image, ImageEnhance, ImageStat
from src.ML_training.preprocessing.step_06_mean_std_calculation import calculate_mean_std
from pathlib import Path
import os
from tqdm import tqdm

# %%
year_old = '1992'

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


# create txt file with all files in folder with 1937 in name
with open(path_txt_file, 'w') as f:
    for file in os.listdir(path_old_data):
        if year_old in file:
            f.write(file.replace('.tif', '') + '\n')

# %%
mean_old, std_old = calculate_mean_std(path_old_data, path_txt_file)
mean_old, std_old = int(mean_old[0]*255), int(std_old[0]*255)

# %%
input_dir_BW = root_dir / 'data/model/original/train_BW/image'
path_txt_train = root_dir / 'data/model/original/dataset/train.txt'
mean_train, std_train = calculate_mean_std(input_dir_BW, path_txt_train)
mean_train, std_train = int(mean_train[0]*255), int(std_train[0]*255)

contrast_factor = std_old / std_train
brightness_factor = mean_old / mean_train

# %% Make training images worse
output_dir = root_dir / 'data/model/original/train_poor/image'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(('.tif')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = Image.open(input_path).convert('RGB')
        image_poor = transform_image(image, contrast_factor, brightness_factor)
        image_poor.save(output_path)
# %% Make old images "better"

contrast_factor = std_train / std_old
brightness_factor = mean_train / mean_old

input_dir = path_old_data
output_dir = root_dir / 'data/model/topredict/train_augmented/image'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(('.tif')) and year_old in filename:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image_poor = Image.open(input_path).convert('RGB')
        image = transform_image(image_poor, contrast_factor, brightness_factor)
        image.save(output_path)

# %%
