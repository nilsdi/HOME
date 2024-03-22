# %%
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Directory containing the images
root_dir = Path(__file__).parents[2]
image_dir = root_dir / "data/model/train/image/"

# File specifying the filenames used for training
train_file = root_dir / "data/model/dataset/train.txt"


def read_train_file(train_file):
    # Function to read filenames from train.txt
    with open(train_file, 'r') as f:
        train_filenames = f.readlines()
    train_filenames = [line.strip() + '.tif' for line in train_filenames]
    return train_filenames


def calculate_mean_std(image_dir, train_filenames):
    # Function to calculate mean and standard deviation
    num_images = len(train_filenames)
    mean = np.zeros(3)  # Initialize mean
    std = np.zeros(3)   # Initialize standard deviation

    for filename in tqdm(train_filenames):
        # Read image
        img = cv2.imread(os.path.join(image_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Update mean and standard deviation
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))

    # Compute mean and standard deviation
    mean /= num_images
    std /= num_images

    return mean, std


# Read train.txt to get filenames used for training
train_filenames = read_train_file(train_file)

# %% Calculate mean and standard deviation
mean, std = calculate_mean_std(image_dir, train_filenames)

# %%
