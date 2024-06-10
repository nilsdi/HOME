# Script to augment images using the albumentations library

import os
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm

# %%
# Define the input and output directories
root_dir = Path(__file__).parents[2]
input_dir = root_dir / 'data/model/original/train_poor/image'
output_dir = root_dir / 'data/model/original/train_augmented/image'

# Define the augmentation pipeline
