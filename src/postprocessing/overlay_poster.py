#%%

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#%%

root_dir = Path(__file__).parents[2]
data_path = root_dir / 'data/model/'

tif1= data_path /'test_full_pic/predictions/test_33/test_33.tif'

#%%
#for testing
tif2 = tif1
tif3 = tif1

# Load the TIFF files
def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1), src.transform
   
img1, transform1 = read_tiff(tif1)
img2, transform2 = read_tiff(tif2)
img3, transform3 = read_tiff(tif3)

# Ensure all images have the same dimensions
height, width = img1.shape
img2 = np.resize(img2, (height, width))
img3 = np.resize(img3, (height, width))


# Create figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each image as a texture on a flat surface with increasing Z values
def plot_image(ax, img, z, alpha):
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    x, y = np.meshgrid(x, y)
    z = np.ones_like(img) * z
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(img), shade=False, alpha=alpha)

# Adjust this shift to control the space between layers
shift = 10

# Plot each image
plot_image(ax, img1, 0, 1.0)        # Bottom image, fully opaque
plot_image(ax, img2, shift, 0.2)   # Middle image, slightly transparent
#plot_image(ax, img3, shift * 2, 0.5) # Top image, more transparent

# Customize the view
ax.set_zlim(0, shift * 3)
ax.view_init(elev=30, azim=-60)

# Hide the axes
ax.set_axis_off()

# Show plot
plt.show()

#%%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

# Generate random images
height, width = 100, 100  # Adjust dimensions as needed
"""img1 = generate_random_image(height, width)
img2 = generate_random_image(height, width)
img3 = generate_random_image(height, width)
"""

img1 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1937.png')
img2 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1979.png')
img3 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/2023.png')


# Create figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each image as a texture on a flat surface with increasing Z values
def plot_image(ax, img, z, alpha):
    x = np.linspace(0, 1, img.shape[1])
    y = np.linspace(0, 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    z = np.ones_like(img[:, :, 0]) * z
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False, alpha=alpha)

# Adjust this shift to control the space between layers
shift = 10

# Plot each image
plot_image(ax, img1, 0, 1.0)        # Bottom image, fully opaque
plot_image(ax, img2, shift, 0)    # Middle image, slightly transparent
plot_image(ax, img3, shift * 2, 0.1) # Top image, more transparent

# Customize the view
ax.set_zlim(0, shift * 3)
ax.view_init(elev=30, azim=-60)

# Hide the axes
ax.set_axis_off()

# Show plot
plt.show()

#%%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

# Load images

img1 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1937.png')
img2 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1979.png')
img3 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/2023.png')

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each image as a texture on a flat surface with increasing Z values
def plot_image(ax, img, z, alpha):
    x = np.linspace(0, 1, img.shape[1])
    y = np.linspace(0, 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    z = np.ones_like(img[:, :, 0]) * z
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors = img, shade=False, alpha=alpha)

# Adjust this shift to control the space between layers
shift = 10

# Plot each image
plot_image(ax, img1, 0, 1.0)        # Bottom image, fully opaque
plot_image(ax, img2, shift, 0.5)    # Middle image, slightly transparent
plot_image(ax, img3, shift * 2, 0.3) # Top image, more transparent

# Customize the view
ax.set_zlim(0, shift * 3)
ax.view_init(elev=30, azim=-60)

# Hide the axes
ax.set_axis_off()

# Show plot
plt.show()

#%%

#%%
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

# Function to crop the image to the center 512x512 pixels
def crop_center(image, cropx, cropy):
    y, x = image.shape[0], image.shape[1]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return image[starty:starty+cropy, startx:startx+cropx]

# Function to load, crop, and rotate the image
def load_crop_rotate_image(img):
    
    img_cropped = crop_center(img, 690, 690)
    img_rotated = np.rot90(img_cropped, 2)  # Rotate 180 degrees
    return img_rotated


img1 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1937.png')
img2 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/1979.png')
img3 = mpimg.imread('/scratch/mueller_andco/demolition_footprints/demolition_footprints/data/temp/test_zoe/overlay_poster/2023.png')


# Load, crop, and rotate images
img1 = load_crop_rotate_image(img1)
img2 = load_crop_rotate_image(img2)
img3 = load_crop_rotate_image(img3)

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each image as a texture on a flat surface with increasing Z values
def plot_image(ax, img, z, alpha):
    x = np.linspace(0, 1, img.shape[1])
    y = np.linspace(0, 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    z = np.ones_like(img[:, :, 0]) * z
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False, alpha=alpha, edgecolor='none')

# Adjust this shift to control the space between layers
shift = 10

# Plot each image
plot_image(ax, img1, 0, 1.0)        # Bottom image, fully opaque
plot_image(ax, img2, shift, 0.5)    # Middle image, slightly transparent
plot_image(ax, img3, shift * 2, 0.3) # Top image, more transparent

# Customize the view
ax.set_zlim(0, shift * 3)
ax.view_init(elev=30, azim=-60)

# Hide the axes
ax.set_axis_off()

# Show plot
plt.show()

