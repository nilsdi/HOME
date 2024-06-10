# %%

import os
from osgeo import gdal
import numpy as np
from pathlib import Path

# Function to parse filename and extract its coordinates
def parse_filename(filename):
    parts = filename.split('_')
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])
    return row, col

root_dir = Path(__file__).parents[2]
current_dir = Path(__file__).parents[0]
test_file = open(root_dir / 'data/model/trondheim_1979/test.txt', 'w')
data_path = root_dir / 'data/model/trondheim_1979/tiles/images'
# Directory containing the TIFF files
input_dir = root_dir / 'data/model/trondheim_1979/tiles/images'
output_file = root_dir / 'data/model/trondheim_1979/tiles/reassembled_tile/test.tif'
file_list_path = root_dir / 'data/model/trondheim_1979/dataset/test.txt'

# Read filenames from the text file
with open(file_list_path, 'r') as file:
    filenames = [line.strip() for line in file]

# Determine grid size
cols, rows = 0, 0
for filename in filenames:
    col, row = parse_filename(filename)
    if row + 1 > rows:
        rows = row + 1
    if col + 1 > cols:
        cols = col + 1

def extract_tile_numbers(filename):
    parts = filename.split('_')
    row = int(parts[-2])
    col = int(parts[-1].split('.')[0])
    return row, col

def order_files_by_xy(files):
    file_info = [extract_tile_numbers(filename) for filename in files]
    ordered_files = sorted(files, key=lambda x: (file_info[files.index(x)][0], file_info[files.index(x)][1]))
    return ordered_files

def get_columns_from_ordered_files(ordered_files):
    columns = {}
    for filename in ordered_files:
        x, y = extract_tile_numbers(filename)
        if x not in columns:
            columns[x] = []
        columns[x].append(filename)
    sorted_columns = [columns[key] for key in sorted(columns.keys())]
    return sorted_columns

def combine_column_tiles(file_list, column, output_file, num_rows=32, overlap_rate=0.01):
    sample_path = os.path.join(input_dir, file_list[0])
    sample_dataset = gdal.Open(sample_path)

    if not sample_dataset:
        raise FileNotFoundError(f"Unable to open sample TIFF file: {sample_path}")

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    geo_transform = sample_dataset.GetGeoTransform()
    projection = sample_dataset.GetProjection()

    effective_tile_height = int(tile_height * (1 - overlap_rate))

    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        tile_width,
        effective_tile_height * num_rows + (tile_height - effective_tile_height),
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    combined_dataset.SetGeoTransform(geo_transform)
    combined_dataset.SetProjection(projection)

    white_tile = np.full((tile_height, tile_width, sample_dataset.RasterCount), 255, dtype=np.uint8)
    tile_dict = {extract_tile_numbers(filename): filename for filename in file_list}

    for y in range(num_rows):
        tile_pos = (column, y)
        y_offset = y * effective_tile_height

        if tile_pos in tile_dict:
            tile_path = tile_dict[tile_pos]
            tile_path = os.path.join(input_dir, tile_path)
            tile_dataset = gdal.Open(tile_path)
            if tile_dataset:
                for band in range(1, tile_dataset.RasterCount + 1):
                    data = tile_dataset.GetRasterBand(band).ReadAsArray()
                    combined_dataset.GetRasterBand(band).WriteArray(data, 0, y_offset)
        else:
            for band in range(1, sample_dataset.RasterCount + 1):
                combined_dataset.GetRasterBand(band).WriteArray(white_tile[:, :, band - 1], 0, y_offset)

    sample_dataset = None
    combined_dataset = None

    print(f"Combined column image saved as {output_file}")

def stitch_columns_together(column_files, output_file, overlap_rate=0.01):
    sample_dataset = gdal.Open(column_files[0])

    tile_width = sample_dataset.RasterXSize
    tile_height = sample_dataset.RasterYSize
    effective_tile_width = int(tile_width * (1 - overlap_rate))

    final_width = effective_tile_width * len(column_files) + (tile_width - effective_tile_width)
    final_height = tile_height

    driver = gdal.GetDriverByName('GTiff')
    combined_dataset = driver.Create(
        str(output_file),
        final_width,
        final_height,
        sample_dataset.RasterCount,
        sample_dataset.GetRasterBand(1).DataType
    )

    combined_dataset.SetGeoTransform(sample_dataset.GetGeoTransform())
    combined_dataset.SetProjection(sample_dataset.GetProjection())

    for i, column_file in enumerate(column_files):
        column_dataset = gdal.Open(column_file)
        if column_dataset:
            for band in range(1, column_dataset.RasterCount + 1):
                data = column_dataset.GetRasterBand(band).ReadAsArray()
                x_offset = i * effective_tile_width
                y_offset = 0
                combined_dataset.GetRasterBand(band).WriteArray(data, x_offset, y_offset)

    sample_dataset = None
    combined_dataset = None

    print(f"Final stitched image saved as {output_file}")

# Example usage on the prediction
with open(file_list_path, 'r') as file:
    file_list = [line.strip() for line in file]

file_list = [filename if filename.endswith('.tif') else f"{filename}.tif" for filename in file_list]
ordered_files = order_files_by_xy(file_list)
sorted_columns = get_columns_from_ordered_files(ordered_files)

output_base_path = root_dir / 'data/model/trondheim_1979/tiles/reassembled_tile'

for i, column in enumerate(sorted_columns):
    output_file = output_base_path / f'reassembled_tile_{i}.tif'
    if not output_file.exists():
        combine_column_tiles(column, i, output_file)
        print(f"Created {output_file}")

column_files = [str(output_base_path / f'reassembled_tile_{i}.tif') for i in range(len(sorted_columns))]
output_file = output_base_path / 'final_stitched_image.tif'
stitch_columns_together(column_files, output_file)



"""# %%
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
# Increase the maximum number of pixels OpenCV can handle
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2  # noqa

root_dir = str(Path(__file__).parents[2])

# %%


def partition_and_crop_images(input_dir_images, input_dir_labels,
                              output_dir_images, output_dir_labels,
                              tile_size=512, overlap_rate=0.01,
                              image_size=None,
                              imbalance_threshold=(0.005, 0.9)):
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # Create archive directories if they don't exist
    archive_dir_images = os.path.join(input_dir_images, 'archive')
    archive_dir_labels = os.path.join(input_dir_labels, 'archive')
    os.makedirs(archive_dir_images, exist_ok=True)
    os.makedirs(archive_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(
        input_dir_images) if f.endswith('.tif')]

    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the image size if not given
    total_iterations = 0
    if image_size is None:
        for file in image_files:
            label_path = os.path.join(input_dir_labels, file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1
            total_iterations += num_tiles_x * num_tiles_y
    else:
        height, width = image_size, image_size

    skipped = 0

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_file in image_files:
            # Load the label
            label_path = os.path.join(input_dir_labels, image_file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape
            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1

            # Calculate the ratio of positive to negative pixels
            num_positive_pixels = np.sum(label > 0)
            num_negative_pixels = np.sum(label == 0)
            ratio = num_positive_pixels / num_negative_pixels

            # Skip this image if the ratio is above or below the threshold
            if (ratio < imbalance_threshold[0] or
                    ratio > imbalance_threshold[1]):
                skipped += num_tiles_x * num_tiles_y
                pbar.update(num_tiles_x * num_tiles_y)
                continue

            # Load the image
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)

            # Calculate the required padding
            padding_x = ((num_tiles_x-1) * effective_tile_size +
                         tile_size - width)
            padding_y = ((num_tiles_y-1) * effective_tile_size +
                         tile_size - height)

            # Pad the image and label
            image = np.pad(image, ((0, int(padding_y)),
                                   (0, int(padding_x)), (0, 0)))
            label = np.pad(label, ((0, int(padding_y)),
                                   (0, int(padding_x))))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the label
                    label_tile = label[y:y+tile_size, x:x+tile_size]

                    # Calculate the ratio of positive to negative pixels
                    num_positive_pixels = np.sum(label_tile > 0)
                    num_negative_pixels = np.sum(label_tile == 0)
                    ratio = num_positive_pixels / num_negative_pixels

                    # Skip this image if the ratio is above or below the
                    # threshold
                    if ratio < imbalance_threshold[0] or (
                            ratio > imbalance_threshold[1]):
                        skipped += 1
                        pbar.update(1)
                        continue

                    # Crop the tile from the image
                    image_tile = image[y:y+tile_size, x:x+tile_size]

                    # Save the image tile to the output directory
                    image_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    image_tile_path = os.path.join(
                        output_dir_images, image_tile_filename)
                    cv2.imwrite(image_tile_path, image_tile)

                    # Save the label tile to the output directory
                    label_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    label_tile_path = os.path.join(
                        output_dir_labels, label_tile_filename)
                    cv2.imwrite(label_tile_path, label_tile)
                    pbar.update(1)

            # Move the processed image and label to the archive directory
            shutil.move(os.path.join(input_dir_images, image_file),
                        os.path.join(archive_dir_images, image_file))
            shutil.move(os.path.join(input_dir_labels, image_file),
                        os.path.join(archive_dir_labels, image_file))

    print(f"Skipped {skipped} tiles due to class imbalance")


input_dir_images = root_dir + '/data/temp/pretrain/images/'
input_dir_labels = root_dir + '/data/temp/pretrain/labels'
output_dir_images = root_dir + '/data/model/original/train/image/'
output_dir_labels = root_dir + '/data/model/original/train/label/'

print("Partitioning and cropping images with labels")
partition_and_crop_images(input_dir_images, input_dir_labels,
                          output_dir_images, output_dir_labels,
                          tile_size=512, overlap_rate=0.01)


# %%
def tile_images_no_labels(input_dir_images,
                          output_dir_images,
                          tile_size=512, overlap_rate=0.01,
                          image_size=None):
    # Create output directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)

    # Create archive directories if they don't exist
    archive_dir_images = os.path.join(input_dir_images, 'archive')
    os.makedirs(archive_dir_images, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(
        input_dir_images) if f.endswith('.tif')]

    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the image size if not given
    total_iterations = 0
    if image_size is None:
        for file in image_files:
            image_path = os.path.join(input_dir_images, file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1
            total_iterations += num_tiles_x * num_tiles_y
    else:
        height, width = image_size, image_size

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for image_file in image_files:
            # Load the image
            image_path = os.path.join(input_dir_images, image_file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                              (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                              (effective_tile_size))) + 1

            # Calculate the required padding
            padding_x = ((num_tiles_x-1) * effective_tile_size +
                         tile_size - width)
            padding_y = ((num_tiles_y-1) * effective_tile_size +
                         tile_size - height)

            # Pad the image and label
            image = np.pad(image, ((0, int(padding_y)),
                                   (0, int(padding_x)), (0, 0)))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the image
                    image_tile = image[y:y+tile_size, x:x+tile_size]

                    # Save the image tile to the output directory
                    image_tile_filename = f"{image_file[:-4]}_{i}_{j}.tif"
                    image_tile_path = os.path.join(
                        output_dir_images, image_tile_filename)
                    cv2.imwrite(image_tile_path, image_tile)

                    pbar.update(1)

            # Move the processed image to the archive directory
            shutil.move(os.path.join(input_dir_images, image_file),
                        os.path.join(archive_dir_images, image_file))


input_dir_images = root_dir + '/data/temp/prepred/images/'
output_dir_images = root_dir + '/data/model/topredict/train/image/'

print("Partitioning and cropping images without labels")
tile_images_no_labels(input_dir_images,
                      output_dir_images,
                      tile_size=512,
                      overlap_rate=0.01)

# %% Similar functions but for labels without images


def tile_labels_no_images(input_dir_labels,
                          output_dir_labels,
                          tile_size=512, overlap_rate=0.01,
                          image_size=None):
    # Create output directories if they don't exist
    os.makedirs(output_dir_labels, exist_ok=True)

    # Get list of all image files in the input directory
    label_files = [f for f in os.listdir(
        input_dir_labels) if f.endswith('.tif')]

    effective_tile_size = tile_size * (1 - overlap_rate)

    # Calculate the image size if not given
    total_iterations = 0
    if image_size is None:
        for file in label_files:
            label_path = os.path.join(input_dir_labels, file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                                      (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                                      (effective_tile_size))) + 1
            total_iterations += num_tiles_x * num_tiles_y
    else:
        height, width = image_size, image_size

    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for label_file in label_files:
            # Load the label
            label_path = os.path.join(input_dir_labels, label_file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            height, width = label.shape

            num_tiles_x = int(np.ceil((width - tile_size) /
                                      (effective_tile_size))) + 1
            num_tiles_y = int(np.ceil((height - tile_size) /
                                      (effective_tile_size))) + 1

            # Calculate the required padding
            padding_x = ((num_tiles_x-1) * effective_tile_size +
                         tile_size - width)
            padding_y = ((num_tiles_y-1) * effective_tile_size +
                         tile_size - height)

            # Pad the image and label
            label = np.pad(label, ((0, int(padding_y)),
                                   (0, int(padding_x))))

            # Iterate over each tile
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    # Calculate the tile coordinates
                    x = int(i * effective_tile_size)
                    y = int(j * effective_tile_size)

                    # Crop the tile from the label
                    label_tile = label[y:y+tile_size, x:x+tile_size]

                    # Save the label tile to the output directory
                    label_tile_filename = f"{label_file[:-4]}_{i}_{j}.tif"
                    label_tile_path = os.path.join(
                        output_dir_labels, label_tile_filename)
                    cv2.imwrite(label_tile_path, label_tile)

                    pbar.update(1)


input_dir_labels = root_dir + '/data/temp/prepred/labels/'
output_dir_labels = root_dir + '/data/model/topredict/train/label/'

print("Partitioning and cropping labels without images")
tile_labels_no_images(input_dir_labels,
                      output_dir_labels,
                      tile_size=512,
                      overlap_rate=0.01)

# %%
"""