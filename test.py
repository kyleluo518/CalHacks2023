# %%
import os
import random
import shutil

# Source directory containing the images
source_directory = r'C:\calhacks_hackathon\ImageSegmentation\archive\images\images'

# Destination directory where you want to copy the 80% of the images
destination_directory = r'C:\calhacks_hackathon\ImageSegmentation\archive\images\80_percent_of_images'

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# List all files in the source directory
image_files = os.listdir(source_directory)

# Calculate the number of images to select (80%)
num_images_to_select = int(0.8 * len(image_files))

# Randomly shuffle the list of image files
random.shuffle(image_files)

# Select the first 'num_images_to_select' files
selected_images = image_files[:num_images_to_select]

# Copy the selected images to the destination directory
for image_file in selected_images:
    source_path = os.path.join(source_directory, image_file)
    destination_path = os.path.join(destination_directory, image_file)
    shutil.copy(source_path, destination_path)

print(f"Extracted {num_images_to_select} images to the destination directory.")


# %%