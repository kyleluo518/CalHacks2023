# %%
import os
import cv2

# Define the folder path
folder_path = r'C:\calhacks_hackathon\ImageSegmentation\archive\masks\masks_tests'

# Initialize a counter for images with white pixels
white_pixel_count = 0

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (e.g., with a ".png" extension)
    if filename.endswith('.tif'):
        # Read the image using OpenCV
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if there are white pixels (pixel value of 255) in the image
        if 255 in image:
            white_pixel_count += 1

# Print the count of images with white pixels
print(f"Number of images with white pixels: {white_pixel_count}")



# %%