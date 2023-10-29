# %%

import os
import cv2
import numpy as np

# Function to calculate Dice Similarity Coefficient
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = mask1.sum()
    mask2_sum = mask2.sum()
    
    if mask1_sum == 0 and mask2_sum == 0:
        return 1.0  # Both masks are empty, consider them identical
    
    return 2.0 * intersection.sum() / (mask1_sum + mask2_sum)

# Folder path containing masks
mask_folder = "C:/calhacks_hackathon/ImageSegmentation/archive/masks/masks_trains"

# Get a list of mask file paths
mask_files = os.listdir(mask_folder)

# Create an empty dictionary to store matching pairs
matching_pairs = {}

# Compare each mask with all other masks
for i, mask_file_1 in enumerate(mask_files):
    mask1 = cv2.imread(os.path.join(mask_folder, mask_file_1), cv2.IMREAD_GRAYSCALE)
    mask1 = (mask1 == 255).astype(np.uint8)  # Convert to binary mask (0 or 1)

    matching_pairs[mask_file_1] = []

    for j, mask_file_2 in enumerate(mask_files):
        if i != j:
            mask2 = cv2.imread(os.path.join(mask_folder, mask_file_2), cv2.IMREAD_GRAYSCALE)
            mask2 = (mask2 == 255).astype(np.uint8)  # Convert to binary mask (0 or 1)

            similarity = dice_coefficient(mask1, mask2)

            if similarity == 1.0:  # Both masks are identical
                matching_pairs[mask_file_1].append(mask_file_2)

# Print matching pairs
for mask_file, matches in matching_pairs.items():
    if matches:
        print(f"{mask_file} matches with: {matches}")





# %%