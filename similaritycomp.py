# %%
import os
import cv2
import numpy as np

# Function to calculate Dice Similarity Coefficient
# Function to calculate Dice Similarity Coefficient
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = mask1.sum()
    mask2_sum = mask2.sum()
    
    if mask1_sum == 0 and mask2_sum == 0:
        return 1.0  # Both masks are empty, consider them identical
    
    return 2.0 * intersection.sum() / (mask1_sum + mask2_sum)

# Folder containing mask images
mask_folder = r'C:\calhacks_hackathon\ImageSegmentation\archive\masks\masks_trains'

# Get a list of mask file paths
mask_files = [os.path.join(mask_folder, file) for file in os.listdir(mask_folder)]

# Initialize a similarity matrix
num_masks = len(mask_files)
similarity_matrix = np.zeros((num_masks, num_masks))

# Load and calculate Dice Similarity Coefficients
for i in range(num_masks):
    for j in range(i, num_masks):
        mask1 = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask_files[j], cv2.IMREAD_GRAYSCALE)
        mask1 = mask1 > 0  # Convert to binary mask
        mask2 = mask2 > 0  # Convert to binary mask
        dsc = dice_coefficient(mask1, mask2)
        similarity_matrix[i][j] = dsc
        similarity_matrix[j][i] = dsc  # Similarity matrix is symmetric

# Print the similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

# %%