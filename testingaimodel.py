# %%
import os
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import torch
import numpy as np

# my laptop
modelPath = r"C:\Kaggle_Project\AI_Code_Training\10000.torch"

# Sherlock
#modelPath = r"r/scratch/groups/ogevaert/dhruv/1225.torch"
 

try:
    checkpoint = torch.load(modelPath)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", str(e))


# My Laptop
test_image_folder = r"C:\Kaggle_Project\AI_training\splits\test_1"  # Path to test image folder

# Sherlock
#test_image_folder = r"/scratch/groups/ogevaert/dhruv/train_1"  # Path to test image folder

# my laptop
test_mask_folder = r"C:\Kaggle_Project\AI_training\Mask_Images\test_masks"  # Path to test mask folder

# Sherlock
#test_mask_folder = r"/scratch/groups/ogevaert/dhruv/test_masks"   # Path to test mask folder

height = width = 900 # a lower value still didn't work even when I tried 224

transformImg = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((height, width)),
    tf.ToTensor(),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath))  # Load trained model
Net.eval()  # Set to evaluation mode

# Get list of test image and mask paths
test_image_paths = [os.path.join(test_image_folder, file) for file in os.listdir(test_image_folder)]
test_mask_paths = [os.path.join(test_mask_folder, file.replace(".tif", ".png")) for file in os.listdir(test_image_folder)]
# %%
for image_path, mask_path in zip(test_image_paths, test_mask_paths):
    
    Img = cv2.imread(image_path)  # Load test image
    Mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load test mask (as grayscale)
    Mask[Mask == 255] = 1
    height_orgin, width_orgin, d = Img.shape  # Get image original size
    #plt.imshow(Img[:, :, ::-1])  # Show image
    #plt.show()

    Img = transformImg(Img)  # Transform image to pytorch tensor
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    # %%
    with torch.no_grad():
        Prd = Net(Img)['out']  # Run net

    Prd = tf.Resize((height_orgin, width_orgin))(Prd[0])  # Resize to original size

    # Converting probability to class map
    seg = torch.argmax(Prd, 0).cpu().detach().numpy()

    # Code to calculate the Dice Score
    intersection = np.sum(seg * Mask)
    union = np.sum(seg) + np.sum(Mask)
    dice_score = (2.0 * intersection) / union

    #plt.imshow(seg)  # Display segmentation result
    print(f"Dice Score: {dice_score:.4f}")
    #plt.title(f"Dice Score: {dice_score:.4f}")
    #plt.show()

# %%
