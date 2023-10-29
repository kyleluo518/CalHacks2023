# %%
import os
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import numpy as np

# %%
# Model path
modelPath = r"/kaggle/input/torch-files/9500.torch"

# Load the model
try:
    checkpoint = torch.load(modelPath)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", str(e))

# Test image folder
test_image_folder = r"/kaggle/input/testing-dataset/test_images/test_images"

# Test mask folder
test_mask_folder = r"/kaggle/input/testing-dataset/masks_tests/masks_tests"

# Output folder for segmented images (create this folder if it doesn't exist)
output_folder = "segmented_images"
os.makedirs(output_folder, exist_ok=True)

height = width = 900

transformImg = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((height, width)),
    tf.ToTensor(),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
Net = Net.to(device)
Net.load_state_dict(torch.load(modelPath))
Net.eval()

test_image_paths = [os.path.join(test_image_folder, file) for file in os.listdir(test_image_folder)]

for image_path in test_image_paths:
    Img = cv2.imread(image_path)
    height_orgin, width_orgin, d = Img.shape

    Img = transformImg(Img)
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)

    with torch.no_grad():
        Prd = Net(Img)['out']

    Prd = tf.Resize((height_orgin, width_orgin))(Prd[0])

    seg = torch.argmax(Prd, 0).cpu().detach().numpy()

    # Save the segmented image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path).replace(".tif", ".png"))
    cv2.imwrite(output_path, seg)

print("Segmented images saved to", output_folder)


# %%
