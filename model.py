# %%
import os
import numpy as np
from PIL import Image
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import cv2

# %%

with open("cuda_available.txt", "w") as f:
    f.write("CUDA Available: " + str(torch.cuda.is_available()))

Learning_Rate = 1e-5
width = height = 900  # image width and height
batchSize = 3

# My Laptop
#TrainFolder = r"C:\Kaggle_Project\AI_training\splits\train_1"
#MaskFolder = r"C:\Kaggle_Project\AI_training\Mask_Images\train_masks"  # Path to the folder containing masks

# Sherlock
TrainFolder = r"C:\calhacks_hackathon\ImageSegmentation\archive\images\images"
MaskFolder = r"C:\calhacks_hackathon\ImageSegmentation\archive\masks\masks"  # Path to the folder containing masks

# Create a list of image filenames without the '_overlay' part
TrainImages = os.listdir(TrainFolder)
MaskImages = os.listdir(MaskFolder)

# %%
#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose([
    tf.ToPILImage(),   # Converting the input tensor to a PIL Image
    tf.Resize((height, width)),  # Resize the PIL Image to the specified height and width
    tf.ToTensor(), # Then convert the resized PIL Image back to a tensor
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize the tensor values
])

# %%
transformAnn = tf.Compose([
    tf.ToPILImage(),             # Converting the input tensor into a PIL Image
    tf.Resize((height, width), tf.InterpolationMode.NEAREST),  # Resizing the PIL Image using nearest interpolation
    tf.ToTensor()                # Converting the resized PIL Image back into a tensor
])
# %%
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(TrainImages, MaskImages):
    # Generating a random index
    idx = np.random.randint(0, len(TrainImages))
    
    # Getting the image filename corresponding to the random index
    img_name = TrainImages[idx]

    # Creating the image file paths
    img_path_tif = os.path.join(TrainFolder, img_name)
    img_path_png = os.path.join(MaskFolder, img_name.replace(".tif", ".png"))  # Correct the mask file extension

    # Reading the tif image
    img = cv2.imread(img_path_tif)

    # Checking for errors in image loading
    if img is None:
        print("Error loading image:", img_path_tif)
        return None, None

    # Initializing the annotation map
    ann_map = np.zeros(img.shape[0:2], np.float32)

    # Checking if the mask image exists
    if os.path.exists(img_path_png):
        # Reading and processing the mask image
        vessel_mask = cv2.imread(img_path_png, 0)
        ann_map[vessel_mask == 255] = 1

    # Applying image and annotation transformations
    img = transformImg(img)
    ann_map = transformAnn(ann_map)

    # Return the transformed image and annotation map
    return img, ann_map
# %%
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    # Creating empty tensors to store the images and annotations
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])
    
    # Going through the batch size to load images and annotations
    for i in range(batchSize):
        # Call the ReadRandomImage function to get a random image and annotation
        img, ann_map = ReadRandomImage(TrainImages, MaskImages)
        
        # Check if the image and annotation are not None
        if img is not None and ann_map is not None:
            # Step 5: Assign the image and annotation to the corresponding tensors in the batch
            images[i] = img
            ann[i] = ann_map
    
    # Return the batch of images and annotations
    return images, ann

# %%
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #identifying whether the computer has GPU or CPU
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate) # Create adam optimizer
# %%
#----------------Train--------------------------------------------------------------------------
for itr in range(10000): # Training loop
   images, ann = LoadBatch() # Load training batch
   images = torch.autograd.Variable(images, requires_grad=False).to(device) # Load image
   if ann is not None:
       ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
   Pred = Net(images)['out'] # make prediction
   Net.zero_grad()
   criterion = torch.nn.CrossEntropyLoss() # Set loss function
   if ann is not None:
       Loss = criterion(Pred, ann.long()) # Calculate cross entropy loss
       Loss.backward() # Backpropagate loss
       optimizer.step() # Apply gradient descent change to weight
   seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get prediction classes
   print(itr, ") Loss=", Loss.data.cpu().numpy()) if ann is not None else print(itr)
   print("Iteration:", itr)
   print("Training Image Shape:", images.shape)
   print("Annotation Shape:", ann.shape)
   print("Prediction Shape:", Pred.shape)

   if itr % 5 == 0: # Save model weight once every 1000 steps to a permanent file
        print("Saving Model " + str(itr) + ".torch")
        torch.save(Net.state_dict(), str(itr) + ".torch")

# %%

'''''''''''

'''''''''