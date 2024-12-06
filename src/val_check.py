#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import config
from model import unet,ResUNet  # Import your model architecture
from glob import glob
from torch.utils.data import DataLoader, Dataset
import rasterio
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import pandas as pd

 
# Move the model to the appropriate device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (device)


# In[2]:


# Load the saved model weights

# Define the model architecture
model_name = 'resunet'
if model_name == 'unet':
    model = unet(num_bands=6, num_classes=3,dropout_prob=0.0).to(device) # Use U-net model
elif model_name == 'resunet':
    model = ResUNet(num_bands=6, num_classes=3).to(device) # Use ResUNet model
else:
    print("Model not found.")

trained_file = f'model/trained_model/{model_name}_trained_1_weights.pth'
if device.type == 'cpu':
    # Map to CPU
    model.load_state_dict(torch.load(trained_file, map_location=torch.device('cpu'),weights_only=True), strict=False)
else:
    # Load normally for GPU
    model.load_state_dict(torch.load(trained_file))

# Move the model to the selected device
model.to(device)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully.")


# In[3]:


print (model)


# In[4]:


print (config.dir_s2)


# In[5]:


## --------- Validation data paths -------- #
paths_scene = sorted(glob(config.dir_s2 + '/val_scene/*')) 
print(len (paths_scene))  # number of validation scenes
i_valset = [i for i in range(len(paths_scene))]  # number of validation sites
paths_truth = [path_scene.replace('_scene/', '_labels/').replace('_6Bands', '').split('.')[0] + '_classified.tif'  for path_scene in paths_scene]
i_valset
paths_val_scene, paths_val_truth = [paths_scene[i] for i in i_valset], [paths_truth[i] for i in i_valset]
site = int(0)
print(paths_val_scene[site])
print(paths_val_truth[site])
print('Number of val scenes:', len(paths_val_scene))
print('Number of total scenes:', len(paths_scene))


# In[6]:


df =pd.read_csv("Trained_bands_mean_std.csv")
mean_list=df["Mean"].tolist() 
std_list=df["Std"].tolist() 


# In[7]:


# Function to crop tensor to match target dimensions
def crop_to_match(tensor, target_height, target_width):
    """Crop a tensor to match the target height and width."""
    _, _, height, width = tensor.shape
    return tensor[:, :, :target_height, :target_width]

# Function to calculate top-left crop dimensions
def top_left_crop(tensor, target_height, target_width):
    """Crop a tensor to the target height and width from the top-left corner."""
    _, _, height, width = tensor.shape
    # Ensure the tensor size is large enough for the crop
    assert height >= target_height and width >= target_width, "Target dimensions exceed tensor size!"
    return tensor[:, :, :target_height, :target_width]

Workspace=os.getcwd() # HPC
Photospace=os.path.join(Workspace, 'Photo')

# Load and preprocess the input scene and ground truth
for index, (scene, truth) in enumerate(zip(paths_val_scene, paths_val_truth)):
    print(f"Index: {index}")
    print(f"Scene: {scene}")
    print(f"Truth: {truth}")

    # Open and check alignment
    with rasterio.open(scene) as scene_src, rasterio.open(truth) as truth_src:
        assert scene_src.crs == truth_src.crs, f"CRS mismatch: {scene_src.crs} vs {truth_src.crs}"
        assert scene_src.transform == truth_src.transform, f"Transform mismatch: {scene_src.transform} vs {truth_src.transform}"

        scene = scene_src.read()  # (bands, height, width)
        truth = truth_src.read(1)  # (height, width)

    # Normalize scene tensor
    scene_tensor = torch.tensor(scene, dtype=torch.float32).unsqueeze(0).to(device)
    bands_mean = torch.tensor(mean_list).view(1, -1, 1, 1).to(device)
    bands_std = torch.tensor(std_list).view(1, -1, 1, 1).to(device)
    scene_tensor = (scene_tensor - bands_mean) / bands_std

    # Prepare truth tensor
    truth_tensor = torch.tensor(truth, dtype=torch.long).unsqueeze(0).unsqueeze(1).to(device)

    # Log original dimensions
    print(f"Original Scene Tensor Shape: {scene_tensor.shape}")
    print(f"Original Truth Tensor Shape: {truth_tensor.shape}")

    # Calculate target crop dimensions
    original_height, original_width = scene_tensor.size(2), scene_tensor.size(3)
    target_height, target_width = (original_height // 16) * 16, (original_width // 16) * 16
    print(f"Target Crop Dimensions: Height={target_height}, Width={target_width}")

    # Apply top-left cropping
    scene_tensor = top_left_crop(scene_tensor, target_height, target_width)
    truth_tensor = top_left_crop(truth_tensor, target_height, target_width)

    # Log cropped dimensions
    print(f"Cropped Scene Tensor Shape: {scene_tensor.shape}")
    print(f"Cropped Truth Tensor Shape: {truth_tensor.shape}")

    # Get the model's prediction
    with torch.no_grad():
        pred = model(scene_tensor)

        # Log prediction dimensions
        print(f"Prediction Tensor Shape: {pred.shape}")

        # Convert prediction to class labels
        pred_class = torch.argmax(pred, dim=1).cpu().numpy().squeeze()

        # Dynamically crop truth based on prediction shape
        pred_height, pred_width = pred_class.shape
        truth_tensor = crop_to_match(truth_tensor, pred_height, pred_width)

        # Convert truth to numpy
        truth_np = truth_tensor.cpu().numpy().squeeze()

        # Log final dimensions
        print(f"Final Prediction Shape: {pred_class.shape}")
        print(f"Final Truth Shape: {truth_np.shape}")

    # Ensure alignment
    assert pred_class.shape == truth_np.shape, "Mismatch between prediction and ground truth dimensions!"

    # Plot the input, ground truth, and prediction
    output_png_path = os.path.join(Photospace, f"predicted_output_site{index}.png")
    plt.figure(figsize=(12, 6))

    # Plot ground truth
    plt.subplot(1, 2, 1)
    plt.title("Cropped Ground Truth")
    plt.imshow(truth_np, cmap='viridis')
        # Add legend with colors from the 'viridis' colormap
    cmap = plt.colormaps.get_cmap('viridis')  # Get the 'viridis' colormap
    legend_patches = [
        mpatches.Patch(color=cmap(0.0), label='Others'),
        mpatches.Patch(color=cmap(0.5), label='Water'),
        mpatches.Patch(color=cmap(1.0), label='Vegetation')
    ]
    plt.legend(handles=legend_patches, loc='upper right')

    # Plot prediction
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(pred_class, cmap='viridis')
    
    # Save and show the plot
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.show()


# In[8]:


# class TiffDataset(Dataset):
#     def __init__(self, scene_paths, truth_paths):
#         self.scene_paths = scene_paths
#         self.truth_paths = truth_paths

#         for path in self.scene_paths + self.truth_paths:
#             print(f"Checking path: {path}")  # Debugging print
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"File does not exist: {path}")

#     def __len__(self):
#         return len(self.scene_paths)

#     def __getitem__(self, idx):
#         scene_path = self.scene_paths[idx]
#         truth_path = self.truth_paths[idx]

#         # Debug print
#         print(f"Loading scene: {scene_path}")
#         print(f"Loading truth: {truth_path}")

#         # Load the scene TIFF
#         with rasterio.open(scene_path) as src:
#             scene = src.read()  # Shape: (bands, height, width)

#         # Load the ground truth TIFF
#         with rasterio.open(truth_path) as src:
#             truth = src.read(1)  # Assuming single-band truth data

#         return torch.tensor(scene), torch.tensor(truth)


# In[9]:


# # Create the dataset and dataloader
# scene_paths = [paths_val_scene[0]]  # Wrap in a list
# truth_paths = [paths_val_truth[0]]  # Wrap in a list

# batch_size = 1  # Adjust as needed
# dataset = TiffDataset(scene_paths, truth_paths)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # Check if paths are correctly passed
# print(f"Scene Paths: {scene_paths}")
# print(f"Truth Paths: {truth_paths}")


# In[10]:


# # Create the dataset and dataloader
# batch_size = 1  # Adjust as needed
# dataset = TiffDataset(scene_paths, truth_paths)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# print(scene_paths)


# In[11]:


# # Process the data in batches
# for batch_idx, (scene_batch, truth_batch) in enumerate(dataloader):
#     # Convert to tensors and move to the device
#     scene_tensor = scene_batch.float().to(device)  # Shape: (batch_size, bands, height, width)
#     truth_tensor = truth_batch.long().unsqueeze(1).to(device)  # Shape: (batch_size, 1, height, width)

#     print(f"Batch {batch_idx + 1}")
#     print(f"Scene Tensor Shape: {scene_tensor.shape}")
#     print(f"Truth Tensor Shape: {truth_tensor.shape}")


# In[12]:


# scene_tensor.shape


# In[13]:


# # Get the model's prediction
# with torch.no_grad():
#     pred = model(scene_tensor)
#     pred_class = torch.argmax(pred, dim=1).cpu().numpy().squeeze()  # Convert to class labels and remove batch dimension

# # Convert ground truth to numpy array
# truth_np = truth_tensor.cpu().numpy().squeeze()

# # Plot the input, ground truth, and prediction
# plt.figure(figsize=(12, 6))

# # Plot ground truth
# plt.subplot(1, 2, 1)
# plt.title("Ground Truth")
# plt.imshow(truth_np, cmap='viridis')

# # Plot prediction
# plt.subplot(1, 2, 2)
# plt.title("Prediction")
# plt.imshow(pred_class, cmap='viridis')

# plt.show()


# In[ ]:




