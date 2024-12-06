#!/usr/bin/env python
# coding: utf-8
# CRMS2Map
# Original author: xin luo, 2023.9.21
# Developed by the Center for Computation & Technology at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified Dec 2, 2024

# import modules
import os
import time
import numpy as np
import pandas as pd
import rasterio
import random
from tqdm import tqdm  # For progress bar
from glob import glob
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# internal modules
import config
from model import unet,ResUNet, ResUnetPlusPlus
from utils.metrics import oa_binary, miou_binary,oa_multi, miou_multi
from dataloader.pyrsimg import *
from dataloader.preprocess import read_normalize
from dataloader.loader import patch_tensor_dset, scene_dset
from dataloader.parallel_loader import threads_scene_dset

Val_patch_flags = False # True: create patches for validation data
model_name = 'resunet' # 'unet', 'resunet', etc

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if model_name == 'unet':
    model = unet(num_bands=6, num_classes=3,dropout_prob=0.0).to(device) # Use U-net model
elif model_name == 'resunet':
    model = ResUNet(num_bands=6, num_classes=3).to(device) # Use ResUNet model
elif model_name == 'resunetpp':
    model = ResUnetPlusPlus(num_bands=6, num_classes=3).to(device) # Use ResUNet++ model
else:
    print("Model not found.")

print (model)

# make some directories
Workspace=os.getcwd() # HPC
Photospace=os.path.join(Workspace, 'Photo')
Modelspace = os.path.join(Workspace, 'model')
Trainedspace = os.path.join(Modelspace, 'trained_model')

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

try:
    os.makedirs(Photospace, exist_ok=True)
    os.makedirs(Modelspace, exist_ok=True)
    os.makedirs(Trainedspace, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")



########################################################################################################################
### Functions
########################################################################################################################
'''------train step------'''
def train_step(model, loss_fn, optimizer, x, y,num_classes):
    model.train()
    optimizer.zero_grad()

    # Ensure `y` is in the correct format: [batch_size, height, width]
    if y.dim() == 4:  # If `y` has an extra channel dimension
        y = y.squeeze(1).long()  # Remove channel dimension if present

    # Forward pass
    pred = model(x)  # Ensure `pred` has shape [batch_size, num_classes, height, width]

    # # Debug shapes
    # print(f"Pred shape: {pred.shape}, Pred dtype: {pred.dtype}")
    # print(f"Y shape: {y.shape}, Y dtype: {y.dtype}")

    # Compute loss
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print (loss)
    miou = miou_multi(pred=pred, truth=y, num_classes=num_classes)
    oa = oa_multi(pred=pred, truth=y, num_classes=num_classes)
    return loss, miou, oa

'''------validation step------'''
def val_step(model, loss_fn, x, y, num_classes):
    model.eval()
    with torch.no_grad():
        # Ensure `y` is in the correct format: [batch_size, height, width]
        if y.dim() == 4:  # If `y` has an extra channel dimension
            y = y.squeeze(1).long()  # Remove channel dimension

        # Forward pass
        pred = model(x)  # Ensure `pred` has shape [batch_size, num_classes, height, width]

        # # Debug shapes
        # print(f"Pred shape: {pred.shape}, Y shape: {y.shape}, Y dtype: {y.dtype}")

        # Compute loss
        loss = loss_fn(pred, y)
        miou = miou_multi(pred=pred, truth=y, num_classes=num_classes)
        oa = oa_multi(pred=pred, truth=y, num_classes=num_classes)

    return loss, miou, oa, pred

'''------ train loops ------'''
def train_loops(model, loss_fn, optimizer, tra_loader, val_loader, epoches,num_classes, lr_scheduler=None, Photospace=Photospace):
    """
    Train the model using the provided training and validation data loaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        loss_fn (callable): The loss function to be used.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        tra_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epoches (int): Number of epochs to train the model.
        num_classes (int): Number of classes in the dataset.
        lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        Photospace (str, optional): Directory to save visualizations. Defaults to Photospace.

    Returns:
        dict: Dictionary containing training and validation loss and mIoU metrics for each epoch.
    """

    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    tra_loss_loops, tra_miou_loops = [], []
    val_loss_loops, val_miou_loops = [], []

    indices = [0,1,2,3]  # Use fixed indices for visualization (can be changed)

    for epoch in range(epoches):
        start = time.time()
        print (epoch)
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0

        '''----- 1. train the model -----'''
        for x_batch, y_batch in tra_loader:
            x_batch, y_batch = x_batch.to(device, dtype=torch.float32), y_batch.to(device)
            # y_batch = config.label_smooth(y_batch)  # Apply label smoothing (not sure do we need it for multi-class)
            y_batch = y_batch.squeeze(1).long()
            loss, miou, oa = train_step(model=model, loss_fn=loss_fn,                                             optimizer=optimizer, x=x_batch, y=y_batch,num_classes=num_classes)
            tra_loss += loss.item()
            tra_miou += miou
            tra_oa += oa.item()
        if lr_scheduler:
          lr_scheduler.step(tra_loss)         # using learning rate scheduler

        '''----- 2. validate the model -----'''
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device, dtype=torch.float32), y_batch.to(device)
            y_batch = y_batch.squeeze(1).long()
            loss, miou, oa, pred = val_step(model=model, loss_fn=loss_fn, x=x_batch, y=y_batch,num_classes=num_classes)
            val_loss += loss.item()
            val_miou += miou.item()
            val_oa += oa.item()

        if epoch % 25 == 0:  # Visualize every 25 epochs
            pred_class = torch.argmax(pred, dim=1).cpu().numpy()  # Convert predictions to class labels
            y_val_np = y_batch.cpu().numpy()  # Ground truth to numpy
            print("Shape of y_val_sample:", y_val_np.shape)

            plt.figure(figsize=(24, 12))
            for i, idx in enumerate(indices):
                y_val_sample = y_val_np[idx].squeeze()  # Shape: (256, 256)
                pred_class_sample = pred_class[idx]  # Shape: (256, 256)

                # Plot ground truth
                plt.subplot(2, 4, i + 1)
                plt.title(f"Ground Truth {i + 1}")
                plt.imshow(y_val_sample, cmap='viridis')  # Use colormap for classification

                # Plot prediction
                plt.subplot(2, 4, i + 5)
                plt.title(f"Prediction {i + 1}")
                plt.imshow(pred_class_sample, cmap='viridis')  # Use the same colormap

            plt.savefig(os.path.join(Photospace,f"epoch_{epoch}_samples.png"))
            plt.close()

        '''------ 3. print mean accuracy ------'''
        tra_loss = tra_loss/size_tra_loader
        val_loss = val_loss/size_val_loader
        tra_miou = tra_miou/size_tra_loader
        val_miou = val_miou/size_val_loader
        tra_oa = tra_oa/size_tra_loader
        val_oa = val_oa/size_val_loader
        tra_loss_loops.append(tra_loss), tra_miou_loops.append(tra_miou)
        val_loss_loops.append(val_loss), val_miou_loops.append(val_miou)
        format = 'Ep{}: Tra-> Loss:{:.3f}, Oa:{:.3f}, Miou:{:.3f}, Val-> Loss:{:.3f}, Oa:{:.3f}, Miou:{:.3f}, Time:{:.1f}s'
        print(format.format(epoch+1, tra_loss, tra_oa, tra_miou, val_loss, val_oa, val_miou, time.time()-start))
    metrics = {'tra_loss':tra_loss_loops, 'tra_miou':tra_miou_loops, 'val_loss': val_loss_loops, 'val_miou': val_miou_loops}
    return metrics


def compute_mean_std(paths_img):
    """
    Calculate per-band mean and standard deviation for the entire training dataset.

    :param paths_img: list of paths to training images.
    :return: per-band mean and standard deviation.
    """
    total_sum = None
    total_squared_sum = None
    total_pixels = 0

    for path in tqdm(paths_img, desc="Calculating mean and std"):
        # Read image
        scene_ins = readTiff(path)
        scene_arr = scene_ins['array']  # Shape: (bands, height, width)

        # Replace nan and inf values
        scene_arr = np.nan_to_num(scene_arr, nan=0.0, posinf=1e5, neginf=-1e5)
        scene_arr = np.clip(scene_arr, 0, 1e5)  # Clip values to a valid range

        # Debug: Print max and min values
        print(f"Max value: {np.max(scene_arr)}, Min value: {np.min(scene_arr)}")

        # Sum across all pixels in each band
        if total_sum is None:
            total_sum = np.sum(scene_arr, axis=(1, 2))  # Sum for each band
            total_squared_sum = np.sum(scene_arr ** 2, axis=(1, 2))  # Sum of squares for each band
        else:
            total_sum += np.sum(scene_arr, axis=(1, 2))
            total_squared_sum += np.sum(scene_arr ** 2, axis=(1, 2))

        # Add the number of valid pixels
        total_pixels += np.prod(scene_arr.shape[1:])  # height * width

    # Safeguard against zero total pixels
    if total_pixels == 0:
        raise ValueError("No valid pixels found in the dataset. Check the input data for issues.")

    # Calculate mean and std for each band
    mean = total_sum / total_pixels
    std = np.sqrt(total_squared_sum / total_pixels - mean ** 2)

    # Replace zero std with a small value
    std = np.where(std == 0, 1e-6, std)

    return mean, std

def read_normalize_with_torchvision(paths_img, paths_truth, bands_mean, bands_std):
    '''
    Satellite image reading, normalization, and transformation using PyTorch torchvision.transforms.Normalize.
    Args:
        paths_img: list of paths to satellite images.
        paths_truth: list of paths to truth images.
        bands_mean: per-band mean values for normalization.
        bands_std: per-band standard deviation values for normalization.
    Returns:
        tra_scenes: list of normalized and transformed scene tensors.
        tra_truths: list of truth tensors.
    '''
    tra_scenes, tra_truths = [], []

    # Define normalization transform
    normalize = transforms.Normalize(mean=bands_mean.tolist(), std=bands_std.tolist())

    for i in range(len(paths_img)):
        # Load and preprocess scene
        scene_ins = readTiff(paths_img[i])  # Read scene TIFF
        scene_arr = scene_ins['array']  # Shape: [bands, height, width]
        scene_tensor = torch.tensor(scene_arr, dtype=torch.float32)  # Convert to tensor
        scene_tensor = normalize(scene_tensor)  # Apply normalization
        tra_scenes.append(scene_tensor)

        # Load and preprocess ground truth
        truth_ins = readTiff(paths_truth[i])  # Read truth TIFF
        truth_tensor = torch.tensor(truth_ins['array'], dtype=torch.long)  # Convert to tensor
        tra_truths.append(truth_tensor)

    return tra_scenes, tra_truths


# Function to apply the color map to the truth tensor
def apply_color_map(truth, color_map):
    """
    Applies the color map to a truth tensor, converting it into an RGB image.

    Args:
        truth (numpy.ndarray): The truth tensor of shape (H, W), with class values (e.g., 0, 1, 2).
        color_map (dict): A dictionary mapping class values to RGB colors.

    Returns:
        numpy.ndarray: An RGB image of shape (H, W, 3).
    """
    h, w = truth.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_value, color in color_map.items():
        mask = (truth == class_value)  # Boolean mask for the class
        color_image[mask] = color  # Assign the color to the mask
    return color_image


# Function to visualize a subset of the 6-band patch
def get_rgb_from_patch(patch, bands=(0, 1, 2)):
    """
    Selects specific bands from a 6-band patch for visualization as an RGB image.

    Args:
        patch (numpy.ndarray): The input patch of shape (H, W, 6).
        bands (tuple): Indices of the bands to use for RGB visualization.

    Returns:
        numpy.ndarray: An RGB image of shape (H, W, 3).
    """
    rgb_patch = patch[:, :, bands]  # Select specific bands
    rgb_patch = (rgb_patch - np.min(rgb_patch)) / (np.max(rgb_patch) - np.min(rgb_patch))  # Normalize to [0, 1]
    rgb_patch = (rgb_patch * 255).astype(np.uint8)  # Scale to [0, 255] for visualization
    return rgb_patch


# Display images with their titles
def imsShow(img_list, img_name_list):
    """
    Displays a list of images with their respective titles.

    Args:
        img_list (list): List of images to display.
        img_name_list (list): List of titles for the images.
    """
    for i, (img, title) in enumerate(zip(img_list, img_name_list)):
        plt.subplot(1, len(img_list), i + 1)
        if len(img.shape) == 3:  # RGB image
            plt.imshow(img)
        else:  # Grayscale image
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

########################################################################################################################
# Deep Learning Model Dataset Preparation
########################################################################################################################



'''--------- 1. Data loading --------'''
'''----- 1.1 training data loading (from scenes path) '''
## Train Data paths
### The whole dataset.
print (config.dir_s2)
paths_scene = sorted(glob(config.dir_s2 + '/tra_scene/*'))
#print (paths_scene)
paths_truth = [path_scene.replace('_scene/', '_labels/').replace('_6Bands', '').split('.')[0] + '_classified.tif' for path_scene in paths_scene]
#print (paths_truth)
### Select training part from the dataset.
id_scene = [i for i in range(len(paths_scene))]
print (id_scene)

# Randomly select training sites (not used same bag now)
# id_tra_scene = random.sample(id_scene, 31) # number of training sites
# i_valset = [i for i in range(31)] # number of validation sites
# id_tra_scene = list(set(id_scene) - set(i_valset))

id_tra_scene = list(set(id_scene))
paths_tra_scene, paths_tra_truth = [paths_scene[i] for i in id_tra_scene], [paths_truth[i] for i in id_tra_scene]
print(len(paths_tra_scene))
# print(paths_tra_scene)

'''----- 1.2 validation data loading (from scenes path) '''
## --------- Validation data paths -------- #
paths_scene = sorted(glob(config.dir_s2 + '/val_scene/*'))
print(len (paths_scene))  # number of validation scenes
i_valset = [i for i in range(len(paths_scene))]  # number of validation sites
paths_truth = [path_scene.replace('_scene/', '_labels/').replace('_6Bands', '').split('.')[0] + '_classified.tif'  for path_scene in paths_scene]
paths_val_scene, paths_val_truth = [paths_scene[i] for i in i_valset], [paths_truth[i] for i in i_valset]
print(paths_val_scene[0])
print(paths_val_truth[0])
print('Number of val scenes:', len(paths_val_scene))
print('Number of total scenes:', len(paths_scene))


'''----- 1.3 calculate the training mean and std for normalization'''
bands_mean, bands_std = compute_mean_std(paths_tra_scene)
print("Mean:", bands_mean)
print("Std:", bands_std)

# Create a DataFrame
data = {
    'Band': [f'Band_{i+1}' for i in range(len(bands_mean))],
    'Mean': bands_mean,
    'Std': bands_std
}
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'Trained_bands_mean_std.csv'
df.to_csv(csv_path, index=False)
print(f"Mean and Std values are saved to {csv_path}")

# Normalize and prepare train dataset
tra_scenes, tra_truths = read_normalize_with_torchvision(
    paths_img=paths_tra_scene,
    paths_truth=paths_tra_truth,
    bands_mean=bands_mean,
    bands_std=bands_std
)

# Normalize and prepare validation dataset
val_scenes, val_truths = read_normalize_with_torchvision(
    paths_img=paths_val_scene,
    paths_truth=paths_val_truth,
    bands_mean=bands_mean,
    bands_std=bands_std
)

'''----- 1.4 Create patches for validation data'''
# Preparation for patches
ziped_val_data = list(zip(val_scenes, val_truths))
len(ziped_val_data)

if Val_patch_flags:
    # Create patches for validation data
    num_patch = 0
    for i in range(50):
        print(i)
        for scene_arr, truth_arr in ziped_val_data:
            # Print initial shapes for debugging
            # print(f"scene_arr shape: {scene_arr.shape}")
            # print(f"truth_arr shape: {truth_arr.shape}")

            # Squeeze unnecessary dimensions from truth_arr
            truth_arr = np.squeeze(truth_arr, axis=0)  # Remove first dimension
            # print(f"truth_arr shape after squeeze: {truth_arr.shape}")

            # Add channel dimension to truth_arr
            truth_arr = truth_arr[:, :, np.newaxis]  # Add new axis
            print(f"truth_arr shape after adding channel axis: {truth_arr.shape}")

            # Ensure scene has the shape (H, W, bands)
            scene_arr = np.transpose(scene_arr, (1, 2, 0))  # Shape: (H, W, 6)

            # Verify dimensions match for concatenation
            if scene_arr.shape[:2] != truth_arr.shape[:2]:
                raise ValueError(f"Shape mismatch: scene_arr shape={scene_arr.shape}, truth_arr shape={truth_arr.shape}")

            # Concatenate scene and truth arrays
            img_truth = np.concatenate((scene_arr, truth_arr), axis=2)
            print(f"Concatenated img_truth shape: {img_truth.shape}")

            patch_img_truth = crop2size(img=img_truth, channel_first=False).toSize(size=(256,256))
            patch_img_truth = patch_img_truth.transpose(2,0,1)
            patch = torch.from_numpy(patch_img_truth[0:-1,:,:]).to(dtype=torch.float16)
            ptruth = torch.from_numpy(patch_img_truth[-1:,:,:]).long()
            path_save = os.path.join(config.dir_patch_val_s2, 'patch_'+ str(num_patch).rjust(3,'0')+'.pt')
            num_patch+=1
            print(path_save)
            torch.save((patch, ptruth), path_save)

else:
    print("Validation patches are not created.")

# Load patch for validation
paths_patch_val = sorted(glob(config.dir_patch_val_s2 +'/patch_*'))
patch_list_val = [torch.load(path,weights_only=True) for path in paths_patch_val]
print(len(patch_list_val))

# Read a .pt file
pt_file_path = paths_patch_val[1] # instead of 0
data = torch.load(pt_file_path,weights_only=True)
label_tensor = data[1]
print(label_tensor) # label
# Get unique classes
unique_classes = torch.unique(label_tensor)

# Get the number of unique classes
num_classes = unique_classes.numel()

print(f"Unique classes: {unique_classes}")
print(f"Number of unique classes: {num_classes}")

# Define a color map for the classes
color_map = {
    0: [255, 0, 0],   # Red for class 0
    1: [0, 0, 255],    # Blue for class 2
    2: [0, 255, 0],   # Green for class 1
}

    ####################################################################################################################
    # Visualize a subset of the patches
    ####################################################################################################################

for idx in range(100, 105):  # Adjust range as needed
    # Access patch and truth
    patch, truth = patch_list_val[idx]  # (patches, truth)
    
    # Convert patch and truth tensors to numpy arrays
    patch = patch.numpy().transpose(1, 2, 0).astype(np.float32)  # Shape (H, W, 6)
    truth = truth.numpy().squeeze()  # Shape (H, W)
    
    # Create an RGB visualization of the 6-band patch
    rgb_patch = get_rgb_from_patch(patch, bands=(0, 1, 2))  # Select bands (0, 1, 2) for RGB
    
    # Apply color map to the truth
    truth_color = apply_color_map(truth, color_map)

    # Combine patch and truth for visualization
    patch_truth = [rgb_patch, truth_color]
    patch_truth_name = ['Patch (RGB Composite)', 'Truth (Color)']

    # Display the images
    plt.figure(figsize=(10, 4))
    imsShow(img_list=patch_truth, img_name_list=patch_truth_name)

########################################################################################################################
# Deep Learning Model training
########################################################################################################################
# Train data
num_epoch = 300
model_name_save = f'{model_name}_trained_1'

########################################################################################################################
### Step 2 #############################################################################################################
print ('Step 2: Read a DL architecture and define the training strategy')
########################################################################################################################
time_start = time.time()
''' ----- 2.1. Auto augmentation for Training data'''
# tra_dset = threads_scene_dset(scene_list = tra_scenes, \
#                               truth_list = tra_truths, 
#                               transforms=config.transforms_tra, 
#                               num_thread=1)          ###  num_thread(30) patches per scene.

tra_dset = scene_dset(scene_list = tra_scenes, 
                             truth_list = tra_truths,
                             transforms = config.transforms_tra, 
                             patch_size = [256, 256])
print('size of training data:  ', len(tra_dset))
print('time comsuming:  ', time.time()-time_start)


''' ----- 2.2. validation data loading (validation patches) ------ '''
patch_list_val = [torch.load(path,weights_only=True) for path in paths_patch_val]
val_dset = patch_tensor_dset(patch_pair_list = patch_list_val)
print('size of validation data:', val_dset.__len__())

''' ----- 2.3. Automatic batching ------ '''
tra_loader = torch.utils.data.DataLoader(tra_dset, batch_size=config.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=16)

for batch_idx, (scenes, truths) in enumerate(val_loader):
    print(f"Batch {batch_idx}")
    print(f"Scenes shape: {scenes.shape}")
    print(f"Truths shape: {truths.shape}")
    break

''' -------- 3. Model loading and training strategy ------- '''
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=20)


''' -------- 4. Model training for loops ------- '''
metrics = train_loops(model=model,
                    loss_fn=config.loss_ce,
                    optimizer=optimizer,
                    tra_loader=tra_loader,
                    val_loader=val_loader,
                    epoches=num_epoch,
                    num_classes=num_classes,
                    lr_scheduler=lr_scheduler,
                    Photospace=Photospace)

''' -------- 5. trained model and accuracy metric saving  ------- '''

# model saving
path_weights = os.path.join (Trainedspace, model_name_save + '_weights.pth')
torch.save(model.state_dict(), path_weights)
print('Model weights are saved to --> ', path_weights)
## metrics saving
path_metrics = os.path.join (Trainedspace, model_name_save + '_metrics.csv')

metrics = {key: [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in values] # avoid gpu working
           for key, values in metrics.items()} # Convert tensors to numpy arrays

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(path_metrics, index=False, sep=',')
metrics_df = pd.read_csv(path_metrics)
print('Training metrics are saved to --> ', path_metrics)

print("Training complete.")





