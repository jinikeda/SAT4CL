## original author: xin luo (created: 2023.9.21, https://github.com/xinluo2018/WatNetv2)
# This file is developed for configuration parameters.
# Developed by the Center for Computation & Technology at Louisiana State University (LSU).
# Developer: Jin Ikeda
# Last modified Dec 5, 2024

########################################################################################################################
### Import modules ###
########################################################################################################################
import torch
import torch.nn as nn
import os

# internal modules
from dataloader.img_aug import rotate, flip, torch_noise, numpy2tensor
from dataloader.img_aug import colorjitter

### Step 1 #############################################################################################################
print ('Step 1: Set work directory')
########################################################################################################################
# Set work directory
Workspace = os.getcwd()

Parentspace = os.path.dirname(Workspace)  # Go up to the parent directory
Dataspace = os.path.join(Parentspace, 'datasets')  # Train and Validation datasets (Optional for testing)
dir_s2 = os.path.join(Dataspace, 's2')  # Sentinel-2 dataset
dir_patch_val_s2 = os.path.join(dir_s2,'patch')  # Patch dataset for validation

try:
    os.makedirs(Dataspace, exist_ok=True)
    os.makedirs(dir_s2, exist_ok=True)
    os.makedirs(dir_s2, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")

os.chdir(Workspace)
print ('Workspace:', Workspace)

### Step 2 #############################################################################################################
# Set the configuration parameters for the model training
########################################################################################################################
print ('Step 2: Set configuration parameters')
# transform for training datasets
transforms_tra = [
        colorjitter(prob=0.25, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small
        rotate(prob=0.25),           # numpy-based
        flip(prob=0.25),             # numpy-based
        numpy2tensor(),
        torch_noise(prob=0.25, std_min=0, std_max=0.1),
]     # tensor-based


### learning rate and batch size for training datasets
lr = 0.002  # initial training rate
batch_size = 32   # batch size for training

### loss function
loss_ce = nn.CrossEntropyLoss()   # selected for multi-class classification
loss_bce = nn.BCELoss()    # selected for binary classification








