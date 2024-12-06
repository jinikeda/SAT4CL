## author: xin luo,
## modified by Jin Ikeda
## created: 2023.9.21, modify: 2024.11.28

# Import the necessary libraries
import config  # Import the configuration file (config.py)
import numpy as np
import torch
import time
import pandas as pd
import rasterio
from glob import glob
from model import unet, deeplabv3plus, deeplabv3plus_mobilev2, hrnet
#from model.seg_model import unet, deeplabv3plus, deeplabv3plus_mobilev2, hrnet
from utils.metrics import oa_binary, miou_binary,oa_multi, miou_multi
from dataloader.preprocess import read_normalize
from dataloader.loader import patch_tensor_dset, scene_dset
from dataloader.parallel_loader import threads_scene_dset
#from utils.plot_dset_one import plot_dset_one
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# Number of epochs and training model name
num_epoch = 300
model_name_save = 'unet_trained_1'

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = unet(num_bands=6, num_classes=3,dropout_prob=0.0).to(device) # use unet model
print (model) # Print the model architecture

## Data paths
### The whole dataset.
print (config.dir_s2)
paths_scene = sorted(glob(config.dir_s2 + '/tra_scene/*'))
#print (paths_scene)
paths_truth = [path_scene.replace('_scene/', '_labels/').replace('_6Bands', '').split('.')[0] + '_classified.tif' for path_scene in paths_scene]
#print (paths_truth)
### Select training part from the dataset.
id_scene = [i for i in range(len(paths_scene))]
print (id_scene)
id_tra_scene = list(set(id_scene) - set(config.i_valset))
id_tra_scene = list(set(id_scene))
paths_tra_scene, paths_tra_truth = [paths_scene[i] for i in id_tra_scene], [paths_truth[i] for i in id_tra_scene]
len(paths_tra_scene)
print(paths_tra_scene)

### Validation part of the dataset (patch format)
paths_patch_val = sorted(glob(config.dir_patch_val_s2+'/*'))   ## Validation patches
print(paths_patch_val)

# Read the first .pt file
pt_file_path = paths_patch_val[0]
data = torch.load(pt_file_path,weights_only=False)
label_tensor = data[1]
print(label_tensor) # label
# Get unique classes
unique_classes = torch.unique(label_tensor)

# Get the number of unique classes
num_classes = unique_classes.numel()

print(f"Unique classes: {unique_classes}")
print(f"Number of unique classes: {num_classes}")

'''--------- 1. Data loading --------'''
'''----- 1.1 training data loading (from scenes path) '''
tra_scenes, tra_truths = read_normalize(paths_img=paths_tra_scene,paths_truth=paths_tra_truth, max_bands=config.bands_max, min_bands=config.bands_min)

''' ----- 1.2. Training data loading and auto augmentation'''
time_start = time.time()
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


''' ----- 1.3. validation data loading (validation patches) ------ '''
patch_list_val = [torch.load(path) for path in paths_patch_val]
val_dset = patch_tensor_dset(patch_pair_list = patch_list_val)
print('size of validation data:', val_dset.__len__())
tra_loader = torch.utils.data.DataLoader(tra_dset, batch_size=config.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=16)

for batch_idx, (scenes, truths) in enumerate(val_loader):
    print(f"Batch {batch_idx}")
    print(f"Scenes shape: {scenes.shape}")
    print(f"Truths shape: {truths.shape}")
    break


'''------train step------'''
def train_step(model, loss_fn, optimizer, x, y,num_classes):
    model.train()
    optimizer.zero_grad()

    # Ensure `y` is in the correct format: [batch_size, height, width]
    if y.dim() == 4:  # If `y` has an extra channel dimension
        y = y.squeeze(1).long()  # Remove channel dimension if present

    # Forward pass
    pred = model(x)  # Ensure `pred` has shape [batch_size, num_classes, height, width]

    # Debug shapes
    print(f"Pred shape: {pred.shape}, Pred dtype: {pred.dtype}")
    print(f"Y shape: {y.shape}, Y dtype: {y.dtype}")

    # Compute loss
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print (loss)
    # miou = miou_binary(pred=pred, truth=y)
    # oa = oa_binary(pred=pred, truth=y)
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
        # miou = miou_binary(pred=pred, truth=y)
        # oa = oa_binary(pred=pred, truth=y)
        miou = miou_multi(pred=pred, truth=y, num_classes=num_classes)
        oa = oa_multi(pred=pred, truth=y, num_classes=num_classes)

    return loss, miou, oa, pred

'''------ train loops ------'''
def train_loops(model, loss_fn, optimizer, tra_loader, val_loader, epoches,num_classes, lr_scheduler=None):
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    tra_loss_loops, tra_miou_loops = [], []
    val_loss_loops, val_miou_loops = [], []

    indices = [1, 2, 3, 0]  # Use fixed indices for visualization
    for epoch in range(epoches):
        start = time.time()
        print (epoch)
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0

        '''----- 1. train the model -----'''
        for x_batch, y_batch in tra_loader:
            x_batch, y_batch = x_batch.to(device, dtype=torch.float32), y_batch.to(device)
            # y_batch = config.label_smooth(y_batch)
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

            plt.savefig(f"epoch_{epoch}_samples.png")
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

# from torchvision import transforms
#
# # Data Augmentation for Training
# train_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(90),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
# ])
#
# # Validation Transforms
# val_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# # Training Loop
# num_epochs = 300
# lr = 0.002
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Define model, optimizer, and learning rate scheduler
# model = unet(num_bands=6, num_classes=3,dropout_prob=0.0).to(device)
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.6,  # Reduce learning rate by multiplying by this factor
#     patience=20,  # Number of epochs with no improvement to wait
#     verbose=True
# )
#
# loss_fn = nn.CrossEntropyLoss()
#
# # Initialize variables for early stopping
# early_stop_patience = 10
# best_val_loss = float('inf')
# epochs_without_improvement = 0
#
# # Define fixed indices for visualization
# val_dataset_size = len(val_loader.dataset)  # Get the total number of validation samples
# # indices = random.sample(list(range(val_dataset_size)), 4)  # Select 4 random sample indices
# # print("Visualization indices:", indices)
# indices = [1, 2, 3, 0]  # Use fixed indices for visualization
#
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0.0
#
#     for x_batch, y_batch in tra_loader:
#         x_batch, y_batch = x_batch.to(device, dtype=torch.float32), y_batch.to(device)
#
#         # Ensure the target tensor is 3D
#         y_batch = y_batch.squeeze(1).long()  # Remove the singleton channel dimension
#
#         # Forward pass
#         pred = model(x_batch)
#         loss = loss_fn(pred, y_batch)  # Compute the loss
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#     # Validation step
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val, y_val = x_val.to(device, dtype=torch.float32), y_val.to(device)
#             y_val = y_val.squeeze(1).long()  # Ensure the target tensor is 3D
#             val_pred = model(x_val)
#             val_loss += loss_fn(val_pred, y_val).item()
#
#         # Visualization during validation
#
#         if epoch % 25 == 0:  # Visualize every 25 epochs
#             pred_class = torch.argmax(val_pred, dim=1).cpu().numpy()  # Convert predictions to class labels
#             y_val_np = y_val.cpu().numpy()  # Ground truth to numpy
#
#
#             plt.figure(figsize=(24, 12))
#             for i, idx in enumerate(indices):
#                 # Select the i-th sample and remove extra dimensions
#                 y_val_sample = y_val_np[idx].squeeze()  # Shape: (256, 256)
#                 pred_class_sample = pred_class[idx]  # Shape: (256, 256)
#
#                 # Plot ground truth
#                 plt.subplot(2, 4, i + 1)
#                 plt.title(f"Ground Truth {i + 1}")
#                 plt.imshow(y_val_sample, cmap='viridis')  # Ground truth visualization
#
#                 # Plot prediction
#                 plt.subplot(2, 4, i + 5)
#                 plt.title(f"Prediction {i + 1}")
#                 plt.imshow(pred_class_sample, cmap='viridis')  # Prediction visualization
#
#             plt.savefig(f"epoch_{epoch}_samples.png")
#             plt.close()
#
#
#     # Average validation loss
#     val_loss /= len(val_loader)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
#
#     # Step the learning rate scheduler
#     lr_scheduler.step(val_loss)
#     print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    # # Early stopping logic
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     epochs_without_improvement = 0
    # else:
    #     epochs_without_improvement += 1
    #
    # if epochs_without_improvement >= early_stop_patience:
    #     print("Early stopping triggered.")
    #     break


# # Training Loop
# num_epochs  = 60
# loss_fn = nn.CrossEntropyLoss()  # Create an instance of the loss function
# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0.0
#
#     for x_batch, y_batch in tra_loader:
#         x_batch, y_batch = x_batch.to(device,dtype=torch.float32), y_batch.to(device)
#
#         # Ensure the target tensor is 3D
#         y_batch = y_batch.squeeze(1).long()  # Remove the singleton channel dimension
#
#         # Forward pass
#         pred = model(x_batch)
#         loss = loss_fn(pred, y_batch)  # Compute the loss
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#
#         # Validation step
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val, y_val = x_val.to(device,dtype=torch.float32), y_val.to(device)
#             y_val = y_val.squeeze(1).long()  # Ensure the target tensor is 3D
#             val_pred = model(x_val)
#             val_loss += loss_fn(val_pred, y_val).item()
#
#             # Visualize predictions (add this block)
#             if epoch % 20 == 0:  # Visualize every 10 epochs
#                 pred_class = torch.argmax(val_pred, dim=1).cpu().numpy()  # Convert predictions to class labels
#                 y_batch_np = y_batch.cpu().numpy()  # Ground truth to numpy
#
#                 # Select the first sample and remove extra dimensions
#                 y_batch_np = y_batch_np[0].squeeze()  # Shape: (256, 256)
#                 pred_class = pred_class[0]  # Shape: (256, 256)
#
#                 # Plot the first batch only
#                 plt.figure(figsize=(12, 6))
#                 plt.subplot(1, 2, 1)
#                 plt.title("Ground Truth")
#                 plt.imshow(y_batch_np, cmap='viridis')  # Now this will be 2D
#                 plt.subplot(1, 2, 2)
#                 plt.title("Prediction")
#                 plt.imshow(pred_class, cmap='viridis')  # Now this will be 2D
#                 plt.savefig(f"epoch_{epoch}_batch_{x_batch.shape[0]}.png")
#                 plt.close()
#                 # plt.close()
#                 # Only visualize one batch for efficiency
#
#     val_loss /= len(val_loader)  # Average validation loss
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
#
#     # Step the learning rate scheduler
#     lr_scheduler.step(val_loss)
#     print("Learning Rate:", lr_scheduler.get_last_lr())

# In[13]:


# dataset = scene_dset(scene_list=['/work/jinikeda/ETC/WatNetv2/datasets/s2//val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S1.tif','/work/jinikeda/ETC/WatNetv2/datasets/s2//val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S2.tif'], truth_list=['/work/jinikeda/ETC/WatNetv2/datasets/s2//val_labels/S2A_L2A_20190318_N0211_R061_S1_classified.tif','/work/jinikeda/ETC/WatNetv2/datasets/s2//val_labels/S2A_L2A_20190318_N0211_R061_S2_classified.tif'],
#     transforms=None
# )


# In[14]:


# # Test the dataset
# for i in range(len(dataset)):
#     scene, truth = dataset[i]
#     print(f"Sample {i}")
#     print(f"Scene Shape: {scene.shape}")  # Should be (H, W, 6)
#     print(f"Truth Shape: {truth.shape}")  # Should be (H, W)
#     print(f"Unique Classes in Truth: {np.unique(truth)}")  # Should contain 0, 1, 2
#     # Break after testing the first sample
#     break
#
#
# # In[15]:
#
#
# tra_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#
# for batch_idx, (scenes, truths) in enumerate(tra_loader):
#     print(f"Batch {batch_idx}")
#     print(f"Scenes shape: {scenes.shape}")
#     print(f"Truths shape: {truths.shape}")
#     break
#
#
# # In[16]:


''' -------- 2. Model loading and training strategy ------- '''
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=20)

''' -------- 3. Model training for loops ------- '''
metrics = train_loops(model=model,
                    loss_fn=config.loss_ce,
                    optimizer=optimizer,
                    tra_loader=tra_loader,
                    val_loader=val_loader,
                    epoches=num_epoch,
                    num_classes=num_classes,
                    lr_scheduler=lr_scheduler
                    )


''' -------- 4. trained model and accuracy metric saving  ------- '''
# model saving
path_weights = 'model/trained_model/' + model_name_save + '_weights.pth'
torch.save(model.state_dict(), path_weights)
print('Model weights are saved to --> ', path_weights)
## metrics saving
path_metrics = 'model/trained_model/' + model_name_save + '_metrics.csv'

metrics = {key: [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in values]
           for key, values in metrics.items()}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(path_metrics, index=False, sep=',')
metrics_df = pd.read_csv(path_metrics)
print('Training metrics are saved to --> ', path_metrics)

print("Training complete.")




