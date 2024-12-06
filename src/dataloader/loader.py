## author: xin luo
## create: 2021.9.15
## des: data loading throuth 3 ways: 
#       1) through scene pathes, 
#       2) through torch.Tensor, 
#       3) through patch paths.

import torch
import numpy as np
from .pyrsimg import readTiff, crop2size


class scene_dset(torch.utils.data.Dataset):
    '''
    des: scene and truth image reading from the np.array(): read data from memory.
    '''
    def __init__(self, scene_list, truth_list, transforms, patch_size=[256, 256], channel_first=False,):
        '''input arrs_scene, arrs_truth are list'''
        if channel_first: 
            self.scene_list = [scene_tensor.transpose(1,2,0) for scene_tensor in scene_list]
        else:
            self.scene_list = scene_list

        self.truth_list = truth_list
        self.patch_size = patch_size
        self.transforms = transforms
        self.channel_first = channel_first

    def __getitem__(self, index):
        '''load images and truths'''
        scene = self.scene_list[index]
        truth = self.truth_list[index]

        # Load scene and truth images using readTiff
        # print(f"Scene Shape: {scene.shape}")
        # print(f"Truth Shape: {truth.shape}")

        # Ensure scene has the shape (H, W, bands)
        scene = np.transpose(scene, (1, 2, 0))  # Shape: (H, W, 6)
        truth = np.transpose(truth, (1, 2, 0))  # Shape: (H, W, 1)

        # Ensure truth has the shape (H, W, 1) by adding an axis
        if truth.ndim == 2:  # If truth is (H, W), add a new axis
            truth = truth[:, :, np.newaxis]  # Shape: (H, W, 1)

        # Concatenate scene and truth along the last axis
        scene_truth = np.concatenate((scene, truth), axis=-1)  # Shape: (H, W, 7)

        '''pre-processing (e.g., random crop)'''
        patch_ptruth = crop2size(img=scene_truth, channel_first=False).toSize(size=self.patch_size)
        # Split into patch and ptruth
        patch, ptruth = patch_ptruth[:, :, 0:-1], patch_ptruth[:, :, -1]
        patch = patch.transpose(2, 0, 1)  # Set the channel first: (C, H, W)

        # Debug print before applying transforms
        # print(f"Patch shape before transforms: {patch.shape}")
        # print(f"Ptruth shape before transforms: {ptruth.shape}")

        # Image augmentation using transforms
        if self.transforms:
            for transform in self.transforms:
                patch, ptruth = transform(patch, ptruth)
                # print(f"After transform {transform}: patch shape = {patch.shape}, ptruth shape = {ptruth.shape}")

        # Convert ptruth to a PyTorch tensor before unsqueezing
        ptruth = ptruth.clone().detach().to(dtype=torch.int)  # Convert to Tensor and ensure it's in integer format
        ptruth = torch.unsqueeze(ptruth, 0)
        return patch, ptruth


        # Return patch and ptruth
        return patch, ptruth

    def __len__(self):
        return len(self.truth_list)


class scene_path_dset(torch.utils.data.Dataset):
    '''
    des: pair-wise image and truth image reading from the data path (read data from disk).
    '''
    def __init__(self, paths_scene, paths_truth, transforms, patch_size=[256, 256], channel_first=False):
        self.paths_scene = paths_scene
        self.paths_truth = paths_truth
        self.transforms = transforms
        self.patch_size = patch_size
        self.channel_first = channel_first
    def __getitem__(self, index):
        '''load images and truths'''
        scene_ins = readTiff(self.paths_scene[index])
        truth_ins = readTiff(self.paths_truth[index])
        # Extract the arrays from the readTiff outputs
        scene = scene_ins['array']  # Scene image array
        truth = truth_ins['array'][:, :, np.newaxis]  # Add a new axis to truth for concatenation

        # Combine scene and truth into a single array for processing
        scene_truth = np.concatenate((scene, truth), axis=-1)

        '''pre-processing (e.g., random crop)'''
        patch, truth = crop2size(img=scene_truth, channel_first=self.channel_first).toSize(size=self.patch_size)
        for transform in self.transforms:
            patch, truth = transform(patch, truth)
        return patch, torch.unsqueeze(truth, 0)

    def __len__(self):
        return len(self.paths_truth)


class patch_path_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from data paths (in SSD)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->1.2 s model train -> 2.9 s 
    '''
    def __init__(self, paths_patch):
        self.paths_patch = paths_patch
    def __getitem__(self, index):
        '''load patches and truths'''
        patch_pair = torch.load(self.paths_patch[index])
        patch = patch_pair[0]
        truth = patch_pair[1]
        return patch, truth
    def __len__(self):
        return len(self.paths_patch)

class patch_tensor_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from memory (in RAM)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->0.7 s model train -> 2.9 s 
    '''
    def __init__(self, patch_pair_list):
        self.patch_pair_list = patch_pair_list
    def __getitem__(self, index):
        '''load patches and truths'''
        patch = self.patch_pair_list[index][0]
        truth = self.patch_pair_list[index][1]
        return patch, truth
    def __len__(self):
        return len(self.patch_pair_list)
