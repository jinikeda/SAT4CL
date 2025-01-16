#!/usr/bin/env python
# coding: utf-8
# Sentinental2 Classification labels for water and vegetation, and other
# water and other classifications are used on the labels of the WatNet datasets
# Developed by the Center for Computation & Technology  at Louisiana State University (LSU).
# Developer: Jin Ikeda,
# Last modified Nov 15, 2024

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt



########################################################################################################################
### Define functions to save the tiff file
########################################################################################################################

def save_tiff(output_path, z_array, src, dtype='uint8'):
    """
    Save the z_array map as a new TIFF file.

    Parameters:
    - output_path (str): The path where the output TIFF file will be saved.
    - z_array (numpy.ndarray): The z_array map to be saved.
    - src (rasterio.DatasetReader): The source dataset to copy metadata from.
    """
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=z_array.shape[0],
        width=z_array.shape[1],
        count=1,
        dtype=dtype,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(z_array, 1)


def save_tiff_workflow(outputspace, output_file, ndwi, src, dtype='uint8'):
    """
    Save the NDWI map as a new TIFF file.

    Parameters:
    - outputspace (str): Directory where the output TIFF file will be saved.
    - output_file (str): Name of the output TIFF file.
    - ndwi (numpy.ndarray): The NDWI map to be saved.
    - src (rasterio.DatasetReader): The source dataset to copy metadata from.
    """
    output_path = os.path.join(outputspace, output_file)
    save_tiff(output_path, ndwi, src, dtype='float32')
    print("NDWI map saved to", output_path)

def scatter_plot(ndwi, ndvi, output_path, xlabel='NDWI', ylabel='NDVI'):
    """
    Create and save a scatter plot of NDWI vs. NDVI.

    Parameters:
    - ndwi (numpy.ndarray): Array of x.
    - ndvi (numpy.ndarray): Array of y.
    - output_path (str): Path where the plot will be saved.
    - xlabel (str): Label for the x-axis. Default is 'NDWI'.
    - ylabel (str): Label for the y-axis. Default is 'NDVI'.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(ndwi, ndvi, c='blue', alpha=0.5, edgecolors='none')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    #plt.show()
    plt.close()
    print(f"Scatter plot saved to {output_path}")
########################################################################################################################


# Define indices for bands based on their positions in your TIFF file
# The bands are ordered as: B2, B3, B4, B8, B11, B12
B2_INDEX, B3_INDEX, B4_INDEX, B8_INDEX, B11_INDEX, B12_INDEX = 0, 1, 2, 3, 4, 5

Workspace = "Y:/Share_public/Dataset/WaterNet/5205674/dset-s2/dset-s2/val_scene"
Workspace2 = "Y:/Share_public/Dataset/WaterNet/5205674/dset-s2/dset-s2/val_truth"
Outputspace = os.path.abspath(os.path.join(Workspace, '..', 'val_lables'))
photospace = os.path.abspath(os.path.join(Workspace, '..', 'Photos'))

try:
    os.makedirs(Outputspace, exist_ok=True)
    os.makedirs(photospace, exist_ok=True)

except Exception as e:
    print(f"An error occurred while creating directories: {e}")

# Load the classification labels
# Define thresholds for classification
vegetation_threshold = 0.4  # Adjust based on visual inspection or known thresholds

# Get a list of all TIFF files in the directory
tiff_files = glob.glob(os.path.join(Workspace, '*.tif'))

# Load the TIFF file
# tiff_file = 'S2A_L2A_20190811_N0213_R013_6Bands_S1.tif'


for tiff_file in tiff_files:
    print(f"Processing {tiff_file}...")
    base_name = os.path.basename(tiff_file)
    labels_file = base_name.split('.')[0].replace('_6Bands', '') + '_Truth.tif'
    labels_path = os.path.join(Workspace2, labels_file)
    print(labels_path)
    with rasterio.open(labels_path) as src2:
        labels = src2.read(1)
        nodata = src2.nodata
        if nodata is not None:
            labels = np.where(labels == nodata, 0, labels)


    input_path = os.path.join(Workspace, tiff_file)
    with rasterio.open(input_path) as src:
        bands = src.read()  # Load all bands

    # Extract individual bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
    blue = bands[B2_INDEX].astype('float32')
    green = bands[B3_INDEX].astype('float32')
    red = bands[B4_INDEX].astype('float32')
    nir = bands[B8_INDEX].astype('float32')
    swir1 = bands[B11_INDEX].astype('float32')
    swir2 = bands[B12_INDEX].astype('float32')

    # Calculate NDVI (Normalized Difference Vegetation Index)
    ndvi = (nir - red) / (nir + red)


    labels[(ndvi > vegetation_threshold)] = 2

    # Save the classification map as a new TIFF
    base_name = os.path.basename(tiff_file)
    output_file = base_name.split('.')[0].replace('_6Bands', '') + '_classified.tif'
    save_tiff_workflow(Outputspace, output_file, labels, src)
