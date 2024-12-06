import rasterio
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def readTiff(path):
    """
    Reads a GeoTIFF file and returns a dictionary with array data and metadata.
    :param path: str, path to the GeoTIFF file.
    :return: dictionary with 'array' (numpy array) and metadata.
    """

    with rasterio.open(path) as src:
        array = src.read()  # Read all bands as a numpy array
        meta = src.meta  # Metadata
    return {'array': array, 'meta': meta}


class crop2size():
    '''
    des: crop image with specific size.
    args:
      img: np.array()
      channel_first: True or False.
    '''
    def __init__(self, img, channel_first=False):
      self.channel_first = channel_first
      if self.channel_first:
        self.img = np.transpose(img, (1,2,0))
      else:
        self.img = img

    def toSize(self, size=(256, 256)):
      '''
        des: randomly crop corresponding to specific size
        input:
          size: tuble/list, (height, width)
        return: patch, the cropped patch from the image.
      '''
      start_h = random.randint(0, self.img.shape[0]-size[0])
      start_w = random.randint(0, self.img.shape[1]-size[1])
      patch = self.img[start_h:start_h+size[0], start_w:start_w+size[1],:]
      if self.channel_first:
        patch = np.transpose(patch, (2,0,1))
      return patch

    def toScales(self, scales=(2048, 512, 256)):
        '''
        des: randomly crop multiple-scale patches (from high to low) from remote sensing image.
        input:
            scales: tuple or list (high scale -> low scale)
        return: patches_group_down: list of multiscale patches.
        '''
        height, width = self.img.shape[:-1]
        if height<scales[0] or width<scales[0]:
          raise Exception('The input scale overpass the size of image!')
        patches_group = []
        patch_high = self.toSize(size=(scales[0], scales[0]))
        patches_group.append(patch_high)
        for scale in scales[1:]:
            start_offset = (scales[0]-scale)//2
            patch_lower = patch_high[start_offset:start_offset+scale, start_offset:start_offset+scale, :]
            patches_group.append(patch_lower)
        patches_group_down = []
        for patch in patches_group[:-1]:
            patch_down=[cv2.resize(patch[:,:,num], dsize=(scales[-1], scales[-1]), \
                                interpolation=cv2.INTER_LINEAR) for num in range(patch.shape[-1])]
            patches_group_down.append(np.stack(patch_down, axis=-1))
        patches_group_down.append(patch_lower)
        if self.channel_first:
          patches_group_down = [np.transpose(patch_down, (2,0,1)) for patch_down in patches_group_down]
        return patches_group_down

import matplotlib.pyplot as plt
import numpy as np


def imgShow(img, ax=None, extent=None, color_bands=(2,1,0), clip_percent=2, per_band_clip=False):
    '''
    Description: show the single image.
    args:
        img: (row, col, band) or (row, col), DN range should be in [0,1]
        ax: axes for showing image.
        extent: list, the coordinates of the extent.
        num_bands: a list/tuple, [red_band,green_band,blue_band]
        clip_percent: for linear strech, value within the range of 0-100.
        per_band_clip: if True, the band values will be clipped by each band respectively.
    return: None
    '''
    img = img.copy()
    img[np.isnan(img)]=0
    img = np.squeeze(img)

    if np.min(img) == np.max(img):
        if len(img.shape) == 2:
            if ax: ax.imshow(np.clip(img, 0, 1), extent=extent, vmin=0,vmax=1)
            else: plt.imshow(np.clip(img, 0, 1), extent=extent, vmin=0,vmax=1)
        else:
            if ax: ax.imshow(np.clip(img[:,:,0], 0, 1), extent=extent, vmin=0, vmax=1)
            else: plt.imshow(np.clip(img[:,:,0], 0, 1), extent=extent, vmin=0, vmax=1)
    else:
        if len(img.shape) == 2:
            img_color = np.expand_dims(img, axis=2)
        else:
            img_color = img[:,:,[color_bands[0], color_bands[1], color_bands[2]]]
        img_color_clip = np.zeros_like(img_color)
        if per_band_clip == True:
            for i in range(img_color.shape[-1]):
                if clip_percent == 0:
                    img_color_hist = [0,1]
                else:
                    img_color_hist = np.percentile(img_color[:,:,i], [clip_percent, 100-clip_percent])
                img_color_clip[:,:,i] = (img_color[:,:,i]-img_color_hist[0])\
                                    /(img_color_hist[1]-img_color_hist[0]+0.0001)
        else:
            if clip_percent == 0:
                    img_color_hist = [0,1]
            else:
                img_color_hist = np.percentile(img_color, [clip_percent, 100-clip_percent])
            img_color_clip = (img_color-img_color_hist[0])\
                                     /(img_color_hist[1]-img_color_hist[0]+0.0001)

        if ax: ax.imshow(np.clip(img_color_clip, 0, 1), extent=extent, vmin=0, vmax=1)
        else: plt.imshow(np.clip(img_color_clip, 0, 1), extent=extent, vmin=0, vmax=1)


def imsShow(img_list, img_name_list, clip_list=None, \
                            color_bands_list=None, axis=True, row=None, col=None):
    ''' des: visualize multiple images.
        input:
            img_list: containes all images
            img_names_list: image names corresponding to the images
            clip_list: percent clips (histogram) corresponding to the images
            color_bands_list: color bands combination corresponding to the images
            row, col: the row and col of the figure
            axis: True or False
        return: None
    '''
    if not clip_list:
        clip_list = [2 for i in range(len(img_list))]
    if not color_bands_list:
        color_bands_list = [[2, 1, 0] for i in range(len(img_list))]
    if row == None:
        row = 1
    if col == None:
        col = len(img_list)
    for i in range(row):
        for j in range(col):
            ind = (i*col)+j
            if ind == len(img_list):
                break
            plt.subplot(row, col, ind+1)
            imgShow(img=img_list[ind], color_bands=color_bands_list[ind], \
                                                        clip_percent=clip_list[ind])
            plt.title(img_name_list[ind])
            if not axis:
                plt.axis('off')


class img2patch():
    def __init__(self, img, patch_size, edge_overlay):
        '''
        args:
            img: np.array()
            patch_size: size of the patch
            edge_overlay: an even number, single-side overlay of the neighboring images.
        '''

        if edge_overlay % 2 != 0:
            raise ValueError('Argument edge_overlay should be an even number')
        self.edge_overlay = edge_overlay
        self.patch_size = patch_size
        self.img = img[:,:,np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]
        self.img_patch_row = np.nan    # valid when call toPatch
        self.img_patch_col = np.nan
        self.start_list = []           #

    def toPatch(self):
        '''
        des:
            convert img to patches.
        return:
            patch_list, contains all generated patches.
            start_list, contains all start positions(row, col) of the generated patches.
        '''
        patch_list = []
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, self.patch_size),
                                          (self.edge_overlay, self.patch_size), (0,0)), 'constant')
        self.img_patch_row = (img_expand.shape[0]-self.edge_overlay)//patch_step
        self.img_patch_col = (img_expand.shape[1]-self.edge_overlay)//patch_step
        for i in range(self.img_patch_row):
            for j in range(self.img_patch_col):
                patch_list.append(img_expand[i*patch_step:i*patch_step+self.patch_size,
                                                        j*patch_step:j*patch_step+self.patch_size, :])
                self.start_list.append([i*patch_step-self.edge_overlay, j*patch_step-self.edge_overlay])
        return patch_list

    def higher_patch_crop(self, higher_patch_size):
        '''
        des:
            crop the higher-scale patch (centered by the given low-scale patch)
                (!!Note: the toPatch() usually should be firstly called when use higher_patch_crop())
        input:
            higher_patch_size, int, the lager patch size compared the low-scale patch size.
        return:
            higher_patch_list, list, contains higher-scale patches corresponding to the lower-scale patches.
        '''
        higher_patch_list = []
        radius_bias = higher_patch_size//2-self.patch_size//2
        img_expand = np.pad(self.img, ((self.edge_overlay, self.patch_size), \
                                            (self.edge_overlay, self.patch_size), (0,0)), 'constant')
        img_expand_higher = np.pad(img_expand, ((radius_bias, radius_bias), (radius_bias, radius_bias), (0,0)), 'constant')
        start_list_new = list(np.array(self.start_list)+self.edge_overlay+radius_bias)
        for start_i in start_list_new:
            higher_row_start, higher_col_start = start_i[0]-radius_bias, start_i[1]-radius_bias
            higher_patch = img_expand_higher[higher_row_start:higher_row_start+higher_patch_size, \
                                                            higher_col_start:higher_col_start+higher_patch_size,:]
            higher_patch_list.append(higher_patch)
        return higher_patch_list

    def toImage(self, patch_list):
        '''
        des:
            merge patches into one image.
            (!!note: the toPatch() usually should be firstly called when use toImage())
        args:
            patch_list: list of the all patches.
        return:
            img_array: the merged image by patches
        '''
        patch_list = [patch[self.edge_overlay//2:-self.edge_overlay//2, self.edge_overlay//2:-self.edge_overlay//2,:]
                                                        for patch in patch_list]
        patch_list = [np.hstack((patch_list[i*self.img_patch_col:i*self.img_patch_col+self.img_patch_col]))
                                                        for i in range(self.img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[self.edge_overlay//2:self.img_row+self.edge_overlay//2, \
            self.edge_overlay//2:self.img_col+self.edge_overlay//2,:]
        return img_array