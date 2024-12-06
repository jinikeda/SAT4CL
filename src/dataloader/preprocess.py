## author: xin luo
## create: 2021.9.9
## des: simple pre-processing for the dset data(image and truth pair).

import numpy as np
from .pyrsimg import readTiff

def img_normalize(img, max_bands, min_bands):
    '''
    des: normalization with the given per-band max and min values
    input:
        img: input multiple spectral satellite image, shape: (row, col, bands)
        max_bands: int or list, maximum value of bands.
        min_bands: int or list, minimum value of bands.
    return:
        img_nor: the normalized statellite image.
    '''
    img_nor = []
    if isinstance(max_bands, int):
        max_bands = [max_bands for i in range(img.shape[-1])]
        min_bands = [min_bands for i in range(img.shape[-1])]
    for i_band in range(img.shape[-1]):
        band_nor = (img[:,:,i_band]-min_bands[i_band])/(max_bands[i_band]-min_bands[i_band]+0.0001)
        img_nor.append(band_nor)
    img_nor = np.stack(img_nor, axis=-1)
    img_nor = np.clip(img_nor, 0., 1.)
    return img_nor


# from pyrsimg import readTiff, img_normalize
#
def read_normalize(paths_img, paths_truth, max_bands, min_bands):
    ''' des: satellite image reading and normalization
        input:
            paths_img: list, paths of the satellite image.
            paths_truth: list, paths of the truth image.
            max_bands: int or list, the determined max values for the normalization.
            min_bands: int or list, the determined min values for the normalization.
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    for i in range(len(paths_img)):
        ## --- data reading
        scene_ins = readTiff(paths_img[i])
        truth_ins = readTiff(paths_truth[i])
        ## --- data normalization
        scene_arr = img_normalize(img=scene_ins['array'], max_bands=max_bands, min_bands=min_bands)
        scene_arr[np.isnan(scene_arr)]=0          ### remove nan value
        scene_list.append(scene_arr)
        truth_list.append(truth_ins['array'])
    return scene_list, truth_list

