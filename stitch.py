# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:04:10 2020

@author: Rijad
"""


from _stitch import _stitch
from skimage.transform import rescale
import imageio
import numpy as np
import matplotlib.pyplot as plt

def stitch(ims, order = [1,0,2], mask_idx = None, tf_model = 'auto', n_keypoints = 1000, min_samples = 4, residual_threshold = 2, **kwargs):
    
    #sorting
    i = 0
    ims_help = np.zeros_like(ims)
    for j in order:
        ims_help[i] = ims[j]
        i += 1
    ims = ims_help
    
    #apply _stitch
    merged = ims[0];
    for i,im in enumerate(ims,1):
        if i == 1:
            continue
        merged = _stitch(merged, ims[i-1],mask_idx = mask_idx, show = False, tf_model=tf_model, n_keypoints = n_keypoints, min_samples = min_samples,residual_threshold = residual_threshold, **kwargs)
    return merged

files = ['DFM_4209.jpg','DFM_4210.jpg','DFM_4211.jpg']
ims = []
for i,file in enumerate(files):
    im = imageio.imread('.\\' + file)
    im = im[:,500:500+1987, :]
    ims.append(rescale(im,0.25,anti_aliasing = True, multichannel = True))
    
merged = stitch(ims)
plt.figure("Panorama")
plt.imshow(merged)

