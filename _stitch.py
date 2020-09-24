# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:02:49 2020

@author: Rijad
"""
#import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import warp
#from skimage.transform import ProjectiveTransform,SimilarityTransform
#from AffineGeometrijskeTransformacije import get_tf_model
from ProjectiveGemetrijskeTransformacije import _get_stitch_images,_merge_stitch_images
#import imageio

def _stitch(im0, im1, mask_idx = None, show = True, tf_model = None, n_keypoints = 500, min_samples = 4, residual_threshold = 2, **kwargs):
    # combine get stitch images i megre stitch images
    im0_ , im1_ = _get_stitch_images(im0, im1, tf_model = tf_model, n_keypoints = n_keypoints, min_samples = min_samples, residual_threshold = residual_threshold)
    image = _merge_stitch_images(im0_, im1_, mask_idx = mask_idx)
    if show is True:
        plt.figure()
        plt.imshow(image)
    return image
