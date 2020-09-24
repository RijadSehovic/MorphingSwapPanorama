# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:17:53 2020

@author: Rijad
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage.transform import ProjectiveTransform,SimilarityTransform
from AffineGeometrijskeTransformacije import get_tf_model
import imageio



def _get_stitch_images(im0, im1, tf_model = None, n_keypoints = 500, min_samples = 4, residual_threshold = 2, **kwargs):
    
    if tf_model is None:
        tf_model = SimilarityTransform()
    elif tf_model == 'auto':
        tf_model = get_tf_model(im1, im0, xTransform = ProjectiveTransform, n_keypoints = n_keypoints, min_samples = min_samples, residual_threshold = residual_threshold, **kwargs)
    
    r,c = im0.shape[:2]
    corners0 = np.array([[0,0],[0,r],[c,0],[c,r]])
    r,c = im1.shape[:2]
    corners1 = np.array([[0,0],[0,r],[c,0],[c,r]])
    wcorners1 = tf_model(corners1)
    
    all_corners = np.vstack((corners0,corners1,wcorners1))
    min_corner = all_corners.min(axis = 0)
    max_corner = all_corners.max(axis = 0)
    new_shape = max_corner - min_corner
    
    new_shape = np.ceil(new_shape[::-1]).astype(np.int)   #ronund number
    shift = SimilarityTransform(translation = -min_corner)  #to ensure all positive coord
    im0_ = warp(im0, shift.inverse, output_shape = new_shape, cval = -1)
    im1_ = warp(im1, (tf_model + shift).inverse, output_shape = new_shape, cval = -1)
    return im0_, im1_

    
   
    
    

def _merge_stitch_images(im0_, im1_, mask_idx = None):
    
    if mask_idx is not None:
        if mask_idx == 0:
            im1_[im0_ > -1] = 0
        elif mask_idx == 1:
            im0_[im1_ > -1] = 0
    else:
        alpha = 1.0 * (im0_[:,:,0] != -1) + 1.0 * (im1_[:,:,0] != -1)
    
    im0_[im0_ == -1] = 0
    im1_[im1_ == -1] = 0
    
    merged = im0_ + im1_
    
    if mask_idx is None:
        merged /= np.maximum(alpha, 1)[...,None]
    return merged
    

