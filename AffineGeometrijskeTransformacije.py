#import numpy as np
#import matplotlib.pyplot as plt
from skimage.transform import warp,AffineTransform
#import imageio
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
#im = imageio.imread('..\GENERALNO\KULeuven\ProcesiranjeSlika\imgs\yoda.jpg')

def get_matches(im_or,im_tf, n_keypoints = 500, ax = None, title = 'Original vs transformed'):
    descriptor_extractor = ORB(n_keypoints = n_keypoints)
    
    descriptor_extractor.detect_and_extract(rgb2gray(im_or))
    keypoints_or = descriptor_extractor.keypoints
    descriptors_or = descriptor_extractor.descriptors
    
    descriptor_extractor.detect_and_extract(rgb2gray(im_tf))
    keypoints_tf = descriptor_extractor.keypoints
    descriptors_tf = descriptor_extractor.descriptors
    
    matches = match_descriptors(descriptors_or,descriptors_tf,cross_check = True)
    
    if ax is not None:
        plot_matches(ax,im_or,im_tf,keypoints_or,keypoints_tf,matches)
        ax.axis('off')
        ax.set_title(title)
        
    return matches, keypoints_or, keypoints_tf

def get_tf_model(src, dst, xTransform = AffineTransform, n_keypoints = 500, min_samples = 4, residual_threshold = 2, **kwargs):
   # fig , ax1 = plt.subplots(nrows =1, ncols =1,figsize=(6,3))
    matches, kp_src, kp_dst = get_matches(src, dst, n_keypoints = n_keypoints)  #keypoints (row,column)
    src = kp_src[matches[:,0]][:,::-1]
    dst = kp_dst[matches[:,1]][:,::-1]
    tf_model, _ = ransac((src,dst), xTransform, min_samples = min_samples, residual_threshold = residual_threshold, **kwargs)
    
    return tf_model

