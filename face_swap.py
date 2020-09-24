"""

@author: Rijad
"""

#-------------------IMPORTS-------------------------------------
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize,AffineTransform,pyramid_reduce,pyramid_expand
import numpy as np
from skimage import img_as_ubyte;
from skimage.color import rgb2gray;
import dlib
from imutils import face_utils
from scipy.spatial import Delaunay;
from skimage.transform import warp
from skimage.draw import polygon
from cv2 import VideoWriter,VideoWriter_fourcc,cvtColor,COLOR_RGB2BGR
import scipy

#---------------FUNCTIONS----------------------------
def add_corners(pts,img):
    
    #assuiming only one face is recognized
    #images are provided in a same size so division with // is safe
    
    r,c = img.shape[:2]
    corners = np.array([[0,0],[0,r-1],[c-1,0],[c-1,r-1],[0,r//2],[c//2,0],[c-1,r//2],[c//2,r-1]]) #r and c are subtracted by 1, becouse of out of bounds exception
    pts[0] = np.vstack((pts[0],corners))
    
    return pts;

def get_bounding_box(t):
    
    topleft = t.min(axis = 0)
    bottomright = t.max(axis = 0)
    width = bottomright[1] - topleft[1];
    height =bottomright[0] - topleft[0];
    return np.array([topleft[0],topleft[1], width, height])

def wrap_triangle(img, bb, transf, shape):     
    
    rect = img[bb[1]:bb[1] + bb[2], bb[0]:bb[0] + bb[3], :]
    
    wp = warp(rect, transf,output_shape = shape)    
    
    return wp;


def give_pts(img):
    p = "./shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    
    gray = img_as_ubyte(rgb2gray(img));
    rects = detector(gray, upsample_num_times = 0);
    pts = []
    for (i,rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        pts.append(shape)
        
    pts = add_corners(pts, img)
    pts = pts[0]    #assuming only one face present   #when printing on image, have to inverse col1 and col2

    return pts;

def plot_landmarks(im,pts):
    gray = img_as_ubyte(rgb2gray(im));
               #for printing found points on image, it inverses columns when printing!
    h = gray.shape[0]
    w = gray.shape[1]
     
        # loop over the image, pixel by pixel
    for x in range(0, h):
        for y in range(0, w):
            for (i,bla) in enumerate(pts):
                if (bla == np.array([x,y])).all():
                    gray[y,x] = 0;
                      
    plt.figure();
    plt.imshow(gray);
    return;
    
def plot_tri(im,pts,tris):
    
    fig,(ax) = plt.subplots(nrows=1,ncols=1)    #ploting triangles found with Delaunay
    ax.imshow(im)
    ax.triplot(pts[:,0], pts[:,1], tris.simplices)
    
    return;
    
def get_triangle_mask(tm, bbm, output_shape):   #triangle center was moved to top left
    #tm = tm + bbm[:2];
    mask = np.zeros(output_shape)
    rr, cc = polygon(tm[:,1] ,tm[:,0], shape = output_shape)
    mask[rr, cc] = 1
    mask = mask[:,:,None];  #adding third dimension becouse of multiplaying with 3 dimensional matrix
    return mask;

    
def warp_image(im, pts, tris, ptsm, shape):
    """ (im, pts, tris, ptsm, shape) """
    warped = np.zeros(shape)

    for tri in tris.simplices:
        t1 = np.array([pts[tri[0]], pts[tri[1]], pts[tri[2]]]);
        tm = np.array([ptsm[tri[0]], ptsm[tri[1]], ptsm[tri[2]]]);
        
       
        
        bb1 = get_bounding_box(t1);
        bbm = get_bounding_box(tm);
        
        
        M = AffineTransform();
        M.estimate(t1-bb1[:2], tm-bbm[:2]);    #like triangle has its own image
          
        if not np.linalg.det(M.params):  #determinant is null (no inverse)
               continue;
        else:
            M = np.linalg.inv(M.params);
                
            output_shape = warped[...].shape[:2];
                
            wt1 = wrap_triangle(im, bb1, M, (bbm[2],bbm[3])); 
            
            wtnew = np.zeros(warped.shape)
            wtnew[bbm[1]:bbm[1]+bbm[2], bbm[0] : bbm[0]+bbm[3],:] = wt1;
            
            mask = get_triangle_mask(tm, bbm, output_shape);
              
            
            warped[...] = warped[...] * (1-mask) + mask * wtnew;
          
    
    return warped;

def swap_faces(im1,im2, blendmode = 'pyramid', faceorder = (0,1), show_landmarks = False, show_tri = False):
    """
    (im1,im2, blendmode = 'pyramid', faceorder = (0,1), show_landmarks = False, show_tri = False)
    
    Parameters
    ----------
    im1 : image
        Image with face which will be changed, so basically it is a frame. Unless rolls are changed by faceorder.
        Also two faces can be on im1.
    im2 : image
        Image with face which will be on first taken to put on im1.
    blendmode : 'alpha' or 'pyramid', optional
        The default is 'pyramid'. Secound option is 'alpha' which is just blending without adjusting
        edges for smooth look.
    faceorder : touple, optional
        The default is (0,1). With this parameter we can select rolls of images.
    show_landmarks : bool, optional
        The default is False. If set true is going to show 68 face landmarks shown as black dots on gray 
        scale image, in different figure, for both images.
    show_tri : bool, optional
        The default is False. If set true is going to show Delaunay triangulation on both image
        in separate figures.


    Returns
    -------
    None.

    """
    
    
    if faceorder != (0,1):
            im = im1
            im1 = im2
            im2 = im
            
    pts1 = give_pts(im1);
    pts2 = give_pts(im2);
        
    tris1 = Delaunay(pts1);
    tris2 = Delaunay(pts2);
        
    if show_landmarks == True:
        plot_landmarks(im1,pts1)
        plot_landmarks(im2,pts2)
            
    if show_tri == True:
        plot_tri(im1,pts1,tris1)
        plot_tri(im2, pts2, tris2)
    
    warped = warp_image(im2, pts2, tris2, pts1, im2.shape)
    
    pts_w = give_pts(warped)
    pts_w = pts_w[:-8,:]
    tris_w = Delaunay(pts_w)
    
    
    mmask = np.zeros(im2.shape);    #mmask as in main mask
    
    for tri in tris_w.simplices:
        t1 = np.array([pts_w[tri[0]], pts_w[tri[1]], pts_w[tri[2]]]);
            
        mask = np.zeros(im2.shape)
        rr, cc = polygon(t1[:,1] ,t1[:,0], shape = im2.shape)
        mask[rr, cc] = 1
            
        mmask[...] =  mmask[...] + mask
    
    if im1.shape > warped.shape:
        warped = resize(warped,im1.shape) 
        mmask = resize(mmask, im1.shape)
    else:
        warped = warped[:im1.shape[0], :im1.shape[1], :]
        mmask = mmask[:im1.shape[0], :im1.shape[1], :]
            
    if (blendmode == 'alpha'):
        blend = im1/256 * (1-mmask) + warped * mmask;
            
    if (blendmode == 'pyramid'):
        #let's play!
        blend = np.zeros(im1.shape)
        pyramid = []
        lpyramid1 = get_laplacian_pyramid(get_gaussian_pyramid(im1));
        lpyramid2 = get_laplacian_pyramid(get_gaussian_pyramid(warped));
        gpyramid = get_gaussian_pyramid(mmask);
        
        for i in range(0,len(lpyramid1)):
            pyramid.append((lpyramid1[i] * (1-gpyramid[i])) + (lpyramid2[i] * gpyramid[i]))
        
       
        for i in range(1,len(pyramid)):
            blend = blend + ((resize(pyramid[i],blend.shape)))
           
    plt.figure()
    plt.imshow(blend)
    
    return;


def get_gaussian_pyramid(image, downscale = 2, **kwargs):
    pom = image
    pyramid = []
    pyramid.append(image)
    while(True):
        pom = pyramid_reduce(pom,downscale = downscale,multichannel = True)
        pyramid.append(pom)
        #print(pom.shape[:2])
        if pom.shape[:2] < (6,6):      
            break;
            
    return pyramid;


def get_laplacian_pyramid(img_pyr, upscale = 2, **kwargs):
    pyramid = []
    
    img_pyr = img_pyr[::-1]
    
    for i in range(0,len(img_pyr)):
        if i == 0:
            pyramid.append(img_pyr[i])
            pom = pyramid_expand(img_pyr[i],upscale = upscale, multichannel = True)
        else:
            pyramid.append(img_pyr[i] - pom)
            pom = pyramid_expand(img_pyr[i],upscale = upscale, multichannel = True)
            
        
    pyramid = pyramid[::-1]
    return pyramid;
    

#---------------MAIN---------------------------------

im1 = imageio.imread('.\superman.jpg');
im2 = imageio.imread('.\\nicolas_cage.jpg');

swap_faces(im1,im2, blendmode = 'alpha')
swap_faces(im1,im2,blendmode = 'pyramid')

