"""
@author: Rijad
"""
#-------------------IMPORTS-------------------------------------
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize,AffineTransform
import numpy as np
from skimage import img_as_ubyte;
from skimage.color import rgb2gray;
import dlib
from imutils import face_utils
from scipy.spatial import Delaunay;
from skimage.transform import warp
from skimage.draw import polygon
from cv2 import VideoWriter,VideoWriter_fourcc,cvtColor,COLOR_RGB2BGR

#---------------FUNCTIONS----------------------------

def add_corners(pts,img):
    
    #assuming only one face is recognized
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

def get_triangle_mask(tm, bbm, output_shape):   #
    #tm = tm + bbm[:2];
    mask = np.zeros(output_shape)
    rr, cc = polygon(tm[:,1] ,tm[:,0], shape = output_shape)
    mask[rr, cc] = 1
    mask = mask[:,:,None];  #adding third dimension becouse of multiplaying with 3 dimensional matrix
    return mask;

def warp_image(im, pts, tris, ptsm, shape):
    wraped = np.zeros(shape)

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
                
            output_shape = wraped[...].shape[:2];
                
            wt1 = wrap_triangle(im, bb1, M, (bbm[2],bbm[3]));
            
            wtnew = np.zeros(wraped.shape)
            wtnew[bbm[1]:bbm[1]+bbm[2], bbm[0] : bbm[0]+bbm[3],:] = wt1;
            
            mask = get_triangle_mask(tm, bbm, output_shape);
            
            wraped[...] = wraped[...] * (1-mask) + mask * wtnew;
          
    
    return wraped;

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
    
    
def morphing(im1,im2,alpha, show_landmarks = False, show_tri = False):
    """
    (im1,im2,alpha, show_landmarks = False, show_tri = False)

    Parameters
    ----------
    im1 : image
        image from which morphing begins
    im2 : image
        image to which it morphs
    alpha : float in range of 0:1
        Coefficient for calculating intermediate positions of all key points
    show_landmarks : bool, optional
        The default is False.
        If set true is going to show 68 face landmarks shown as black dots on gray scale image, in different figure, for both images.
    show_tri : bool, optional
        The default is False.
        If set true is going to show Delaunay triangulation on both image in separate figures.
    Returns
    -------
    morphed : TYPE
        DESCRIPTION.
        Image morphed between two input images.

    """
    
    if im1.shape > im2.shape:           #resize for better morphing
        im2 = resize(im2,im1.shape) 
    else:
        im1 = resize(im1,im2.shape) 
    
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
    
    ptsm = (1-alpha) * pts1 + alpha*pts2;   #calculate intermidiete position of all key points
    ptsm = ptsm.astype(int)
    
    warped1 = warp_image(im1, pts1, tris1, ptsm, im2.shape)
    warped2 = warp_image(im2, pts2, tris2, ptsm, im1.shape);

    morphed = img_as_ubyte((1-alpha)*warped1 + alpha*warped2)

        
    return morphed

def face_morph(im1,im2,alphas):
    
    frames = [];
    
    for (i,alpha) in enumerate(alphas):
        frames.append(morphing(im1,im2,alpha))
    
    return frames;

def save_frames_to_video(loc,frames):
    
    
    
    out = VideoWriter(loc, VideoWriter_fourcc(*'mp4v'), 1, frames[0].shape[:2])
    
    for(i,frame) in enumerate(frames):
        frame = cvtColor(frame, COLOR_RGB2BGR)    #had to use one more cv function becouse writer demands bgr type
        out.write(frame)
        
    
    out.release();
        
    
#------------MAIN----------------------------------

im1 = imageio.imread('.\daenerys.jpg');
im2 = imageio.imread('.\gal_gadot.jpg');

#morphed = morphing(im1, im2, 0.5)
#fig,(ax) = plt.subplots(nrows=1, ncols=1);
#ax.imshow(morphed);

frames = face_morph(im1, im2, alphas = np.linspace(0,1,15))        # it's 15 frames
save_frames_to_video('.\projectmain.mp4',frames)
#for (i,frame) in enumerate(frames):
 #   plt.figure(i);
  #  plt.imshow(frame);

