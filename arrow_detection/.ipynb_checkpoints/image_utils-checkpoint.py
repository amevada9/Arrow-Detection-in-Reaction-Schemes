'''
Here is a module that has the basic image manipulation methods that 
we need to get the information we need. Has segmentation, conversions, etc.
and can be used to change images around. 
'''

import cv2
import copy

import scipy
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import skimage
from skimage import io
from skimage.util import pad
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.util import crop as crop_skimage
from skimage.morphology import binary_closing, disk
from skimage.morphology import skeletonize as skeletonize_skimage

def convert_colors(image):
    '''
    Function that gives a clear interface to convert colors 
    from BGR in cv2 to RGB for normal use. Make sure to only 
    use this function if the 
    
    @PARAM:
        - image: a cv2 image that is in BGR
    @RETURN:
        - new_image that has been converted
    '''
    # Convert from BGR to RGB
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return new_image

def convert_greyscale(img):
    '''
    If we find an image that is in RGB/3 Channel
    we want to convert it to grayscale
    '''
    # Convert to greyscale if needed
    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        grey_img = rgb2gray(img)
    else:
        grey_img = img
    return grey_img

def binarize(fig, threshold = 0.85):
    '''
    Converts image to binary
    RGB images are converted to greyscale using :class:`skimage.color.rgb2gray` before binarizing.

    @PARAM:
        - numpy.ndarray img: Input image
        - float|numpy.ndarray threshold: Threshold to use.
    @RETURN:
        - Binary image as a numpy array 
    '''
    bin_fig = copy.deepcopy(fig)
    img = bin_fig

    # Skip if already binary
    if img.ndim <= 2 and img.dtype == bool:
        return img

    img = convert_greyscale(img)

    # Binarize with threshold (default of 0.85 empirically determined)
    binary = img < threshold
    bin_fig = binary
    return bin_fig

def binary_close(fig, size = 20):
    '''
    Performs a binary close of the image
    Joins unconnected pixel by dilation and erosion
    Used to smoothen image contours and ensure that images are as 
    dense and high-definition as possible
    
    @PACKAGES: 
        - numpy: used to store images
        - Scikit-Image: functions used to perform binary close, specifically
            - disk
            - pad
            - binary_closing
            - crop
    @PARAM:
        - fig: figure that we want binary closed needed in numpy array
    @RETURN:
        - fig: floodfilled figure
    '''
    
    selem = disk(size)

    fig = pad(fig, size, mode='constant')
    fig = binary_closing(fig, selem)
    fig = crop_skimage(fig, size)
    return fig

def binary_floodfill(fig):
    '''
    Converts all pixels inside closed contour to 1
    
    @PACKAGES: 
        - numpy: used to store images
        - ndi: functions used to fill
    @PARAM:
        - fig: figure that we want floodfilled, needed in numpy array
    @RETURN:
        - fig: floodfilled figure
    '''
    fig = ndi.binary_fill_holes(fig)
    return fig

def skeletonize(fig):
    """
    Erode pixels down to skeleton of a figure's img object
    :param fig :
    :return: Figure : binarized figure
    """

    skel_fig = binarize(fig)
    skel_fig = skeletonize_skimage(skel_fig)

    return skel_fig

def pixel_ratio(img):
    '''
    Finds the pixel ratio of white to black, used to decide
    the kernel size of the binary closing
    
    @PACKAGES:
        - numpy: used to perform the pixel calculations and such for the image
    
    @PARAM:
        - img: a loaded image in Scikit-Image, preferrbably binarized 
    
    @RETURN:
        - ratio: a double of the ratio of white (1) to black (0) pixels
    '''
    ones = np.count_nonzero(img)
    all_pixels = np.size(img)
    ratio = ones / all_pixels
    return ratio

def skeletonize_area_ratio(image):
    '''
    Skeletonizes and image and returns the pixel 
    ratio for the new image 
    
    @PACKAGES:
        - numpy: used to perform the pixel calculations and such for the image
        - Scikit-Image: used for image manipulations
    
    @PARAM:
        - image: a loaded image in Scikit-Image, preferrbably binarized 
    
    @RETURN:
        - pixel)ratio: a double of the ratio of white (1) to black (0) pixels of skeltonized image
    '''
    skel_fig = skeletonize(image)
    return pixel_ratio(skel_fig)

