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

from contour_utils import split_contours

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
    """ 
    Converts to greyscale if RGB
    
    """
    # Convert to greyscale if needed
    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        grey_img = rgb2gray(img)
    else:
        grey_img = img
    return grey_img

def binarize(fig, threshold = 0.85):
    """ 
    Converts image to binary
    RGB images are converted to greyscale using :class:`skimage.color.rgb2gray` before binarizing.
    
    @PARAM:
        - numpy.ndarray img: Input image
        - float|numpy.ndarray threshold: Threshold to use.
    @RETURN:
        - Binary image as a numpy array 
    """
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
    """
    Joins unconnected pixel by dilation and erosion
    """
    
    selem = disk(size)

    fig = pad(fig, size, mode='constant')
    fig = binary_closing(fig, selem)
    fig = crop_skimage(fig, size)
    return fig

def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    # log.debug('Binary floodfill initiated...')
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
    ones = np.count_nonzero(img)
    all_pixels = np.size(img)
    ratio = ones / all_pixels
    return ratio

def skeletonize_area_ratio(image):
    skel_fig = skeletonize(image)
    return pixel_ratio(skel_fig)


def small_to_large_pad(image, size = 275):
    shape = image.shape
    image = cv2.copyMakeBorder(image, size - shape[0], size - shape[0], size - shape[1], size - shape[1], cv2.BORDER_CONSTANT, value = (255, 255, 255))
    return image