import os
import cv2
import math 
import imutils
import json
import copy
import logging

import scipy
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output

import sklearn
from sklearn.cluster import KMeans

import skimage
from skimage import io
from skimage.util import pad
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.util import crop as crop_skimage
from skimage.morphology import binary_closing, disk
from skimage.morphology import skeletonize as skeletonize_skimage

from image_utils import binary_close, binarize, binary_floodfill, skeletonize, pixel_ratio, skeletonize_area_ratio

def show_contours(image, contours, num = -1):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    if num == -1:
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth = 3)
    else:
        for n, contour in enumerate(contours):
            if n == num:
                ax.plot(contour[:, 1], contour[:, 0], linewidth = 3)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def arrow_centroid(arrow_contours):
    '''
    Function that finds the centroid of the arrow
    Rather than find the literal center of mass of the arrow, 
    We found the average of the largest x and y and smallest 
    x and y to get a point closer to the middle of the entire shape. 
    
    We will be using this centroid to find the middle point of a reaction 
    or set of reactions and this can be used to label the molecules around the arrow
    as products, intermediates, or reactants. 
    
    @PARAM:
        - arrow_contours: an array of contours of arrows from the find_arrows() method
    @RETURN:
        - centroids: an array of tuples which we can use to locate the centroids of the arrows
                    Will be of size number of arrows identified
    '''
    centroids = []
    # For each arrow in contour array
    for i in range(len(arrow_contours)):
        x_values = []
        y_values = []  
        # For each list of points in contour
        for j in range(len(arrow_contours[i])):
            x_values.append(arrow_contours[i][j][0])
            y_values.append(arrow_contours[i][j][1])
        max_x_val = max(x_values)
        min_x_val = min(x_values)
        average_x = (max_x_val + min_x_val) * 0.5
        
        max_y_val = max(y_values)
        min_y_val = min(y_values)
        average_y = (max_y_val + min_y_val) * 0.5
        centroids.append((average_x, average_y))
    return centroids 

def pad_image(input_contours, num, size = 500):
    '''
    Function that will pad a contour in a square image so that 
    we can plug into a formula or neural network. Useful for isolating contours,
    and making a generalized square shape that is easy to work with.
    
    @PACKAGES:
        - cv2: used for the image manipulation and drawing
        - NumPy: used for matrix iperations and intializations
    @PARAM:
        - input_contours: points of contours that we want to draw on the 
                          padded image. 
        - num: the index of the contour we want to pad. Useful for loops
        - size: size of the square that we want. Image will end up being
                of shape (size, size). Defaults to 500
    @RETURN:
        - padded_image: the new image with the contour drawn in
        
    Note: If you input more than one contour, a ValueError will not be raised
    However, specifiy the index of the contour that we want 
    '''
            
    # start with a normal white square with size  
    padded_image = np.ones((size, size))
    # Center of the image and the contour 
    center = arrow_centroid(input_contours)
    image_center = (size/2, size/2)
    # Find the difference from the center of the contour to the center 
    # of the of the padded image
    y_change = center[num][1] - image_center[1]
    x_change = center[num][0] - image_center[0]
    for i in range(len(input_contours[num])):
        # Adjust the contours to align it close to the center
        x = int(input_contours[num][i][0] - x_change)
        y = int(input_contours[num][i][1] - y_change)
        padded_image[x, y] = 0
    # Draw the adjusted contours on the new image
    #show_contours(padded_image, input_contours, num = num)
    return padded_image


def segment_image(image, kernel = None):
    bin_fig = binarize(image)
    
    if kernel == None:
        skel_pixel_ratio = skeletonize_area_ratio(image)

        if skel_pixel_ratio > 0.025:
            kernel = 1
            closed_fig = binary_close(bin_fig, size=kernel)
            #print("Segmentation kernel size =" , kernel)

        elif 0.02 < skel_pixel_ratio <= 0.025:
            kernel = 2
            closed_fig = binary_close(bin_fig, size=kernel)
            #print("Segmentation kernel size =", kernel)

        elif 0.015 < skel_pixel_ratio <= 0.02:
            kernel = 4
            closed_fig = binary_close(bin_fig, size=kernel)
            #print("Segmentation kernel size =", kernel)

        else:
            kernel = 5
            closed_fig = binary_close(bin_fig, size=kernel)
            #print("Segmentation kernel size =", kernel)
    else:
        closed_fig = binary_close(bin_fig, size=kernel)
    
    fill_img = binary_floodfill(closed_fig)

    return fill_img.astype(int)

def get_image_contours(image):
    seg_im = segment_image(image)
    cnts = find_contours(seg_im, False) 
    return cnts


