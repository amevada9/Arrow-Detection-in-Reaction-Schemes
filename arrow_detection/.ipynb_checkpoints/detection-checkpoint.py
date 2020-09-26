'''
Module containing the methods for the detection of the arrows
Here is where the Neural Network can be applied and the images are passed through
Methods here are:

load_default_model(): Loads the model that is defualt, works well for most reaction images

get_direction(): gets the direction of the arrow, recursively uses pipeline() to ensure that text is
                 not merged into the arrow 
                 
pipeline(): final pipeline, insert an image that needs to be detected and spits out the arrow
'''


import os
import cv2
import math 
import imutils
import json
import copy
import time
import logging

import scipy
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd

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
from skimage.util import random_noise
from skimage.morphology import binary_closing, disk
from skimage.morphology import skeletonize as skeletonize_skimage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from arrow import *
from scikit_arrow_utils import arrow_average, arrow_centroid, line_mag, get_contour_height, get_contour_length, get_orientation
from image_utils import binary_close, binarize, binary_floodfill, skeletonize, pixel_ratio, skeletonize_area_ratio
from scikit_scripts import pad_image, segment_image, show_contours, get_image_contours

def load_default_model():
    '''
    Loads the default model for the functions to use
    
    @RETURNS:
        - model: default neural network 
    '''
    model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'notOverfittedModel2'))
    return model


def get_direction(arrow_contours, doc_name, count, model, image = None):
    '''
    Gets the direction of an arrow given its contour of points. Will be important for getting
    the products and the reactants of the arrow. Can use the centroid and direction of the 
    arrow to see whether the compunds on either side are products, reactants or intermediates
    
    If the arrow height is seen to be above 25 pixels, we can assume that either
        a) It is not an arrow
        b) Or it has text merged in
    Either way we will reevaluate the image and contours without full segmentation, so not 
    binary closing and skeletonzing
    
    @PARAM:
        - arrow_contours: the contours of the arrows extracted from find_arrow()
        - doc_name: Name of the document
        - count: page count
        - model: model used for detection
        - image: if using on only one set of contours, can leave this none, however for 
                 full capability use this to allow for recurvie checking
    @RETURN:
        - a dictionary with label of the arrow and the direction in the form of a string
    '''
   
    directions = {}
#     averages = arrow_average(arrow_contours)
    centroids = arrow_centroid(arrow_contours)
    orientations = get_orientation(arrow_contours)
    
    for arrow in range(len(orientations)):
        name = 'Arrow ' + str(arrow + 1)
        
        if orientations[arrow] == "Horizontal":
            height, extreme_idx = get_contour_height(arrow_contours[arrow])
            # If the height of the "arrow" is above 25 pixels, we need to ensure that text did not
            # Close it in, thus we will run the pipeline again without the binary closing
            if height > 15 and image != None:
                info, arrow_contours, averages, directions = pipeline(image, doc_name, count, model = model, segment = False)
            x_min = arrow_contours[arrow][extreme_idx[0]][1]
            x_max = arrow_contours[arrow][extreme_idx[1]][1]
            
            if (x_min + x_max) * 0.5 >= centroids[arrow][0]:
                directions[name] = 'Right'
            else:
                directions[name] = 'Left' 
                
        else:
            length, extreme_idx = get_contour_length(arrow_contours[arrow])
            y_min = arrow_contours[arrow][extreme_idx[0]][0]
            y_max = arrow_contours[arrow][extreme_idx[1]][0]
            
            if (y_min + y_max) * 0.5 >= centroids[arrow][1]:
                directions[name] = 'Up'
            else:
                directions[name] = 'Down'
    return directions

def prepare_padded_images(image, segment = True):
    '''
    Given a reaction image, create the set of padded images
    of all the contours. Like the pipeline, there is the option 
    to segment or just binarize the image, and the default is just 
    to segment
    
    @PARAM:
        - image: image we want to get the padded image from
        - segment: True if we want full segment (binarize, binary close)
                   False if we only want binarization. Default is True
    @RETURN:
        - padded_images: a Numpy Array of padded images that are in the shape of:
                        (num images, 500, 500, 1)
    '''
    # Segment the image if segment == true, else 
    # just binarize 
    if segment:
        image = segment_image(image)
    else:
        image = binarize(image)
    
    cnts1 = find_contours(image, 0)
    if len(cnts1) > 400:
        return [], [], [], []

    padded_images = [] 
    for i, cnt in enumerate(cnts1):
        padded = pad_image(cnts1, i)
        padded = segment_image(padded)
        padded_images.append(padded)

    padded_images = np.array(padded_images)
    padded_images = padded_images.reshape(padded_images.shape[0], 500, 500, 1)
    return cnts1, padded_images

def pipeline(image, doc_name, count, model = None, segment = True, verbose=1):
    '''
    Full extraction pipeline from reaction image to coordinates and direction of the 
    arrows (if there are any) in the image
    
    Steps:
        (1) If segment is true, binarize and segment the image, else just binarize it
        (2) Find all the contours in the image
        (3) Pad all the contours onto 500 x 500 images. 
        (4) Run the padded images through the model and get a confidence as to if it is
            an arrow or not
        (5) If the result is greater than .875, then we are confident it is an arrow
        (6) If it is abouve .575, but below .875, then we check the ratio of the contour height
            to the height of the image, if that is below .15, then we can assume it is skinny enought to be an arrow
        (7) WE also check the length, and if the length is less than .1 of the image size (vertical arrows), then we also add it
        (8) Take the isolate arrow contours and run them through to find their average, centroids, and directions. Create arrow object
        (9) Print time for extractions, and return all necessary info
    
    @PARAM:
        - image: if using on only one set of contours, can leave this none, however for 
                 full capability use this to allow for recurvie checking
        - doc_name: Name of the document
        - count: page count
        - model: model used for detection, if None then load default model
        - segment: Boolean telling us whether we want to segment. Defualt is true, false is used 
                   when we dont want binary closing
        - verbose: verbosity of information being printed out
                   - 0: No output
                   - 1: Print things out
    @RETURN:
        - info: List of Arrow objects that are returned (refer to Arrow Doc for info on what is included)
        - final_contours: Actual contours for the arrows that are on the page
        - centroids: Centroids for all the arrow objects that are present
        - directions: Directions for all the arrows
        
    '''
    
    if model == None:
        model = load_default_model()
        
    times = []
    times.append(time.time())
    
    cnts1, padded_images = prepare_padded_images(image, segment = segment)
    
    results = model.predict(padded_images)
    
    final_contours = []
    final_index = []
    conf = []

    for contour in range(len(results)):
        if results[contour] >= 0.6:
            height,_ = get_contour_height(cnts1[contour])
            length, _ = get_contour_length(cnts1[contour])
            if (height / image.shape[0]) <= 0.125:
                final_contours.append(cnts1[contour])
                final_index.append(contour)
                conf.append(results[contour])
                
    centroids = arrow_centroid(final_contours)
    averages = arrow_average(final_contours)
    directions = get_direction(final_contours, doc_name, count, model, image)
    info = []

    for i, arrow in enumerate(final_contours):
        arrow = Arrow(final_contours[i], centroids[i], averages[i], directions['Arrow ' + str(i + 1)])
        info.append(arrow.to_dictionary())
    
    if len(final_contours) == 0 and verbose == 1:
        if segment == True:
            info, arrow_contours, averages, directions = pipeline(image, doc_name, count, model = model, segment = False)
        if len(info) == 0:
            print('Label ' + str(count) + ' ' + doc_name + ': No Arrows were Identified')
            return [], cnts1, [], []
        else:
            return info, arrow_contours, averages, directions
        
    times.append(time.time())
    
    if verbose == 1:
        print('Label ' + str(count) + ' ' + doc_name + ' ' + str(len(final_contours)) + 
              " Arrows Extracted! Time Elapsed: %.2fs"%(times[-1] - times[-2]))
        
    return info, final_contours, centroids, directions