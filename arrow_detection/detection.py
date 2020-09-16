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
    model = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'notOverfittedModel2'))
    return model


def get_direction(arrow_contours, doc_name, count, model, image = None):
    '''
    Gets the direction of an arrow given its contour of points. Will be important for getting
    the products and the reactants of the arrow. Can use the centroid and direction of the 
    arrow to see whether the compunds on either side are products, reactants or intermediates
    
    @PARAM:
        - arrow_contours: the contours of the arrows extracted from find_arrow()
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
            if height > 25 and image != None:
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


def pipeline(image, doc_name, count, model = None, segment = True, verbose=1):
    if model == None:
        model = load_default_model()
        
    times = []
    times.append(time.time())
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
    #padded_images = padded_images / 255.0
    padded_images = padded_images.reshape(padded_images.shape[0], 500, 500, 1)
    results = model.predict(padded_images)
    
    final_contours = []
    final_index = []
    conf = []

    for contour in range(len(results)):
        if results[contour] >= 0.875:
            final_contours.append(cnts1[contour])
            final_index.append(contour)
            conf.append(results[contour])
    
        elif 0.575 < results[contour] < 0.875:
            height,_ = get_contour_height(cnts1[contour])
            length, _ = get_contour_length(cnts1[contour])
            if (height / image.shape[0]) <= 0.15:
                final_contours.append(cnts1[contour])
                final_index.append(contour)
                conf.append(results[contour])
    
    if len(final_contours) == 0 and verbose == 1:
        print('Label ' + str(count) + ' ' + doc_name + ': No Arrows were Identified')
        return [], cnts1, [], []
    
    centroids = arrow_centroid(final_contours)
    averages = arrow_average(final_contours)
    directions = get_direction(final_contours, doc_name, count, model, image)
    info = []
    
    for i, arrow in enumerate(final_contours):
        #print(type(directions))
        arrow = Arrow(final_contours[i], centroids[i], averages[i], directions['Arrow ' + str(i + 1)])
        info.append(arrow.to_dictionary())

    times.append(time.time())
    if verbose == 1:
        print('Label ' + str(count) + ' ' + doc_name + ' ' + str(len(final_contours)) + 
              " Arrows Extracted! Time Elapsed: %.2fs"%(times[-1] - times[-2]))
    return info, final_contours, averages, directions