import os
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

import sklearn
from sklearn.cluster import KMeans

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

def define_model():
    ''' 
    Function that creates the Convolutional Neural Network that we need.
    Has 6 convolutional layers, and 2 flattened layers at the end for binary
    classification. Uses a binary crossentropy loss function, and a RMSprop 
    optimizer with a learning rate of 0.001. After 8 training epochs has an accuracy of 
    0.9427. Can definitly add more contours if needed later 
    
    Summary:
    Model: "Sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 498, 498, 16)      448       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 249, 249, 16)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 247, 247, 32)      4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 123, 123, 32)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 121, 121, 64)      18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 60, 60, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 58, 58, 64)        36928     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 29, 29, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 27, 27, 64)        36928     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 11, 11, 64)        36928     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1600)              0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               409856    
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 544,481
    Trainable params: 544,481
    Non-trainable params: 0
    _________________________________________________________________
    
    @RETURN: the compiled version of this model. Still needs fitting and training later 
             (8-10 epochs, should take about 20-25 min)
    
    '''
    
    model = keras.Sequential()
    # First Convolution
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 1), activity_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))
    # Second Convolution
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.01), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.25))
    # Third Convolution 
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.01),  activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.25))
    # The fourth convolution
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.01), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.25))
    # The fifth convolution
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.01), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.25))
    # Sixth Convolution (Wow)
    model.add(Conv2D(32, (3, 3), activity_regularizer=l2(0.01), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(keras.layers.Dropout(0.25))
    # Flatten and feed to a normal DNN
    model.add(Flatten())
    model.add(Dense(256, activity_regularizer=l2(0.01), activation= 'relu'))
    # Gives us value between 0 and 1 for what it thinks
    model.add(Dense(1, activity_regularizer=l2(0.01), activation = 'sigmoid'))
    
    # Use binary crossentropy loss function to create binary classifer
    loss_fn = 'binary_crossentropy'
    opt = RMSprop()
    model.compile(optimizer= opt, loss=loss_fn, metrics=['accuracy'])
    
    return model