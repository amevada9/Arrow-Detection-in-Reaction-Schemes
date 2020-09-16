'''
Module that defines the model that we used

Model is not a class but rather a neural network

define_model() should be used if wanting to create a new model
'''

import os
import math 


import scipy
import numpy as np
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

def define_model():
    ''' 
    Function that creates the Convolutional Neural Network that we need.
    Has 6 convolutional layers, and 2 flattened layers at the end for binary
    classification. Uses a binary crossentropy loss function, and a RMSprop 
    optimizer with a learning rate of 0.01 (the default. After 15 training epochs has an accuracy of 
    0.93. 
    
    Used L2 Activity Regularization to prevent overfitting here
    
    Summary:
    
    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_12 (Conv2D)           (None, 498, 498, 16)      160       
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 249, 249, 16)      0         
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 249, 249, 16)      0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 247, 247, 32)      4640      
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 123, 123, 32)      0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 123, 123, 32)      0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 121, 121, 32)      9248      
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 60, 60, 32)        0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 60, 60, 32)        0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 58, 58, 32)        9248      
    _________________________________________________________________
    max_pooling2d_15 (MaxPooling (None, 29, 29, 32)        0         
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 29, 29, 32)        0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 27, 27, 32)        9248      
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 13, 13, 32)        0         
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 11, 11, 32)        9248      
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 5, 5, 32)          0         
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 5, 5, 32)          0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 800)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               205056    
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 247,105
    Trainable params: 247,105
    Non-trainable params: 0
    _________________________________________________________________
    
    @RETURN: the compiled version of this model. Still needs fitting and training later 
    
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
    # RMSprop optimizer, can also use SGD
    opt = RMSprop()
    model.compile(optimizer= opt, loss=loss_fn, metrics=['accuracy'])
    
    return model