'''
Arrow working methods that we can use to get information, orientations,
and directions from as well as basic math operations. Specilized for the contours in 
Scikit-Image
'''

import os
import cv2
import skimage
import math 
import imutils
import copy
import logging
import numpy as np


def line_mag(arr):
    '''
    Simple function that gets the magntide of a "vector", or 2D coordinate point that is given
    Generalized for all 2D vectors and will be used for finding distance of points on the contour
    
    Formula: dist = sqrt(arr[0]^2 + arr[1]^2) or x^2 + y^2 = (dist)^2
    
    @PACKAGES:
        - math: python math function is used to calculate the magntiude here
    @PARAM:
        - arr: a 2D array (length will be checked) which we want to find the magntiude of 
    @RETURN:
        - If length of the array is 2, then we get the length of the array, otherwise a printed error
    '''
    if len(arr) != 2:
        print('Input of Mag does not have expected length of 2')
        return 1
    else:
        return math.sqrt(arr[0] * arr[0] + arr[1] * arr[1])

def arrow_average(arrow_contours):
    '''
    Find the weighted average of the points in a set of arrows
    This allows us to get the the orientation of the arrow. 
    This function is general and we can use the results to find the orientation 
    in another function based on rules we assign
    
    @PARAM:
        - arrow_contours: the contours of the arrows extracted from find_arrow()
                          we will find averages of all the arrows to keep simplicity
    @RETURN:
        - averages: list of tuples that are the average of all the points in an arrow 
    '''
    averages = []
    point_count = 0
    x_mean = 0
    y_mean = 0
    # For each arrow in contour array
    for i in range(len(arrow_contours)):
        point_count = 0
        x_mean = 0
        y_mean = 0
        # For each list of points in contour
        for j in range(len(arrow_contours[i])):
                x_mean += arrow_contours[i][j][1]
                y_mean += arrow_contours[i][j][0]
                point_count += 1
        x_mean = x_mean / point_count
        y_mean = y_mean / point_count
        averages.append((x_mean, y_mean))
    return averages

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
                # For each coordinate 
                    # Element 0 is the x, element 1 is the y value
                    x_values.append(arrow_contours[i][j][1])
                    y_values.append(arrow_contours[i][j][0])
        max_x_val = max(x_values)
        min_x_val = min(x_values)
        average_x = (max_x_val + min_x_val) * 0.5
        
        max_y_val = max(y_values)
        min_y_val = min(y_values)
        average_y = (max_y_val + min_y_val) * 0.5
        centroids.append((average_x, average_y))
    return centroids 

def get_orientation(arrow_contours):
    '''
    A function that gets whether the arrow is vertical or horizontal
    Assuming normally ratioed arrow, if the longer side of the arrow is
    the x side, it is horizontal and if the y side is longer then we know it
    is horizontal. This is an assumption and will need tests
    
    @PARAM:
        - arrow_contours: the contours of the arrows extracted from find_arrow()
    @RETURN:
        - A array of strings that tells how each one of the arrows is oriented
        Choices: 
            - "Horizontal"
            - "Vertical "
    '''
    orientations = []
    # For each arrow in contour array
    for i in range(len(arrow_contours)):
        x_values = []
        y_values = []
        # For each list of points in contour
        for j in range(len(arrow_contours[i])):
                # For each coordinate 
                for k in range(len(arrow_contours[i][j])):
                    # Element 0 is the x, element 1 is the y value
                        x_values.append(arrow_contours[i][j][1])
                        y_values.append(arrow_contours[i][j][0])
        max_x_val = max(x_values)
        min_x_val = min(x_values)
        diff_x = abs(max_x_val - min_x_val)
        
        max_y_val = max(y_values)
        min_y_val = min(y_values)
        diff_y = abs(max_y_val - min_y_val)
        
        # If the differences in x are larger than y
        # then we know that it is horizontal. Otherwise
        # it is vertical
        # We will assume no diagonal or bent arrows yet 
        if diff_x > diff_y:
            orientations.append('Horizontal')
        else:
            orientations.append('Vertical')
    return orientations

def get_contour_height(contour):
    '''
    Function that gets the linear height from min to max y-val of a contour
    Will be used to differentiate between moleucules and arrows in less 
    confident idenification

    @PACKAGES:
        - Scikit-Image: image processing library that allows for the extraction
                        of contours
    @PARAM:
        - contour: a single contour that is extracted from the image: Should be an 
                   indexed contour, so do not put list of contours in, put a single 
                   contour
                   Ex: get_contour_height(all_contours[2])
    @RETURN:
        - height: Unsigned value of the height of the contour. Difference between max y and min y
    '''
    y_vals = []
    for i, pair in enumerate(contour):
        y_vals.append(contour[i][0])
    max_y = max(y_vals)
    idx_maxY = y_vals.index(max_y)
    min_y = min(y_vals)
    idx_minY = y_vals.index(min_y)
    
    return abs(max_y - min_y), (idx_minY, idx_maxY)

def get_contour_length(contour):
    '''
    Function that gets the linear length from min to max x-val of a contour
    Will be used to differentiate between different sized contours
    @PACKAGES:
        - Scikit-Image: image processing library that allows for the extraction
                        of contours
    @PARAM:
        - contour: a single contour that is extracted from the image: Should be an 
                   indexed contour, so do not put list of contours in, put a single 
                   contour
                   Ex: get_contour_height(all_contours[2])
    @RETURN:
        - length: Unsigned value of the height of the contour. Difference between max y and min y
    '''
    x_vals = []
    for i, pair in enumerate(contour):
        x_vals.append(contour[i][1])
    max_x = max(x_vals)
    idx_maxX = x_vals.index(max_x)
    min_x = min(x_vals)
    idx_minX = x_vals.index(min_x)
    
    return abs(max_x - min_x), (idx_minX, idx_maxX)

def distance(point1, point2):
    '''
    Function that can find the distance between 2 points 
    on the Cartesian coordinate plane. Useful for finding 
    distances between two points and can be used to establish
    distance between contour centers
    
    @PACKAGES:
        - math: python math function is used to calculate the magntiude here
    @PARAM:
        - point1: a point that has two dimensions (order does not matter)
        - point1: another point that has two dimensions (order does not matter)
    @RETURN:
        - If length of the array is 2, then we get the distance between the points 
          otherwise a printed error
    '''
    
    x_diff = abs(point1[0] - point2[0])
    y_diff = abs(point1[1] - point2[1])
    vector = [x,diff, y_diff]
    return line_mag(vector)
