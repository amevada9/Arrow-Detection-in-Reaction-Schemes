'''
This script will allow us to input a reaction image and then identify the arrows that 
might be in the image and return its contour coordinates. 

'''

import os
import sys
import cv2
import math 
import imutils
import json
import numpy as np
import skimage as ski
from skimage import io

def convert_colors(image):
    '''
    Function that gives a clear interface to convert colors 
    from BGR in cv2 to RGB for normal use. Make sure to only 
    use this function if the image was loaded with CV2

    @PACKAGES:
        - Cv2 
    @PARAM:
        - image: a cv2 image that is in BGR
    @RETURN:
        - new_image that has been converted
    '''
    # Convert from BGR to RGB
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return new_image

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

def is_arrow(approx_list):
    ''' 
    This function was written to determine in a reaction scheme whether or not a given 
    contour is an arrow. We will do this by finding the ratio of the sides of a shape. Since 
    the contour finder will only give us 6 sided shapes, we can narrow it down even further by 
    finding the ratio of the shortest side to longest side. If this ratio is below an arbitarary value
    (we are setting it to 0.15 for now) then it is an arrow, indicating that there is very large side
    and a comparitively smaller side. This is indiciative of an arrow as other shapes will have ratios closer to 1
    given that the computer reaction drawings are more or less equilateral shapes. 
    
    @PARAM:
        - approx_list: list of points that are of the verticies of the contour
    @RETURN:
        - True if the ratio of shortest side to longest side is below 0.15 (it is most likely an arrow)
        - False if the ratio of shortest side to longest side is above 0.15
    '''
    diff_arr = []
    length_approx = len(approx_list)
    # find all the distance between consecutive verticies
    for i in range(length_approx):
        arr = []
        x_diff = abs(approx_list[(i + 1) % length_approx, 0, 0] - 
                     approx_list[i % length_approx, 0, 0]) 
        
        y_diff = abs(approx_list[(i + 1) % length_approx, 0, 1] - 
                     approx_list[i % length_approx, 0, 1])
        arr.append(x_diff)
        arr.append(y_diff)
        diff_arr.append(arr)
    length_arr = []
    # Find the magntiude of the differences 
    for vector in diff_arr:
        vec_mag = line_mag(vector)
        length_arr.append(vec_mag)
    # Find the max and min of the magntiudes and then find the ratio
    max_list = max(length_arr)
    min_list = min(length_arr)
    ratio = abs(min_list / max_list)
    if ratio <= 0.075:
        return True
    else:
        return False

def find_arrows(image):
    ''' 
    Gets and draws arrow contours from the image onto the image
    Need to work and making it save an image and/or getting the points so we 
    can get the centroid of the shape, thus knowing which side is products and which sides is reactants
    
    @PACKAGES:
        - CV2: needed for contour extraction
        - numpy: image array handling 
    @PARAM: 
        - image: a numpy array from a cv2 loaded image. Needs to be loaded by cv2 to allow for BGR conversion
                 also will be converted from here
    @RETURN:
        - the image with the drawn in contours. Later will make it return the contours so that we can get the 
          
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    list_contours = []
    _,threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY) 
    # Get all the contours on the image
    cnts,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for cnt in cnts: 
        area = cv2.contourArea(cnt) 
        # Shortlisting the regions based on there area. 
        if area > 200:  
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) 
            # Reaction arrows have 5, 6, 7, or 8 sides so we will use that to narrow it down
            # Also check if it is an arrow after that before drawing contours
            if (len(approx) in [5, 6, 7, 8]) and is_arrow(approx): 
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)
                list_contours.append(approx)
    return list_contours

def show_contours(image, list_contours):
    '''
    Draws the contours on the image that is given
    This function will be used when find_arrows only returns 
    the set of contours instead of the image
    
    @PACKAGES:
        - Cv2
    @PARAM:
        - image that will be turned to grayscale of that we have the contours of
        - list_contours: lsit of the contours from find_arrows
    '''
    for cnts in list_contours:
        cv2.drawContours(image, [cnts], 0, (0, 255, 0), 5)
    return image

def show_all_contours(image):
    '''
    Draws all the  contours on the image that is given

    @PACKAGES:
        - CV2: needed for contour extraction
        - numpy: image array handling 
    @PARAM: 
        - image: a numpy array from a cv2 loaded image. Needs to be loaded by cv2 to allow for BGR conversion
                 also will be converted from here
    @RETURN:
        - the image with the drawn in contours. 
    
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY) 
    cnts,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    image = show_contours(image, cnts)
    return image

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
    x_values = []
    y_values = []
    # For each arrow in contour array
    for i in range(len(arrow_contours)):
        # For each list of points in contour
        for j in range(len(arrow_contours[i])):
            # For each tuple of points 
            for k in range(len(arrow_contours[i][j])):
                # For each coordinate 
                for l in range(len(arrow_contours[i][j][k])):
                    # Element 0 is the x, element 1 is the y value
                    if l == 0:
                        x_values.append(arrow_contours[i][j][k][l])
                    else:
                        y_values.append(arrow_contours[i][j][k][l])
        max_x_val = max(x_values)
        min_x_val = min(x_values)
        average_x = (max_x_val + min_x_val) * 0.5
        
        max_y_val = max(y_values)
        min_y_val = min(y_values)
        average_y = (max_y_val + min_y_val) * 0.5
        centroids.append((average_x, average_y))
        x_values = []
        y_values = []     
    return centroids 


