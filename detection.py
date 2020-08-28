
import os
import numpy as np
import cv2
import math 
import imutils
import copy


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

