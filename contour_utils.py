import os
import numpy as np
import cv2

def split_contours(input_contours):
    ''' 
    Method to pad the contours and allow for multiple contours to be passed through 
    the pad image function one at a time. Need this method because of how weird the 
    OpenCV outputs contours 
    
    @PARAM:
        - input_contours: contours that we want to split up. Need to be at least length 1
    @RETURN:
        - if the len(input_contours) > 1, then we return a new list with contours split up
        - Else we return just return the original list for simplicity. 
    
    Note: if input_contours do not have the needed number of contours then the function will
          raise a value error. 
    '''
    if len(input_contours) == 0:
        raise ValueError("Input Contours Need at least 1 contour in them")
    elif len(input_contours) > 1:
        # Return list is the list of all contours
        return_list = []
        for cnt in input_contours:
            # New list is just compartmentalizing all the singular contours
            # Each one should have just length 1
            new_list = []
            new_list.append(cnt)
            return_list.append(new_list)
        return np.array(return_list)
    return np.array(input_contours)

def show_contours(image, list_contours):
    '''
    Draws the contours on the image that is given
    This function will be used when find_arrows only returns 
    the set of contours instead of the image
    
    @PACKAGES:
        - Cv2
    @PARAM:
        - image turned to grayscale of that we have the contours of
        - list_contours: lsit of the contours from find_arrows
    @RETURN:
        - image: the new image with the drawn on contours
    '''
    for cnts in list_contours:
        approx = cv2.approxPolyDP(cnts, 0.01 * cv2.arcLength(cnt, True), True) 
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
    new_image = np.copy(image)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(new_image, 110, 255, cv2.THRESH_BINARY) 
    cnts,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    image = show_contours(image, cnts)
    return image

def find_all_contours(image):
    '''
    Finds all the  contours on the image that is given
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY) 
    cnts,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    return cnts