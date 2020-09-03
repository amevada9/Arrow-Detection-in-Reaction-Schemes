# Arrow-Detection-in-Reaction-Schemes

## This is part of my SURF in Summer 2020. My role was to create a model that will eventually be able to identify and pull the location and direction of the the yield arrow to allow the computer to understand which way the reaction is proceeding. Due to the variety of shapes we can deal with, the majority of this repository is images and sets of data relating to arrows, as well as sample documents and such. 

### We used a combination of image processing from Scikit-Image, Deepfigures to extract reaction schemes, and finally, a large Convolutional Neural Network (CNN) to process the padded contours and identify whether a particular contour is an arrow or not. An example is shown below as to generally how it can work 

<p align="center">
  <img width=422 height=74 src="/other/CroppedPage10-1.png?raw=True">
</p> 

## Which is converted to this, where the image is processed and contours are identified:
<p align="center">
  <img width=341 height=68 src="/other/all_conts.png?raw=True">
</p>

## From which the Arrow can be isolated and the information returned and stored

<p align="center">
  <img width=341 height=68 src="/other/arrow_cont.png?raw=True">
</p>

## Updates
#### For the most current code and information, check out the Jupyter Notebook named SciKit_Arrow_Detection.ipynb. That has the latest information and code for the project
#### Also included in the root directory are scripts relating to image manipulation and augmentation. 

**August 31st:** created new model with Dropout Layers after each Convolutional/Pooling Layer, as well as adding L2 Activity Regularization to the Convolutional Layers to reduce over fitting. Caused top end training accuracy to drop from last push however there was a substanial increase in testing accuracy. As seen in the main Jupyter Notebook, testing accuracy is now over 90%, which puts it withing 3 points of training accuracy in a 10 Epoch training set. The graph below shows the new training and validation set accuracy.

<p align="center">
  <img width=631 height=365 src="/other/newNotOverfitGraph.png?raw=True">
</p>

**August 18th:** we are seeing CNN models that have a 96% Training accuracy and 75-80% accuracy when dealing with new processed reaction scheme images. However on similar training set we are seeing a lower accuracy (around 65%) showing that the model is currently overfitting the data/ 
