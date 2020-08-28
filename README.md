# Arrow-Detection-in-Reaction-Schemes

## This is part of my SURF in Summer 2020. My role was to create a model that will eventually be able to identify and pull the location and direction of the the yield arrow to allow the computer to understand which way the reaction is proceeding. Due to the variety of shapes we can deal with, the majority of this repository is images and sets of data relating to arrows, as well as sample documents and such. An example is shown below as to generally how it can work 

![Original Reaction Image](https://github.com/amevada9/Arrow-Detection-in-Reaction-Schemes/tree/master/other/CroppedPage10-1.png?raw=True)
## Which is converted to this:
![Image with Contours](https://github.com/amevada9/Arrow-Detection-in-Reaction-Schemes/tree/master/other/Screen-Shot-2020-08-28-at-6.28.46-PM.png?raw=True)
# Where the Arrow can be isolated and the information returned and stored
![Image with just Arrow](https://github.com/amevada9/Arrow-Detection-in-Reaction-Schemes/tree/master/other/Screen-Shot-2020-08-28-at-6.29.06-PM.png?raw=True)

### Updates

For the most current information, checkout the Jupyter Notebook named SciKit_Arrow_Detection.ipynb. That has the latest information and code for the project
Also included are scripts relating to image manipulation and augmentation. As of Aug 18th, we are seeing models that have a 96% Training accuracy and 75-80% accuracy when dealing with new processed reaction scheme images. However on similar training set we are seeing a lower accuracy (around 65%) showing that the model is not correctly fitting the data. 
