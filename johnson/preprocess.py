'''
This file includes preprocessing calls for the images.
'''

import os
import cv2
import numpy as np
from glob import glob
import skimage
import scipy.ndimage as ndimg
import skimage.filters as filters
import skimage.morphology as morphology

def load(setName):
    '''
    Reading all images from a file directory. argument neuroset indicates the name
    of the dataset. For instance, 00.00, 00.01, 02.00, etc.
    Return a matrix instantiating all images
    '''
    # get the full name of the path; (Maybe not hard code it?)
    dirname = '/media/data2TB/jeremyshi/neurofinder.{}.test/images/'.format(setName)
    # load images using opencv
    images = [cv2.imread(file) for file in glob('{}*.tiff'.format(dirname))]
    # change the images to grayscale
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]  # convert to greyscale

    return images
    
def medianFilter(img):
    '''
    Add a median filter into a image using _median filters_ and _morphology_ in skimage.
    Input: an image array.
    Output: an image array after the median filter.
    The method is adopted from:
    https://github.com/eds-uga/cbio4835-sp17/blob/master/lectures/Lecture23.ipynb
    '''
    # i is the result of only keeping the values higher than the mean in the image
    i = img > np.mean(img)
    # Morphology has options on different shapes and values. In practice, square(3) or square(4) perform well.
    bin_median = filters.median(i, morphology.square(3))
    return bin_median
