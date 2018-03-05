'''
This file includes preprocessing calls for the images.
This preprocessing file provides the functions of loading and transforming
image globs. Say, to load an image glob on Caesar server, just call:
    images = load('01.01', 'caesar')
To preprocess the images, we can just apply the functions on the loaded image glob:
For instance,
    images = grayScale(images)
    images = medianFilter(images)
'''

import os
import cv2
import numpy as np
from glob import glob
import skimage
import scipy.ndimage as ndimg
import skimage.filters as filters
import skimage.morphology as morphology

def load(setName, base):
    '''
    Reading all images from a file directory. argument neuroset indicates the name
    of the dataset. For instance, 00.00, 00.01, 02.00, etc.
    Return a matrix instantiating all images
    '''
    # get the full name of the path;
    if base == 'local':
        dirname = "/Users/yuanmingshi/downloads/johnson/neurofinder.01.01/images/"
    else:
        dirname = '/media/data2TB/jeremyshi/neurofinder.{}.test/images/'.format(setName)
    # load images using opencv
    images = [cv2.imread(file) for file in glob('{}*.tiff'.format(dirname))]
    print (len(images))
    return images

def grayScale(images):
    '''
    Change the images to grayscale
    Input: an array of image arrays.
    Output: an array of image arrays in gray scale.
    '''
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]  # convert to greyscale
    return images

def medianFilter(images):
    '''
    Add a median filter into a image using _median filters_ and _morphology_ in skimage.
    There's a helper function for it because we need to apply the filter on every img.
    Input: an array of image arrays.
    Output: an array of image arrays after the median filter.
    The method is adopted from:
    https://github.com/eds-uga/cbio4835-sp17/blob/master/lectures/Lecture23.ipynb
    '''
    images = [_median(image) for image in images]
    return images

def _median(img):
    '''
    helper function for medianFilter method.
    Input: an image.
    Output: an image after the median filter.
    '''
    # i is the result of only keeping the values higher than the mean in the image
    i = img > np.median(img)
    # Morphology has options on different shapes and values. In practice, square(3) or square(4) perform well.
    bin_median = filters.median(i, morphology.square(3))
    return bin_median
