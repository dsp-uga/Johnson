"""
This code is an attempt to solve the Image segmentation problem using Spectral Clustering from the scikit-learn package.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from numpy import array, zeros
from scipy.misc import imread,imsave
import scipy
from glob import glob
import cv2
from sklearn.preprocessing import normalize
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


#Input file
path1 = '/media/data4TbExt4/neuron/neurofinder.00.00.test/images'
path2 = '/media/data4TbExt4/neuron/neurofinder.00.01.test/images'

gfiles = sorted(glob("traindata/[0-9][0-9]_mask.tif"))
files = sorted(glob("traindata/[0-9][0-9].tif"))

#Get the training set
imgs = array([imread(f) for f in files])
imgs1 = imgs.sum(axis=0)
files1 = sorted(glob(path1+'/*.tiff'))
stuff = array([imread(f) for f in files])
s1 = stuff.sum(axis=0)
files2 = sorted(glob(path2+'/*.tiff'))
stuff2 = array([imread(f) for f in files])
s2 = stuff2.sum(axis=0)

#Final training set  = train + test files.
images = imgs1 + s1

# Create a mask
mask = images.astype(bool)

#Fit the images. Find 400 neurons in the data. 
graph = image.img_to_graph(img, mask=m)
graph.data = np.exp(-graph.data / graph.data.std())
labels = spectral_clustering(graph, n_clusters=400, eigen_solver='arpack')
label_im = -np.ones(m.shape)
label_im[m] = labels
plt.matshow(img)
plt.matshow(label_im)
plt.show()










