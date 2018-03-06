"""
Generates .tif movie from .tiff images for each frame.
Runs CNMF extraction algorithm with CaImAn (https://github.com/flatironinstitute/CaImAn)
Evaluates the selected components and Plots the neurons on the original plot.
Generates neurons' coordinates as format.
"""

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tifffile
import os
from cnmf_process import CNMF_PROCESS
import scipy.sparse as ss
import json

def tocoord(pixels_array):
    """
    Transforms pixels array into original shape of image,
    Returns the coordinates of existed neurons.
    """
    coordinates = np.argwhere(pixels_array.reshape(512, 512)!=0)
    return coordinates.tolist()

#### Specifys testing frames
print('***** Inputting datasets **********************************************')
datasets = ['00.00.test', '00.01.test', '01.00.test', '01.01.test',
            '02.00.test', '02.01.test', '03.00.test', '04.00.test',
            '04.01.test']
# datasets = ['00.00.test', '00.01.test']

prediction = []
os.makedirs('figures/')
os.makedirs('pixels_npz/')
os.makedirs('predictions/')
for sample in datasets:

    print('***** Generating .tif movie from .tiff images *********************')
    files = sorted(glob('neurofinder.'+sample+'/images/*.tiff'))
    images = [tifffile.imread(f) for f in files]
    name = sample.replace('.','')
    tifffile.imsave(name+'.tif', np.array(images))

    print('***** Constrained NMF *********************************************')
    pixels = CNMF_PROCESS(name+'.tif') #scipy sparse matrix

    print('***** Saving selected neurons pixels location *********************')
    ss.save_npz('pixels_npz/pixels_'+name+'.npz', pixels)

    print('***** Generates neurons coordinates *******************************')
    neurons_pixels = np.array(pixels.todense()) # sparse -> dense -> numpy array
    regions = [{"coordinates": tocoord(pix)} for pix in list(neurons_pixels.T)]
    result = {"dataset": sample, "regions": regions}
    json.dump(result, open("predictions/prediction_"+name+".json", "w"))
    prediction.append(result)

print('***** Saving predictions **********************************************')
json.dump(prediction, open("prediction.json", "w"))

print('***** Mission Complete! ***********************************************')
