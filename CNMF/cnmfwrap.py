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
from .cnmf_process import CNMF_PROCESS, tocoord
import scipy.sparse as ss
from scipy.sparse import csr_matrix
import json

def main(setName = ['00.00', '00.01','01.00','01.01','02.00','02.01','03.00','04.00','04.01'],
            _k = 1000, _g = 5, _merge = 0.8):
    """
    Main method for neuron segmentation using CNMF approach.
    """
    prediction = []
    os.makedirs('figures/')
    os.makedirs('pixels_npz/')
    os.makedirs('predictions/')
    for data in setName:

        print('***** Generating .tif movie from .tiff images *****************')
        files = sorted(glob('neurofinder.'+data+'.test/images/*.tiff'))
        images = [tifffile.imread(f) for f in files]
        name = data + '.test'
        tifffile.imsave(name+'.tif', np.array(images))

        print('***** Constrained NMF *****************************************')
        pixels, dims = CNMF_PROCESS(name+'.tif', _k, _g, _merge) #scipy sparse matrix

        print('***** Saving selected neurons pixels location *****************')
        ss.save_npz('pixels_npz/pixels_'+name+'.npz', pixels)

        print('***** Generates neurons coordinates ***************************')
        neurons_pixels = np.array(pixels.todense()) # sparse > dense > numpy array
        regions = [{"coordinates": tocoord(pix,dims)} for pix in list(neurons_pixels.T)]
        result = {"dataset": name, "regions": regions}

        print('***** Saving individual sparse matrix *************************')
        json.dump(result, open("predictions/prediction_"+name+".json", "w"))
        prediction.append(result)
        os.remove(name+'.tif')

    print('***** Saving predictions ******************************************')
    json.dump(prediction, open("prediction.json", "w"))

    print('***** Mission Complete! *******************************************')
