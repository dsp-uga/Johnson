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

#### Specifys testing frames
print('***** Inputting datasets **********************************************')
datasets = ['00.00.test/', '00.01.test/', '01.00.test/', '01.01.test/',
            '02.00.test/', '02.01.test/', '03.00.test/', '04.00.test/',
            '04.01.test/']
# datasets = ['neurofinder.00.00/']

for sample in datasets:

    print('***** Generating .tif movie from .tiff images *********************')
    files = sorted(glob(sample+'images/*.tiff'))
    images = [tifffile.imread(f) for f in files]
    name = os.path.dirname(sample.replace('.',''))
    tifffile.imsave(name+'.tif', np.array(images))

    print('***** Constrained NMF *********************************************')
    neurons = CNMF_PROCESS(name+'.tif')

    print('***** Saving selected neurons pixels location *********************')
    ss.save_npz(name+'_neurons.npz', neurons)

    print('***** Generates neurons coordinates *******************************')
    # A = ss.load_npz('A.npz')
    # A = np.array(sparse_matrix.todense())
