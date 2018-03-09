'''
This file includes the Unet code: Using the package Tensorflow Unet.
This is a framework used to build Unet architectures. 
The package provides the following modules that we have used:
unet Module: Creates a Unet architecture by specifying the number of layers, the optimizer function, the dropout (optional), the
number of classes, and the number of features 
train Module: Train your unet module, by specfying the number of epochs, number of iterations.
predict Module: Predict on a given query image, using the trained module.
'''

import sys
import cv2
from PIL import Image
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from numpy import array, zeros, savetxt
from scipy.misc import imread,imsave
import scipy
from glob import glob
from tf_unet import image_gen, image_util, unet, image_util

def main(trainPath='traindata', testPath='/media/data4TbExt4/neuron/neurofinder.00.00.test/',
        layerNum=4, features=64, bsize=4, opm='adam',
        iter=120, ep=220, display=60):
    '''
    Driver function. Provides the required inputs for all the modules of the tf_unet package.
    Input:
    trainPath: The path to the training data. This is the path to the directory. All .tif training images must be stored in this directory.
    testPath: The path to the test data. All _mask.tif files must be stored in this directory.
    layerNum: Number of layers in the Unet architecture.
    features: Length of the feature map in the Unet architecture.
    bsize: The batch size for the input.
    opm: The type of optimizer used.
    iter: The number of iterations during training. 
    ep: Number of epochs to be used for training. 
    display: This is used during display. Number of epochs after which the accuracy should be displayed.
    
    '''
    if sys.version_info[0] >= 3:
        raise ("Must be using Python 2.7!")
    if (tf.test.gpu_device_name()):
        print('GPU detected')
    else:
        print('No GPU!')

    # Train using Unet
    data_provider = image_util.ImageDataProvider('{}/*.tif'.format(trainPath))
    net = unet.Unet(channels=1, n_class=2,
                    layers=layerNum, features_root=features)
    trainer = unet.Trainer(net, batch_size=bsize, optimizer=opm)
    path = trainer.train(data_provider, "./unet_trained",
                        training_iters=iter, epochs=ep, display_step=display)

    # Test using the trained result
    path = '{}/images'.format(testPath)
    files = sorted(glob(path+'/*.tiff'))
    testimg = array([imread(f) for f in files])
    concatArray = testimg.sum(axis=0)
    print('The dimension of testing image is {}'.format(concatArray.shape))
    plt.imshow(concatArray)
    concatArray = concatArray.reshape( (1,) + s.shape + (1,))
    prediction = net.predict("./unet_trained/model.cpkt", concatArray)
    prediction = prediction[0, :, :, 1]
    print('The output dimension is {}'.format(prediction.shape))
    savetxt('predictedArray.txt', prediction)

    # Plot the results
    fig, ax = plt.subplots(1, 2 , figsize=(12,5))
    ax[0].imshow(s[0,...,0], cmap='gray')
    ax[1].imshow(prediction, aspect="auto",cmap='gray')
    ax[0].set_title("Input")
    ax[1].set_title("Prediction")
    plt.show()

if __name__ == '__main__':
    main()
