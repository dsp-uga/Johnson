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
    Need more docs on params here.
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
