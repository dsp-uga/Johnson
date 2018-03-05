import argparse
import johnson
import tensorflow as tf

def info(args):
    '''
    Print system info.
    '''
    print ('Python version:')
    print (sys.version)
    print ('Tensorflow version:')
    print (tf.__version__)
    print ('Tensorflow GPU Support')
    print (tf.test.gpu_device_name())

def main():
    parser = argparse.ArgumentParser(
        description='Neuron Segmentation',
        argument_default=argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # Print info
    op = options.add_parser('info', description='print system info')
    op.set_defaults(func = info)

    # johnson nmf 

    images = load('_', 'local')
    # print (images)
    print(len(images), len(images[1]))
    print(images[1].shape)
    images = grayScale(images)
    print(images[1].shape)
    print(len(images), len(images[1]))
    print(images[1])
    images = medianFilter(images)
    print(images[1].shape)
    print(images[1])
