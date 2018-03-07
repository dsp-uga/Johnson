'''
Main file for UNET. The structure is largely inspired from
https://github.com/dsp-uga/elizabeth/blob/master/elizabeth/__main__.py
Credits to @cbarrick and @zachdj
'''

import sys
import argparse
import UNET
import tensorflow as tf

def info():
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
    op.set_defaults(func=info)

    # UNET
    parser.add_argument('--trainPath', help='the folder of training images')
    parser.add_argument('--testPath', help='the folder of testing images')
    parser.add_argument('--layerNum', help='the number of layers in the Unet. Suggested 3, 4, or 5')
    parser.add_argument('--features', help='the number of features in the Unet')
    parser.add_argument('--bsize', help='training batch size; suggested 2 or 4')
    parser.add_argument('--opm', help='optimizer in Unet; default adam; alternatively you can use momentum')
    parser.add_argument('--iter', help='training iterations in one epoch of all data')
    parser.add_argument('--ep', help='the number of epoches in training')
    parser.add_argument('--display', help='the number of steps per display')
    parser.set_defaults(func=UNET.unet.main)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
