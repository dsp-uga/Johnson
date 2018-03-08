"""
Main file for CNMF using CaImAn package (https://github.com/flatironinstitute/CaImAn)
File structure credits to @cbarrick and @zachdj
"""

import sys
import argparse
import cv2
import tensorflow as tf
import keras
from . import cnmfwrap

def info():
    """
    System information
    """
    print('Python version:', sys.version)
    print('Tensorflow version:', tf.__version__)
    print('Keras version:', keras.__version__)
    print('OpenCV version:', cv2.__version__)

def main():
    parser = argparse.ArgumentParser(
        description = 'Neuron Segmentation',
        argument_default = argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # print information
    op = options.add_parser('info', description = 'print system information')
    op.set_defaults(func = info)

    # Optional args
    parser.add_argument('-setName',
        help = 'Folder names of testing images [Default: all 9 testing folders]')
    parser.add_argument('-_k', default = 1000, type = int,
        help = 'K value, number of neurons expected, [Default: K=1000]')
    parser.add_argument('-_g', default = 5, type = int,
        help = 'gSig value, expected half size of neurons, [Default: g=5]')
    parser.add_argument('-_merge', default = 0.8, type = float,
        help = 'merging threshold, max correlation allowed, [Default: merge=0.8]')
    parser.set_defaults(func = cnmfwrap.main)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
