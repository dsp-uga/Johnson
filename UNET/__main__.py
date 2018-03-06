

def main():
    parser = argparse.ArgumentParser(
        description='Neuron Segmentation',
        argument_default=argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # Print info
    op = options.add_parser('info', description='print system info')
    op.set_defaults(func = info)

    # johnson unet
    op = options.add_parser('unet', description='Unet Image Segmentation', argument_default=argparse.SUPPRESS)
    op.add_argument('--trainPath', help='the folder of training images')
    op.add_argument('--testPath', help='the folder of testing images')
    op.add_argument('--layerNum', help='the number of layers in the Unet. Suggested 3, 4, or 5')
    op.add_argument('--features', help='the number of features in the Unet')
    op.add_argument('--bsize', help='training batch size; suggested 2 or 4')
    op.add_argument('--opm', help='optimizer in Unet; default adam; alternatively you can use momentum')
    op.add_argument('--iter', help='training iterations in one epoch of all data')
    op.add_argument('--ep', help='the number of epoches in training')
    op.add_argument('--display', help='the number of steps per display')
    op.set_defaults(func=johnson.unet.main)
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()
