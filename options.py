import argparse
import const


def Options():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--datasets', dest='dataset',choices=const.DATASETS,
                    default='CIFAR10',type=str)
    parser.add_argument('--models',dest='model',choices=const.MODELS,
                    default='resnet18',type=str,
                    help='datasets')
    parser.add_argument('--bs',dest='batch_size',choices=const.BATCHSIZE,
                    default=64,type=int,
                    help='batch_size')
    parser.add_argument('--epochs',dest='epochs',
                    default=30,type=int,
                    help='epochs of training')
    parser.add_argument('--lr',dest='lr',
                    default=1e-2,type=float,
                    help='learning rate')
    parser.add_argument('--sp',dest='shuffle_percentage',
                    default=0.05,type=float,
                    help='percentage of training sample to shuffle labels')
    args = parser.parse_args()
    return args
