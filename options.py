import argparse
import const


def Options():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--dataset', dest='dataset',choices=const.DATASETS,
                    default='CIFAR10',type=str)
    parser.add_argument('--model',dest='model',choices=const.MODELS,
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
    parser.add_argument('--k',dest='top_k',
                    default=0.05,type=float,
                    help='track top k percentage of samples with highest loss')
    parser.add_argument('--ls',dest='label_shuffle',
                    default=True,type=bool,
                    help='wheter to shuffle labels of k percent of training samples')
    parser.add_argument('--rp',dest='recording_point',
                    default=0.8,type=float,
                    help='begin loss recoding at rp * optimal_total_epochs')
    args = parser.parse_args()
    return args
