import random
import torch
import torchvision
import torchvision.transforms as transforms


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print(len(trainset))