import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from config import Config

class ShuffledDataset():
    def __init__(self, mode, root, shuffle_percentage, train=True, download=True, transform=transforms.ToTensor()):
        if mode == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
            self.label_len = 10
        elif mode == "CIFAR100":
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
            self.label_len = 100
        else:
            raise NotImplementedError
        
        percentage = train * shuffle_percentage
        dataset_len = len(self.dataset)
        indices_to_shuffle = random.sample(range(dataset_len), int(percentage * dataset_len))
        self._create_shuffle_mapping(indices_to_shuffle)
    
    def _create_shuffle_mapping(self, indices):
        self.mapping = {}
        for index in indices:
            _, label = self.dataset[index]
            label_range = list(range(self.label_len))
            label_range.remove(label)
            new_label = random.choice(label_range)
            self.mapping[index] = new_label

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        label = self.mapping.get(index, label)
        return (sample, label)
    
    def get_shuffle_mapping(self):
        return self.mapping


