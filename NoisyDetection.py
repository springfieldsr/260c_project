from h11 import Data
from utils import *

mean = torch.tensor([0.4914, 0.4822, 0.4465])
std = torch.tensor([0.2009, 0.2009, 0.2009])
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean = mean, std = std)])

cifar10_train_data = ShuffledDataset('CIFAR10', './data', 0.05, train=True, transform=transform)
cifar10_test_data = ShuffledDataset('CIFAR10', './data', 0, train=False, transform=transform)

cifar10_train_dl = DataLoader(cifar10_train_data, batch_size=Config.batch_size, shuffle=True)
cifar10_test_dl = DataLoader(cifar10_test_data, batch_size=Config.batch_size)