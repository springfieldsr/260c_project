import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
import random
import string
import os
import json
from os import path
from torch.utils.data import DataLoader, Subset
from const import EVAL_INTERVAL
# from config import Config

class ShuffledDataset():
    def __init__(self, dataset_name, root, shuffle_percentage, transform=None, train=True, download=True):
        """
        Input:
        dataset_name
            string of desired dataset name
        root
            string of data download destination
        shuffle_percentage
            float between [0, 1] to specify what fraction of train data to have incorrect labels
        train
            bool to specify wheter the dataset is for train or test
        download
            bool to specify whether to download the dataset specified by dataset_name
        transform
            operations to perform transformation on the dataset
        """
        if dataset_name == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
            self.label_len = 10
        elif dataset_name == "CIFAR100":
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
            self.label_len = 100
        else:
            raise NotImplementedError

        # Force the test set to be intact
        percentage = train * shuffle_percentage
        dataset_len = len(self.dataset)
        indices_to_shuffle = random.sample(range(dataset_len), int(percentage * dataset_len))
        self._create_shuffle_mapping(indices_to_shuffle)

    def _create_shuffle_mapping(self, indices):
        """
        Input:
        indices
            list of int to specify which samples to shuffle label
        """
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

    def __len__(self):
        return len(self.dataset)


def eval(model, test_dataloader, device):
    """
    Input:
    model
        pytorch model object to eval
    test_dataloader
        pytorch dataloader
    device
        string of either 'cuda' or 'cpu'
    
    Return:
        float accuracy on the validation set
    """
    model.eval()

    match_count, total_count = 0, 0
    for (X, y) in test_dataloader:
        with torch.no_grad:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            match_count += torch.sum(torch.argmax(logits, dim=1) == y)
            total_count += len(X)

    return match_count / total_count


def train(model, epoch, train_dataset, test_dataloader, device, args):
    """
    Input:
    model
        pytorch model object to train
    epoch
        int number of total training epochs
    train_dataset
        dataset object which satisfies pytorch dataset format
    test_dataloader
        pytorch dataloader
    device
        string of either 'cuda' or 'cpu'
    """

    # Important! Set the reduction to be None which allows single sample loss recording
    criterion = nn.CrossEntropyLoss(reduction='none')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(epoch):
        model.train()

        train_loss = 0
        batch_size = args.batch_size # TODO: this will modified by new methods later
        # Manually shuffle the training dataset for loss recording later
        feed_indices = torch.randperm(len(train_dataset)).tolist()
        shuffled_dataset = Subset(train_dataset, feed_indices)
        dataloader = DataLoader(shuffled_dataset, batch_size, shuffle=False)

        with tqdm.tqdm(total=len(dataloader)*batch_size, unit='it', unit_scale=True, unit_divisor=1000) as pbar:
            sum_acc = 0
            for step, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits = model(X)
                loss_list = criterion(logits, y)
                acc = torch.sum(torch.argmax(logits, dim=1) == y).item()
                loss_reduced = torch.mean(loss_list)
                loss_reduced.backward()
                opt.step()

                # progress bar counter
                train_loss += loss_reduced.item()
                pbar.update(batch_size)
                sum_acc += acc
                pbar.set_postfix(loss=loss_reduced.item(),
                                     acc=sum_acc/((step+1)))
        print("Epoch {} - Training loss: {}".format(e, train_loss/len(dataloader)))
        if (1 + e) % EVAL_INTERVAL == 0:
            validation_accuracy = eval(model, test_dataloader, device)
            print(f"Test accuracy at epoch {e}: {validation_accuracy:.4f}")
    print("Training Finished...")



def SaveEnvironment():

    pass

def GenerateEnvironment(args):
    seeds = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    result_dir = './results'
    if not path.isdir(result_dir):
        os.mkdir(result_dir)
    expr_path = path.join(result_dir,seeds).replace("\\","/")
    args_path = path.join(expr_path,'args.txt').replace("\\","/")
    os.mkdir(expr_path)
    print('Create enviornment at : {}'.format(expr_path))
    with open(args_path,'w') as F:
        DumpOptionsToFile(args, F)

def DumpOptionsToFile(args,fp):
    d = vars(args)
    for key,value in d.items():
        if type(value) == str:
            fp.write('{} = "{}"\n'.format(key,value))
        else:
            fp.write('{} = {}\n'.format(key,value))