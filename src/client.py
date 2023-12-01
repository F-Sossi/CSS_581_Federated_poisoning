from collections import OrderedDict

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import warnings

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def random_label_flip(labels):
    labels = torch.randint(0, 10, labels.shape)
    return labels


def constant_label_flip(labels, offset):
    unique_values = torch.unique(labels)
    # print(unique_values)
    num_unique = list(unique_values.size())[0]
    # print(num_unique)
    max_label = int(torch.max(unique_values))
    # print(max_label)

    for i in range(0, num_unique):
        labels[labels == i] = i - offset
    labels[labels < 0] += max_label + 1
    return labels


def targeted_label_flip(labels, original, target):
    labels[labels == original] = target
    return labels


def train(net, trainloader, poisoned_dataset, epochs, is_malicious=False, attack_type='none'):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if attack_type == 'gan_attack':
        dataloader = poisoned_dataset
    else:
        dataloader = trainloader

    for _ in range(epochs):
        for images, labels in tqdm(dataloader):
            # If the client is malicious, run the specified attack
            if is_malicious:

                if 'constant_flip' in attack_type:
                    # print('constant flip')
                    offset = int(attack_type.split('_'[2]))
                    labels = constant_label_flip(labels, offset)

                elif 'targeted' in attack_type:
                    # print('targeted flip')
                    split = attack_type.split('T')
                    target_class = int(split[1])
                    new_label = int(split[2])
                    # print(target_class, new_label)
                    labels = targeted_label_flip(labels, target_class, new_label)

                elif 'random_flip' in attack_type:
                    # print('random flip')
                    labels = random_label_flip(labels)
                else:
                    # no attack_type
                    print('no attack type:', attack_type)
                    pass
                # print('modified labels')
                # print(labels)

            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            pred = int(torch.argmax(outputs))
            # print('outputs',pred)
            y_pred.append(pred)

            labels = labels.to(DEVICE)
            # print('labels', int(labels))
            y_true.append(int(labels))

            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    if max(set(y_true)) > 9:
        print(set(y_true))
        raise Exception



    accuracy = correct / len(testloader.dataset)
    return loss, accuracy, y_pred, y_true


def load_data(poisoned=False):
    # load poisoned data
    if poisoned:
        poisoned_data = torch.load('../fakeData/poisoned_data_epoch_004.pt')  # adjust the filename as necessary
        # Resize the poisoned data to match the CIFAR-10 data
        poisoned_data_resized = F.interpolate(poisoned_data, size=(32, 32))
        # add random labels to the poisoned data 1-10
        poisoned_labels = torch.randint(0, 10, (len(poisoned_data_resized),))
        # combine the poisoned data and labels
        poisoned_dataset = torch.utils.data.TensorDataset(poisoned_data_resized, poisoned_labels)
        DataLoaderPoisoned=DataLoader(poisoned_dataset)
        print('Poisoned data shape:', poisoned_data_resized.shape)
    else:
        DataLoaderPoisoned=None



    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("../data", train=True, download=True, transform=trf)
    testset = CIFAR10("../data", train=False, download=True, transform=trf)

        # Print the shapes of the datasets and a few samples
    print('CIFAR-10 trainset shape:', len(trainset), 'x', trainset[0][0].shape)
    print('CIFAR-10 testset shape:', len(testset), 'x', testset[0][0].shape)

    DataLoaderTrain=DataLoader(trainset, batch_size=32, shuffle=True)
    DataLoaderTest=DataLoader(testset)
    return DataLoaderTrain, DataLoaderTest, DataLoaderPoisoned


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)

trainloader, testloader, poisoned_dataset = load_data()


# Define Flower client
def set_parameters(parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self) -> None:
        self.round_number=0
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(parameters)
        # print("\n Starting training. Malicious:", os.environ.get("IS_MALICIOUS") == "1")
        is_malicious = os.environ.get("IS_MALICIOUS") == "1"
        attack_type = os.environ.get("ATTACK")
        train(net, trainloader, poisoned_dataset, epochs=1, is_malicious=is_malicious, attack_type=attack_type)
        # print("\n Finished training.")
        return self.get_parameters(config={}), len(trainloader.dataset), {}


    """
    This function should create CSV files with y_pred/Y_True
    for Each round of testing.
    Number of files created should be num_clients x n_rounds x (max_malicious_clients +1)
    """
    def log_metrics(self, y_pred, y_true):
        print('logging metrics')
        cwd = os.getcwd()
        cwd = cwd.replace('\\src', '')
        print('cwd', cwd)

        round_number = self.round_number
        self.round_number += 1
        client_number = os.environ.get("CLIENT_ID")
        attack_type = os.environ.get("ATTACK")
        num_mal = os.environ.get("NUM_MAL")
        is_mal = os.environ.get("IS_MALICIOUS")
        exp_id = os.environ.get("EXP_ID")
        exp_id = attack_type + exp_id

        cwd = os.getcwd()
        path = cwd.replace('\\src', '')
        path += '\\log_metrics\\' + exp_id + '\\'
        try:
            os.makedirs(path)
        except FileExistsError:
            # directory already exists
            pass

        # B for benign, or M for malicious
        designation='B'
        if is_mal == '1':
            designation='M'

        df = pd.DataFrame(columns=['y_pred', 'y_true'])
        df['y_pred'] = y_pred
        df['y_true'] = y_true

        # Construct the output filename
        filename = f'{designation}{num_mal}{attack_type}Round{round_number}_ID{client_number}_.csv'
        outputfilename = os.path.join(cwd, '../', 'log_metrics', filename)

        # Save the dataframe to the specified CSV file
        df.to_csv(outputfilename)



    def evaluate(self, parameters, config):
        set_parameters(parameters)
        loss, accuracy, y_pred, y_true = test(net, testloader)  # Use the actual test function for all clients

        self.log_metrics(y_pred, y_true)

        return float(loss), len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8081",
    client=FlowerClient(),
)
