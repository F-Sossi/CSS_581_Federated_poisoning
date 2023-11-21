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
    labels[labels < 0] += max_label+1
    return labels

def targeted_label_flip(labels, original, target):
    labels[labels==original]=target
    return labels


def train(net, trainloader, epochs, is_malicious=False, attack_type='none'):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            # If the client is malicious, flip the labels
            if is_malicious:

                if 'constant_flip' in attack_type:
                    #print('constant flip')
                    offset=int(attack_type.split('_'[2]))
                    labels=constant_label_flip(labels, offset)


                elif 'targeted_flip' in attack_type:
                    #print('targeted flip')
                    split = attack_type.split('T')
                    target_class=int(split[1])
                    new_label=int(split[2])
                    #print(target_class, new_label)
                    labels=targeted_label_flip(labels, target_class, new_label)

                elif 'random_flip' in attack_type:
                    #print('random flip')
                    labels = random_label_flip(labels)
                else:
                    #no attack_type
                    #print('no attack type:', attack_type)
                    pass
                #print('modified labels')
                #print(labels)

            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    y_true=[]
    y_pred=[]
    #Hard coded the classes here for cifar10 dataset
    classes=list(range(0,10))
    with torch.no_grad():
        for images, labels in tqdm(testloader):

            outputs = net(images.to(DEVICE))
            pred=int(torch.argmax(outputs))
            #print('outputs',pred)
            y_pred.append(pred)

            labels = labels.to(DEVICE)
            #print('labels', int(labels))
            y_true.append(int(labels))

            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    extra_displays=False
    if extra_displays:
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sns.heatmap(df_cm, annot=True)
        plt.savefig('output'+str(time.time())+'.png')


    accuracy = correct / len(testloader.dataset)
    return loss, accuracy, y_pred, y_true


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
def set_parameters(parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(parameters)
        # print("\n Starting training. Malicious:", os.environ.get("IS_MALICIOUS") == "1")
        is_malicious = os.environ.get("IS_MALICIOUS") == "1"
        attack_type=os.environ.get("ATTACK")
        train(net, trainloader, epochs=1, is_malicious=is_malicious, attack_type=attack_type)
        # print("\n Finished training.")
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def log_metrics(self, y_pred, y_true):

        print('logging metrics')
        cwd=os.getcwd()
        cwd=cwd.replace('\\src','')
        print('cwd',cwd)
        round_number=os.environ.get("ROUND")
        client_number=os.environ.get("CLIENT_ID")
        attack_type=os.environ.get("ATTACK")


        df=pd.DataFrame(columns=['y_pred','y_true'])
        df['y_pred']=y_pred
        df['y_true']=y_true

        outputfilename=cwd+'\\log_metrics\\'
        outputfilename += attack_type
        outputfilename+='Round'+str(round_number)+'_'
        outputfilename += 'ID' + str(client_number) + '_'
        outputfilename+='.csv'
        df.to_csv(outputfilename)


    def evaluate(self, parameters, config):
        set_parameters(parameters)
        loss, accuracy, y_pred, y_true = test(net, testloader)  # Use the actual test function for all clients

        self.log_metrics(y_pred, y_true)

        return float(loss), len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(),
)
