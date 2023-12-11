from collections import OrderedDict
import flwr as fl
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
    """
    A simple convolutional neural network model adapted from the 'PyTorch: A 60 Minute Blitz' tutorial.

    Inherits from nn.Module.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer, producing the final output.

    Methods:
        forward(x): Defines the forward pass of the model.
    """

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
    """
    Randomly flips the labels of a given tensor to new random values.

    Parameters:
        labels (torch.Tensor): The original labels tensor.

    Returns:
        torch.Tensor: Tensor with randomly flipped labels.
    """
    labels = torch.randint(0, 10, labels.shape)
    return labels


def constant_label_flip(labels, offset):
    """
    Flips the labels of a tensor by a constant offset.

    Parameters:
        labels (torch.Tensor): The original labels tensor.
        offset (int): The offset to apply to each label.

    Returns:
        torch.Tensor: Tensor with labels flipped by the specified offset.
    """
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
    """
    Flips specific labels from an original value to a target value.

    Parameters:
        labels (torch.Tensor): The original labels tensor.
        original (int): The original label value to be flipped.
        target (int): The target label value to flip to.

    Returns:
        torch.Tensor: Tensor with specific labels flipped from original to target.
    """
    labels[labels == original] = target
    return labels


def train(net, trainloader, poisoned_dataset, epochs, is_malicious=False, attack_type='none'):
    """
    Trains the neural network model on a dataset.

    Parameters:
        net (Net): The neural network model to train.
        trainloader (DataLoader): DataLoader for the training dataset.
        poisoned_dataset (DataLoader): DataLoader for the poisoned dataset (if any).
        epochs (int): Number of epochs to train the model.
        is_malicious (bool, optional): Indicates if the training should include a malicious attack. Defaults to False.
        attack_type (str, optional): Type of attack to simulate if is_malicious is True. Defaults to 'none'.

    Returns:
        None
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if attack_type == 'gan_attack':
        dataloader = poisoned_dataset
    else:
        dataloader = trainloader

    # Check if dataloader is not None
    if dataloader is None:
        raise ValueError("Dataloader is None. Check the data loading process.")

    for _ in range(epochs):
        for images, labels in tqdm(dataloader):
            # If the client is malicious, run the specified attack
            if is_malicious:

                if 'constantFlip' in attack_type:
                    # print('constant flip')
                    # Corrected line below
                    offset = int(attack_type.split('_')[1])
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
                    # print('no attack type:', attack_type)
                    pass
                # print('modified labels')
                # print(labels)

            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """
    Tests the neural network model on a test dataset.

    Parameters:
        net (Net): The neural network model to test.
        testloader (DataLoader): DataLoader for the test dataset.

    Returns:
        Tuple: A tuple containing the total loss, accuracy, predictions, and true labels of the test dataset.
    """
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


import os


def load_data():
    """
    Loads the CIFAR-10 dataset and a poisoned dataset if available.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing DataLoaders for the training dataset,
        test dataset, and the poisoned dataset (if available).
    """
    poisoned_data_loader = None
    poisoned_data_path = '../fakeData/poisoned_data.pt'

    # Check if poisoned data file exists and load it
    if os.path.exists(poisoned_data_path):
        try:
            poisoned_data = torch.load(poisoned_data_path)
            # Resize the poisoned data to match the CIFAR-10 data
            poisoned_data_resized = F.interpolate(poisoned_data, size=(32, 32))
            # Add random labels to the poisoned data
            poisoned_labels = torch.randint(0, 10, (len(poisoned_data_resized),))
            # Combine the poisoned data and labels
            poisoned_dataset = torch.utils.data.TensorDataset(poisoned_data_resized, poisoned_labels)
            poisoned_data_loader = DataLoader(poisoned_dataset)
            print('Poisoned data shape:', poisoned_data_resized.shape)
        except Exception as e:
            print(f"Error loading poisoned data: {e}")

    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("../data", train=True, download=True, transform=trf)
    testset = CIFAR10("../data", train=False, download=True, transform=trf)

    # Print the shapes of the datasets
    print('CIFAR-10 trainset shape:', len(trainset), 'x', trainset[0][0].shape)
    print('CIFAR-10 testset shape:', len(testset), 'x', testset[0][0].shape)

    data_loader_train = DataLoader(trainset, batch_size=32, shuffle=True)
    data_loader_test = DataLoader(testset)
    return data_loader_train, data_loader_test, poisoned_data_loader


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
    """
    A Flower client for federated learning, implementing the necessary methods for federated training and evaluation.

    Inherits from fl.client.NumPyClient.

    Methods:
        get_parameters(config): Returns the model parameters.
        fit(parameters, config): Trains the model.
        evaluate(parameters, config): Evaluates the model.
    """

    def __init__(self) -> None:
        self.round_number = 0

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

    def evaluate(self, parameters, config):
        set_parameters(parameters)
        loss, accuracy, y_pred, y_true = test(net, testloader)  # Use the actual test function for all clients

        return float(loss), len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8081",
    client=FlowerClient(),
)
