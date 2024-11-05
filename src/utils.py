"""
utils.py

This module contains utility functions for loading data, managing model parameters,
and training and testing the neural network. It is designed to support the
federated learning framework implemented in this project.

Key functions include:
- load_data: Loads and preprocessed the CIFAR-10 dataset, partitioning it for federated learning.
- create_dataloaders: Creates DataLoaders for the given datasets.
- get_parameters: Extracts model parameters as NumPy arrays.
- set_parameters: Sets model parameters from NumPy arrays.
- train: Trains the model on the provided DataLoader.
- test: Evaluates the model on the provided test DataLoader.
"""


from typing import List, OrderedDict
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

from src.blockchain import CONTRACT_ABI, CONTRACT_ADDRESS, W3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_contract = W3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

num_clients = 10

def load_data(num_clients: int = 10, batch_size: int = 32):
    """
    Loads and preprocesses the CIFAR-10 dataset, partitioning the training set
    into `num_clients` partitions for federated learning simulation.

    Returns:
        tuple: List of DataLoaders for each client, a list of validation loaders,
               and a test DataLoader for global evaluation.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)

    # Partition training set into `num_clients` parts
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Create train and validation loaders for each client
    trainloaders = create_dataloaders(datasets, batch_size)
    valloaders = create_dataloaders([testset] * num_clients, batch_size, shuffle=False)

    # Define global test DataLoader
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders, testloader


def create_dataloaders(datasets, batch_size, shuffle=True):
    """
    Helper function to create DataLoaders for each dataset in `datasets`.

    Args:
        datasets (list): List of datasets for each client.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle data in DataLoader.

    Returns:
        list: List of DataLoaders.
    """
    return [DataLoader(ds, batch_size=batch_size, shuffle=shuffle) for ds in datasets]


def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

client_ids = [str(i) for i in range(num_clients)]


def train(net, trainloader, epochs: int, optimizer, client_id: int, dataset_contract, W3):
    """Train the model for a given number of epochs on the provided DataLoader."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    accuracy_log, loss_log = [], []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total
        loss_log.append(epoch_loss)
        accuracy_log.append(epoch_accuracy)

        print(f"""Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, 
              Accuracy: {epoch_accuracy:.4f}""")
        
        # Tokenizing training metrics, encrypting them and appending on the blockchain (smart contract) for each epoch
        try:
            # Tokenizing based on client_id, epoch, and metrics like loss and accuracy
            tx_hash = dataset_contract.functions.tokenizeTrainingProgress(
                client_id,
                epoch + 1,  # Current epoch (1-based indexing)
                int(epoch_loss * 1e6),  # Convert loss to integer (to avoid floating point)
                int(epoch_accuracy * 1e6)  # Convert accuracy to integer (scaled for precision)
            ).transact({'from': W3.eth.accounts[0]})
            # Optionally, wait for the transaction to be mined
            receipt = W3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Tokenized training metrics for client {client_id}, epoch {epoch+1}: TxHash = {tx_hash.hex()}")
        except Exception as e:
            print(f"Error tokenizing training metrics for client {client_id}, epoch {epoch+1}: {e}")

    return loss_log, accuracy_log


def test(net, testloader):
    """Evaluate the model on the provided test DataLoader."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
