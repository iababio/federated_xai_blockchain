from typing import List, OrderedDict
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    # Load and preprocess the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)

    # Split training set into 10 partitions to simulate the distributed setting
    num_clients = 10
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Create DataLoaders for each partition
    trainloaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets]
    valloaders = [DataLoader(testset, batch_size=32) for _ in range(num_clients)]  # Use the same validation loader for each client

    # Define testloader for global evaluation
    testloader = DataLoader(testset, batch_size=32)

    return trainloaders, valloaders, testloader

def get_parameters(net) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int, optimizer, client_id: int):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    accuracy_log = []
    loss_log = []

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return loss_log, accuracy_log

def test(net, testloader):
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