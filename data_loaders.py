import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np


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
