"""
model.py

This module defines the Convolutional Neural Network architecture used for
image classification tasks. The network consists of two convolutional layers,
adaptive pooling, and three fully connected layers. It is designed to work
with the CIFAR-10 dataset, which contains 32x32 color images in 10 classes.

The architecture is as follows:
- Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> FC -> ReLU -> FC
"""

from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Convolutional Neural Network model with two convolutional layers
    followed by adaptive pooling, and three fully connected layers.

    The model architecture is:
    - Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> FC -> ReLU -> FC
    """

    def __init__(self):
        """Initialize the network layers with specified input-output dimensions."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AdaptiveAvgPool2d((5, 5))  # Ensures the output size is always 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Output logits for each class.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)  # 16 channels, 5x5 spatial dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
