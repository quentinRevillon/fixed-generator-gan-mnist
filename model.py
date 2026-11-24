"""Generator and discriminator imposed by the assignment (architecture must not be changed).

Usage:
    from model import Generator, Discriminator
    G = Generator(g_output_dim=784)      # z of shape (N, 100) -> images of shape (N, 784) in [-1, 1]
    D = Discriminator(d_input_dim=784)   # images of shape (N, 784) -> probability of being real, shape (N, 1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Fully connected generator 100 -> 256 -> 512 -> 1024 -> g_output_dim, tanh output."""
    def __init__(self, g_output_dim):
        """Build the four linear layers."""
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        """Map latent vectors z (N, 100) to flattened images (N, g_output_dim) in [-1, 1]."""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    """Fully connected discriminator d_input_dim -> 1024 -> 512 -> 256 -> 1, sigmoid output."""
    def __init__(self, d_input_dim):
        """Build the four linear layers."""
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        """Return the probability (N, 1) that each flattened image is real."""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
