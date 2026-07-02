import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import SpectralConv2d

class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator (FNO-2D) for weather nowcasting.
    Transforms spatial weather states into frequency domain using SpectralConvs
    to solve spatial transport operators efficiently.
    """
    def __init__(self, modes1=8, modes2=8, width=20, in_channels=5, out_channels=24):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Projection layer: maps input parameters (rain, wind components + 2D grid coordinates) to channels
        self.fc0 = nn.Linear(in_channels, self.width)

        # Spectral Convolutions
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Linear skip connections
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        # Output projection layers
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)
