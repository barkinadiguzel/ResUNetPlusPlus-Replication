import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()

        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)

        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)

        out = self.relu(theta_x + phi_g)
        out = self.psi(out)
        attention = self.sigmoid(out)

        return x * attention
