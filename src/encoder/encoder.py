import torch
import torch.nn as nn
from ..blocks.stem_block import StemBlock
from ..blocks.residual_block import ResidualBlock
from ..blocks.se_block import SEBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256]):
        super().__init__()
        self.stem = StemBlock(in_channels, features[0])
        
        self.enc1 = nn.Sequential(
            ResidualBlock(features[0], features[0]),
            SEBlock(features[0])
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(features[0], features[1], stride=2),
            SEBlock(features[1])
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(features[1], features[2], stride=2),
            SEBlock(features[2])
        )
        
    def forward(self, x):
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e1, e2, e3
