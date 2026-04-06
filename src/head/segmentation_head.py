import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        return self.act(self.conv(x))
