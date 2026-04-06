import torch
import torch.nn as nn
from ..blocks.residual_block import ResidualBlock
from ..blocks.attention_block import AttentionBlock
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, features=[256, 128, 64]):
        super().__init__()
        self.att3 = AttentionBlock(features[2], features[1])
        self.dec3 = ResidualBlock(features[2]+features[1], features[1])
        
        self.att2 = AttentionBlock(features[1], features[0])
        self.dec2 = ResidualBlock(features[1]+features[0], features[0])
        
        self.att1 = AttentionBlock(features[0], features[0])
        self.dec1 = ResidualBlock(features[0]*2, features[0])
        
    def forward(self, e1, e2, e3):
        d3 = self.att3(e3, e2)
        d3 = torch.cat([F.interpolate(e3, scale_factor=2, mode='nearest'), d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.att2(d3, e1)
        d2 = torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.att1(d2, e1)
        d1 = torch.cat([F.interpolate(d2, scale_factor=2, mode='nearest'), d1], dim=1)
        d1 = self.dec1(d1)
        
        return d1, d2, d3
