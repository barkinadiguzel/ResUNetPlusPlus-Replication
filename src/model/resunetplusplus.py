import torch
import torch.nn as nn
from ..encoder.encoder import Encoder
from ..decoder.decoder import Decoder
from ..blocks.aspp import ASPP
from ..head.segmentation_head import SegmentationHead

class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, aspp_rates=[1,6,12,18]):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.aspp = ASPP(256, 256, rates=aspp_rates) 
        self.decoder = Decoder()
        self.head = SegmentationHead(64, out_channels)
        
    def forward(self, x):
        e1, e2, e3 = self.encoder(x)
        b = self.aspp(e3)
        d1, d2, d3 = self.decoder(e1, e2, b)
        out = self.head(d3)
        return out
