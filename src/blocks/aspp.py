import torch
import torch.nn as nn


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()

        self.blocks = nn.ModuleList()

        for rate in rates:
            self.blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
            )

        self.project = nn.Conv2d(
            len(rates) * out_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        features = [block(x) for block in self.blocks]
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x
