import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Tuple, List


class ResConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple = (3, 3),
        dropout: float = 0,
    ):
        super().__init__()
        self.init_kwargs = {}
        args = inspect.getfullargspec(self.__init__)
        for k, v in zip(args.annotations.keys(), args.args[1:]):
            self.init_kwargs[k] = locals()[k]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, (1, 1))
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size
        )
        self.pool = nn.AvgPool2d(self.kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)

        residual = self.downsample(x)
        residual = self.dropout(residual)

        out = out + residual
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


if __name__ == "__main__":
    B = 32
    C = 3
    H = 800
    W = 1280
    x = torch.rand(B, C, H, W)
    model = ResConv2d(3, 6, dropout=0.1)
    out = model(x)
    pass
