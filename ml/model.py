"""Day 1 residual CNN for contract-shaped observation windows."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation):
        super().__init__()
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation,
                              padding=dilation * (kernel // 2), bias=False)
        self.norm = nn.BatchNorm1d(outputs)
        self.skip = (nn.Identity() if inputs == outputs
                     else nn.Conv1d(inputs, outputs, 1, bias=False))

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)) + self.skip(x))


class PeriodCNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.window_length = cfg["window_length"]
        channels, dilations, kernel = cfg["channels"], cfg["dilations"], cfg["kernel_size"]
        if (self.window_length != 1024 or len(channels) != 6 or len(dilations) != 6
                or type(kernel) is not int or kernel < 1 or kernel % 2 != 1
                or any(type(v) is not int or v < 1 for v in (*channels, *dilations))):
            raise ValueError("invalid CNN window or residual block configuration")
        inputs = 3
        blocks = []
        for outputs, dilation in zip(channels, dilations):
            blocks.append(ResidualBlock(inputs, outputs, kernel, dilation))
            inputs = outputs
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(inputs, 129)

    def forward(self, windows):
        if windows.ndim != 3 or windows.shape[1:] != (3, self.window_length):
            raise ValueError("expected (batch, 3, 1024) windows")
        return self.head(self.pool(self.blocks(windows)).squeeze(-1))
