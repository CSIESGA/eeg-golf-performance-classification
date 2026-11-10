"""1D ResNet-Tiny adapted from the researcher's original new_main.py.

The model architecture is preserved: a strided 1D convolutional stem,
three residual stages, global average pooling, and one binary logit.
"""

import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return F.relu(out + identity, inplace=True)


class ResNetTiny1D(nn.Module):
    """Return one logit per sample for arrays shaped (batch, channels, time)."""

    def __init__(self, in_channels: int, num_blocks: tuple[int, int, int] = (1, 1, 1),
                 base_channels: int = 32, kernel_size: int = 7, num_classes: int = 1):
        super().__init__()
        if len(num_blocks) != 3 or any(n < 1 for n in num_blocks):
            raise ValueError("num_blocks must contain three positive integers")
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, stride=2,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = (base_channels, base_channels * 2, base_channels * 4)
        stages = []
        current = base_channels
        for out_ch, nblk in zip(channels, num_blocks):
            blocks = [BasicBlock1D(current, out_ch, kernel_size, stride=2)]
            blocks.extend(BasicBlock1D(out_ch, out_ch, kernel_size) for _ in range(nblk - 1))
            stages.append(nn.Sequential(*blocks))
            current = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(channels[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
