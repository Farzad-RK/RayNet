"""
Coordinate Attention Module (Hou et al., 2021).

Encodes spatial direction information (row and column) with depthwise
pooling, preserving geometric context critical for landmark localisation.
Preferred over SE/CBAM for spatial tasks on mobile architectures.
"""

import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # H x 1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # 1 x W
        self.conv1 = nn.Conv2d(ch, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid, ch, 1)
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.pool_h(x)                          # (B, C, H, 1)
        w = self.pool_w(x).permute(0, 1, 3, 2)      # (B, C, W, 1)
        y = torch.cat([h, w], dim=2)                 # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        h, w = torch.split(y, [H, W], dim=2)
        w = w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(h)) * torch.sigmoid(self.conv_w(w))
