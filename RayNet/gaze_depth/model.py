# gaze_depth/model.py

import torch
import torch.nn as nn
from RayNet.coordatt import CoordAtt  # Adjust import path if needed

class GazeDepthRegressionHead(nn.Module):
    """
    Regresses gaze depth (scalar distance) with coordinate attention.
    """
    def __init__(self, in_channels=256, hidden_dim=128, reduction=32):
        super().__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single scalar depth

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.coord_att(x)
        x = self.pool(x).flatten(1)       # [B, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)                   # [B, 1]
        return x.squeeze(-1)              # [B] (scalar)
