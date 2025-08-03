# head_gaze_point/model.py

import torch
import torch.nn as nn
from RayNet.coordatt import CoordAtt  # or wherever your CoordAtt lives

class GazePointRegressionHead(nn.Module):
    """
    Gaze Point Regression Head with Coordinate Attention.
    Accepts feature maps, applies coordinate attention, pools, then regresses a 3D point.
    """
    def __init__(self, in_channels=256, hidden_dim=128, reduction=32):
        super().__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 3)  # Output is 3D point

    def forward(self, x):
        x = self.coord_att(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x  # Output: [B, 3]
