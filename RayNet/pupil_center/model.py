# pupil_center/model.py

import torch
import torch.nn as nn
from RayNet.coordatt import CoordAtt  # Make sure the path matches your project

class PupilCenterRegressionHead(nn.Module):
    """
    Predicts 3D pupil center for each eye from fused feature maps.
    Output: [B, 2, 3] (left and right eye, in camera coordinates)
    """

    def __init__(self, in_channels=256, hidden_dim=128, reduction=32):
        super().__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 6)  # 3D coords × 2 eyes

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] (fused feature map)
        Returns:
            out: [B, 2, 3] (left and right pupil center in 3D camera space)
        """
        x = self.coord_att(x)
        x = self.pool(x).flatten(1)      # [B, C]
        x = self.act(self.fc1(x))        # [B, hidden_dim]
        x = self.fc2(x)                  # [B, 6]
        out = x.view(x.size(0), 2, 3)    # [B, 2, 3]
        return out
