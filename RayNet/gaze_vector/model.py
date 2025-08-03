# head_gaze/model.py

import torch
import torch.nn as nn
from RayNet.coordatt import CoordAtt

class GazeVectorRegressionHead(nn.Module):
    """
    Predicts a 6D rotation representation per view, from which
    we extract the 3D gaze direction (the 3rd column of the rotation).
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128, reduction: int = 32):
        """
        Args:
            in_channels: Number of channels in the fused feature map.
            hidden_dim:  Hidden size of the intermediate FC layer.
            reduction:   Reduction ratio for the CoordAtt bottleneck.
        """
        super().__init__()
        # Coordinate Attention to reweight spatial & channel cues
        self.coord_att = CoordAtt(in_channels, in_channels, reduction)
        # Global pooling → MLP → 6D representation
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(in_channels, hidden_dim)
        self.act  = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(hidden_dim, 6)  # 6D rotation rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] fused feature map for *one* view
        Returns:
            rot6d: [B, 6] six‐dim rotation representation per view
        """
        x = self.coord_att(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
