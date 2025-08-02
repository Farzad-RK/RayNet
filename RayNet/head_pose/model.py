import torch
import torch.nn as nn
from RayNet.coordatt import CoordAtt  # Import your efficient CoordAtt block

class HeadPoseRegressionHead(nn.Module):
    """
    SOTA Head Pose Regression Head with Coordinate Attention.
    Accepts feature maps, applies coordinate attention, pools, then regresses 6D pose.
    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", ECCV 2019.
    """

    def __init__(self, in_channels=256, hidden_dim=128,reduction=32):
        """
        Args:
            in_channels (int): Number of input feature channels from the fused feature map.
            hidden_dim (int): Hidden dimension for the regression MLP.
            attn_groups (int): Number of groups for CoordAtt (usually 32 or 64).
        """
        super().__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)  # Efficient CoordAtt block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 6)  # Output is 6D rotation rep

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, C, H, W], fused feature maps
        Returns:
            rot6d: [B, 6], 6D rotation representation for each sample in batch
        """
        x = self.coord_att(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x  # Output is [B, 6]
