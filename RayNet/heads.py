"""
Task-specific heads for RayNet.

IrisPupilLandmarkHead: 14 landmarks (10 iris + 4 pupil) via heatmap + soft-argmax.
OpticalAxisHead: Optical axis regression as pitch/yaw -> unit 3D vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisPupilLandmarkHead(nn.Module):
    """
    Heatmap-based landmark detection with subpixel refinement.

    14 landmarks: 10 iris contour + 4 pupil boundary points.
    Uses soft-argmax over heatmaps + learned offset for subpixel accuracy.
    """

    def __init__(self, in_ch=256, n_landmarks=14):
        super().__init__()
        self.n_landmarks = n_landmarks

        # Heatmap branch: predicts spatial probability for each landmark
        self.heatmap = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, n_landmarks, 1)
        )

        # Offset branch: subpixel refinement (dx, dy per landmark)
        self.offset = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, n_landmarks * 2, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: (B, C, H, W) feature map from PANet P2 + CoordAtt

        Returns:
            coords: (B, 14, 2) landmark pixel coordinates in feature map space
            heatmaps: (B, 14, H, W) raw logit heatmaps (for loss computation)
        """
        hm = self.heatmap(feat)     # (B, 14, H, W)
        off = self.offset(feat)     # (B, 28, H, W)
        coords = self._soft_argmax(hm, off)  # (B, 14, 2)
        return coords, hm

    def _soft_argmax(self, hm, off):
        """Differentiable soft-argmax with offset refinement."""
        B, N, H, W = hm.shape

        # Softmax over spatial dims with high temperature for peaky distribution
        flat = hm.view(B, N, -1)
        weight = F.softmax(flat * 100.0, dim=-1).view(B, N, H, W)

        # Grid coordinates
        device = hm.device
        gx = torch.arange(W, dtype=torch.float32, device=device)
        gy = torch.arange(H, dtype=torch.float32, device=device)

        # Weighted sum of coordinates
        x = (weight * gx[None, None, None, :]).sum(dim=[2, 3])  # (B, N)
        y = (weight * gy[None, None, :, None]).sum(dim=[2, 3])  # (B, N)

        # Subpixel offset refinement
        off2 = off.view(B, N, 2, H * W)
        idx = (y.long() * W + x.long()).clamp(0, H * W - 1)
        dx = off2[:, :, 0, :].gather(2, idx.unsqueeze(2)).squeeze(2)
        dy = off2[:, :, 1, :].gather(2, idx.unsqueeze(2)).squeeze(2)

        return torch.stack([x + dx, y + dy], dim=-1)  # (B, N, 2)


class OpticalAxisHead(nn.Module):
    """
    Optical axis regression head.

    Predicts pitch and yaw angles, then converts to a unit 3D vector.
    Attaches to P5 (stride=32, 14x14 for 448 input) via GAP.

    Split into pool_features() and predict_from_pooled() to allow
    CrossViewAttention to be inserted between them.
    """

    def __init__(self, in_ch=256, hidden_dim=128):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # pitch, yaw
        )

    def pool_features(self, feat):
        """(B, C, H, W) -> (B, C) pooled feature vector."""
        return self.pool(feat)

    def predict_from_pooled(self, pooled):
        """
        (B, C) -> (gaze_vector (B, 3), angles (B, 2))

        Runs FC layers and converts pitch/yaw to unit 3D vector.
        """
        angles = self.fc(pooled)  # (B, 2)
        pitch = angles[:, 0]
        yaw = angles[:, 1]

        # Convert spherical (pitch, yaw) to unit 3D vector
        # Convention: x = -cos(pitch)*sin(yaw), y = -sin(pitch), z = -cos(pitch)*cos(yaw)
        x = -torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)

        gaze_vector = torch.stack([x, y, z], dim=-1)  # (B, 3)
        gaze_vector = F.normalize(gaze_vector, dim=-1)

        return gaze_vector, angles

    def forward(self, feat):
        """
        Args:
            feat: (B, C, H, W) feature map from PANet P5 + CoordAtt

        Returns:
            gaze_vector: (B, 3) unit vector in camera coordinate space
            angles: (B, 2) predicted pitch and yaw in radians
        """
        return self.predict_from_pooled(self.pool_features(feat))
