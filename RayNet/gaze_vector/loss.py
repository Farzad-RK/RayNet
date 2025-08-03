# gaze_vector_loss.py

import torch
import torch.nn as nn

from sixdrepnet.utils import compute_rotation_matrix_from_ortho6d, normalize_vector
from RayNet.utils import orthogonalize_rotmat

class GeodesicLoss(nn.Module):
    """
    Geodesic Loss for 3x3 rotation matrices (batch-wise).
    Computes the geodesic distance (angle, in radians) between rotation matrices.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        """
        Args:
            m1: torch.Tensor, [B, 3, 3], predicted rotation matrices
            m2: torch.Tensor, [B, 3, 3], ground-truth rotation matrices
        Returns:
            mean geodesic loss (scalar)
        """
        m = torch.bmm(m1, m2.transpose(1,2))  # [B, 3, 3]
        cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1) / 2
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))  # [B]
        return torch.mean(theta)

def gaze_vector_to_rotmat(gaze_vec):
    """
    Convert a batch of 3D gaze vectors into rotation matrices.
    Args:
        gaze_vec: torch.Tensor, [B, 3]
    Returns:
        rotmat: torch.Tensor, [B, 3, 3]
    """
    z = normalize_vector(gaze_vec)
    # Create an arbitrary "up" vector for cross product
    up = torch.zeros_like(z)
    up[:, 1] = 1
    mask = (z[:, 1].abs() > 0.99)
    up[mask] = torch.tensor([1, 0, 0], dtype=z.dtype, device=z.device)
    x = normalize_vector(torch.cross(up, z, dim=1))
    y = normalize_vector(torch.cross(z, x, dim=1))
    rotmat = torch.stack([x, y, z], dim=2)
    return rotmat

def multiview_gaze_vector_geodesic_losses(gaze6d_pred, gaze_vec_gt):
    """
    Multi-view gaze vector loss for MGDA.
    Computes per-view geodesic loss (accuracy) and consistency loss.

    Args:
        gaze6d_pred: [B, 9, 6] predicted gaze in 6D rotation rep (per view)
        gaze_vec_gt: [B, 9, 3] GT gaze direction in 3D (per view, normalized)

    Returns:
        dict: {'accuracy': ..., 'consistency': ...}
    """
    B, N, _ = gaze6d_pred.shape
    geo = GeodesicLoss()

    # 1. Convert 6D to rotmat (predicted)
    pred_rotmat = compute_rotation_matrix_from_ortho6d(
        gaze6d_pred.reshape(-1, 6)
    ).reshape(B, N, 3, 3)  # [B, N, 3, 3]

    # 2. Convert GT gaze vectors to rotmat
    gt_rotmat = gaze_vector_to_rotmat(
        gaze_vec_gt.reshape(-1, 3)
    ).reshape(B, N, 3, 3)

    # 3. Accuracy loss: geodesic between prediction and GT
    acc_loss = geo(
        pred_rotmat.reshape(-1, 3, 3),
        gt_rotmat.reshape(-1, 3, 3)
    )

    # 4. Consistency loss: distance to sample mean
    mean_pred = pred_rotmat.mean(dim=1)  # [B, 3, 3]
    mean_pred = orthogonalize_rotmat(mean_pred)
    cons_loss = geo(
        pred_rotmat.reshape(-1, 3, 3),
        mean_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, 3, 3)
    )

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }

