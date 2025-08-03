# loss.py

import torch
import torch.nn as nn
from RayNet.utils import ortho6d_to_rotmat

class GeodesicLoss(nn.Module):
    """
    Geodesic Loss for 3x3 rotation matrices (batch-wise).
    Computes the geodesic distance (angle, in radians) between rotation matrices.

    Args:
        eps (float): Small epsilon for numerical stability in acos.

    Usage:
        loss_fn = GeodesicLoss()
        loss = loss_fn(predicted_rotmat, gt_rotmat)  # both are [B, 3, 3]
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

def multiview_headpose_losses(pred_6d, gt_rotmats):
    """
    Args:
        pred_6d: [B, 9, 6]   (raw 6D head pose predictions)
        gt_rotmats: [B, 9, 3, 3]  (ground truth rotmats)
    Returns:
        dict: { 'accuracy': ..., 'consistency': ... }
    """

    B, N, _ = pred_6d.shape
    geo = GeodesicLoss()

    # 1. Convert 6D to rotmat
    pred_rotmats = ortho6d_to_rotmat(pred_6d.reshape(-1, 6)).reshape(B, N, 3, 3)

    # 2. Per-view accuracy
    acc_loss = geo(
        pred_rotmats.reshape(-1, 3, 3),
        gt_rotmats.reshape(-1, 3, 3)
    )

    # 3. Consistency: distance to mean prediction
    mean_pred = pred_rotmats.mean(dim=1)  # [B, 3, 3]
    cons_loss = geo(
        pred_rotmats.reshape(-1, 3, 3),
        mean_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, 3, 3)
    )

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }


