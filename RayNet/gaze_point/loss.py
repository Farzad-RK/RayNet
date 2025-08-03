# head_gaze_point/loss.py

import torch
import torch.nn as nn

class L2PointLoss(nn.Module):
    """
    Euclidean (L2) loss for 3D points.
    """
    def forward(self, p_pred, p_gt):
        """
        Args:
            p_pred: [B, 3] or [B*N, 3]
            p_gt:   [B, 3] or [B*N, 3]
        Returns:
            scalar: mean L2 distance
        """
        return torch.norm(p_pred - p_gt, dim=-1).mean()

def multiview_gazepoint_losses(points_pred, points_gt):
    """
    Multi-view gaze point loss for MGDA: per-view L2 loss (accuracy), and inter-view consistency.

    Args:
        points_pred: [B, 9, 3] (predicted points for all views)
        points_gt:   [B, 9, 3] (ground truth)

    Returns:
        dict:
            - 'accuracy': mean L2 loss between prediction and GT
            - 'consistency': mean L2 loss to sample mean prediction
    """
    B, N, _ = points_pred.shape
    l2 = L2PointLoss()

    # Per-view accuracy loss
    acc_loss = l2(points_pred.reshape(-1, 3), points_gt.reshape(-1, 3))

    # Consistency loss: distance to mean prediction (per sample)
    mean_pred = points_pred.mean(dim=1, keepdim=True)  # [B,1,3]
    cons_loss = l2(points_pred, mean_pred.expand(-1, N, -1))

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }
