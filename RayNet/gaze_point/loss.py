# gaze_point/loss.py

import torch
import torch.nn.functional as F

def multiview_gaze_point_losses(points_pred, points_gt):
    """
    Multi-view L2 loss for gaze point regression (3D point):
      - Accuracy: Per-view L2 (Euclidean) error to ground truth.
      - Consistency: L2 error to mean prediction across all views.

    Args:
        points_pred: [B, 9, 3] (predicted points for all views)
        points_gt:   [B, 9, 3] (ground truth)

    Returns:
        dict of {'accuracy': scalar, 'consistency': scalar}
    """
    B, N, _ = points_pred.shape

    # Per-view accuracy (mean squared L2)
    acc_loss = F.mse_loss(points_pred, points_gt, reduction='mean')

    # Consistency: distance to mean prediction (per sample)
    mean_pred = points_pred.mean(dim=1, keepdim=True)   # [B, 1, 3]
    cons_loss = F.mse_loss(points_pred, mean_pred.expand(-1, N, -1), reduction='mean')

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }
