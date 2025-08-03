# pupil_center/loss.py

import torch
import torch.nn.functional as F

def multiview_pupil_center_losses(pred, gt):
    """
    Multi-view L2 (Euclidean) loss for pupil center regression (left/right eye):
      - Accuracy: Per-view L2 error to ground truth.
      - Consistency: L2 error to mean prediction across all views.

    Args:
        pred: [B, 9, 2, 3]  # 9 views per batch, 2 eyes, 3D
        gt:   [B, 9, 2, 3]
    Returns:
        dict of {'accuracy': scalar, 'consistency': scalar}
    """
    B, N, _, _ = pred.shape

    # Per-view accuracy (mean L2 distance per sample)
    acc_loss = F.mse_loss(pred, gt, reduction='mean')

    # Consistency: distance to mean prediction (per sample)
    mean_pred = pred.mean(dim=1, keepdim=True)       # [B, 1, 2, 3]
    cons_loss = F.mse_loss(pred, mean_pred.expand(-1, N, -1, -1), reduction='mean')

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }
