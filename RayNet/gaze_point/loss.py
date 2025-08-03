# gaze_point/loss.py

import torch

def multiview_gaze_point_losses(points_pred: torch.Tensor, points_gt: torch.Tensor):
    """
    Multi-view L2 loss for gaze point regression (3D point):
      - Accuracy: Per-view L2 (Euclidean) error to ground truth.
      - Consistency: L2 error to mean prediction across all views.

    Args:
        points_pred: [B, N, 3] (predicted points for all views) in cm
        points_gt:   [B, N, 3] (ground truth) in cm

    Returns:
        dict of {'accuracy': scalar cm, 'consistency': scalar cm}
    """
    # Compute per-view Euclidean distances (cm)
    errors = torch.norm(points_pred - points_gt, dim=-1)  # [B, N]
    # Accuracy: mean over batch and views
    accuracy = errors.mean()

    # Consistency: deviation from mean prediction
    mean_pred = points_pred.mean(dim=1, keepdim=True)     # [B, 1, 3]
    cons_errors = torch.norm(points_pred - mean_pred, dim=-1)  # [B, N]
    consistency = cons_errors.mean()

    return {
        'accuracy': accuracy,
        'consistency': consistency
    }
