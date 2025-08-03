# pupil_center/loss.py

import torch

def multiview_pupil_center_losses(pred: torch.Tensor, gt: torch.Tensor):
    """
    Multi-view L2 (Euclidean) loss for pupil center regression (left/right eye):
      - Accuracy: Per-view L2 error to ground truth.
      - Consistency: L2 error to mean prediction across all views.

    Args:
        pred: [B, N, 2, 3]  # views × eyes × 3D coords (cm)
        gt:   [B, N, 2, 3]  # same shape, ground truth in cm

    Returns:
        dict of {'accuracy': scalar cm, 'consistency': scalar cm}
    """
    # Compute per-eye per-view Euclidean distances (cm)
    eye_errors = torch.norm(pred - gt, dim=-1)     # [B, N, 2]
    # Average over eyes → per-view error
    errors = eye_errors.mean(dim=-1)               # [B, N]
    # Accuracy: mean over batch and views
    accuracy = errors.mean()

    # Consistency: deviation from mean prediction
    mean_pred = pred.mean(dim=1, keepdim=True)             # [B, 1, 2, 3]
    cons_eye_errors = torch.norm(pred - mean_pred, dim=-1)  # [B, N, 2]
    cons_errors = cons_eye_errors.mean(dim=-1)              # [B, N]
    consistency = cons_errors.mean()

    return {
        'accuracy': accuracy,
        'consistency': consistency
    }
