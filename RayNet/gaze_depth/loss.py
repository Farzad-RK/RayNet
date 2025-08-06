import torch


def multiview_gaze_depth_losses(pred_depth: torch.Tensor, gt_depth: torch.Tensor):
    """
    Multi-view loss for gaze depth regression:
      - Accuracy: L2 error to ground truth depth (averaged across views)
      - Consistency: L2 deviation from mean prediction across views

    Args:
        pred_depth: [B, N] predicted gaze depths for each view
        gt_depth:   [B, N] ground truth depths for each view

    Returns:
        dict: {'accuracy': scalar, 'consistency': scalar}
    """
    # Accuracy loss (per-view ground truth)
    acc_errors = (pred_depth - gt_depth).pow(2)
    accuracy = acc_errors.mean()

    # Consistency loss (distance to mean prediction per sample)
    mean_pred = pred_depth.mean(dim=1, keepdim=True)  # [B, 1]
    cons_errors = (pred_depth - mean_pred).pow(2)
    consistency = cons_errors.mean()

    return {
        'accuracy': accuracy,
        'consistency': consistency
    }
