# pupil_center/loss.py

import torch
import torch.nn as nn

class MultiViewPupilCenterLoss(nn.Module):
    """
    Computes multi-view L2 (Euclidean) loss for pupil center regression:
      - Accuracy: Per-view L2 error to ground truth.
      - Consistency: L2 error to mean prediction across all views.
    """

    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss(reduction='mean')  # You could also use SmoothL1Loss for robustness

    def forward(self, pred, gt):
        """
        Args:
            pred: [B, 9, 2, 3]  # 9 views per batch, 2 eyes, 3D
            gt:   [B, 9, 2, 3]
        Returns:
            dict of {'accuracy': scalar, 'consistency': scalar}
        """
        B, N, _, _ = pred.shape  # B=batch, N=views

        # Per-view accuracy
        acc_loss = self.l2(pred, gt)

        # Consistency: distance to sample mean
        mean_pred = pred.mean(dim=1, keepdim=True)    # [B, 1, 2, 3]
        cons_loss = self.l2(pred, mean_pred.expand(-1, N, -1, -1))

        return {
            'accuracy': acc_loss,
            'consistency': cons_loss
        }
