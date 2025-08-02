# loss.py

import torch
import torch.nn as nn

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

def multiview_headpose_losses(rotmats_pred, rotmats_gt):
    """
    Compute multi-view head pose loss for MGDA: per-view geodesic loss (accuracy) and
    inter-view geodesic consistency loss.

    Args:
        rotmats_pred: torch.Tensor, [B, 9, 3, 3]  (predicted rotation matrices for all views)
        rotmats_gt:   torch.Tensor, [B, 9, 3, 3]  (ground truth)

    Returns:
        dict:
            - 'accuracy': mean geodesic loss between each predicted rotation and ground truth
            - 'consistency': mean geodesic loss between each view prediction and the sample mean prediction
    """
    B, N, _, _ = rotmats_pred.shape  # B=batch, N=views (usually 9)
    geo = GeodesicLoss()

    # 1. Per-view accuracy loss
    acc_loss = geo(
        rotmats_pred.reshape(-1, 3, 3),   # [B*N, 3, 3]
        rotmats_gt.reshape(-1, 3, 3)      # [B*N, 3, 3]
    )

    # 2. Consistency loss: distance to mean prediction (per sample)
    mean_pred = rotmats_pred.mean(dim=1)  # [B, 3, 3]
    mean_pred = orthogonalize_rotmat(mean_pred)  # Optional, can skip for speed
    cons_loss = geo(
        rotmats_pred.reshape(-1, 3, 3),                                   # [B*N, 3, 3]
        mean_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, 3, 3)    # [B*N, 3, 3]
    )

    return {
        'accuracy': acc_loss,
        'consistency': cons_loss
    }

def orthogonalize_rotmat(rotmat):
    """
    Projects a batch of 3x3 matrices to the closest rotation matrix using SVD.

    Args:
        rotmat: [B, 3, 3]
    Returns:
        [B, 3, 3] (rotation matrices)
    """
    # SVD-based projection
    U, _, Vt = torch.linalg.svd(rotmat)
    R = torch.matmul(U, Vt)
    # Enforce det(R) = +1 (rotation, not reflection)
    det = torch.det(R)
    det_sign = det.sign().unsqueeze(-1).unsqueeze(-1)
    Vt = Vt * det_sign
    R = torch.matmul(U, Vt)
    return R

# Example usage in your training loop (MGDA-ready):
# losses = multiview_headpose_losses(preds, gt)  # preds, gt: [B, 9, 3, 3]
# mgda_losses = [losses['accuracy'], losses['consistency']]
