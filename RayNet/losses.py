"""
Loss functions for RayNet.

Two core losses:
  1. Landmark loss: heatmap MSE + coordinate L1
  2. Angular loss: L1 on angle (more robust than cosine for large errors)
"""

import torch
import torch.nn.functional as F


def gaussian_heatmaps(coords, H, W, sigma=2.0):
    """
    Generate ground-truth Gaussian heatmaps from landmark coordinates.

    Args:
        coords: (B, N, 2) landmark pixel coordinates (x, y) in feature map space
        H: heatmap height
        W: heatmap width
        sigma: Gaussian standard deviation in pixels

    Returns:
        heatmaps: (B, N, H, W) ground-truth heatmaps
    """
    B, N, _ = coords.shape
    device = coords.device

    gx = torch.arange(W, dtype=torch.float32, device=device)
    gy = torch.arange(H, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)

    # Expand for broadcasting: (1, 1, H, W)
    grid_x = grid_x[None, None, :, :]
    grid_y = grid_y[None, None, :, :]

    # Landmark coords: (B, N, 1, 1)
    cx = coords[:, :, 0:1].unsqueeze(-1)
    cy = coords[:, :, 1:2].unsqueeze(-1)

    # Gaussian
    heatmaps = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
    return heatmaps  # (B, N, H, W)


def landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma=2.0):
    """
    Combined heatmap MSE + coordinate L1 loss.

    Args:
        pred_hm: (B, N, H, W) predicted logit heatmaps
        pred_coords: (B, N, 2) predicted landmark coordinates
        gt_coords: (B, N, 2) ground-truth landmark coordinates in feature map space
        feat_H: feature map height (for GT heatmap generation)
        feat_W: feature map width
        sigma: Gaussian sigma for GT heatmaps

    Returns:
        loss: scalar tensor
    """
    gt_hm = gaussian_heatmaps(gt_coords, feat_H, feat_W, sigma)
    hm_loss = F.mse_loss(torch.sigmoid(pred_hm), gt_hm)
    coord_loss = F.l1_loss(pred_coords, gt_coords)
    return hm_loss + coord_loss


def angular_loss(pred_gaze, gt_gaze):
    """
    Angular error loss in radians. L1 on angle -- more robust than cosine
    for large errors early in training.

    Args:
        pred_gaze: (B, 3) predicted gaze direction (unit vector)
        gt_gaze: (B, 3) ground-truth gaze direction (unit vector)

    Returns:
        loss: scalar tensor (mean angular error in radians)
    """
    cos_sim = F.cosine_similarity(pred_gaze, gt_gaze, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos_sim).mean()


def total_loss(pred_hm, pred_coords, pred_gaze,
               gt_coords, gt_gaze,
               feat_H, feat_W,
               lam_lm=1.0, lam_gaze=0.5, sigma=2.0):
    """
    Total training loss combining landmarks and gaze.

    Args:
        pred_hm: (B, N, H, W) predicted logit heatmaps
        pred_coords: (B, N, 2) predicted landmark coordinates
        pred_gaze: (B, 3) predicted optical axis (unit vector)
        gt_coords: (B, N, 2) GT landmark coordinates in feature map space
        gt_gaze: (B, 3) GT optical axis (unit vector)
        feat_H, feat_W: feature map spatial dims
        lam_lm: landmark loss weight
        lam_gaze: gaze loss weight
        sigma: heatmap Gaussian sigma

    Returns:
        total: scalar loss
        components: dict of individual loss values (detached, for logging)
    """
    lm = landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma)
    gz = angular_loss(pred_gaze, gt_gaze)
    total = lam_lm * lm + lam_gaze * gz

    components = {
        'landmark_loss': lm.detach(),
        'angular_loss': gz.detach(),
        'angular_loss_deg': torch.rad2deg(gz.detach()),
        'total_loss': total.detach(),
    }
    return total, components
