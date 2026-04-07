"""
Loss functions for RayNet.

Two core losses:
  1. Landmark loss: heatmap MSE + coordinate L1
  2. Gaze loss: L1 on unit gaze vectors (following GazeGene paper Sec 4.1.1)

Angular error is computed for metrics only (not backpropagated).
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


def gaze_loss(pred_gaze, gt_gaze):
    """
    L1 loss on unit gaze vectors, following GazeGene paper (Sec 4.1.1).

    This is numerically stable everywhere (no acos singularity) and matches
    the paper's training procedure.

    Args:
        pred_gaze: (B, 3) predicted gaze direction (unit vector)
        gt_gaze: (B, 3) ground-truth gaze direction (unit vector)

    Returns:
        loss: scalar tensor (mean L1 error on vector components)
    """
    return F.l1_loss(pred_gaze, gt_gaze)


def ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth):
    """
    Ray-to-target constraint: origin + depth * direction ≈ target.

    Uses ground-truth depth to reconstruct the 3D gaze target from
    the predicted gaze direction and known eye center. Provides an
    explicit geometric constraint that ties gaze direction to a
    physical target location.

    Args:
        pred_gaze: (B, 3) predicted gaze direction (unit vector)
        eyeball_center: (B, 3) eyeball center in CCS (ray origin)
        gaze_target: (B, 3) ground-truth 3D gaze target in CCS
        gaze_depth: (B,) ground-truth depth along gaze ray

    Returns:
        loss: scalar ray-target consistency loss
    """
    # Reconstruct target: target_hat = origin + depth * direction
    target_hat = eyeball_center + gaze_depth.unsqueeze(-1) * pred_gaze  # (B, 3)
    return F.l1_loss(target_hat, gaze_target)


def angular_error(pred_gaze, gt_gaze):
    """
    Angular error in radians between predicted and GT gaze vectors.

    Uses atan2 for numerical stability (no gradient singularity at 0° or 180°).
    This is for METRICS ONLY — not used as a training loss.

    Args:
        pred_gaze: (B, 3) predicted gaze direction
        gt_gaze: (B, 3) ground-truth gaze direction

    Returns:
        error: scalar tensor (mean angular error in radians)
    """
    # atan2(||cross||, dot) is stable everywhere unlike acos(dot)
    cross = torch.cross(pred_gaze, gt_gaze, dim=-1)
    dot = (pred_gaze * gt_gaze).sum(dim=-1)
    angle = torch.atan2(cross.norm(dim=-1), dot)
    return angle.mean()


def total_loss(pred_hm, pred_coords, pred_gaze,
               gt_coords, gt_gaze,
               feat_H, feat_W,
               lam_lm=1.0, lam_gaze=0.5, sigma=2.0,
               lam_ray=0.0,
               eyeball_center=None, gaze_target=None, gaze_depth=None):
    """
    Total training loss combining landmarks, gaze, and ray-to-target constraint.

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
        lam_ray: ray-to-target loss weight (0 = disabled)
        eyeball_center: (B, 3) eyeball center in CCS (for ray loss)
        gaze_target: (B, 3) GT 3D gaze target in CCS (for ray loss)
        gaze_depth: (B,) GT depth along gaze ray (for ray loss)

    Returns:
        total: scalar loss
        components: dict of individual loss values (detached, for logging)
    """
    lm = landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma)
    gz = gaze_loss(pred_gaze, gt_gaze)
    total = lam_lm * lm + lam_gaze * gz

    # Angular error for metrics only (detached, not in computation graph)
    with torch.no_grad():
        ang_err = angular_error(pred_gaze, gt_gaze)

    components = {
        'landmark_loss': lm.detach(),
        'angular_loss': ang_err,
        'angular_loss_deg': torch.rad2deg(ang_err),
        'total_loss': total.detach(),
    }

    # Ray-to-target constraint (v4)
    if lam_ray > 0 and eyeball_center is not None and gaze_target is not None:
        ray = ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth)
        total = total + lam_ray * ray
        components['ray_target_loss'] = ray.detach()
        components['total_loss'] = total.detach()

    return total, components
