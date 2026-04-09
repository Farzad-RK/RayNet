"""
Loss functions for RayNet v4.1.

Core losses:
  1. Landmark loss: heatmap MSE + coordinate L1
  2. Gaze loss: L1 on unit gaze vectors (following GazeGene paper Sec 4.1.1)
  3. Pose rotation loss: geodesic loss on SO(3) with 6D rotation representation
  4. Pose translation loss: SmoothL1 on xy + log-space SmoothL1 on z (depth)
  5. Ray-to-target loss: geometric constraint tying gaze direction to 3D target

Angular error is computed for metrics only (not backpropagated).
"""

import torch
import torch.nn.functional as F
import math


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


def rotation_6d_to_matrix(r6d):
    """
    Convert 6D rotation representation to 3×3 rotation matrix.

    Uses Gram-Schmidt orthogonalization (Zhou et al., "On the Continuity
    of Rotation Representations in Neural Networks", CVPR 2019).

    The 6D representation is the optimal continuous representation of SO(3)
    for neural networks — unlike quaternions, Euler angles, or axis-angle,
    it has no discontinuities or singularities.

    Args:
        r6d: (B, 6) — first two columns of the rotation matrix

    Returns:
        R: (B, 3, 3) proper rotation matrix (orthogonal, det=+1)
    """
    a1 = r6d[:, 0:3]  # first column
    a2 = r6d[:, 3:6]  # second column

    # Gram-Schmidt: orthogonalize and normalize
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)


def matrix_to_rotation_6d(R):
    """
    Convert 3×3 rotation matrix to 6D representation.

    Args:
        R: (B, 3, 3) rotation matrix

    Returns:
        r6d: (B, 6) first two columns of R, flattened
    """
    return R[:, :, :2].reshape(-1, 6)


def geodesic_loss(pred_R, gt_R):
    """
    Geodesic loss on SO(3): angular distance between two rotations.

    L = arccos( (trace(R_pred^T @ R_gt) - 1) / 2 )

    This measures the actual angle of the rotation needed to go from
    pred_R to gt_R, respecting the SO(3) manifold geometry. Unlike L1/L2
    on matrix elements, geodesic loss treats all rotation axes equally.

    Numerically stabilized with clamp to avoid NaN from arccos at ±1.

    Args:
        pred_R: (B, 3, 3) predicted rotation matrix
        gt_R: (B, 3, 3) ground-truth rotation matrix

    Returns:
        loss: scalar mean geodesic distance in radians
    """
    # R_diff = R_pred^T @ R_gt
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

    # arccos((trace - 1) / 2), clamped for numerical stability
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    return angle.mean()


def pose_prediction_loss(pred_6d, gt_head_R):
    """
    Auxiliary pose prediction loss with 6D representation + geodesic loss.

    The PoseEncoder predicts 6D rotation (first two columns of R),
    which is reconstructed to a proper rotation matrix via Gram-Schmidt,
    then compared to GT using geodesic distance on SO(3).

    This is the same approach as SixDRepNet (Hempel et al., 2022).

    Args:
        pred_6d: (B, 6) predicted 6D rotation representation
        gt_head_R: (B, 3, 3) ground-truth head rotation matrix

    Returns:
        loss: scalar geodesic loss in radians
    """
    pred_R = rotation_6d_to_matrix(pred_6d)  # (B, 3, 3)
    return geodesic_loss(pred_R, gt_head_R)


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
               eyeball_center=None, gaze_target=None, gaze_depth=None,
               lam_pose=0.0,
               pred_pose_6d=None, gt_head_R=None,
               lam_trans=0.0,
               pred_pose_t=None, gt_head_t=None
               ):
    """
    Total training loss combining landmarks, gaze, ray-to-target, and pose.

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
        lam_pose: pose prediction auxiliary loss weight (0 = disabled)
        pred_pose_6d: (B, 6) predicted 6D rotation from PoseEncoder
        gt_head_R: (B, 3, 3) GT head rotation matrix

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
    if lam_ray > 0 and eyeball_center is not None and gaze_target is not None and gaze_depth is not None:
        ray = ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth)
        total = total + lam_ray * ray
        components['ray_target_loss'] = ray.detach()

    # Auxiliary pose prediction loss (v4.1, 6D repr + geodesic)
    if lam_pose > 0 and pred_pose_6d is not None and gt_head_R is not None:
        pose = pose_prediction_loss(pred_pose_6d, gt_head_R)
        total = total + lam_pose * pose
        components['pose_loss'] = pose.detach()
        components['pose_loss_deg'] = torch.rad2deg(pose.detach())

    if lam_trans > 0 and pred_pose_t is not None and gt_head_t is not None:
        trans = translation_loss(pred_pose_t, gt_head_t)
        total = total + lam_trans * trans
        components['translation_loss'] = trans.detach()

    components['landmark_loss'] *= 1.0 / (feat_H * feat_W)
    components['total_loss'] = total.detach()

    return total, components


def translation_loss(pred_t, gt_t, eps=1e-6):
    """
    Translation loss with log-depth normalization.

    gt_t must be:
      - tx, ty normalized to [-1, 1]
      - tz in metric depth (positive)

    Args:
        pred_t: (B, 3) predicted translation
        gt_t: (B, 3) ground-truth translation

    Returns:
        loss: scalar
    """
    pred_xy = pred_t[:, :2]
    pred_z = pred_t[:, 2:3]

    gt_xy = gt_t[:, :2]
    gt_z = gt_t[:, 2:3]

    # XY (image-plane)
    loss_xy = F.smooth_l1_loss(pred_xy, gt_xy)

    # Z (log space → scale invariant)
    pred_z = pred_z.clamp(min=eps)
    gt_z = gt_z.clamp(min=eps)

    loss_z = F.smooth_l1_loss(torch.log(pred_z), torch.log(gt_z))

    return loss_xy + loss_z
