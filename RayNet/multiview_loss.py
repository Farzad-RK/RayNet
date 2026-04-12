"""
Multi-view consistency losses for RayNet v3.

Ray-based approach: works with unit gaze vectors in camera coordinate
space (CCS), transformed to world frame via R_cam (camera extrinsics).
All values are unit vectors (bounded [-1, 1]), ensuring numerical
stability under AMP.

Two losses:
  1. Gaze ray consistency: predicted gaze vectors from different camera
     views of the same subject, transformed to world coordinates via
     R_cam, should agree in direction.
  2. Landmark shape consistency: the spatial pattern (relative positions)
     of predicted landmarks should be consistent across views.
"""

import torch
import torch.nn.functional as F
import random


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def reshape_multiview(tensor, n_views=9):
    """Reshape flat batch [B*V, ...] -> grouped [G, V, ...]."""
    B_total = tensor.shape[0]
    G = B_total // n_views
    return tensor.view(G, n_views, *tensor.shape[1:])


def _normalize_vec(v, dim=-1, eps=1e-4):
    """L2-normalize vectors along given dimension.

    eps=1e-4 is the minimum safe value for FP16 (min normal ~6e-5).
    """
    return v / (v.norm(dim=dim, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# 1. Gaze Ray Consistency Loss
# ---------------------------------------------------------------------------

def gaze_ray_consistency_loss(pred_gaze, R_cam, n_pairs=3):
    """
    Enforce that predicted gaze vectors from different camera views of the
    same subject point in the same world-frame direction.

    Each view predicts a gaze vector g_v in camera coordinate space (CCS).
    We transform to world frame: g_world_v = R_cam_v^T @ g_v. All g_world_v
    for the same group should be identical.

    R_cam is static per camera (from camera_info.pkl extrinsics), providing
    more stable consistency targets than the old per-frame R_norm.

    Args:
        pred_gaze: (G, V, 3) predicted gaze unit vectors in CCS
        R_cam:     (G, V, 3, 3) camera extrinsic rotation matrices
        n_pairs:   number of random camera pairs to sample per group

    Returns:
        loss: scalar angular consistency loss (radians)
    """
    G, V, _ = pred_gaze.shape
    device = pred_gaze.device

    if V < 2:
        return torch.tensor(0.0, device=device)

    # Transform predicted gaze to world frame: g_world = R_cam^T @ g_pred
    # R_cam is orthogonal, so R_cam^T = R_cam^{-1}
    g_world = torch.einsum('gvji,gvj->gvi', R_cam, pred_gaze)  # (G, V, 3)
    g_world = _normalize_vec(g_world, dim=-1)

    # Compute mean world gaze direction per group as reference
    g_mean = _normalize_vec(g_world.mean(dim=1), dim=-1)  # (G, 3)
    g_mean_exp = g_mean.unsqueeze(1).expand_as(g_world)   # (G, V, 3)

    # L1 loss on unit vectors (stable, matches paper's approach)
    return F.l1_loss(g_world, g_mean_exp.detach())


# ---------------------------------------------------------------------------
# 2. Landmark Shape Consistency Loss
# ---------------------------------------------------------------------------

def landmark_shape_consistency_loss(pred_coords, n_pairs=3):
    """
    Enforce that the spatial pattern of predicted landmarks is consistent
    across views. Since each view has different normalization warps, we
    compare translation-and-scale-invariant landmark shapes (Procrustes).

    For each sampled pair (i, j):
      1. Center both landmark sets (subtract centroid)
      2. Normalize by scale (divide by RMS distance from centroid)
      3. L1 loss between the normalized shapes

    This catches inconsistencies in landmark structure without needing
    to unproject through 3D (which causes numerical issues).

    Args:
        pred_coords: (G, V, N, 2) predicted landmark coordinates in
                     feature map space
        n_pairs: number of random camera pairs to sample

    Returns:
        loss: scalar shape consistency loss
    """
    G, V, N, _ = pred_coords.shape
    device = pred_coords.device

    if V < 2:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for _ in range(n_pairs):
        i, j = random.sample(range(V), 2)

        pts_i = pred_coords[:, i]  # (G, N, 2)
        pts_j = pred_coords[:, j]  # (G, N, 2)

        # Center
        c_i = pts_i.mean(dim=1, keepdim=True)  # (G, 1, 2)
        c_j = pts_j.mean(dim=1, keepdim=True)
        pts_i_c = pts_i - c_i
        pts_j_c = pts_j - c_j

        # Scale-normalize (RMS distance from centroid)
        scale_i = pts_i_c.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)  # (G,1,1)
        scale_j = pts_j_c.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
        pts_i_n = pts_i_c / scale_i
        pts_j_n = pts_j_c / scale_j

        pair_loss = F.smooth_l1_loss(pts_i_n, pts_j_n)
        if torch.isfinite(pair_loss):
            total_loss = total_loss + pair_loss
            n_valid += 1

    return total_loss / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Combined Multi-View Consistency Loss
# ---------------------------------------------------------------------------

def multiview_consistency_loss(pred_gaze, pred_coords, R_cam,
                               lam_gaze_consist=1.0, lam_shape=0.5,
                               n_views=9):
    """
    Orchestrates ray-based multi-view consistency losses.

    Args:
        pred_gaze:   (B_total, 3) predicted gaze unit vectors (CCS)
        pred_coords: (B_total, N, 2) predicted landmark coords (feature space)
        R_cam:       (B_total, 3, 3) camera extrinsic rotation matrices
        lam_gaze_consist: weight for gaze ray consistency loss
        lam_shape:        weight for landmark shape consistency loss
        n_views:     number of camera views per group (9)

    Returns:
        total_mv_loss: scalar
        components: dict of individual loss values (detached, for logging)
    """
    device = pred_gaze.device
    B_total = pred_gaze.shape[0]

    # Check batch is divisible by n_views
    if B_total % n_views != 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {'gaze_consist_loss': zero, 'shape_loss': zero}

    # Reshape to multi-view groups: (G, V, ...)
    gaze_mv = reshape_multiview(pred_gaze, n_views)     # (G, V, 3)
    coords_mv = reshape_multiview(pred_coords, n_views)  # (G, V, N, 2)
    R_cam_mv = reshape_multiview(R_cam, n_views)            # (G, V, 3, 3)

    # Gaze ray consistency
    gaze_consist = gaze_ray_consistency_loss(gaze_mv, R_cam_mv, n_pairs=3)

    # Landmark shape consistency
    shape = landmark_shape_consistency_loss(coords_mv, n_pairs=3)

    total = lam_gaze_consist * gaze_consist + lam_shape * shape

    components = {
        'gaze_consist_loss': gaze_consist.detach(),
        'shape_loss': shape.detach(),
    }
    return total, components
