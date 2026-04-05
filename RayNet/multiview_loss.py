"""
Multi-view consistency losses for RayNet.

Enforces geometric constraints across synchronized camera views:
  1. Reprojection consistency: landmarks from cam_i reprojected to cam_j
     should match cam_j's own predictions.
  2. Auxiliary triangulation: mask one camera, triangulate from two others,
     project into masked view — should match the masked camera's predictions.

All geometric operations happen in original camera space (not normalized).
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


def denormalize_landmarks_to_original_px(pred_coords_feat, M_norm_inv,
                                         img_size=224, feat_size=56):
    """
    Convert predicted landmarks from feature-map space back to original
    camera pixel space by undoing the Zhang normalization warp.

    Args:
        pred_coords_feat: (G, V, N, 2) landmarks in feature-map coords
        M_norm_inv:       (G, V, 3, 3) inverse normalization homography
        img_size:  normalized image size (224)
        feat_size: feature map size (56)

    Returns:
        coords_original_px: (G, V, N, 2) in original camera pixel space
    """
    scale = img_size / feat_size  # 4.0
    coords_px = pred_coords_feat * scale  # -> normalized pixel space

    # Homogeneous coordinates: append 1
    ones = torch.ones(*coords_px.shape[:-1], 1,
                       device=coords_px.device, dtype=coords_px.dtype)
    coords_h = torch.cat([coords_px, ones], dim=-1)  # (G, V, N, 3)

    # Apply inverse warp: M_inv @ [u, v, 1]^T per point
    # einsum: (G,V,3,3) x (G,V,N,3) -> (G,V,N,3)   (contract last dim)
    warped = torch.einsum('gvij,gvnj->gvni', M_norm_inv, coords_h)

    # Dehomogenize
    coords_original = warped[..., :2] / (warped[..., 2:3] + 1e-8)
    return coords_original


def unproject_2d_to_3d(pts_2d, K_inv, depth):
    """
    Unproject 2D pixel coordinates to 3D camera coordinates.

    Args:
        pts_2d: (..., N, 2) pixel coordinates
        K_inv:  (..., 3, 3) inverse intrinsic matrix
        depth:  (..., 1, 1) depth value (Z) for all points

    Returns:
        pts_3d: (..., N, 3) 3D points in camera frame
    """
    ones = torch.ones(*pts_2d.shape[:-1], 1,
                       device=pts_2d.device, dtype=pts_2d.dtype)
    pts_h = torch.cat([pts_2d, ones], dim=-1)  # (..., N, 3)

    # K_inv @ [u, v, 1]^T gives normalized ray direction
    rays = torch.einsum('...ij,...nj->...ni', K_inv, pts_h)  # (..., N, 3)

    # Scale by depth
    pts_3d = rays * depth  # broadcast depth over N points
    return pts_3d


def project_3d_to_2d(pts_3d, K):
    """
    Project 3D camera-frame points to 2D pixel coordinates.

    Args:
        pts_3d: (..., N, 3) points in camera frame
        K:      (..., 3, 3) intrinsic matrix

    Returns:
        pts_2d: (..., N, 2) pixel coordinates
    """
    # K @ P^T -> (u*z, v*z, z)
    proj = torch.einsum('...ij,...nj->...ni', K, pts_3d)  # (..., N, 3)
    pts_2d = proj[..., :2] / (proj[..., 2:3].clamp(min=1.0))
    return pts_2d


def transform_points(pts, R_src, T_src, R_dst, T_dst):
    """
    Transform 3D points from source camera frame to destination camera frame.

    P_world = R_src^T @ (P_src - T_src)
    P_dst   = R_dst @ P_world + T_dst

    Args:
        pts:   (..., N, 3) points in source camera frame
        R_src: (..., 3, 3) source camera rotation
        T_src: (..., 3)    source camera translation
        R_dst: (..., 3, 3) destination camera rotation
        T_dst: (..., 3)    destination camera translation

    Returns:
        pts_dst: (..., N, 3) points in destination camera frame
    """
    # Relative rotation: R_dst @ R_src^T
    R_rel = torch.einsum('...ij,...kj->...ik', R_dst, R_src)  # (..., 3, 3)

    # Relative translation: T_dst - R_rel @ T_src
    t_rel = T_dst - torch.einsum('...ij,...j->...i', R_rel, T_src)  # (..., 3)

    # Transform: R_rel @ P + t_rel
    pts_dst = torch.einsum('...ij,...nj->...ni', R_rel, pts) + t_rel.unsqueeze(-2)
    return pts_dst


# ---------------------------------------------------------------------------
# Reprojection Consistency Loss
# ---------------------------------------------------------------------------

def reprojection_consistency_loss(pred_coords_feat, M_norm_inv, K, R_cam, T_cam,
                                  eyeball_center_3d, n_pairs=2,
                                  img_size=224, feat_size=56):
    """
    Cross-view reprojection consistency loss.

    Samples random camera pairs. For each pair (i, j), unprojects cam_i's
    predicted landmarks to 3D (using GT depth), transforms to cam_j's frame,
    projects to 2D, and penalizes discrepancy with cam_j's own predictions.

    Args:
        pred_coords_feat: (G, V, N, 2) predicted landmarks in feature space
        M_norm_inv:       (G, V, 3, 3) inverse normalization warp
        K:                (G, V, 3, 3) camera intrinsics
        R_cam:            (G, V, 3, 3) camera extrinsic rotation
        T_cam:            (G, V, 3)    camera extrinsic translation
        eyeball_center_3d:(G, V, 3)    eye center in each camera's coords
        n_pairs:          number of random camera pairs to sample
        img_size, feat_size: image and feature map sizes

    Returns:
        loss: scalar reprojection consistency loss
    """
    G, V, N, _ = pred_coords_feat.shape
    device = pred_coords_feat.device

    # Denormalize all predictions to original camera pixel space
    pts_orig = denormalize_landmarks_to_original_px(
        pred_coords_feat, M_norm_inv, img_size, feat_size)  # (G, V, N, 2)

    # Precompute inverse intrinsics
    K_inv = torch.linalg.inv(K)  # (G, V, 3, 3)

    # Reasonable pixel range for clamping reprojected points
    max_px = img_size * 4.0  # allow some margin beyond image bounds

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for _ in range(n_pairs):
        # Sample two distinct cameras
        i, j = random.sample(range(V), 2)

        # Depth from GT eyeball center (Z component), clamped for stability
        Z_i = eyeball_center_3d[:, i, 2:3].clamp(min=100.0)  # (G, 1)
        Z_j = eyeball_center_3d[:, j, 2:3].clamp(min=100.0)  # (G, 1)

        # --- Direction i -> j ---
        pts_i_3d = unproject_2d_to_3d(
            pts_orig[:, i],          # (G, N, 2)
            K_inv[:, i],             # (G, 3, 3)
            Z_i[:, :, None])         # (G, 1, 1)

        pts_i_in_j = transform_points(
            pts_i_3d,                # (G, N, 3)
            R_cam[:, i], T_cam[:, i],
            R_cam[:, j], T_cam[:, j])

        pts_i_proj_j = project_3d_to_2d(pts_i_in_j, K[:, j])  # (G, N, 2)
        pts_i_proj_j = pts_i_proj_j.clamp(-max_px, max_px)

        loss_ij = F.smooth_l1_loss(pts_i_proj_j, pts_orig[:, j])

        # --- Direction j -> i ---
        pts_j_3d = unproject_2d_to_3d(
            pts_orig[:, j],
            K_inv[:, j],
            Z_j[:, :, None])

        pts_j_in_i = transform_points(
            pts_j_3d,
            R_cam[:, j], T_cam[:, j],
            R_cam[:, i], T_cam[:, i])

        pts_j_proj_i = project_3d_to_2d(pts_j_in_i, K[:, i])  # (G, N, 2)
        pts_j_proj_i = pts_j_proj_i.clamp(-max_px, max_px)

        loss_ji = F.smooth_l1_loss(pts_j_proj_i, pts_orig[:, i])

        pair_loss = (loss_ij + loss_ji) * 0.5
        if torch.isfinite(pair_loss):
            total_loss = total_loss + pair_loss
            n_valid += 1

    return total_loss / max(n_valid, 1)


# ---------------------------------------------------------------------------
# DLT Triangulation (batched)
# ---------------------------------------------------------------------------

def triangulate_dlt_batch(pts_a, pts_b, P_a, P_b):
    """
    Triangulate 3D world points from two 2D observations using DLT.

    The result is detached so gradients do not flow through the SVD,
    ensuring stable training. Supervision flows only through the
    prediction that is compared against this triangulated pseudo-GT.

    Args:
        pts_a: (G, 2) 2D pixel coords in camera A
        pts_b: (G, 2) 2D pixel coords in camera B
        P_a:   (G, 3, 4) projection matrix for camera A  [K @ [R | t]]
        P_b:   (G, 3, 4) projection matrix for camera B

    Returns:
        X_world: (G, 3) triangulated 3D points (detached)
    """
    G = pts_a.shape[0]

    # Build the 4x4 linear system  A @ X = 0
    # Row 0: x_a * P_a[2] - P_a[0]
    # Row 1: y_a * P_a[2] - P_a[1]
    # Row 2: x_b * P_b[2] - P_b[0]
    # Row 3: y_b * P_b[2] - P_b[1]
    x_a, y_a = pts_a[:, 0:1], pts_a[:, 1:2]  # (G, 1)
    x_b, y_b = pts_b[:, 0:1], pts_b[:, 1:2]

    A = torch.zeros(G, 4, 4, device=pts_a.device, dtype=pts_a.dtype)
    A[:, 0] = x_a * P_a[:, 2] - P_a[:, 0]
    A[:, 1] = y_a * P_a[:, 2] - P_a[:, 1]
    A[:, 2] = x_b * P_b[:, 2] - P_b[:, 0]
    A[:, 3] = y_b * P_b[:, 2] - P_b[:, 1]

    # Normalize rows of A for numerical stability (Hartley-style)
    row_norms = A.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    A = A / row_norms

    # Solve via SVD: X is the last column of V (smallest singular value)
    # CUDA batched SVD does not support float16; cast to float32 under AMP.
    _, _, Vh = torch.linalg.svd(A.float())
    X_h = Vh[:, -1, :]  # (G, 4) homogeneous

    # Dehomogenize — clamp w away from zero to avoid explosion
    w = X_h[:, 3:4]
    w = torch.where(w.abs() < 1e-6, torch.ones_like(w), w)
    X_world = X_h[:, :3] / w

    return X_world.detach()


# ---------------------------------------------------------------------------
# Auxiliary Masking / Triangulation Loss
# ---------------------------------------------------------------------------

def triangulation_masking_loss(pred_coords_feat, M_norm_inv, K, R_cam, T_cam,
                               img_size=224, feat_size=56):
    """
    Auxiliary triangulation loss: mask one camera, triangulate the eye center
    from two other cameras, project into the masked camera's view, and
    penalize the discrepancy with the masked camera's own prediction.

    Args:
        pred_coords_feat: (G, V, N, 2) predicted landmarks in feature space
        M_norm_inv:       (G, V, 3, 3) inverse normalization warp
        K:                (G, V, 3, 3) camera intrinsics
        R_cam:            (G, V, 3, 3) camera extrinsic rotation
        T_cam:            (G, V, 3)    camera extrinsic translation

    Returns:
        loss: scalar triangulation masking loss
    """
    G, V, N, _ = pred_coords_feat.shape
    device = pred_coords_feat.device

    if V < 3:
        return torch.tensor(0.0, device=device)

    # Denormalize to original pixel space
    pts_orig = denormalize_landmarks_to_original_px(
        pred_coords_feat, M_norm_inv, img_size, feat_size)  # (G, V, N, 2)

    # Compute landmark centroids as eye center proxy
    centroids = pts_orig.mean(dim=2)  # (G, V, 2)

    # Randomly select 3 distinct cameras
    cam_indices = random.sample(range(V), 3)
    mask_cam = cam_indices[0]
    tri_a = cam_indices[1]
    tri_b = cam_indices[2]

    # Build projection matrices P = K @ [R | T] for triangulation cameras
    # P is (3, 4): [K @ R | K @ T]
    def build_projection_matrix(K_v, R_v, T_v):
        """K_v: (G,3,3), R_v: (G,3,3), T_v: (G,3) -> P: (G,3,4)"""
        KR = torch.bmm(K_v, R_v)                     # (G, 3, 3)
        KT = torch.bmm(K_v, T_v.unsqueeze(-1))       # (G, 3, 1)
        return torch.cat([KR, KT], dim=-1)            # (G, 3, 4)

    P_a = build_projection_matrix(K[:, tri_a], R_cam[:, tri_a], T_cam[:, tri_a])
    P_b = build_projection_matrix(K[:, tri_b], R_cam[:, tri_b], T_cam[:, tri_b])

    # Triangulate eye center in world coordinates (detached)
    eye_3d_world = triangulate_dlt_batch(
        centroids[:, tri_a],  # (G, 2)
        centroids[:, tri_b],  # (G, 2)
        P_a, P_b)             # (G, 3)

    # Project triangulated point into masked camera's image
    # Transform world -> masked camera frame: P_mask = R_mask @ P_world + T_mask
    eye_in_mask = (torch.bmm(R_cam[:, mask_cam], eye_3d_world.unsqueeze(-1)).squeeze(-1)
                   + T_cam[:, mask_cam])  # (G, 3)

    # Project to 2D: K_mask @ eye_in_mask
    proj = torch.bmm(K[:, mask_cam], eye_in_mask.unsqueeze(-1)).squeeze(-1)  # (G, 3)
    proj_2d = proj[:, :2] / (proj[:, 2:3].clamp(min=1.0))  # (G, 2)

    # Compare with masked camera's actual predicted centroid
    gt_centroid = centroids[:, mask_cam]  # (G, 2)

    # Filter out samples where triangulation produced outliers
    error = (proj_2d - gt_centroid).abs()
    max_px = img_size * 4.0
    valid = (error < max_px).all(dim=-1)  # (G,)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=proj_2d.device)

    return F.smooth_l1_loss(proj_2d[valid], gt_centroid[valid])


# ---------------------------------------------------------------------------
# Combined Multi-View Consistency Loss
# ---------------------------------------------------------------------------

def multiview_consistency_loss(pred_coords_feat, batch_meta,
                               lam_reproj=0.2, lam_mask=0.1,
                               n_views=9, n_reproj_pairs=2,
                               img_size=224, feat_size=56):
    """
    Orchestrates all multi-view consistency losses.

    Args:
        pred_coords_feat: (B_total, N, 2) raw model landmark predictions
        batch_meta: dict with keys:
            K:                (B_total, 3, 3)
            R_cam:            (B_total, 3, 3)
            T_cam:            (B_total, 3)
            M_norm_inv:       (B_total, 3, 3)
            eyeball_center_3d:(B_total, 3)
        lam_reproj: weight for reprojection consistency loss
        lam_mask:   weight for auxiliary triangulation loss
        n_views:    number of camera views per group (9)
        n_reproj_pairs: number of camera pairs to sample for reprojection
        img_size, feat_size: image and feature map sizes

    Returns:
        total_mv_loss: scalar
        components: dict of individual loss values (detached, for logging)
    """
    device = pred_coords_feat.device
    B_total = pred_coords_feat.shape[0]

    # Check batch is divisible by n_views
    if B_total % n_views != 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {'reproj_loss': zero.detach(), 'mask_loss': zero.detach()}

    # Reshape to multi-view groups: (G, V, ...)
    coords = reshape_multiview(pred_coords_feat, n_views)
    K = reshape_multiview(batch_meta['K'], n_views)
    R_cam = reshape_multiview(batch_meta['R_cam'], n_views)
    T_cam = reshape_multiview(batch_meta['T_cam'], n_views)
    M_inv = reshape_multiview(batch_meta['M_norm_inv'], n_views)
    eye_3d = reshape_multiview(batch_meta['eyeball_center_3d'], n_views)

    # Reprojection consistency
    reproj = reprojection_consistency_loss(
        coords, M_inv, K, R_cam, T_cam, eye_3d,
        n_pairs=n_reproj_pairs, img_size=img_size, feat_size=feat_size)

    # Auxiliary triangulation masking
    mask = triangulation_masking_loss(
        coords, M_inv, K, R_cam, T_cam,
        img_size=img_size, feat_size=feat_size)

    total = lam_reproj * reproj + lam_mask * mask

    components = {
        'reproj_loss': reproj.detach(),
        'mask_loss': mask.detach(),
    }
    return total, components


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------

def sanity_check_roundtrip(dataset, n_samples=50, threshold_px=2.0):
    """
    Verify normalization invertibility: warp 2D landmarks with M then
    unwarp with M_inv. Round-trip error should be < threshold_px.

    Args:
        dataset: GazeGeneDataset instance
        n_samples: number of samples to test
        threshold_px: maximum acceptable round-trip error in pixels

    Returns:
        max_error: maximum pixel error across all tested samples
        passed: bool
    """
    import numpy as np
    max_error = 0.0
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        M_inv = sample['M_norm_inv'].numpy()  # (3, 3)
        lm_px = sample['landmark_coords_px'].numpy()  # (14, 2) in normalized pixel space

        # Forward: normalized px -> original px via M_inv
        ones = np.ones((lm_px.shape[0], 1), dtype=lm_px.dtype)
        pts_h = np.concatenate([lm_px, ones], axis=1)
        warped_h = (M_inv @ pts_h.T).T
        original_px = warped_h[:, :2] / (warped_h[:, 2:3] + 1e-8)

        # Backward: original px -> normalized px via M (= inv(M_inv))
        M = np.linalg.inv(M_inv)
        ones2 = np.ones((original_px.shape[0], 1), dtype=original_px.dtype)
        pts_h2 = np.concatenate([original_px, ones2], axis=1)
        back_h = (M @ pts_h2.T).T
        back_px = back_h[:, :2] / (back_h[:, 2:3] + 1e-8)

        error = np.max(np.linalg.norm(back_px - lm_px, axis=1))
        max_error = max(max_error, error)

    passed = max_error < threshold_px
    print(f"Roundtrip sanity check: max_error={max_error:.4f}px, "
          f"threshold={threshold_px}px, {'PASSED' if passed else 'FAILED'}")
    return max_error, passed
