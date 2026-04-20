"""
Loss functions for RayNet v4.1 / v5.

Core losses (v4.1):
  1. Landmark loss: heatmap MSE + coordinate L1
  2. Gaze loss: L1 on unit gaze vectors (following GazeGene paper Sec 4.1.1)
  3. Pose rotation loss: geodesic loss on SO(3) with 6D rotation representation
  4. Pose translation loss: SmoothL1 on xy + log-space SmoothL1 on z (depth)
  5. Ray-to-target loss: geometric constraint tying gaze direction to 3D target

GazeGene 3D Eyeball Structure losses (v5, Sec 4.2.2):
  6. Eyeball center L1: L1 on predicted vs GT eyeball center 3D
  7. Pupil center L1: L1 on predicted vs GT pupil center 3D
  8. Geometric angular error: angular error between optical axis derived from
     predicted geometry (normalize(pupil - eyeball)) and GT optical axis
  (Loss 3 from paper = iris contour L1, handled by landmark_loss on the
   10 iris contour points within the 14 landmarks)

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
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
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

    # arccos((trace - 1) / 2), clamped for numerical stability.
    # Cast to float32 before acos: under BF16 autocast, a clamp margin of
    # 1e-7 is below BF16 epsilon (~7.8e-3), so (1 - 1e-7) rounds to exactly
    # 1.0, acos' derivative becomes -1/sqrt(0) = -inf, and every backward
    # produces nan gradients on batch 1.
    cos_angle = ((trace - 1.0) / 2.0).float()
    cos_angle = cos_angle.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
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

    IMPORTANT — scale-invariant formulation:
        The naive version `F.l1_loss(origin + depth * dir, target)` is
        position-space L1 in whatever unit gaze_depth is stored in. For
        the GazeGene data used here, raw values reach 10^4 magnitude, so
        this loss overwhelms every other term the moment it turns on
        (previously epoch 6 of Stage 2 P2, where ray_target jumped to
        ~7000–26000 per batch and poisoned the shared backbone).

        We normalize by per-sample gaze_depth before the L1. This makes
        the loss numerically equivalent to a bounded angular-error term
        on the ray endpoint, independent of scene scale / units:
            |origin/depth + dir - target/depth|
        which is O(|pred_gaze - unit_direction_to_target|).

    Args:
        pred_gaze: (B, 3) predicted gaze direction (unit vector)
        eyeball_center: (B, 3) eyeball center in CCS (ray origin)
        gaze_target: (B, 3) ground-truth 3D gaze target in CCS
        gaze_depth: (B,) ground-truth depth along gaze ray

    Returns:
        loss: scalar, O(1) regardless of gaze_depth unit/magnitude
    """
    # Clamp depth to avoid division explosions on any pathological samples
    # (zero/negative depth shouldn't exist in GT, but be defensive in BF16).
    depth = gaze_depth.unsqueeze(-1).clamp(min=1.0)  # (B, 1)

    # Reconstruct target in depth-normalized coordinates:
    #   target_hat / depth = eyeball_center / depth + pred_gaze
    #   gaze_target / depth (the GT reference)
    target_hat_norm = eyeball_center / depth + pred_gaze          # (B, 3)
    gaze_target_norm = gaze_target / depth                        # (B, 3)
    return F.l1_loss(target_hat_norm, gaze_target_norm)


def eyeball_center_loss(pred_eyeball, gt_eyeball):
    """
    L1 loss on predicted vs GT eyeball center (GazeGene Sec 4.2.2, loss 1).

    Both in CCS, centimeters.

    Args:
        pred_eyeball: (B, 3) predicted eyeball center
        gt_eyeball: (B, 3) GT eyeball center

    Returns:
        loss: scalar L1
    """
    return F.l1_loss(pred_eyeball, gt_eyeball)


def pupil_center_loss(pred_pupil, gt_pupil):
    """
    L1 loss on predicted vs GT pupil center (GazeGene Sec 4.2.2, loss 2).

    Both in CCS, centimeters.

    Args:
        pred_pupil: (B, 3) predicted pupil center
        gt_pupil: (B, 3) GT pupil center

    Returns:
        loss: scalar L1
    """
    return F.l1_loss(pred_pupil, gt_pupil)


def geometric_angular_loss(pred_eyeball, pred_pupil, gt_optical_axis):
    """
    Angular error between optical axis derived from predicted 3D geometry
    and GT optical axis (GazeGene Sec 4.2.2, loss 4).

    optical_axis_pred = normalize(pupil_center - eyeball_center)

    This loss ensures that the predicted 3D structure is GEOMETRICALLY
    CONSISTENT with the gaze direction — the model can't cheat by
    predicting correct eyeball/pupil positions but wrong relative
    direction.

    Uses atan2 for numerical stability.

    Args:
        pred_eyeball: (B, 3) predicted eyeball center
        pred_pupil: (B, 3) predicted pupil center
        gt_optical_axis: (B, 3) GT optical axis (unit vector)

    Returns:
        loss: scalar angular error in radians
    """
    pred_axis = F.normalize(pred_pupil - pred_eyeball, dim=-1)
    cross = torch.cross(pred_axis, gt_optical_axis, dim=-1)
    dot = (pred_axis * gt_optical_axis).sum(dim=-1)
    angle = torch.atan2(cross.norm(dim=-1), dot)
    return angle.mean()


def mask_seg_loss(pred_logits, gt_mask_uint8):
    """
    Binary segmentation BCE-with-logits for AERI iris / eyeball masks.

    GT masks are stored as uint8 {0, 255} in the shards (see
    RayNet.streaming.eye_masks). They're normalised to float {0, 1} here
    before BCE. Pred shape (B, H, W), GT shape (B, H, W); both are
    single-channel binary masks at 56x56.

    Args:
        pred_logits: (B, H, W) raw logits from AERIHead.
        gt_mask_uint8: (B, H, W) GT mask in uint8 {0, 255}.

    Returns:
        loss: scalar BCE-with-logits (mean over all pixels).
    """
    gt = gt_mask_uint8.to(pred_logits.dtype) / 255.0
    return F.binary_cross_entropy_with_logits(pred_logits, gt)


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


def total_loss(
    pred_hm, pred_coords, pred_gaze,
    gt_coords, gt_gaze,
    feat_H, feat_W,
    lam_lm=1.0, lam_gaze=0.5, sigma=2.0,
    # GazeGene 3D eyeball structure (Sec 4.2.2)
    lam_eyeball=0.0,
    pred_eyeball=None, gt_eyeball=None,
    lam_pupil=0.0,
    pred_pupil=None, gt_pupil=None,
    lam_geom_angular=0.0,
    # Ray-to-target
    lam_ray=0.0,
    eyeball_center=None, gaze_target=None, gaze_depth=None,
    # Pose
    lam_pose=0.0,
    pred_pose_6d=None, gt_head_R=None,
    lam_trans=0.0,
    pred_pose_t=None, gt_head_t=None,
    # AERI segmentation (iris + eyeball binary masks @ 56x56)
    lam_iris_seg=0.0,
    pred_iris_mask_logits=None, gt_iris_mask=None,
    lam_eyeball_seg=0.0,
    pred_eyeball_mask_logits=None, gt_eyeball_mask=None,
):
    """
    Total training loss: landmarks + gaze + GazeGene 3D structure + pose.

    GazeGene 4 losses (Sec 4.2.2):
      1. lam_eyeball * L1(pred_eyeball, gt_eyeball)    — eyeball center
      2. lam_pupil * L1(pred_pupil, gt_pupil)           — pupil center
      3. lam_lm (iris subset) — iris contour L1 (handled within landmark_loss)
      4. lam_geom_angular * angular_error(normalize(pupil-eyeball), gt_gaze)

    Args:
        pred_hm: (B, N, H, W) predicted logit heatmaps
        pred_coords: (B, N, 2) predicted landmark coordinates
        pred_gaze: (B, 3) predicted optical axis (unit vector, derived from geometry)
        gt_coords: (B, N, 2) GT landmark coordinates in feature map space
        gt_gaze: (B, 3) GT optical axis (unit vector)
        feat_H, feat_W: feature map spatial dims
        lam_lm, lam_gaze, sigma: standard loss config
        lam_eyeball: weight for eyeball center L1 (0 = disabled)
        pred_eyeball: (B, 3) predicted eyeball center
        gt_eyeball: (B, 3) GT eyeball center
        lam_pupil: weight for pupil center L1 (0 = disabled)
        pred_pupil: (B, 3) predicted pupil center
        gt_pupil: (B, 3) GT pupil center
        lam_geom_angular: weight for geometric angular error (0 = disabled)
        lam_ray, lam_pose, lam_trans: existing loss weights

    Returns:
        total: scalar loss
        components: dict of individual loss values (detached, for logging)
    """
    lm = landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma)
    gz = gaze_loss(pred_gaze, gt_gaze)
    total = lam_lm * lm + lam_gaze * gz

    with torch.no_grad():
        ang_err = angular_error(pred_gaze, gt_gaze)

    components = {
        'landmark_loss': lm.detach(),
        'gaze_loss': gz.detach(),
        'angular_loss': ang_err,
        'angular_loss_deg': torch.rad2deg(ang_err),
        'total_loss': total.detach(),
    }

    # GazeGene loss 1: eyeball center L1
    if lam_eyeball > 0 and pred_eyeball is not None and gt_eyeball is not None:
        eb_loss = eyeball_center_loss(pred_eyeball, gt_eyeball)
        total = total + lam_eyeball * eb_loss
        components['eyeball_center_loss'] = eb_loss.detach()

    # GazeGene loss 2: pupil center L1
    if lam_pupil > 0 and pred_pupil is not None and gt_pupil is not None:
        pc_loss = pupil_center_loss(pred_pupil, gt_pupil)
        total = total + lam_pupil * pc_loss
        components['pupil_center_loss'] = pc_loss.detach()

    # GazeGene loss 4: geometric angular error (from predicted structure)
    if lam_geom_angular > 0 and pred_eyeball is not None and pred_pupil is not None:
        ga_loss = geometric_angular_loss(pred_eyeball, pred_pupil, gt_gaze)
        total = total + lam_geom_angular * ga_loss
        components['geometric_angular_loss'] = ga_loss.detach()
        components['geometric_angular_loss_deg'] = torch.rad2deg(ga_loss.detach())

    # Ray-to-target constraint
    if lam_ray > 0 and eyeball_center is not None and gaze_target is not None and gaze_depth is not None:
        ray = ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth)
        total = total + lam_ray * ray
        components['ray_target_loss'] = ray.detach()

    # Pose rotation loss
    if lam_pose > 0 and pred_pose_6d is not None and gt_head_R is not None:
        pose = pose_prediction_loss(pred_pose_6d, gt_head_R)
        total = total + lam_pose * pose
        components['pose_loss'] = pose.detach()
        components['pose_loss_deg'] = torch.rad2deg(pose.detach())

    # Pose translation loss
    if lam_trans > 0 and pred_pose_t is not None and gt_head_t is not None:
        trans = translation_loss(pred_pose_t, gt_head_t)
        total = total + lam_trans * trans
        components['translation_loss'] = trans.detach()

    # AERI segmentation losses (iris + eyeball masks, BCE-with-logits @ 56x56)
    if (lam_iris_seg > 0 and pred_iris_mask_logits is not None
            and gt_iris_mask is not None):
        iris_seg = mask_seg_loss(pred_iris_mask_logits, gt_iris_mask)
        total = total + lam_iris_seg * iris_seg
        components['iris_seg_loss'] = iris_seg.detach()

    if (lam_eyeball_seg > 0 and pred_eyeball_mask_logits is not None
            and gt_eyeball_mask is not None):
        eyeball_seg = mask_seg_loss(pred_eyeball_mask_logits, gt_eyeball_mask)
        total = total + lam_eyeball_seg * eyeball_seg
        components['eyeball_seg_loss'] = eyeball_seg.detach()

    components['landmark_loss'] *= 1.0 / (feat_H * feat_W)
    components['total_loss'] = total.detach()

    return total, components


def translation_loss(pred_t, gt_t, gt_scale_cm_per_m=100.0):
    """
    Direct-metric SmoothL1 translation loss in METERS.

    pred_t is the raw Linear output of the pose head, interpreted as
    translation in meters. gt_t is GazeGene CCS head translation in
    CENTIMETERS — we divide by `gt_scale_cm_per_m` on the fly so both
    sides live in the same metric space with O(0.1-1) magnitudes.

    Why this replaces the earlier tanh(xy) + exp(z) / log-space split:
        The previous head applied tanh on xy, clipping predictions to
        [-1, 1] while GT head_t sits in cm (typical |tx|, |ty| = 5-30 cm,
        |tz| = 40-100 cm). The loss was mathematically incapable of
        matching GT: at best it plateaued at ~mean(|gt_xy|) - 1, with
        gradients killed by tanh saturation. Observed plateau in
        run_20260411_050522 was 0.93 over 20 epochs of active training.

        Direct SmoothL1 on an unbounded linear head restores full range,
        Huber-clips outliers, and gives O(1) gradients everywhere.

    Args:
        pred_t: (B, 3) predicted translation in meters (raw linear head)
        gt_t:   (B, 3) GT translation in centimeters (GazeGene convention)
        gt_scale_cm_per_m: cm→m conversion factor (default 100.0)

    Returns:
        loss: scalar SmoothL1, units of meters
    """
    gt_m = gt_t / gt_scale_cm_per_m
    return F.smooth_l1_loss(pred_t, gt_m)
