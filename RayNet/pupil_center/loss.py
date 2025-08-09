# pupil_center/loss.py
# Multi-view pupil-center loss with geometry lifting from ellipse+intrinsics.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PupilCenterRegressionHead as _Geom  # for geometry utilities


def huber(x, delta=1.0):
    return torch.where(
        x.abs() <= delta,
        0.5 * x * x,
        delta * (x.abs() - 0.5 * delta),
    )


def sample_ellipse_points(ellipse, num=64):
    """
    Sample points on ellipse in image space.
    ellipse: [B,2,6] (cx,cy,a,b,cosθ,sinθ)  pixels
    returns: pts [B,2,num,2]
    """
    B = ellipse.shape[0]
    device = ellipse.device
    t = torch.linspace(0, 2 * torch.pi, steps=num, device=device).view(1, 1, num, 1)
    cx, cy, a, b, cth, sth = torch.split(ellipse, [1, 1, 1, 1, 1, 1], dim=-1)
    x0 = a * torch.cos(t)
    y0 = b * torch.sin(t)
    x = cth * x0 - sth * y0 + cx
    y = sth * x0 + cth * y0 + cy
    return torch.cat([x, y], dim=-1)  # [B,2,num,2]


def multiview_pupil_center_losses(
    pred: dict,
    gt_pupil_center_3d: torch.Tensor,
    intrinsics_K: torch.Tensor,
    iris_radius_cm: torch.Tensor,
    gt_iris_mesh_2d: torch.Tensor = None,
    w_3d: float = 1.0,
    w_consistency: float = 0.3,
    w_ellipse2d: float = 0.2,
    huber_delta_cm: float = 0.5,
):
    """
    Multi-view loss for pupil center with geometry.

    Args:
      pred: dict with keys:
         - "ellipse": [B,2,6] (cx,cy,a,b,cosθ,sinθ) pixels
         - "delta_cm": [B,2,1] pupil offset along normal
         - optional "logvar": [B,2,1] (aleatoric variance, if model enabled)
      gt_pupil_center_3d: [B,N,2,3]  in cm (CCS); use N=number of views (e.g., 9)
      intrinsics_K:        [B*N,3,3] intrinsics for each view (cropped image)
      iris_radius_cm:      [B*N] iris radius per view (cm)  (from subject_label / metadata)
      gt_iris_mesh_2d:     [B,N,2,100,2] optional iris landmarks in pixels
      weights: scalars

    Returns:
      dict with 'accuracy' (3D), 'consistency', optional 'ellipse2d', and 'total'
    """
    B, N = gt_pupil_center_3d.shape[:2]
    device = gt_pupil_center_3d.device

    # Repeat predicted per-eye ellipse/delta across N views if your forward was per-view.
    # If your model runs per-view already, adapt this to directly use [B*N, ...].
    ellipse = pred["ellipse"]                      # [B,2,6]
    delta_cm = pred["delta_cm"]                    # [B,2,1]
    if ellipse.dim() == 3:
        ellipse = ellipse.unsqueeze(1).expand(B, N, 2, 6).contiguous()
        delta_cm = delta_cm.unsqueeze(1).expand(B, N, 2, 1).contiguous()

    # Flatten views for geometry lifting
    ellipse_f = ellipse.view(B * N, 2, 6)
    delta_f   = delta_cm.view(B * N, 2, 1)
    K_f       = intrinsics_K.view(B * N, 3, 3)
    R_f       = iris_radius_cm.view(B * N)

    # Lift iris plane & center
    iris_center, iris_normal, depth_cm = _Geom.lift_iris_pose_and_center(ellipse_f, K_f, R_f)
    pupil_center = iris_center + delta_f * iris_normal  # [B*N,2,3]
    pupil_center = pupil_center.view(B, N, 2, 3)

    # ----- 1) 3D accuracy (Huber in cm) -----
    err3d = torch.norm(pupil_center - gt_pupil_center_3d, dim=-1)  # [B,N,2]
    loss_3d = huber(err3d, delta=huber_delta_cm).mean()

    # Optional aleatoric weighting (per-eye): L / exp(s) + s
    if "logvar" in pred:
        s = pred["logvar"]           # [B,2,1] or [B,N,2,1]
        if s.dim() == 3:
            s = s.unsqueeze(1).expand(B, N, 2, 1)
        loss_3d = torch.mean(torch.exp(-s.squeeze(-1)) * err3d + s.squeeze(-1))

    # ----- 2) Cross-view consistency (same CCS regularizer) -----
    mean_pc = pupil_center.mean(dim=1, keepdim=True)         # [B,1,2,3]
    cons = torch.norm(pupil_center - mean_pc, dim=-1)        # [B,N,2]
    loss_cons = cons.mean()

    # ----- 3) 2D ellipse regularizer (optional) -----
    loss_e2d = torch.tensor(0.0, device=device)
    if gt_iris_mesh_2d is not None:
        # Sample points on predicted ellipse and pull to GT landmarks (Chamfer-like)
        pts = sample_ellipse_points(ellipse.view(B * N, 2, 6), num=64).view(B, N, 2, 64, 2)  # [B,N,2,64,2]
        # Simple L2 to nearest GT landmark per eye
        # gt_iris_mesh_2d: [B,N,2,100,2]
        diff = pts.unsqueeze(-2) - gt_iris_mesh_2d.unsqueeze(-3)  # [B,N,2,64,100,2]
        dists = torch.norm(diff, dim=-1)                          # [B,N,2,64,100]
        min_d, _ = dists.min(dim=-1)                              # [B,N,2,64]
        loss_e2d = min_d.mean()

    total = w_3d * loss_3d + w_consistency * loss_cons + w_ellipse2d * loss_e2d

    return {
        "accuracy": w_3d * loss_3d,
        "consistency": w_consistency * loss_cons,
        "ellipse2d": w_ellipse2d * loss_e2d,
        "total": total,
        "pred_center_3d": pupil_center.detach(),  # handy for logging
        "pred_depth_cm": depth_cm.view(B, N, 2, 1).detach(),
    }
