# pupil_center/loss.py
# Multi-view pupil-center loss with ellipse->3D lifting from intrinsics + iris radius.

import os
import math
import torch
import torch.nn.functional as F

from .model import PupilCenterRegressionHead as _Geom  # geometry utils

PUPIL_DEBUG = os.environ.get("PUPIL_DEBUG", "0") not in ("0", "", "false", "False", "no", "No")

def huber(x, delta=1.0):
    return torch.where(
        x.abs() <= delta,
        0.5 * x * x,
        delta * (x.abs() - 0.5 * delta),
    )

def _ensure_6param_ellipse(ellipse: torch.Tensor) -> torch.Tensor:
    if ellipse.size(-1) == 6:
        return ellipse
    if ellipse.size(-1) == 5:
        cx, cy, a, b, th = torch.split(ellipse, [1, 1, 1, 1, 1], dim=-1)
        cth, sth = torch.cos(th), torch.sin(th)
        return torch.cat([cx, cy, a, b, cth, sth], dim=-1)
    raise ValueError(f"_ensure_6param_ellipse: expected last dim 5 or 6, got {ellipse.size(-1)}")

def sample_ellipse_points(ellipse: torch.Tensor, num: int = 64) -> torch.Tensor:
    ellipse = _ensure_6param_ellipse(ellipse)
    assert ellipse.dim() == 3 and ellipse.size(1) == 2, \
        f"sample_ellipse_points expects [B,2,6], got {list(ellipse.shape)}"

    B = ellipse.size(0)
    device = ellipse.device

    cx, cy, a, b, cth, sth = torch.split(ellipse, [1, 1, 1, 1, 1, 1], dim=-1)
    cx = cx.unsqueeze(2); cy = cy.unsqueeze(2)
    a  = a.unsqueeze(2);  b  = b.unsqueeze(2)
    cth = cth.unsqueeze(2); sth = sth.unsqueeze(2)

    t = torch.linspace(0.0, 2.0 * math.pi, steps=num + 1, device=device)[:-1]
    cos_t = torch.cos(t).view(1, 1, num, 1)
    sin_t = torch.sin(t).view(1, 1, num, 1)

    x0 = a * cos_t
    y0 = b * sin_t
    x = cth * x0 - sth * y0 + cx
    y = sth * x0 + cth * y0 + cy
    return torch.cat([x, y], dim=-1)  # [B,2,num,2]

def _shape_to_BV(ten: torch.Tensor, B: int, V: int, last: int, name: str):
    if ten.dim() == 4:
        if ten.size(0) == B and ten.size(1) == V and ten.size(2) == 2 and ten.size(3) == last:
            return ten
        raise RuntimeError(f"{name}: unexpected 4D shape {list(ten.shape)}; expected [B,V,2,{last}]")
    if ten.dim() != 3 or ten.size(1) != 2 or ten.size(2) != last:
        raise RuntimeError(f"{name}: expected 3D [*,2,{last}] or 4D [B,V,2,{last}], got {list(ten.shape)}")
    n0 = ten.size(0)
    if n0 == B:
        return ten.unsqueeze(1).expand(B, V, 2, last).contiguous()
    elif n0 == B * V:
        return ten.view(B, V, 2, last).contiguous()
    else:
        raise RuntimeError(f"{name}: first dim {n0} does not match B={B} or B*V={B*V}")

def _project_points(K: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    X = xyz[..., 0]; Y = xyz[..., 1]; Z = xyz[..., 2].clamp_min(1e-6)
    fx, fy = K[:, 0, 0].unsqueeze(-1), K[:, 1, 1].unsqueeze(-1)
    cx, cy = K[:, 0, 2].unsqueeze(-1), K[:, 1, 2].unsqueeze(-1)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return torch.stack([u, v], dim=-1)

def multiview_pupil_center_losses(
    pred: dict,
    gt_pupil_center_3d: torch.Tensor,   # [B, V, 2, 3] in cm
    intrinsics_K: torch.Tensor,         # [B*V, 3, 3] (must be in the model's pixel space!)
    iris_radius_cm: torch.Tensor = None,# [B*V] or None
    image_hw=None,
    gt_iris_mesh_3d: torch.Tensor = None,   # [B, V, 2, K, 3] or None
    gt_pupil_center_2d: torch.Tensor = None,# [B, V, 2, 2] (optional)
    w_uv: float = 0.0,
    w_hm: float = 0.0,
    w_3d: float = 1.0,
    w_consistency: float = 0.3,
    w_plane: float = 0.0,
    huber_delta_cm: float = 0.5,
    z_margin_cm: float = 0.0,
):
    device = intrinsics_K.device
    B, V = gt_pupil_center_3d.shape[:2]

    ellipse = pred["ellipse"]
    delta_cm = pred["delta_cm"]
    ellipse_BV = _shape_to_BV(ellipse, B, V, ellipse.size(-1), "ellipse")
    delta_BV = _shape_to_BV(delta_cm, B, V, 1, "delta_cm")

    ellipse_f = _ensure_6param_ellipse(ellipse_BV.view(B * V, 2, ellipse_BV.size(-1)))
    delta_f = delta_BV.view(B * V, 2, 1)
    K_f = intrinsics_K.view(B * V, 3, 3)

    can_lift = iris_radius_cm is not None

    if PUPIL_DEBUG:
        print("can lift ?", can_lift, iris_radius_cm if iris_radius_cm is None else iris_radius_cm.detach())
        if can_lift:
            fx = K_f[:, 0, 0]; fy = K_f[:, 1, 1]
            print(f"[LOSS][K] fx mean {float(fx.mean()):.2f} | fy mean {float(fy.mean()):.2f}")

    if can_lift:
        R_f = iris_radius_cm.view(B * V)
        iris_center, iris_normal, depth_cm = _Geom.lift_iris_pose_and_center(ellipse_f, K_f, R_f)
        if PUPIL_DEBUG:
            print("[LOSS] can_lift:", iris_radius_cm is not None, "| R_f stats:",
                  float(R_f.min()), float(R_f.mean()), float(R_f.max()))
            print("[LOSS] depth_cm stats (clamped):", float(depth_cm.min()), float(depth_cm.mean()), float(depth_cm.max()))
        pupil_center = iris_center + delta_f * iris_normal
        pupil_center = pupil_center.view(B, V, 2, 3)
    else:
        iris_center = torch.zeros(B * V, 2, 3, device=device)
        iris_normal = torch.zeros(B * V, 2, 3, device=device)
        depth_cm = torch.zeros(B * V, 2, 1, device=device)
        pupil_center = torch.zeros(B, V, 2, 3, device=device)

    # 1) 3D accuracy
    loss_3d = torch.tensor(0.0, device=device)
    if can_lift:
        err3d = torch.norm(pupil_center - gt_pupil_center_3d.to(device), dim=-1)
        if "logvar" in pred:
            s = _shape_to_BV(pred["logvar"], B, V, 1, "logvar").squeeze(-1)
            err2 = (pupil_center - gt_pupil_center_3d.to(device)).pow(2).sum(dim=-1)
            loss_3d = torch.mean(torch.exp(-s) * err2 + s)
        else:
            loss_3d = huber(err3d, delta=huber_delta_cm).mean()
        if PUPIL_DEBUG:
            print(f"[DEBUG] |err3d| cm min/mean/max: "
                  f"{float(err3d.min()):.2f} / {float(err3d.mean()):.2f} / {float(err3d.max()):.2f}")
            gt = gt_pupil_center_3d.to(device)
            print(f"[DEBUG] GT pc3d cm min/mean/max: "
                  f"{float(gt.min()):.2f} / {float(gt.mean()):.2f} / {float(gt.max()):.2f}")

    # 2) cross-view consistency
    loss_cons = torch.tensor(0.0, device=device)
    if can_lift:
        mean_pc = pupil_center.mean(dim=1, keepdim=True)
        loss_cons = torch.norm(pupil_center - mean_pc, dim=-1).mean()

    # 3) ellipse-vs-mesh 2D regularizer
    loss_e2d = torch.tensor(0.0, device=device)
    if gt_iris_mesh_3d is not None:
        mesh3d = gt_iris_mesh_3d.to(device).view(B * V, 2, -1, 3)
        ptsL = _project_points(K_f, mesh3d[:, 0])
        ptsR = _project_points(K_f, mesh3d[:, 1])
        gt2d = torch.stack([ptsL, ptsR], dim=1)
        pts_pred = sample_ellipse_points(ellipse_f, num=64)
        diff = pts_pred.unsqueeze(-2) - gt2d.unsqueeze(-3)
        dists = torch.norm(diff, dim=-1)
        min_d, _ = dists.min(dim=-1)
        loss_e2d = min_d.mean()

    # 4) plane regularizer
    loss_plane = torch.tensor(0.0, device=device)
    if can_lift and gt_iris_mesh_3d is not None and w_plane > 0.0:
        center = iris_center.view(B * V, 2, 1, 3)
        normal = F.normalize(iris_normal, dim=-1).view(B * V, 2, 1, 3)
        mesh = gt_iris_mesh_3d.to(device).view(B * V, 2, -1, 3)
        d = torch.abs(((mesh - center) * normal).sum(dim=-1))
        loss_plane = d.mean()

    loss_uv = torch.tensor(0.0, device=device)
    loss_hm = torch.tensor(0.0, device=device)

    total = (w_3d * loss_3d
             + w_consistency * loss_cons
             + w_plane * loss_plane
             + w_hm * loss_hm
             + w_uv * loss_uv
             + 0.2 * loss_e2d)

    return {
        "accuracy":    w_3d * loss_3d,
        "consistency": w_consistency * loss_cons,
        "plane":       w_plane * loss_plane,
        "ellipse2d":   0.2 * loss_e2d,
        "total":       total,
        "pred_center_3d": pupil_center.detach() if can_lift else None,
        "pred_depth_cm":  depth_cm.view(B, V, 2, 1).detach() if can_lift else None,
    }
