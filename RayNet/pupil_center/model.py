# pupil_center/model.py
# Geometry-aware pupil head: predicts per-eye ellipse in pixels + along-normal offset (scalar).

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from coordatt import CoordAtt

PUPIL_DEBUG = os.environ.get("PUPIL_DEBUG", "0") not in ("0", "", "false", "False", "no", "No")

# Set a generous upper bound so you can *see* if depths are wrong before clamping.
MIN_DEPTH_CM = float(os.environ.get("PUPIL_MIN_DEPTH_CM", "0.5"))
MAX_DEPTH_CM = float(os.environ.get("PUPIL_MAX_DEPTH_CM", "1000.0"))

def _safe_norm(v, dim=-1, eps=1e-8):
    return v / (v.norm(dim=dim, keepdim=True) + eps)

def _ensure_6param(ellipse: torch.Tensor) -> torch.Tensor:
    if ellipse.size(-1) == 6:
        return ellipse
    if ellipse.size(-1) == 5:
        cx, cy, a, b, th = torch.split(ellipse, [1, 1, 1, 1, 1], dim=-1)
        return torch.cat([cx, cy, a, b, torch.cos(th), torch.sin(th)], dim=-1)
    raise ValueError(f"_ensure_6param: expected last dim 5 or 6, got {ellipse.size(-1)}")

def _k_stats_str(K: torch.Tensor) -> str:
    fx = K[:, 0, 0]; fy = K[:, 1, 1]; cx = K[:, 0, 2]; cy = K[:, 1, 2]
    return (f"fx min/mean/max: {float(fx.min()):.2f}/{float(fx.mean()):.2f}/{float(fx.max()):.2f} | "
            f"fy min/mean/max: {float(fy.min()):.2f}/{float(fy.mean()):.2f}/{float(fy.max()):.2f} | "
            f"cx mean: {float(cx.mean()):.2f} cy mean: {float(cy.mean()):.2f}")

def _validate_intrinsics(K: torch.Tensor):
    fx = K[:, 0, 0]; fy = K[:, 1, 1]
    if torch.any(torch.isnan(K)) or torch.any(torch.isinf(K)):
        raise ValueError("[K] contains NaN/Inf.")
    if float(fx.mean()) < 10.0 or float(fy.mean()) < 10.0:
        raise ValueError(f"[K] intrinsics look invalid; {_k_stats_str(K)}")

class PupilCenterRegressionHead(nn.Module):
    """
    For each eye (L, R):
      - ellipse (cx, cy, a, b, cosθ, sinθ)    [pixels]
      - scalar offset δ (cm) along iris-plane normal
      - optional logvar (aleatoric)
    """
    def __init__(self, in_channels=256, hidden_dim=256, reduction=32, dropout=0.0, predict_logvar=False):
        super().__init__()
        self.predict_logvar = predict_logvar

        self.ca = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Per-eye params: [cx, cy, log_a, log_b, ang_x, ang_y, delta, (logvar)]
        per_eye = 7 + (1 if predict_logvar else 0)
        self.fc_out = nn.Linear(hidden_dim, 2 * per_eye)
        nn.init.zeros_(self.fc_out.bias)

    def _unpack_eye(self, t):
        cx = t[..., 0:1]
        cy = t[..., 1:2]
        a = F.softplus(t[..., 2:3]) + 1e-6
        b = F.softplus(t[..., 3:4]) + 1e-6
        ang = _safe_norm(t[..., 4:6], dim=-1)
        cos_t = ang[..., 0:1]
        sin_t = ang[..., 1:2]
        delta = 0.3 * torch.tanh(t[..., 6:7])  # cm
        ellipse = torch.cat([cx, cy, a, b, cos_t, sin_t], dim=-1)
        if self.predict_logvar:
            logvar = t[..., 7:8]
            return ellipse, delta, logvar
        return ellipse, delta, None

    @staticmethod
    def ellipse_to_normalized_conic(ellipse, K):
        """
        x_n^T Cn x_n = 0 with u = K x_n  =>  Cn = K^T C K  (not K^{-T} C K^{-1})
        """
        B = ellipse.shape[0]
        device = ellipse.device
        ellipse = _ensure_6param(ellipse)
        _validate_intrinsics(K)

        cx, cy, a, b, cth, sth = torch.split(ellipse, [1, 1, 1, 1, 1, 1], dim=-1)
        theta = torch.atan2(sth, cth)
        inv_a2 = (1.0 / (a * a)).squeeze(-1)
        inv_b2 = (1.0 / (b * b)).squeeze(-1)
        ct = torch.cos(theta).squeeze(-1)
        st = torch.sin(theta).squeeze(-1)

        A11 = ct * ct * inv_a2 + st * st * inv_b2
        A22 = st * st * inv_a2 + ct * ct * inv_b2
        A12 = ct * st * (inv_a2 - inv_b2)

        C = torch.zeros(B, 2, 3, 3, device=device)
        C[:, :, 0, 0] = A11
        C[:, :, 1, 1] = A22
        C[:, :, 0, 1] = A12
        C[:, :, 1, 0] = A12

        ax = (A11 * cx.squeeze(-1) + A12 * cy.squeeze(-1))
        ay = (A12 * cx.squeeze(-1) + A22 * cy.squeeze(-1))
        C[:, :, 0, 2] = -ax
        C[:, :, 2, 0] = -ax
        C[:, :, 1, 2] = -ay
        C[:, :, 2, 1] = -ay

        cAc = (A11 * cx.squeeze(-1) ** 2 + 2 * A12 * cx.squeeze(-1) * cy.squeeze(-1) + A22 * cy.squeeze(-1) ** 2)
        C[:, :, 2, 2] = cAc - 1.0

        Cn = K.transpose(1, 2).unsqueeze(1) @ C @ K.unsqueeze(1)
        return Cn

    @staticmethod
    def conic_axes_in_normalized(Cn):
        A = Cn[:, :, :2, :2]
        u = Cn[:, :, :2, 2:3]
        mu = torch.linalg.solve(A, -u).squeeze(-1)  # [B,2,2]

        evals, evecs = torch.linalg.eigh(A)
        lam1 = evals[:, :, 0:1]
        lam2 = evals[:, :, 1:2]
        a_n = 1.0 / torch.sqrt(lam1.clamp_min(1e-12))
        b_n = 1.0 / torch.sqrt(lam2.clamp_min(1e-12))

        v = evecs[:, :, :, 0]
        theta_n = torch.atan2(v[..., 1], v[..., 0]).unsqueeze(-1)
        return a_n, b_n, theta_n, mu

    @staticmethod
    def _direct_normalized_from_pixels(ellipse, K):
        ellipse = _ensure_6param(ellipse)
        cx, cy, a, b, _, _ = torch.split(ellipse, [1,1,1,1,1,1], dim=-1)
        fx = K[:, 0, 0].view(-1, 1, 1)
        fy = K[:, 1, 1].view(-1, 1, 1)
        cx0 = K[:, 0, 2].view(-1, 1, 1)
        cy0 = K[:, 1, 2].view(-1, 1, 1)
        a_n = a / fx
        b_n = b / fy
        mu_n = torch.cat([(cx - cx0) / fx, (cy - cy0) / fy], dim=-1)
        return a_n, b_n, mu_n

    @staticmethod
    def lift_iris_pose_and_center(ellipse, K, iris_radius_cm):
        B = ellipse.shape[0]
        ellipse = _ensure_6param(ellipse)
        _validate_intrinsics(K)

        Cn = PupilCenterRegressionHead.ellipse_to_normalized_conic(ellipse, K)
        a_n, b_n, theta_n, mu_n = PupilCenterRegressionHead.conic_axes_in_normalized(Cn)

        if PUPIL_DEBUG:
            a_dir, b_dir, mu_dir = PupilCenterRegressionHead._direct_normalized_from_pixels(ellipse, K)
            a_rel = torch.abs((a_n - a_dir) / a_dir.clamp_min(1e-12))
            b_rel = torch.abs((b_n - b_dir) / b_dir.clamp_min(1e-12))
            mu_err = torch.norm(mu_n - mu_dir, dim=-1)
            def _stat(t): return float(t.min()), float(t.mean()), float(t.max())
            print("[DEBUG] K stats:", _k_stats_str(K))
            print(f"[DEBUG] iris R cm min/mean/max: {float(iris_radius_cm.min()):.4f} / "
                  f"{float(iris_radius_cm.mean()):.4f} / {float(iris_radius_cm.max()):.4f}")
            print(f"[DEBUG] a_n min/mean/max:       {_stat(a_n)[0]:.6f} / {_stat(a_n)[1]:.6f} / {_stat(a_n)[2]:.6f}")
            print(f"[DEBUG] b_n min/mean/max:       {_stat(b_n)[0]:.6f} / {_stat(b_n)[1]:.6f} / {_stat(b_n)[2]:.6f}")
            print(f"[DEBUG] sanity a_n~a/fx rel err mean/max: {float(a_rel.mean()):.4e} / {float(a_rel.max()):.4e}")
            print(f"[DEBUG] sanity b_n~b/fy rel err mean/max: {float(b_rel.mean()):.4e} / {float(b_rel.max()):.4e}")
            print(f"[DEBUG] center μ_n vs direct (norm) mean/max: {float(mu_err.mean()):.4e} / {float(mu_err.max()):.4e}")

        a_n = a_n.clamp_min(1e-6)
        R   = iris_radius_cm.view(B, 1, 1)
        z_raw = R / a_n
        z     = z_raw.clamp(MIN_DEPTH_CM, MAX_DEPTH_CM)

        x_n = mu_n[..., 0:1]
        y_n = mu_n[..., 1:2]
        ray = _safe_norm(torch.cat([x_n, y_n, torch.ones_like(x_n)], dim=-1), dim=-1)
        iris_center_3d = ray * z

        cos_tilt = (b_n / a_n).clamp(0.0, 1.0)
        tilt = torch.acos(cos_tilt)

        maj_dir = torch.cat([torch.cos(theta_n), torch.sin(theta_n)], dim=-1)
        axis = F.pad(maj_dir, (0, 1), value=0.0)
        z_axis = iris_center_3d.new_zeros(B, 2, 3); z_axis[..., 2] = 1.0

        k = _safe_norm(axis, dim=-1)
        ct = torch.cos(tilt); st = torch.sin(tilt)
        kv = torch.cross(k, z_axis, dim=-1)
        kdotv = (k * z_axis).sum(dim=-1, keepdim=True)
        normal = _safe_norm(z_axis * ct + kv * st + k * (kdotv * (1.0 - ct)), dim=-1)

        if PUPIL_DEBUG:
            print(f"[DEBUG] z_raw cm min/mean/max:  {float(z_raw.min()):.2f} / {float(z_raw.mean()):.2f} / {float(z_raw.max()):.2f}")
            print(f"[DEBUG] z_clamp cm mn/mean/mx:  {float(z.min()):.2f} / {float(z.mean()):.2f} / {float(z.max()):.2f}")

        return iris_center_3d, normal, z

    def forward(self, x, K=None, iris_radius_cm=None):
        B = x.size(0)
        x = self.ca(x)
        x = self.pool(x).flatten(1)
        h = self.drop(self.act(self.bn1(self.fc1(x))))
        raw = self.fc_out(h)
        per_eye = raw.size(-1) // 2
        left, right = raw[:, :per_eye], raw[:, per_eye:]

        eL, dL, sL = self._unpack_eye(left)
        eR, dR, sR = self._unpack_eye(right)

        ellipse = torch.stack([eL, eR], dim=1)
        delta_cm = torch.stack([dL, dR], dim=1)

        out = {"ellipse": ellipse, "delta_cm": delta_cm}
        if self.predict_logvar:
            out["logvar"] = torch.stack([sL, sR], dim=1)

        if K is not None and iris_radius_cm is not None:
            iris_center, iris_normal, depth_cm = self.lift_iris_pose_and_center(ellipse, K, iris_radius_cm)
            out["iris_center_3d"] = iris_center
            out["iris_normal"] = iris_normal
            out["pupil_center_3d"] = iris_center + delta_cm * iris_normal
            out["iris_depth_cm"] = depth_cm
        return out
