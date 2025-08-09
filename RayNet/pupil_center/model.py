# pupil_center/model.py
# Geometry-aware pupil center head: predicts per-eye ellipse in image + along-normal offset.
# Works with your CoordAtt and RepNeXt backbone features.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from RayNet.coordatt import CoordAtt


def _safe_norm(v, dim=-1, eps=1e-8):
    return v / (v.norm(dim=dim, keepdim=True) + eps)


class PupilCenterRegressionHead(nn.Module):
    """
    Predicts, for each eye (L,R):
      - an image-space ellipse E = (cx, cy, a, b, cosθ, sinθ)  [pixels]
      - a signed offset δ [cm] along the inferred iris-plane normal (toward the camera if negative)
      - optional aleatoric log-variance for robust training

    Forward can be called in two modes:
      1) params-only (default): returns ellipse params and offset; 3D lifting is done in the loss.
         out = {"ellipse": [B,2,6], "delta_cm": [B,2,1]}
      2) geometry mode: pass K (intrinsics) and iris_radius_cm to also get 3D pupil centers:
         out additionally contains "pupil_center_3d": [B,2,3] and "iris_normal": [B,2,3]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        reduction: int = 32,
        dropout: float = 0.0,
        predict_logvar: bool = False,
    ):
        super().__init__()
        self.predict_logvar = predict_logvar

        # Lightweight attention + global pooling
        self.ca = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP trunk
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Per-eye output (L and R, concatenated)
        # ellipse: (cx, cy, log_a, log_b, cosθ, sinθ) -> 6 params
        # delta: along-normal offset in cm -> 1 param
        # optional: logvar (for heteroscedastic loss) -> 1 param
        per_eye = 6 + 1 + (1 if predict_logvar else 0)
        self.fc_out = nn.Linear(hidden_dim, 2 * per_eye)

        # Friendly initialization
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _pack_ellipse(raw_eye: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        raw_eye: [..., 8] or [..., 7] depending on predict_logvar
        Returns:
          ellipse [...,6]  (cx,cy,a,b,cosθ,sinθ) with a,b > 0 and angle unit components normalized
          delta_cm [...,1]
          (optional logvar [...,1]) handled by caller
        """
        cx, cy, log_a, log_b, cth, sth, delta = torch.split(raw_eye[..., :7], [1, 1, 1, 1, 1, 1, 1], dim=-1)
        a = torch.exp(log_a).clamp_min(1e-3)
        b = torch.exp(log_b).clamp_min(1e-3)
        ang = torch.cat([cth, sth], dim=-1)
        ang = _safe_norm(ang, dim=-1)  # normalize (cosθ, sinθ)
        ellipse = torch.cat([cx, cy, a, b, ang[..., :1], ang[..., 1:2]], dim=-1)
        return ellipse, delta

    @staticmethod
    def ellipse_to_normalized_conic(ellipse, K):
        """
        Convert ellipse params to a 2D conic in normalized camera coords.
        ellipse: [B,2,6] (cx,cy,a,b,cosθ,sinθ) in pixels
        K:       [B,3,3] intrinsics for the (cropped) image (per sample)
        Returns:
          Cn: [B,2,3,3]  conic matrices in normalized coords (homogeneous 2D)
        """
        B = ellipse.shape[0]
        device = ellipse.device

        cx, cy, a, b, cth, sth = torch.split(ellipse, [1, 1, 1, 1, 1, 1], dim=-1)  # [B,2,1] each
        theta = torch.atan2(sth, cth)
        # 2x2 shape matrix A = R diag(1/a^2, 1/b^2) R^T
        c2 = (1.0 / (a * a)).squeeze(-1)  # [B,2]
        d2 = (1.0 / (b * b)).squeeze(-1)  # [B,2]
        ct = torch.cos(theta).squeeze(-1)
        st = torch.sin(theta).squeeze(-1)
        # A entries
        A11 = ct * ct * c2 + st * st * d2
        A22 = st * st * c2 + ct * ct * d2
        A12 = ct * st * (c2 - d2)

        # Conic in pixel coords: (x-c)^T A (x-c) = 1  -> C = [A  -A c; -c^T A  c^T A c - 1]
        C = torch.zeros(B, 2, 3, 3, device=device)
        C[:, :, 0, 0] = A11
        C[:, :, 1, 1] = A22
        C[:, :, 0, 1] = A12
        C[:, :, 1, 0] = A12

        # -A*c
        ax = (A11 * cx.squeeze(-1) + A12 * cy.squeeze(-1))
        ay = (A12 * cx.squeeze(-1) + A22 * cy.squeeze(-1))
        C[:, :, 0, 2] = -ax
        C[:, :, 2, 0] = -ax
        C[:, :, 1, 2] = -ay
        C[:, :, 2, 1] = -ay

        # c^T A c - 1
        cAc = (A11 * cx.squeeze(-1) ** 2
               + 2 * A12 * cx.squeeze(-1) * cy.squeeze(-1)
               + A22 * cy.squeeze(-1) ** 2)
        C[:, :, 2, 2] = cAc - 1.0

        # Normalize to camera coordinates: x_n = T x, with
        # T = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]
        # Conic transforms as Cn = T^{-T} C T^{-1}
        Kinv = torch.inverse(K)  # [B,3,3]
        # Break Kinv for affine part: x_n = K^{-1} x
        # For a 2D conic, full 3x3 works; just apply for each eye
        Cn = torch.zeros_like(C)
        for eye in range(2):
            Cn[:, eye] = torch.matmul(Kinv.transpose(1, 2), torch.matmul(C[:, eye], Kinv))
        return Cn

    @staticmethod
    def conic_axes_in_normalized(Cn):
        """
        Extract semi-axes (a_n, b_n) and angle theta_n (in the normalized plane z=1)
        from conic matrix Cn (2D ellipse).
        Returns: a_n [B,2,1], b_n [B,2,1], theta_n [B,2,1]  (all in normalized units)
        """
        B = Cn.shape[0]
        device = Cn.device
        # Convert Cn -> implicit ellipse (x-μ)^T A (x-μ)=1 to read axes.
        # For conic C = [A  u; u^T  w], the center μ solves A μ = -u
        A = Cn[:, :, :2, :2]      # [B,2,2,2]
        u = Cn[:, :, :2, 2:3]     # [B,2,2,1]
        # Solve for center (per batch, per eye)
        mu = torch.linalg.solve(A, -u)  # [B,2,2,1]
        # Translate to center -> A stays; axes from eigen-decomposition of A
        # A = R diag(λ1, λ2) R^T, semi-axes = 1/sqrt(λi)
        evals, evecs = torch.linalg.eigh(A)  # [B,2,2], [B,2,2,2]
        lam1 = evals[:, :, 0:1]  # smallest
        lam2 = evals[:, :, 1:2]
        # Ensure ordering so that a_n >= b_n
        a_n = (1.0 / torch.sqrt(lam1.clamp_min(1e-12)))  # major
        b_n = (1.0 / torch.sqrt(lam2.clamp_min(1e-12)))  # minor
        # Angle from eigenvector of major axis
        v = evecs[:, :, :, 0]  # [B,2,2], major axis direction in normalized x-y
        theta_n = torch.atan2(v[..., 1], v[..., 0]).unsqueeze(-1)
        return a_n, b_n, theta_n, mu.squeeze(-1)  # mu: [B,2,2]

    @staticmethod
    def lift_iris_pose_and_center(ellipse, K, iris_radius_cm):
        """
        Given ellipse in pixels and intrinsics K, lift to a coarse iris plane pose:
          - depth z via a_n (major axis in normalized coords): z ≈ R / a_n
          - plane tilt via axis ratio: cos(tilt) ≈ b_n / a_n (orthographic approx)
          - plane normal by rotating camera z-axis about the major-axis direction by 'tilt'
          - 3D iris center at depth z along center ray

        Returns:
          iris_center_3d: [B,2,3] (cm, camera coords)
          iris_normal:    [B,2,3] (unit, camera coords)
          depth_cm:       [B,2,1]
        """
        B = ellipse.shape[0]
        device = ellipse.device

        # Conic in normalized coords & axes
        Cn = PupilCenterRegressionHead.ellipse_to_normalized_conic(ellipse, K)   # [B,2,3,3]
        a_n, b_n, theta_n, mu_n = PupilCenterRegressionHead.conic_axes_in_normalized(Cn)  # [B,2,1],..., mu_n [B,2,2]

        # Depth from major axis (normalized plane has focal=1): a_n ≈ R / z  -> z ≈ R / a_n
        R = iris_radius_cm.view(B, 1, 1)  # [B,1,1]
        z = (R / a_n.clamp_min(1e-6))     # [B,2,1]

        # Iris center direction ray (normalized)
        x_n = mu_n[..., 0:1]  # [B,2,1]
        y_n = mu_n[..., 1:2]
        ray = torch.cat([x_n, y_n, torch.ones_like(x_n)], dim=-1)  # [B,2,3]
        ray = _safe_norm(ray, dim=-1)

        iris_center_3d = ray * z  # [B,2,3] in "focal=1, units=cm" because z is cm

        # Plane tilt & normal: cos(tilt) ≈ b_n / a_n
        cos_tilt = (b_n / a_n).clamp(0.0, 1.0)
        tilt = torch.acos(cos_tilt)  # [B,2,1]

        # Major-axis direction in normalized x-y plane
        maj_dir = torch.cat([torch.cos(theta_n), torch.sin(theta_n)], dim=-1)  # [B,2,2]
        # Build rotation axis (major-axis direction in 3D)
        axis = F.pad(maj_dir, (0, 1), value=0.0)  # [B,2,3], z=0

        # Rotate z-axis [0,0,1] by 'tilt' around 'axis' (Rodrigues)
        z_axis = torch.zeros(B, 2, 3, device=device); z_axis[..., 2] = 1.0
        k = _safe_norm(axis, dim=-1)
        ct = torch.cos(tilt)
        st = torch.sin(tilt)
        # Rodrigues: v_rot = v*ct + (k x v)*st + k*(k·v)*(1-ct)
        kv = torch.cross(k, z_axis, dim=-1)
        kdotv = (k * z_axis).sum(dim=-1, keepdim=True)
        normal = z_axis * ct + kv * st + k * (kdotv * (1.0 - ct))
        normal = _safe_norm(normal, dim=-1)  # [B,2,3]

        depth_cm = z
        return iris_center_3d, normal, depth_cm

    def forward(self, x, K: torch.Tensor = None, iris_radius_cm: torch.Tensor = None):
        """
        x: features [B,C,H,W]
        K: intrinsics [B,3,3] (optional)
        iris_radius_cm: [B,] or [B,1] (optional)
        """
        B = x.size(0)
        x = self.ca(x)
        x = self.pool(x).flatten(1)               # [B,C]
        h = self.drop(self.act(self.bn1(self.fc1(x))))  # [B,H]

        raw = self.fc_out(h)                      # [B, 2 * per_eye]
        per_eye = raw.shape[-1] // 2
        left, right = raw[:, :per_eye], raw[:, per_eye:]

        # Unpack eyes (ellipse, delta, optional logvar)
        def unpack_eye(t):
            ellipse, delta = self._pack_ellipse(t)
            if self.predict_logvar:
                logvar = t[..., 7:8]
                return ellipse, delta, logvar
            return ellipse, delta, None

        eL, dL, sL = unpack_eye(left)
        eR, dR, sR = unpack_eye(right)

        ellipse = torch.stack([eL, eR], dim=1)     # [B,2,6]
        delta_cm = torch.stack([dL, dR], dim=1)    # [B,2,1]

        out = {
            "ellipse": ellipse,
            "delta_cm": delta_cm,
        }
        if self.predict_logvar:
            logvar = torch.stack([sL, sR], dim=1)  # [B,2,1]
            out["logvar"] = logvar

        # Optional geometry pass
        if K is not None and iris_radius_cm is not None:
            iris_center, iris_normal, depth_cm = self.lift_iris_pose_and_center(ellipse, K, iris_radius_cm)
            pupil_center = iris_center + delta_cm * iris_normal  # shift along normal
            out["iris_center_3d"] = iris_center
            out["iris_normal"] = iris_normal
            out["pupil_center_3d"] = pupil_center
            out["iris_depth_cm"] = depth_cm

        return out
