# raynet.py
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# backbone helper (only used in the sanity block)
from backbone.repnext_utils import load_pretrained_repnext

# use your stable 6D -> SO(3)
from utils import compute_rotation_matrix_from_ortho6d


# ========= Geometry helpers (kept here; move to utils.py later if you like) =========

def _skew3(v: torch.Tensor) -> torch.Tensor:
    """v: [...,3] -> skew-symmetric [...,3,3]"""
    O = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([O, -vz, vy], dim=-1),
            torch.stack([vz, O, -vx], dim=-1),
            torch.stack([-vy, vx, O], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(rotvec: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Axis-angle exponential map.
    rotvec: [...,3]  (radians)  ->  R: [...,3,3]
    """
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)          # [...,1]
    k = torch.where(theta > eps, rotvec / theta, torch.zeros_like(rotvec))
    K = _skew3(k)                                                     # [...,3,3]
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(K.shape)
    st = torch.sin(theta)[..., None]                                  # [...,1,1]
    ct = torch.cos(theta)[..., None]
    R = I + st * K + (1.0 - ct) * (K @ K)
    return torch.where((theta[..., None] < eps), I, R)


# ========= Parameter bounds (all cm except angles) =========

@dataclass
class GeomBounds:
    eyeball_min_cm: float = 0.9
    eyeball_max_cm: float = 1.6
    iris_min_cm: float = 0.30
    iris_max_cm: float = 0.80
    cornea_min_cm: float = 0.55
    cornea_max_cm: float = 0.90
    cornea2center_min_cm: float = 0.20
    cornea2center_max_cm: float = 0.45

    center_abs_max_cm: float = 4.0         # tanh box for c_eye in HCS
    kappa_max_rad: float = 0.15            # ~8.6 deg

    iris_plane_min_cm: float = 0.0
    iris_plane_max_cm: float = 0.60
    iris_aniso_abs_max: float = 0.30       # ±30%
    phi_abs_max_rad: float = math.pi

    pupil_u_min: float = 0.20
    pupil_u_max: float = 0.95


# ========= Tiny MLP factory =========

def mlp(in_ch: int, out_ch: int, hidden: int = 256, num_layers: int = 2) -> nn.Sequential:
    layers = []
    c = in_ch
    for _ in range(num_layers - 1):
        layers += [nn.Linear(c, hidden), nn.ReLU(inplace=True)]
        c = hidden
    layers += [nn.Linear(c, out_ch)]
    return nn.Sequential(*layers)


# ========= RayNet (fully geometric) =========

class RayNet(nn.Module):
    """
    Inputs:
        x: [B, V, 3, H, W]   (V = number of views, e.g., 9)
    Outputs:
        {
          "per_view": {
            "R_head": [B,V,3,3],
            "t_head": [B,V,3],                 # cm
          },
          "frame": {
            "c_eye": [B,2,3],                  # HCS, cm (L,R)
            "R_eye": [B,2,3,3],
            "kappa": [B,2,3],                  # axis-angle, rad
            "r_eye": [B,1], "r_iris": [B,1], "r_cornea": [B,1], "d_cornea": [B,1],  # cm
            "iris_pts_local": [B,2,N,3],       # eye-local ring, cm
            "iris_pts_h": [B,2,N,3],           # ring in HCS, cm (NEW: 3D eye mesh you asked for)
            "pupil_radius": [B,2,1],           # cm
            "iris_params": { "aniso": [B,2,2], "phi":[B,2,1], "z_plane":[B,2,1], "pupil_u":[B,2,1] }
          }
        }
    """

    def __init__(self, backbone, in_channels_list=None, panet_out_channels: int = 256,
                 bounds: GeomBounds = None, num_iris_pts: int = 100):
        super().__init__()
        self.backbone = backbone
        self.bounds = bounds if bounds is not None else GeomBounds()
        self.num_iris_pts = num_iris_pts

        # Decide per-view feature dimensionality
        self.has_stages = hasattr(self.backbone, "stem") and hasattr(self.backbone, "stages")
        if self.has_stages and in_channels_list is not None:
            # multi-scale concat of GAP(c1..c4)
            self.per_view_dim = int(sum(in_channels_list))
        else:
            # fallback: last feature map only
            self.per_view_dim = int(getattr(self.backbone, "num_features", 512))

        # ---------------- Per-view head pose ----------------
        self.head_rot_fc = nn.Linear(self.per_view_dim, 6)   # 6D rep
        self.head_trans_fc = nn.Linear(self.per_view_dim, 3) # cm

        # ---------------- Frame-shared anatomy & joints -----
        fused_dim = self.per_view_dim                        # we fuse views by mean

        self.eye_center_fc = mlp(fused_dim, 6)               # two eyes × 3D
        self.eye_rot6d_fc = mlp(fused_dim, 12)               # two eyes × 6D
        self.kappa_fc     = mlp(fused_dim, 6)                # two eyes × 3

        self.radii_fc     = mlp(fused_dim, 4)                # r_eye, r_iris, r_cornea, d_cornea (cm)

        self.iris_aniso_fc = mlp(fused_dim, 4)               # (ax,ay) per eye
        self.iris_phi_fc   = mlp(fused_dim, 2)               # in-plane angle per eye
        self.iris_z_fc     = mlp(fused_dim, 2)               # plane z offset per eye (cm)
        self.pupil_u_fc    = mlp(fused_dim, 2)               # dilation fraction per eye

        # Precompute unit circle
        theta = torch.linspace(0, 2 * math.pi, self.num_iris_pts)
        self.register_buffer("unit_circle_xy", torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1))

        # zero-initialize small heads (stable start)
        for m in [
            self.head_rot_fc, self.head_trans_fc,
            self.eye_center_fc, self.eye_rot6d_fc, self.kappa_fc, self.radii_fc,
            self.iris_aniso_fc, self.iris_phi_fc, self.iris_z_fc, self.pupil_u_fc
        ]:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        nn.init.zeros_(mm.weight); nn.init.zeros_(mm.bias)

    # ---- Feature extractor (multi-scale if available) ----
    def _per_view_token(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [B*V, 3, H, W] -> per-view tokens [B,V,C]
        If stages available: concat GAP(c1..c4); else: GAP(last).
        """
        BxV = x_flat.shape[0]
        if self.has_stages:
            c0 = checkpoint(self.backbone.stem, x_flat)       # stride 4
            c1 = checkpoint(self.backbone.stages[0], c0)      # stride 4
            c2 = checkpoint(self.backbone.stages[1], c1)      # stride 8
            c3 = checkpoint(self.backbone.stages[2], c2)      # stride 16
            c4 = checkpoint(self.backbone.stages[3], c3)      # stride 32
            pools = [
                torch.flatten(F.adaptive_avg_pool2d(fm, 1), 1)
                for fm in (c1, c2, c3, c4)
            ]
            vec = torch.cat(pools, dim=1)                     # [B*V, sum(Ci)]
        else:
            # Many RepNeXt wrappers expose forward_features()
            feats = (self.backbone.forward_features(x_flat)
                     if hasattr(self.backbone, "forward_features")
                     else x_flat)  # let it crash loudly if backbone is incompatible
            vec = torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)
        return vec

    # ---- Bounding helpers ----
    def _bound_scalar(self, raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return torch.sigmoid(raw) * (hi - lo) + lo

    def _bound_tanh(self, raw: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.tanh(raw) * scale

    # ---- Iris generator (eye-local) ----
    def _build_iris_points_local(
        self,
        r_iris_per_eye: torch.Tensor,     # [B,2,1] cm
        anisotropy_per_eye: torch.Tensor, # [B,2,2]
        phi_per_eye: torch.Tensor,        # [B,2,1] rad
        z_per_eye: torch.Tensor           # [B,2,1] cm
    ) -> torch.Tensor:
        B = r_iris_per_eye.shape[0]
        base = self.unit_circle_xy.view(1, 1, self.num_iris_pts, 2).to(r_iris_per_eye.dtype)    # [1,1,N,2]
        xy = base * r_iris_per_eye.unsqueeze(2)                                                 # [B,2,N,2]
        xy = xy * (1.0 + anisotropy_per_eye).unsqueeze(2)                                       # ellipse
        c = torch.cos(phi_per_eye).unsqueeze(2); s = torch.sin(phi_per_eye).unsqueeze(2)        # [B,2,1,1]
        x, y = xy[..., 0:1], xy[..., 1:2]
        xr = c * x - s * y; yr = s * x + c * y
        z = z_per_eye.unsqueeze(2).expand(-1, -1, self.num_iris_pts, -1)                        # [B,2,N,1]
        return torch.cat([xr, yr, z], dim=-1)                                                   # [B,2,N,3]

    # ---- Public helpers for transforms/projection (optional use in train.py) ----
    @staticmethod
    def hcs_to_ccs_points(pts_h: torch.Tensor, R_head: torch.Tensor, t_head: torch.Tensor) -> torch.Tensor:
        """
        pts_h:   [B,2,N,3]
        R_head:  [B,V,3,3]
        t_head:  [B,V,3]
        -> [B,V,2,N,3]
        """
        B, V = R_head.shape[0], R_head.shape[1]
        pts_h_v = pts_h.unsqueeze(1).expand(-1, V, -1, -1, -1)                 # [B,V,2,N,3]
        pts_c   = torch.matmul(R_head.unsqueeze(2).unsqueeze(2),               # [B,V,1,1,3,3]
                               pts_h_v.unsqueeze(-1)).squeeze(-1)              # [B,V,2,N,3]
        return pts_c + t_head.unsqueeze(2).unsqueeze(2)

    @staticmethod
    def project_points(pts_c: torch.Tensor, K: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        pts_c: [B,V,2,N,3], K: [B,V,3,3] -> uv [B,V,2,N,2]
        """
        B, V, E, N, _ = pts_c.shape
        X = pts_c.view(B, V, E * N, 3)
        x = torch.matmul(K, X.transpose(-1, -2)).transpose(-1, -2)            # [B,V,E*N,3]
        uv = x[..., :2] / x[..., 2:].clamp_min(eps)
        return uv.view(B, V, E, N, 2)

    @staticmethod
    def compose_axes(R_head: torch.Tensor, R_eye: torch.Tensor, kappa: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Returns optic and visual axes in CCS for each view.
        R_head: [B,V,3,3], R_eye: [B,2,3,3], kappa: [B,2,3]
        -> (a_optic, a_visual): [B,V,2,3] each
        """
        B, V = R_head.shape[:2]
        ez = torch.tensor([0.0, 0.0, 1.0], device=R_head.device, dtype=R_head.dtype).view(1, 1, 1, 3, 1)
        a_eye_local = torch.matmul(R_eye.unsqueeze(2), ez).squeeze(-1).squeeze(-1)        # [B,2,3]
        R_kappa = so3_exp(kappa.view(B * 2, 3)).view(B, 2, 3, 3)
        a_visual_local = torch.matmul(R_kappa, a_eye_local.unsqueeze(-1)).squeeze(-1)     # [B,2,3]
        a_eye_v   = a_eye_local.unsqueeze(1).expand(-1, V, -1, -1)                        # [B,V,2,3]
        a_vis_v   = a_visual_local.unsqueeze(1).expand(-1, V, -1, -1)                     # [B,V,2,3]
        a_optic   = torch.matmul(R_head.unsqueeze(2), a_eye_v.unsqueeze(-1)).squeeze(-1)  # [B,V,2,3]
        a_visual  = torch.matmul(R_head.unsqueeze(2), a_vis_v.unsqueeze(-1)).squeeze(-1)  # [B,V,2,3]
        return a_optic, a_visual

    # ---- Forward ----
    def forward(self, x: torch.Tensor):
        """
        x: [B, V, 3, H, W]
        """
        assert x.dim() == 5, "Expected input [B,V,3,H,W]"
        B, V = x.shape[0], x.shape[1]

        # Per-view tokens (multi-scale concat if possible)
        x_flat = x.view(B * V, x.shape[2], x.shape[3], x.shape[4])
        per_view_flat = self._per_view_token(x_flat)                          # [B*V, per_view_dim]
        per_view_vec  = per_view_flat.view(B, V, -1)                          # [B,V,C]

        # Frame-level fused token (mean over views)
        frame_vec = per_view_vec.mean(dim=1)                                  # [B,C]

        # ----- Per-view head pose -----
        rot6d = self.head_rot_fc(per_view_vec).view(B * V, 6)                 # [B*V,6]
        R_head = compute_rotation_matrix_from_ortho6d(rot6d).view(B, V, 3, 3) # [B,V,3,3]
        t_head = self.head_trans_fc(per_view_vec).view(B, V, 3)               # [B,V,3]  (cm)

        # ----- Frame-shared anatomy & joints -----
        # Eye centers in HCS (cm), bounded via tanh box
        c_eye_raw = self.eye_center_fc(frame_vec).view(B, 2, 3)
        c_eye     = self._bound_tanh(c_eye_raw, self.bounds.center_abs_max_cm)           # [B,2,3]

        # Eye rotations (6D -> SO(3))
        eye_rot6d = self.eye_rot6d_fc(frame_vec).view(B, 2, 6)
        R_eye     = compute_rotation_matrix_from_ortho6d(eye_rot6d.view(B * 2, 6)).view(B, 2, 3, 3)

        # Kappa (axis-angle, radians)
        kappa_raw = self.kappa_fc(frame_vec).view(B, 2, 3)
        kappa     = self._bound_tanh(kappa_raw, self.bounds.kappa_max_rad)               # [B,2,3]

        # Radii / cornea distances (cm)
        radii_raw = self.radii_fc(frame_vec)                                             # [B,4]
        r_eye     = self._bound_scalar(radii_raw[:, 0:1], self.bounds.eyeball_min_cm,  self.bounds.eyeball_max_cm)
        r_iris    = self._bound_scalar(radii_raw[:, 1:2], self.bounds.iris_min_cm,     self.bounds.iris_max_cm)
        r_cornea  = self._bound_scalar(radii_raw[:, 2:3], self.bounds.cornea_min_cm,   self.bounds.cornea_max_cm)
        d_cornea  = self._bound_scalar(radii_raw[:, 3:4], self.bounds.cornea2center_min_cm, self.bounds.cornea2center_max_cm)

        # Iris ellipse parameters
        aniso_raw = self.iris_aniso_fc(frame_vec).view(B, 2, 2)
        aniso     = self._bound_tanh(aniso_raw, self.bounds.iris_aniso_abs_max)
        phi_raw   = self.iris_phi_fc(frame_vec).view(B, 2, 1)
        phi       = self._bound_tanh(phi_raw, self.bounds.phi_abs_max_rad)
        z_raw     = self.iris_z_fc(frame_vec).view(B, 2, 1)
        z_plane   = self._bound_scalar(z_raw, self.bounds.iris_plane_min_cm, self.bounds.iris_plane_max_cm)

        # Pupil radius from dilation fraction
        pupil_u_raw  = self.pupil_u_fc(frame_vec).view(B, 2, 1)
        pupil_u      = self._bound_scalar(pupil_u_raw, self.bounds.pupil_u_min, self.bounds.pupil_u_max)
        pupil_radius = pupil_u * r_iris.unsqueeze(1)                                     # [B,2,1] cm

        # Build iris ring in eye-local (cm) -> in HCS (cm)  *** 3D eye mesh ***
        r_iris_per_eye = r_iris.unsqueeze(1).expand(-1, 2, -1)                           # [B,2,1]
        iris_pts_local = self._build_iris_points_local(r_iris_per_eye, aniso, phi, z_plane)  # [B,2,N,3]
        iris_pts_h     = torch.matmul(R_eye.unsqueeze(2), iris_pts_local.unsqueeze(-1)).squeeze(-1) + c_eye.unsqueeze(2)  # [B,2,N,3]

        # Pack outputs
        out = {
            "per_view": {
                "R_head": R_head,                  # [B,V,3,3]
                "t_head": t_head,                  # [B,V,3] cm
            },
            "frame": {
                "c_eye": c_eye,                    # [B,2,3] HCS, cm
                "R_eye": R_eye,                    # [B,2,3,3]
                "kappa": kappa,                    # [B,2,3] axis-angle, rad
                "r_eye": r_eye, "r_iris": r_iris, "r_cornea": r_cornea, "d_cornea": d_cornea,  # [B,1] each
                "iris_pts_local": iris_pts_local,  # [B,2,N,3] cm
                "iris_pts_h": iris_pts_h,          # [B,2,N,3] cm  (3D iris mesh in head coords)
                "pupil_radius": pupil_radius,      # [B,2,1] cm
                "iris_params": {
                    "aniso": aniso, "phi": phi, "z_plane": z_plane, "pupil_u": pupil_u
                },
            },
        }
        return out


# ========= Sanity check =========

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone_name = "repnext_m3"
    weight_path = "./repnext_m3_pretrained.pt"

    repnext = load_pretrained_repnext(backbone_name, weight_path).to(device)
    in_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }
    model = RayNet(repnext, in_channels_list=in_channels_dict[backbone_name], bounds=GeomBounds()).to(device)

    B, V, H, W = 2, 9, 256, 256
    x = torch.randn(B, V, 3, H, W, device=device)
    with torch.no_grad():
        y = model(x)
    print("R_head", y["per_view"]["R_head"].shape,
          "t_head", y["per_view"]["t_head"].shape,
          "c_eye",  y["frame"]["c_eye"].shape,
          "iris_h", y["frame"]["iris_pts_h"].shape)
