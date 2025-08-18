import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from coordatt import CoordAtt
from depth_from_iris import depth_from_iris_cm       # analytic depth from iris (MediaPipe-style)


# -----------------------------
# Utils
# -----------------------------
def ortho6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Zhou et al., CVPR'19: 6D -> 3x3 rotation matrix."""
    a1 = F.normalize(x[:, 0:3], dim=1)
    a2 = x[:, 3:6]
    a2 = F.normalize(a2 - (a1 * a2).sum(1, keepdim=True) * a1, dim=1)
    a3 = torch.cross(a1, a2, dim=1)
    return torch.stack([a1, a2, a3], dim=2)


def make_unit_ring(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.linspace(0.0, 2.0 * math.pi, steps=n, device=device, dtype=dtype)
    x = torch.cos(t); y = torch.sin(t); z = torch.zeros_like(t)
    return torch.stack([x, y, z], dim=1)  # [N,3]


# -----------------------------
# Minimal Neck: 1x1 + BN/ReLU + CoordAtt on c4
# -----------------------------
class CoordNeck(nn.Module):
    """
    Lightweight adapter (kept as 'fusion' so GradNorm continues to work).
    """
    def __init__(self, in_ch: int, out_ch: int = 256, reduction: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.ca   = CoordAtt(out_ch, out_ch, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.proj(x)))
        return self.ca(x)


# -----------------------------
# Heads
# -----------------------------
class GlobalRegHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, hidden: int = 128, use_coord_att: bool = False, reduction=32):
        super().__init__()
        self.use_ca = use_coord_att
        if use_ca := use_coord_att:
            self.ca = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        if hasattr(self, "ca"):
            x = self.ca(x)
        x = self.pool(x).flatten(1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class EyeParamHead(nn.Module):
    """
    Identity-aware eye parameters (sizes come from identity, not learned):
      rot6d        [B,6]    eye rotation
      globe_center [B,3]    CCS position (cm)
      iris_offset  [B,1]    along optical axis (0..radius]
      kappa_deg    [B,2]    yaw,pitch for visual axis (±8°)
    """
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 6 + 3 + 1 + 2)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        y = self.mlp(x)
        rot6d = y[:, 0:6]
        cx = 10.0 * torch.tanh(y[:, 6:7])                 # [-10,10] cm
        cy = 10.0 * torch.tanh(y[:, 7:8])
        cz = 50.0 * torch.tanh(y[:, 8:9]) + 70.0          # [20,120] cm
        globe_center = torch.cat([cx, cy, cz], dim=1)
        iris_offset  = torch.sigmoid(y[:, 9:10])          # (0,1) → multiply by radius later
        kappa_deg    = 8.0 * torch.tanh(y[:, 10:12])      # [-8,+8]°
        return rot6d, globe_center, iris_offset, kappa_deg


class Iris2DHead(nn.Module):
    """
    Predicts per-eye 2D iris ring (100 points) in normalized image coords [-1,1],
    scaled to pixels using input H,W.
    """
    def __init__(self, in_ch: int, n_pts: int = 100, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2 * n_pts * 2)  # (L/R) * points * (x,y)
        )
        self.n_pts = n_pts

    def forward(self, x, H: int, W: int):
        b = x.size(0)
        y = self.pool(x).flatten(1)
        y = self.fc(y).view(b, 2, self.n_pts, 2)
        y = torch.tanh(y)
        u = (y[..., 0] + 1.0) * 0.5 * (W - 1)
        v = (y[..., 1] + 1.0) * 0.5 * (H - 1)
        return torch.stack([u, v], dim=-1)  # [B,2,N,2] in pixels


# -----------------------------
# RayNet (CoordAtt-only + MediaPipe depth)
# -----------------------------
class RayNet(nn.Module):
    def __init__(self, backbone, in_channels_list, n_iris_points: int = 100, reduction: int = 32):
        super().__init__()
        self.backbone = backbone
        self.n_iris_points = n_iris_points

        # Minimal neck on c4 (keep attr name 'fusion' for GradNorm)
        c4_ch = in_channels_list[-1]
        self.fusion = CoordNeck(c4_ch, out_ch=256, reduction=reduction)

        # Heads
        self.head_pose_regression   = GlobalRegHead(256, 6, use_coord_att=True, reduction=reduction)
        self.gaze_vector_regression = GlobalRegHead(256, 6)
        self.eye_left  = EyeParamHead(256)
        self.eye_right = EyeParamHead(256)
        self.iris2d_head = Iris2DHead(256, n_pts=n_iris_points)

        # Geometry buffers
        self.register_buffer("unit_ring",
            make_unit_ring(n_iris_points, device=torch.device("cpu"), dtype=torch.float32),
            persistent=False)

    # backbone: only c4
    def _forward_backbone(self, x: torch.Tensor):
        c0 = checkpoint(self.backbone.stem, x)
        c1 = checkpoint(self.backbone.stages[0], c0)
        c2 = checkpoint(self.backbone.stages[1], c1)
        c3 = checkpoint(self.backbone.stages[2], c2)
        c4 = checkpoint(self.backbone.stages[3], c3)
        return c4

    def _eye_geometry(self, rot6d, globe_center, iris_offset_alpha, kappa_deg, iris_radius_cm, device):
        """
        Build per-eye geometry in CCS (cm), using identity iris radius (cm).
        """
        B = rot6d.shape[0]
        R = ortho6d_to_rotmat(rot6d)                               # [B,3,3]
        z_axis = torch.tensor([0,0,1.0], device=device).view(1,3,1).repeat(B,1,1)
        optic = F.normalize(torch.bmm(R, z_axis).squeeze(-1), dim=1)  # [B,3]

        # If eyeball radius available, limit offset to [0, radius]
        # iris_offset_alpha \in (0,1) from head; multiply by radius proxy
        # Use iris radius as a proxy scale if eyeball radius not passed here
        offset_cm = iris_offset_alpha  # later multiplied externally if eyeball radius available

        # Visual axis = small yaw/pitch from optic
        yaw  = torch.deg2rad(kappa_deg[:, 0]);  pitch = torch.deg2rad(kappa_deg[:, 1])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        Rk = torch.stack([
            torch.stack([ cy, 0*cy,  sy], dim=1),
            torch.stack([ 0*cy,  cp, -sp], dim=1),
            torch.stack([-sy,  sp,  cy], dim=1),
        ], dim=1)
        visual = F.normalize(torch.einsum('bij,bj->bi', Rk, optic), dim=1)

        # Ring in eye-local (XY plane), radius = iris_radius_cm
        ring_local = self.unit_ring.to(device).unsqueeze(0).repeat(B,1,1)
        ring_local[:, :, :2] *= iris_radius_cm.view(B,1,1)

        return {
            "R": R,
            "optic": optic,
            "visual": visual,
            "globe_center": globe_center,
            "offset_alpha": offset_cm,   # (0,1), multiply outside
            "ring_local": ring_local,    # [B,N,3] in eye-local
        }

    def forward(self, x: torch.Tensor, K: Dict = None, identity: Dict = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B,3,H,W]
            K: {'fx': [B] or [B,2]}  focal length(s) in pixels
            identity:
              - 'iris_diam_cm': [B] or [B,2]
              - 'eyeball_radius_cm': [B,2] (optional, used for rendering/offset)
        """
        assert K is not None and identity is not None, "Please pass K (fx px) and identity (iris_diam_cm, ...)."
        B, _, H, W = x.shape

        # features
        c4 = self._forward_backbone(x)
        feats = self.fusion(c4)

        # iris 2D for analytic depth
        iris2d_px = self.iris2d_head(feats, H, W)  # [B,2,N,2]

        # gaze direction from 6D
        gaze_vec6d = self.gaze_vector_regression(feats)
        Rg = ortho6d_to_rotmat(gaze_vec6d)
        direction = F.normalize(Rg[:, :, 2], dim=1)  # [B,3]

        # per-eye params (pose/center/offset/kappa)
        Lp = self.eye_left(feats)
        Rp = self.eye_right(feats)

        # identity sizes
        iris_diam_cm = identity["iris_diam_cm"]  # [B] or [B,2]
        if iris_diam_cm.ndim == 1:
            iris_radius_L = iris_radius_R = iris_diam_cm * 0.5
        else:
            iris_radius_L = iris_diam_cm[:, 0] * 0.5
            iris_radius_R = iris_diam_cm[:, 1] * 0.5

        eyeball_radius_cm = identity.get("eyeball_radius_cm", None)  # [B,2] if provided

        # left/right geometry (eye-local)
        geoL = self._eye_geometry(Lp[0], Lp[1], Lp[2], Lp[3], iris_radius_L, x.device)
        geoR = self._eye_geometry(Rp[0], Rp[1], Rp[2], Rp[3], iris_radius_R, x.device)

        # Compute absolute centers for iris plane (pupil centers) using offsets
        if eyeball_radius_cm is not None:
            offL = geoL["offset_alpha"].squeeze(-1) * eyeball_radius_cm[:, 0]
            offR = geoR["offset_alpha"].squeeze(-1) * eyeball_radius_cm[:, 1]
        else:
            # fallback: scale by iris radius (keeps magnitude reasonable)
            offL = geoL["offset_alpha"].squeeze(-1) * iris_radius_L
            offR = geoR["offset_alpha"].squeeze(-1) * iris_radius_R

        iris_center_L = geoL["globe_center"] + geoL["optic"] * offL.unsqueeze(-1)  # [B,3]
        iris_center_R = geoR["globe_center"] + geoR["optic"] * offR.unsqueeze(-1)  # [B,3]

        # Rotate iris rings to CCS
        ringL = torch.einsum('bij,bnj->bni', geoL["R"], geoL["ring_local"]) + iris_center_L.unsqueeze(1)
        ringR = torch.einsum('bij,bnj->bni', geoR["R"], geoR["ring_local"]) + iris_center_R.unsqueeze(1)

        pupil_centers = torch.stack([iris_center_L, iris_center_R], dim=1)    # [B,2,3]
        origin = pupil_centers.mean(dim=1)                                    # [B,3]

        # --- analytic depth from iris (MediaPipe) ---
        depth_cm = depth_from_iris_cm(iris2d_px, K["fx"], identity["iris_diam_cm"])  # [B]

        # gaze point (hard)
        gaze_point = origin + depth_cm.unsqueeze(-1) * direction

        # optional head pose (if supervised)
        head_pose_6d = self.head_pose_regression(feats)

        # if eyeball radius provided, expose it; else put a proxy (NaN-safe)
        if eyeball_radius_cm is None:
            eyeball_radius_out = torch.stack([iris_radius_L, iris_radius_R], dim=1) * 2.2  # loose proxy
        else:
            eyeball_radius_out = eyeball_radius_cm

        return {
            "head_pose_6d": head_pose_6d,
            "gaze_vector_6d": gaze_vec6d,
            "gaze_vector_normalized": direction,
            "origin": origin,
            "direction": direction,
            "gaze_depth": depth_cm,                                # cm
            "gaze_point_3d": gaze_point,
            "gaze_point_from_ray": gaze_point,
            "pupil_center_3d": pupil_centers,                      # [B,2,3] cm
            "iris2d_px": iris2d_px,                                # [B,2,N,2] px
            "iris_mesh_3d": torch.stack([ringL, ringR], dim=1),    # [B,2,N,3] cm
            "eyeball_center_3d": torch.stack([geoL["globe_center"], geoR["globe_center"]], dim=1),
            "eyeball_radius_cm": eyeball_radius_out,               # [B,2] cm
            "optic_axis_eyes": torch.stack([geoL["optic"], geoR["optic"]], dim=1),     # [B,2,3]
            "visual_axis_eyes": torch.stack([geoL["visual"], geoR["visual"]], dim=1),  # [B,2,3]
            "fused": feats,
        }
