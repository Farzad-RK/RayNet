# raynet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from coordatt import CoordAtt  # your existing module

# --- minimal 6D -> rotation utils (Zhou et al.) ---
def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def ortho6d_to_rotmat(x):
    """
    x: (..., 6) -> (..., 3, 3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)

def small_angle_rpy_to_rot(yaw_pitch_rad):
    """
    yaw about +Y (nasal-temporal), pitch about +X (up-down)
    yaw_pitch_rad: (..., 2) -> (..., 3, 3)
    """
    yaw, pitch = yaw_pitch_rad[..., 0], yaw_pitch_rad[..., 1]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cx, sx = torch.cos(pitch), torch.sin(pitch)
    # R = R_x(pitch) @ R_y(yaw)
    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx,  cx
    ], dim=-1).reshape(*cx.shape, 3, 3)
    Ry = torch.stack([
        cy, torch.zeros_like(cy), sy,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy), cy
    ], dim=-1).reshape(*cy.shape, 3, 3)
    return torch.einsum('...ij,...jk->...ik', Rx, Ry)

def build_orthonormal_basis(z):
    """
    z: (B,3) unit vector -> (B,3), (B,3) two orthonormal vectors u,v s.t. [u,v,z] is right-handed
    """
    B = z.shape[0]
    # choose a helper not parallel to z
    helper = torch.where(torch.abs(z[:, 2:3]) < 0.9,
                         torch.tensor([0., 0., 1.], device=z.device).expand(B, 3),
                         torch.tensor([0., 1., 0.], device=z.device).expand(B, 3))
    u = _normalize(torch.cross(helper, z, dim=-1))
    v = torch.cross(z, u, dim=-1)
    return u, v


class MLPHead(nn.Module):
    def __init__(self, in_ch, out_dim, hidden=512, act=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )
        if not act:
            # for completeness if you want no nonlinearity, but here we ReLU inside
            pass

    def forward(self, x):
        return self.net(x)


class RayNet(nn.Module):
    """
    FLAME-compliant eye module:
      - Uses only CoordAtt after backbone (no PANet/FPN/fusion)
      - Predicts head pose (6D), per-eye local pose (6D), per-eye head-local centers,
        per-eye iris-plane offset alpha in [0,1], small kappa angles (visual vs optical),
        and a 2D iris landmark head (for MediaPipe-style depth and 2D supervision).
      - Builds 3D iris rings analytically from identity (iris diameter, eyeball radius).
    """
    def __init__(
        self,
        backbone,
        in_channels_list,
        n_iris_landmarks=100,
        eyeball_radius_cm=1.2,         # ~12 mm
        iris_diameter_cm=1.17,         # ~11.7 mm
        kappa_max_deg=8.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.C4 = in_channels_list[-1]
        self.coordatt = CoordAtt(self.C4, self.C4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        flat_dim = self.C4

        self.nL = n_iris_landmarks
        self.eyeball_radius_cm = float(eyeball_radius_cm)
        self.iris_diameter_cm = float(iris_diameter_cm)
        self.kappa_max = math.radians(kappa_max_deg)

        # --- heads ---
        self.head_pose_6d = MLPHead(flat_dim, 6)            # head rotation
        self.eye_rot6d_L  = MLPHead(flat_dim, 6)
        self.eye_rot6d_R  = MLPHead(flat_dim, 6)

        self.eye_center_H_L = MLPHead(flat_dim, 3)          # centers in head-local (cm)
        self.eye_center_H_R = MLPHead(flat_dim, 3)

        self.alpha_L = MLPHead(flat_dim, 1)                 # iris-plane offset factor [0,1]
        self.alpha_R = MLPHead(flat_dim, 1)

        self.kappa_L = MLPHead(flat_dim, 2)                 # yaw, pitch (deg->rad by tanh * kappa_max)
        self.kappa_R = MLPHead(flat_dim, 2)

        # 2D iris landmark head (per-eye; normalized to [-1,1], later scaled to pixels)
        self.iris2d_L = MLPHead(flat_dim, self.nL * 2)
        self.iris2d_R = MLPHead(flat_dim, self.nL * 2)

        # Gaze direction head (6D -> rot; use z-axis as direction)
        self.gaze6d = MLPHead(flat_dim, 6)

        # Small learned pupil axial offsets (cm) for entrance pupil mismatch
        self.pupil_offset = nn.Parameter(torch.zeros(2))  # [δ_L, δ_R], will be squashed

    @torch.no_grad()
    def _canonical_ring(self, device):
        """
        Unit circle points in XY plane (N, 3)
        """
        t = torch.linspace(0, 2 * math.pi, self.nL, device=device, dtype=torch.float32)
        ring = torch.stack([torch.cos(t), torch.sin(t), torch.zeros_like(t)], dim=-1)
        return ring  # (N,3)

    def _forward_backbone(self, x):
        # RepNeXt style
        x = self.backbone.stem(x)
        x = self.backbone.stages[0](x)
        x = self.backbone.stages[1](x)
        x = self.backbone.stages[2](x)
        x = self.backbone.stages[3](x)  # C4 stride 32
        return x

    def forward(self, x, K=None, head_pose_gt=None, image_size=None, global_step=0, warmup_steps=1000):
        """
        x: (B,3,H,W)
        K: optional intrinsics (B,3,3) for projected previews (not required to compute outputs)
        head_pose_gt: optional {'R': (B,3,3), 't': (B,3)} if you want to compose eyes with GT translation
        image_size: optional (H,W) to scale 2D landmarks to pixels
        """
        B, _, H, W = x.shape
        feat = self._forward_backbone(x)
        feat = self.coordatt(feat)
        g = self.pool(feat).flatten(1)   # (B,C)

        # raw predictions
        R_H     = ortho6d_to_rotmat(self.head_pose_6d(g))       # (B,3,3)
        R_EL    = ortho6d_to_rotmat(self.eye_rot6d_L(g))         # (B,3,3)
        R_ER    = ortho6d_to_rotmat(self.eye_rot6d_R(g))         # (B,3,3)
        cEL_H   = self.eye_center_H_L(g)                         # (B,3)
        cER_H   = self.eye_center_H_R(g)                         # (B,3)

        aL      = torch.sigmoid(self.alpha_L(g)).squeeze(-1)     # (B,)
        aR      = torch.sigmoid(self.alpha_R(g)).squeeze(-1)     # (B,)

        # clamp kappa within +/- kappa_max
        kL      = torch.tanh(self.kappa_L(g)) * self.kappa_max   # (B,2)
        kR      = torch.tanh(self.kappa_R(g)) * self.kappa_max   # (B,2)

        gaze6d  = self.gaze6d(g)
        Rg      = ortho6d_to_rotmat(gaze6d)                      # (B,3,3)
        gaze_dir= _normalize(Rg[..., 2])                         # (B,3)

        # centers in camera space (use predicted head R; for t, prefer GT if given)
        if head_pose_gt is not None and 't' in head_pose_gt:
            t_H = head_pose_gt['t']                              # (B,3)
        else:
            t_H = torch.zeros_like(cEL_H)

        cEL_C = torch.einsum('bij,bj->bi', R_H, cEL_H) + t_H     # (B,3)
        cER_C = torch.einsum('bij,bj->bi', R_H, cER_H) + t_H     # (B,3)
        REL_C = torch.einsum('bij,bjk->bik', R_H, R_EL)          # (B,3,3)
        RER_C = torch.einsum('bij,bjk->bik', R_H, R_ER)          # (B,3,3)

        # optical axes (z-axis of eye frames)
        z = torch.tensor([0., 0., 1.], device=x.device).expand(B, 3)
        oL = torch.einsum('bij,bj->bi', REL_C, z)                # (B,3)
        oR = torch.einsum('bij,bj->bi', RER_C, z)                # (B,3)
        oL = _normalize(oL); oR = _normalize(oR)

        # visual axes apply kappa rotations
        RLk = small_angle_rpy_to_rot(kL)                         # (B,3,3)
        RRk = small_angle_rpy_to_rot(kR)
        vL = torch.einsum('bij,bj->bi', RLk, oL)                 # (B,3)
        vR = torch.einsum('bij,bj->bi', RRk, oR)                 # (B,3)
        vL = _normalize(vL); vR = _normalize(vR)

        # iris centers along optical axis; add small learnable axial pupil offsets (entrance pupil mismatch)
        rE = torch.tensor(self.eyeball_radius_cm, device=x.device)
        delta = torch.tanh(self.pupil_offset) * 0.2              # clamp +/- 0.2 cm
        pL = cEL_C + oL * (aL.unsqueeze(-1) * rE + delta[0])
        pR = cER_C + oR * (aR.unsqueeze(-1) * rE + delta[1])

        # analytic 3D iris ring for each eye
        ring = self._canonical_ring(x.device)                    # (N,3) in canonical XY plane
        D = torch.tensor(self.iris_diameter_cm, device=x.device)
        radius = 0.5 * D

        # Build ortho bases for each eye
        uL, vLbasis = build_orthonormal_basis(oL)
        uR, vRbasis = build_orthonormal_basis(oR)
        # (B,N,3)
        ringL = pL[:, None, :] + radius * (uL[:, None, :] * ring[None, :, 0:1] +
                                           vLbasis[:, None, :] * ring[None, :, 1:2])
        ringR = pR[:, None, :] + radius * (uR[:, None, :] * ring[None, :, 0:1] +
                                           vRbasis[:, None, :] * ring[None, :, 1:2])

        # separate 2D iris head (normalized to [-1,1] then scaled to pixels if size supplied)
        L2d = self.iris2d_L(g).reshape(B, self.nL, 2)
        R2d = self.iris2d_R(g).reshape(B, self.nL, 2)
        L2d = torch.tanh(L2d)  # [-1,1]
        R2d = torch.tanh(R2d)
        if image_size is None:
            Hn, Wn = H, W
        else:
            Hn, Wn = image_size
        # map to pixel coords
        L2d_px = torch.stack([(L2d[..., 0] + 1) * 0.5 * Wn, (L2d[..., 1] + 1) * 0.5 * Hn], dim=-1)
        R2d_px = torch.stack([(R2d[..., 0] + 1) * 0.5 * Wn, (R2d[..., 1] + 1) * 0.5 * Hn], dim=-1)

        # ray origin (between pupils) and direction
        origin = 0.5 * (pL + pR)                                 # (B,3)
        direction = gaze_dir                                     # (B,3)

        out = {
            # raw heads
            "head_pose_6d": self.head_pose_6d(g),
            "eye_rot6d_L":  self.eye_rot6d_L(g),
            "eye_rot6d_R":  self.eye_rot6d_R(g),
            "eye_center_H_L": cEL_H,
            "eye_center_H_R": cER_H,
            "alpha_L": aL, "alpha_R": aR,
            "kappa_L": kL, "kappa_R": kR,
            "gaze_vector_6d": gaze6d,

            # composed camera-space geometry
            "R_H": R_H, "t_H": t_H,
            "R_EL_C": REL_C, "R_ER_C": RER_C,
            "cEL_C": cEL_C, "cER_C": cER_C,
            "optic_L": oL, "optic_R": oR,
            "visual_L": vL, "visual_R": vR,
            "pupil_L": pL, "pupil_R": pR,

            # 3D iris rings (camera space, cm)
            "iris3d_L": ringL,  # (B,N,3)
            "iris3d_R": ringR,

            # 2D iris landmarks (pixels)
            "iris2d_L_px": L2d_px,  # (B,N,2)
            "iris2d_R_px": R2d_px,

            # gaze ray
            "ray_origin": origin,
            "ray_dir": direction,

            # identity (for completeness)
            "eyeball_radius_cm": rE,
            "iris_diameter_cm": D,
        }
        return out
