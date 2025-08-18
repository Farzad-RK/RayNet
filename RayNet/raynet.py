import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.repnext_utils import load_pretrained_repnext
from torch.utils.checkpoint import checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Utility: 6D -> SO(3)
# ----------------------------
def sixd_to_R(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 6]
    returns rotation matrices [..., 3, 3]
    """
    a1 = F.normalize(x[..., 0:3], dim=-1)
    a2 = x[..., 3:6]
    b2 = F.normalize(a2 - (a1 * a2).sum(-1, keepdim=True) * a1, dim=-1)
    b3 = torch.cross(a1, b2, dim=-1)
    R = torch.stack([a1, b2, b3], dim=-2)
    return R


# ----------------------------
# Lightweight FPN neck
# ----------------------------
class Conv1x1(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv3x3(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyFPN(nn.Module):
    """
    Creates P2..P5 from C1..C4 with lateral 1x1 + top-down, then 3x3 smooth.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        assert len(in_channels_list) == 4
        C1, C2, C3, C4 = in_channels_list

        self.lats = nn.ModuleList([
            Conv1x1(C1, out_channels),
            Conv1x1(C2, out_channels),
            Conv1x1(C3, out_channels),
            Conv1x1(C4, out_channels),
        ])
        self.smooth = nn.ModuleList([
            Conv3x3(out_channels, out_channels),
            Conv3x3(out_channels, out_channels),
            Conv3x3(out_channels, out_channels),
            Conv3x3(out_channels, out_channels),
        ])

    def forward(self, feats):
        # feats = [C1, C2, C3, C4]
        c1, c2, c3, c4 = feats
        p4 = self.lats[3](c4)                     # top
        p3 = self.lats[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lats[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.lats[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")

        p1 = self.smooth[0](p1)
        p2 = self.smooth[1](p2)
        p3 = self.smooth[2](p3)
        p4 = self.smooth[3](p4)
        return [p1, p2, p3, p4]


class GlobalEmbed(nn.Module):
    """
    Global feature from pyramid by GAP + concat.
    """
    def __init__(self, num_levels=4, level_dim=256, out_dim=1024):
        super().__init__()
        in_dim = num_levels * level_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, ps):  # list of [N,C,H,W]
        pools = [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in ps]  # each [N,C]
        x = torch.cat(pools, dim=1)                                   # [N, L*C]
        return self.proj(x)                                           # [N, out_dim]


# ----------------------------
# Constrained parameter projector
# ----------------------------
class GeomBounds:
    """
    Default anatomical-ish bounds (adjust to your dataset units).
    Values are in the same unit as your labels (often mm).
    """
    def __init__(self,
                 r_eye_min=10.0, r_eye_max=15.0,
                 r_iris_min=4.0, r_iris_max=7.5,
                 r_cornea_min=6.5, r_cornea_max=9.0,
                 d_cornea_min=0.0, d_cornea_max=5.0,
                 kappa_max_deg=12.0):
        self.r_eye_min = r_eye_min
        self.r_eye_max = r_eye_max
        self.r_iris_min = r_iris_min
        self.r_iris_max = r_iris_max
        self.r_cornea_min = r_cornea_min
        self.r_cornea_max = r_cornea_max
        self.d_cornea_min = d_cornea_min
        self.d_cornea_max = d_cornea_max
        self.kappa_max = torch.tensor(kappa_max_deg * 3.14159265 / 180.0, dtype=torch.float32)


def _bounded_sigmoid(z, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(z)


class EyeParamProjector(nn.Module):
    """
    Maps raw unconstrained outputs to physically valid parameters.
    - Rotations: 6D -> SO(3)
    - Radii/lengths: bounded to [min,max]
    - Pupil radius: (0, r_iris)
    - Kappa: bounded to (-kappa_max, +kappa_max)
    """
    def __init__(self, bounds: GeomBounds):
        super().__init__()
        self.register_buffer("kappa_max", bounds.kappa_max)
        self.r_eye_min, self.r_eye_max = bounds.r_eye_min, bounds.r_eye_max
        self.r_iris_min, self.r_iris_max = bounds.r_iris_min, bounds.r_iris_max
        self.r_cornea_min, self.r_cornea_max = bounds.r_cornea_min, bounds.r_cornea_max
        self.d_cornea_min, self.d_cornea_max = bounds.d_cornea_min, bounds.d_cornea_max

    def forward(self, raw: dict) -> dict:
        out = {}

        # Per-view head pose
        # raw['head_rot6d']: [B,V,6], raw['head_t']: [B,V,3]
        B, V, _ = raw["head_rot6d"].shape
        R_head = sixd_to_R(raw["head_rot6d"].view(B * V, 6)).view(B, V, 3, 3)
        out["R_head"] = R_head
        out["t_head"] = raw["head_t"]  # [B,V,3]

        # Frame-level (shared across views)
        # eye rotations
        # raw['eye_rot6d']: [B,2,6]
        R_eye_L = sixd_to_R(raw["eye_rot6d"][:, 0])  # [B,3,3]
        R_eye_R = sixd_to_R(raw["eye_rot6d"][:, 1])  # [B,3,3]
        out["R_eye"] = torch.stack([R_eye_L, R_eye_R], dim=1)  # [B,2,3,3]

        # centers in HCS (can be predicted, or you can feed GT β directly during training)
        out["c_eye"] = raw["c_eye"]  # [B,2,3]

        # radii/lengths (shared L/R by default)
        r_eye    = _bounded_sigmoid(raw["z_r_eye"],    self.r_eye_min,    self.r_eye_max)    # [B,1]
        r_iris   = _bounded_sigmoid(raw["z_r_iris"],   self.r_iris_min,   self.r_iris_max)   # [B,1]
        r_cornea = _bounded_sigmoid(raw["z_r_cornea"], self.r_cornea_min, self.r_cornea_max) # [B,1]
        d_cornea = _bounded_sigmoid(raw["z_d_cornea"], self.d_cornea_min, self.d_cornea_max) # [B,1]
        out["r_eye"]    = r_eye
        out["r_iris"]   = r_iris
        out["r_cornea"] = r_cornea
        out["d_cornea"] = d_cornea

        # pupil radius per eye: (0, r_iris)
        # raw['z_pupil']: [B,2,1]
        r_pupil = torch.sigmoid(raw["z_pupil"]).clamp_min(1e-6) * r_iris.unsqueeze(1)  # [B,2,1]
        out["r_pupil"] = r_pupil

        # kappa per eye, bounded
        # raw['z_kappa']: [B,2,3]
        out["kappa"] = self.kappa_max * torch.tanh(raw["z_kappa"])  # [B,2,3]

        return out


# ----------------------------
# MLP heads
# ----------------------------
def mlp(in_dim, hidden, out_dim, dropout=0.0):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class RayNet(nn.Module):
    """
    FLAME-inspired eye parameter regressor with geometric constraints.
    Inputs:
      - Single view:  [B,3,H,W]
      - Multi-view:   [B,V,3,H,W]  (V=9)
    Outputs (dict):
      {
        'per_view': {'R_head':[B,V,3,3], 't_head':[B,V,3]},
        'frame':    {
            'R_eye':[B,2,3,3], 'c_eye':[B,2,3],
            'r_eye':[B,1], 'r_iris':[B,1], 'r_cornea':[B,1], 'd_cornea':[B,1],
            'r_pupil':[B,2,1], 'kappa':[B,2,3]
        },
        'raw': {...}  # optional, for losses/debug
      }
    """
    def __init__(self, backbone, in_channels_list, panet_out_channels=256,
                 embed_dim=1024, bounds: GeomBounds = GeomBounds()):
        super().__init__()

        # Backbone (RepNeXt)
        self.backbone = backbone

        # Neck: Tiny FPN + Global embedding per view
        self.fpn = TinyFPN(in_channels_list, out_channels=panet_out_channels)
        self.embed = GlobalEmbed(num_levels=4, level_dim=panet_out_channels, out_dim=embed_dim)

        # View-level head pose head (per view)
        self.view_head_pose = nn.ModuleDict({
            "rot": mlp(embed_dim, [512, 256], 6),   # 6D rotation
            "t":   mlp(embed_dim, [512, 256], 3),   # translation in CCS
        })

        # Frame-level shared heads (from fused multi-view feature)
        self.eye_rot_head = mlp(embed_dim, [512, 256], 12)   # 2 eyes × 6D
        self.center_head  = mlp(embed_dim, [512, 256], 6)    # c_L (3) + c_R (3) in HCS
        self.radii_head   = mlp(embed_dim, [512, 256], 4)    # z_r_eye, z_r_iris, z_r_cornea, z_d_cornea
        self.kappa_head   = mlp(embed_dim, [512, 256], 6)    # 2×3
        self.pupil_head   = mlp(embed_dim, [512, 256], 2)    # 2×1 (z)

        # Projector for constraints
        self.projector = EyeParamProjector(bounds)

    def _forward_backbone_feats(self, x: torch.Tensor):
        """
        x: [N,3,H,W]
        returns list [C1,C2,C3,C4]
        """
        # These calls follow your template with checkpoint
        c0 = checkpoint(self.backbone.stem, x)                 # stride=4
        c1 = checkpoint(self.backbone.stages[0], c0)           # stride=4
        c2 = checkpoint(self.backbone.stages[1], c1)           # stride=8
        c3 = checkpoint(self.backbone.stages[2], c2)           # stride=16
        c4 = checkpoint(self.backbone.stages[3], c3)           # stride=32
        return [c1, c2, c3, c4]

    def _per_view_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,3,H,W]
        return: [N, D]
        """
        feats = self._forward_backbone_feats(x)
        pyr = self.fpn(feats)
        emb = self.embed(pyr)
        return emb

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B,3,H,W] or [B,V,3,H,W]
        """
        single_view = (x.dim() == 4)
        if single_view:
            B, C, H, W = x.shape
            V = 1
            x_flat = x
        else:
            B, V, C, H, W = x.shape
            x_flat = x.view(B * V, C, H, W)

        # Per-view embeddings
        emb_v = self._per_view_embedding(x_flat)           # [B*V, D]
        emb_v = emb_v.view(B, V, -1)                       # [B, V, D]

        # View-level head pose
        head_rot6d = self.view_head_pose["rot"](emb_v.reshape(B*V, -1)).view(B, V, 6)
        head_t     = self.view_head_pose["t"](emb_v.reshape(B*V, -1)).view(B, V, 3)

        # Frame-level fused feature (mean across views)
        emb_f = emb_v.mean(dim=1)                          # [B, D]

        # Eye rotations (shared across views)
        eye_rot6d = self.eye_rot_head(emb_f).view(B, 2, 6) # [B,2,6]

        # Eye centers in HCS (shared across views)
        c_eye = self.center_head(emb_f).view(B, 2, 3)      # [B,2,3]

        # Radii & cornea depth (shared across views)
        z_r = self.radii_head(emb_f)                       # [B,4]
        z_r_eye, z_r_iris, z_r_cornea, z_d_cornea = torch.split(z_r, 1, dim=-1)

        # Kappa per eye
        z_kappa = self.kappa_head(emb_f).view(B, 2, 3)     # [B,2,3]

        # Pupil per eye
        z_pupil = self.pupil_head(emb_f).view(B, 2, 1)     # [B,2,1]

        raw = {
            "head_rot6d": head_rot6d,  # [B,V,6]
            "head_t":     head_t,      # [B,V,3]
            "eye_rot6d":  eye_rot6d,   # [B,2,6]
            "c_eye":      c_eye,       # [B,2,3] (HCS)
            "z_r_eye":    z_r_eye,     # [B,1]
            "z_r_iris":   z_r_iris,    # [B,1]
            "z_r_cornea": z_r_cornea,  # [B,1]
            "z_d_cornea": z_d_cornea,  # [B,1]
            "z_kappa":    z_kappa,     # [B,2,3]
            "z_pupil":    z_pupil,     # [B,2,1]
        }

        proj = self.projector(raw)

        out = {
            "per_view": {
                "R_head": proj["R_head"],  # [B,V,3,3]
                "t_head": proj["t_head"],  # [B,V,3]
            },
            "frame": {
                "R_eye":    proj["R_eye"],    # [B,2,3,3]
                "c_eye":    proj["c_eye"],    # [B,2,3] (HCS)
                "r_eye":    proj["r_eye"],    # [B,1]
                "r_iris":   proj["r_iris"],   # [B,1]
                "r_cornea": proj["r_cornea"], # [B,1]
                "d_cornea": proj["d_cornea"], # [B,1]
                "r_pupil":  proj["r_pupil"],  # [B,2,1]
                "kappa":    proj["kappa"],    # [B,2,3]
            },
            "raw": raw,  # keep for loss/debug if you want
        }
        return out


if __name__ == "__main__":
    backbone_name = "repnext_m3"
    weight_path = "./repnext_m3_pretrained.pt"
    repnext_model = load_pretrained_repnext(backbone_name, weight_path)
    repnext_model = repnext_model.to(device)

    # Channels from each RepNeXt variant
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }

    in_channels_list = backbone_channels_dict[backbone_name]
    raynet_model = RayNet(repnext_model, in_channels_list, panet_out_channels=256).to(device)

    # sanity: single-view input
    x = torch.randn(2, 3, 256, 256).to(device)
    out = raynet_model(x)
    print("single-view R_head:", out["per_view"]["R_head"].shape)

    # sanity: multi-view input (V=9)
    x_mv = torch.randn(1, 9, 3, 256, 256).to(device)
    out_mv = raynet_model(x_mv)
    print("multi-view R_head:", out_mv["per_view"]["R_head"].shape)
