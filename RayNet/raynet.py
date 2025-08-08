# raynet.py
# Two-stage RayNet with Coordinate Attention (from coordatt.py) and a Cross-Scale Attention neck.
# Stage-1: head pose, gaze vector, pupil center
# Stage-2: gaze depth, gaze point
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backbone.repnext_utils import load_pretrained_repnext
from coordatt import CoordAtt

from head_pose.model import HeadPoseRegressionHead
from gaze_vector.model import GazeVectorRegressionHead
from pupil_center.model import PupilCenterRegressionHead
from gaze_point.model import GazePointRegressionHead
from gaze_depth.model import GazeDepthRegressionHead

from utils import ortho6d_to_rotmat

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# 2D sinusoidal positional encoding (lightweight)
# ---------------------------
def pos_enc_2d(h, w, dim, device):
    assert dim % 4 == 0, "pos_enc_2d: dim must be multiple of 4"
    pe = torch.zeros(1, dim, h, w, device=device)
    dim_half = dim // 2
    dim_quarter = dim // 4

    y = torch.arange(h, device=device).unsqueeze(1).float()  # [H,1]
    x = torch.arange(w, device=device).unsqueeze(0).float()  # [1,W]

    div_term_y = torch.exp(torch.arange(0, dim_quarter, 2, device=device).float()
                           * (-math.log(10000.0) / dim_quarter))
    div_term_x = torch.exp(torch.arange(0, dim_quarter, 2, device=device).float()
                           * (-math.log(10000.0) / dim_quarter))

    pe[:, 0:dim_quarter:2, :, :] = torch.sin(y * div_term_y)[:, None, :, None]
    pe[:, 1:dim_quarter:2, :, :] = torch.cos(y * div_term_y)[:, None, :, None]
    pe[:, dim_quarter:dim_half:2, :, :] = torch.sin(x * div_term_x)[None, None, :, :]
    pe[:, dim_quarter + 1:dim_half:2, :, :] = torch.cos(x * div_term_x)[None, None, :, :]

    # second half kept zero; can be used for level embeddings if needed
    return pe


# ---------------------------
# Cross-Scale Attention Neck (replaces PANet / MultiScaleFusion)
# Enrich highest-resolution feature (C2) using multi-scale context (C2..C5) via efficient attention.
# ---------------------------
class CrossScaleAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, d_model=128, num_heads=4,
                 grid=8, query_stride=4, ffn_expansion=2):
        """
        Args:
            in_channels_list: [C2,C3,C4,C5] channels from backbone stages
            out_channels: unified channels after 1x1 projection per level
            d_model: attention channel size (must be divisible by num_heads)
            num_heads: attention heads
            grid: pooled token grid size per level (g x g)
            query_stride: downsample factor for queries on C2 (e.g., 4)
            ffn_expansion: expansion ratio for conv FFN
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.levels = len(in_channels_list)
        self.query_stride = query_stride
        self.grid = grid

        # Project each level to a common channel dimension
        self.proj_in = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])

        # Q/K/V projections into attention space
        self.q_proj = nn.Conv2d(out_channels, d_model, 1)
        self.k_proj = nn.Conv2d(out_channels, d_model, 1)
        self.v_proj = nn.Conv2d(out_channels, d_model, 1)

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False)

        self.proj_out = nn.Conv2d(d_model, out_channels, 1)

        hidden = out_channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, hidden, 3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1)
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def _tokenize_grid(self, x, g):
        # Average pool to g×g tokens
        return F.adaptive_avg_pool2d(x, (g, g))

    def forward(self, c_feats):
        """
        c_feats: list of 4 tensors [C2,C3,C4,C5] (after CoordAtt), shapes vary
        Returns: enriched high-res map [B, out_channels, H2, W2]
        """
        # Project per level
        p = [proj(f) for proj, f in zip(self.proj_in, c_feats)]

        # Target: level 0 (C2) — highest resolution
        p2 = p[0]
        B, C, H, W = p2.shape

        # Queries on a strided grid for efficiency
        Hq, Wq = math.ceil(H / self.query_stride), math.ceil(W / self.query_stride)
        q_map = F.interpolate(p2, size=(Hq, Wq), mode="bilinear", align_corners=False)
        q = self.q_proj(q_map)  # [B, d_model, Hq, Wq]
        q = q + pos_enc_2d(Hq, Wq, q.shape[1], q.device)
        q = q.flatten(2).permute(2, 0, 1)  # [Hq*Wq, B, d_model]

        # Keys/Values from pooled tokens of each level
        K_tokens, V_tokens = [], []
        for x in p:
            kvi = self._tokenize_grid(x, self.grid)  # [B,C,g,g]
            K_tokens.append(self.k_proj(kvi))
            V_tokens.append(self.v_proj(kvi))

        K = torch.cat([k.flatten(2).permute(2, 0, 1) for k in K_tokens], dim=0)  # [L, B, d_model]
        V = torch.cat([v.flatten(2).permute(2, 0, 1) for v in V_tokens], dim=0)  # [L, B, d_model]

        attn_out, _ = self.attn(q, K, V)  # [Hq*Wq, B, d_model]
        attn_out = attn_out.permute(1, 2, 0).reshape(B, -1, Hq, Wq)  # [B,d,Hq,Wq]

        # Project back + upsample to HxW
        attn_out = self.proj_out(attn_out)
        attn_out = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)

        # Residual + local FFN
        y = self.norm1(p2 + attn_out)
        y = self.norm2(y + self.ffn(y))
        return y  # [B, C, H, W]


# =========================================================
# Stage 1: head pose + gaze vector + pupil center
# =========================================================
class RayNetStage1(nn.Module):
    """
    Backbone + CoordAtt per stage + CrossScaleAttention neck
    -> heads: head_pose_6d, gaze_vector_6d, pupil_center_3d
    Also returns 'feat' (enriched high-res map), 'origin' and 'direction'.
    """
    def __init__(self, backbone, in_channels_list, out_channels=256,
                 ca_reduction=32, d_model=128, num_heads=4, grid=8, query_stride=4):
        super().__init__()
        self.backbone = backbone

        # Coordinate Attention at each stage (no channel change: oup = inp)
        c1, c2, c3, c4 = in_channels_list
        self.ca1 = CoordAtt(c1, c1, reduction=ca_reduction)
        self.ca2 = CoordAtt(c2, c2, reduction=ca_reduction)
        self.ca3 = CoordAtt(c3, c3, reduction=ca_reduction)
        self.ca4 = CoordAtt(c4, c4, reduction=ca_reduction)

        # Cross-Scale Attention neck (replaces PANet / MultiScaleFusion)
        self.xattn = CrossScaleAttention(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            d_model=d_model,
            num_heads=num_heads,
            grid=grid,
            query_stride=query_stride,
        )

        C = out_channels
        self.head_pose    = HeadPoseRegressionHead(in_channels=C, hidden_dim=128, reduction=32)
        self.gaze_vector  = GazeVectorRegressionHead(in_channels=C, hidden_dim=128, reduction=32)
        self.pupil_center = PupilCenterRegressionHead(in_channels=C, hidden_dim=128, reduction=32)

    def _stages(self, x):
        c0 = checkpoint(self.backbone.stem, x)
        c1 = checkpoint(self.backbone.stages[0], c0)
        c2 = checkpoint(self.backbone.stages[1], c1)
        c3 = checkpoint(self.backbone.stages[2], c2)
        c4 = checkpoint(self.backbone.stages[3], c3)
        return [c1, c2, c3, c4]

    def forward(self, x):
        c1, c2, c3, c4 = self._stages(x)

        # Coordinate Attention gates
        c1, c2, c3, c4 = self.ca1(c1), self.ca2(c2), self.ca3(c3), self.ca4(c4)

        # Cross-scale attention to enrich high-res features
        feat = self.xattn([c1, c2, c3, c4])  # [B, C, H, W]

        # Task heads
        head_pose_6d     = self.head_pose(feat)      # [B, 6]
        gaze_vector_6d   = self.gaze_vector(feat)    # [B, 6]
        pupil_center_3d  = self.pupil_center(feat)   # [B, 2, 3]

        # Geometry for downstream
        R = ortho6d_to_rotmat(gaze_vector_6d)        # [B,3,3]
        direction = R[:, :, 2]
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        origin = pupil_center_3d.mean(dim=1)         # [B,3]

        return {
            "feat": feat,  # enriched high-res map
            "head_pose_6d": head_pose_6d,
            "gaze_vector_6d": gaze_vector_6d,
            "direction": direction,
            "pupil_center_3d": pupil_center_3d,
            "origin": origin,
        }


# =========================================================
# Stage 2: gaze depth + gaze point (train after freezing Stage-1)
# =========================================================
class RayNetStage2(nn.Module):
    """
    Consumes Stage-1 'feat' (enriched high-res map) and predicts:
      - gaze_depth [B]
      - gaze_point_3d [B,3]
      - gaze_point_from_ray = origin + depth * direction
    """
    def __init__(self, in_channels=256):
        super().__init__()
        C = in_channels
        self.gaze_point = GazePointRegressionHead(in_channels=C, hidden_dim=128, reduction=32)
        self.gaze_depth = GazeDepthRegressionHead(in_channels=C, hidden_dim=128, reduction=32)

    def forward(self, feat, origin, direction):
        gaze_point_3d = self.gaze_point(feat)           # [B,3]
        gaze_depth    = self.gaze_depth(feat)           # [B]
        gaze_point_from_ray = origin + gaze_depth.unsqueeze(-1) * direction
        return {
            "gaze_point_3d": gaze_point_3d,
            "gaze_depth": gaze_depth,
            "gaze_point_from_ray": gaze_point_from_ray,
        }


# =========================================================
# Simple pipeline wrapper for inference or Stage-2 training
# =========================================================
class RayPipeline(nn.Module):
    def __init__(self, stage1: RayNetStage1, stage2: RayNetStage2, freeze_stage1: bool = False):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        if freeze_stage1:
            for p in self.stage1.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(feat=s1["feat"], origin=s1["origin"], direction=s1["direction"])
        return {**s1, **s2}


# ---------------------------
# Quick sanity run
# ---------------------------
if __name__ == "__main__":
    backbone_name = "repnext_m3"
    weight_path = "./repnext_m3_pretrained.pt"
    repnext_model = load_pretrained_repnext(backbone_name, weight_path).to(device)

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

    stage1 = RayNetStage1(
        backbone=repnext_model,
        in_channels_list=in_channels_list,
        out_channels=256,      # feature channels after neck
        ca_reduction=32,
        d_model=128,           # attention hidden size (divisible by num_heads)
        num_heads=4,
        grid=8,
        query_stride=4
    ).to(device)

    stage2 = RayNetStage2(in_channels=256).to(device)

    pipe = RayPipeline(stage1, stage2, freeze_stage1=False).to(device)

    x = torch.randn(2, 3, 448, 448, device=device)
    out = pipe(x)

    print("feat:", out["feat"].shape)                           # [B,256,H,W]
    print("head_pose_6d:", out["head_pose_6d"].shape)           # [B,6]
    print("gaze_vector_6d:", out["gaze_vector_6d"].shape)       # [B,6]
    print("pupil_center_3d:", out["pupil_center_3d"].shape)     # [B,2,3]
    print("direction:", out["direction"].shape)                 # [B,3]
    print("origin:", out["origin"].shape)                       # [B,3]
    print("gaze_depth:", out["gaze_depth"].shape)               # [B]
    print("gaze_point_3d:", out["gaze_point_3d"].shape)         # [B,3]
