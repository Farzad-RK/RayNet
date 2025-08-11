# raynet.py
# Stage-1: head pose + vMF gaze + pupil (geometric params)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backbone.repnext_utils import load_pretrained_repnext  # used only by your own launcher, not here
from coordatt import CoordAtt  # keep this import style to match your project

from head_pose.model import HeadPoseRegressionHead
from gaze_vector.model import GazeVectorRegressionHead
from pupil_center.model import PupilCenterRegressionHead


def pos_enc_2d(H: int, W: int, dim: int, device=None, dtype=torch.float32):
    """2D sinusoidal positional encoding, returns [1, dim, H, W]. dim must be divisible by 4."""
    assert dim % 4 == 0, "pos_enc_2d: dim must be divisible by 4"
    pe = torch.zeros(1, dim, H, W, device=device, dtype=dtype)
    dim_half = dim // 2
    dim_quarter = dim_half // 2

    y = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)  # [H,1]
    div_y = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=dtype)
                      * (-math.log(10000.0) / dim_half))
    siny = torch.sin(y * div_y)   # [H, dim_quarter]
    cosy = torch.cos(y * div_y)   # [H, dim_quarter]
    pe[:, 0:dim_half:2, :, :] = siny.T.unsqueeze(0).unsqueeze(-1)
    pe[:, 1:dim_half:2, :, :] = cosy.T.unsqueeze(0).unsqueeze(-1)

    x = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)  # [W,1]
    div_x = torch.exp(torch.arange(0, dim_half, 2, device=device, dtype=dtype)
                      * (-math.log(10000.0) / dim_half))
    sinx = torch.sin(x * div_x)   # [W, dim_quarter]
    cosx = torch.cos(x * div_x)   # [W, dim_quarter]
    pe[:, dim_half:dim:2, :, :]   = sinx.T.unsqueeze(0).unsqueeze(2)
    pe[:, dim_half+1:dim:2, :, :] = cosx.T.unsqueeze(0).unsqueeze(2)
    return pe


class CrossScaleAttention(nn.Module):
    """Fuse multi-scale features into the highest-res map via efficient attention."""
    def __init__(self, in_channels_list, out_channels=256, d_model=128, num_heads=4,
                 grid=8, query_stride=4, ffn_expansion=2):
        super().__init__()
        assert d_model % num_heads == 0
        self.query_stride = query_stride
        self.grid = grid

        self.proj_in = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.q_proj = nn.Conv2d(out_channels, d_model, 1)
        self.k_proj = nn.Conv2d(out_channels, d_model, 1)
        self.v_proj = nn.Conv2d(out_channels, d_model, 1)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=False)
        self.proj_out = nn.Conv2d(d_model, out_channels, 1)

        hidden = out_channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def _tokenize_grid(self, x, g):
        return F.adaptive_avg_pool2d(x, (g, g))

    def forward(self, c_feats):
        p = [proj(f) for proj, f in zip(self.proj_in, c_feats)]
        p2 = p[0]
        B, C, H, W = p2.shape

        Hq, Wq = math.ceil(H / self.query_stride), math.ceil(W / self.query_stride)
        q_map = F.interpolate(p2, size=(Hq, Wq), mode="bilinear", align_corners=False)
        q = self.q_proj(q_map)
        q = q + pos_enc_2d(Hq, Wq, q.size(1), device=q.device, dtype=q.dtype)
        q = q.flatten(2).permute(2, 0, 1)  # [Hq*Wq, B, d_model]

        K_tokens, V_tokens = [], []
        for x in p:
            kvi = self._tokenize_grid(x, self.grid)
            K_tokens.append(self.k_proj(kvi))
            V_tokens.append(self.v_proj(kvi))
        K = torch.cat([k.flatten(2).permute(2, 0, 1) for k in K_tokens], dim=0)
        V = torch.cat([v.flatten(2).permute(2, 0, 1) for v in V_tokens], dim=0)

        out, _ = self.attn(q, K, V)
        out = out.permute(1, 2, 0).reshape(B, -1, Hq, Wq)
        out = self.proj_out(out)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        y = self.norm1(p2 + out)
        y = self.norm2(y + self.ffn(y))
        return y


class RayNetStage1(nn.Module):
    """Backbone + CoordAtt per stage + CrossScaleAttention neck -> heads: HP, vMF gaze, Pupil (params)."""
    def __init__(self, backbone, in_channels_list, out_channels=256,
                 ca_reduction=32, d_model=128, num_heads=4, grid=8, query_stride=4):
        super().__init__()
        self.backbone = backbone

        c1, c2, c3, c4 = in_channels_list
        self.ca1 = CoordAtt(c1, c1, reduction=ca_reduction)
        self.ca2 = CoordAtt(c2, c2, reduction=ca_reduction)
        self.ca3 = CoordAtt(c3, c3, reduction=ca_reduction)
        self.ca4 = CoordAtt(c4, c4, reduction=ca_reduction)

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

    def forward(self, x):
        # Backbone with explicit non-reentrant checkpoints (avoid PyTorch 2.4 warning)
        c0 = self.backbone.stem(x)
        c1 = checkpoint(self.backbone.stages[0], c0, use_reentrant=False)
        c2 = checkpoint(self.backbone.stages[1], c1, use_reentrant=False)
        c3 = checkpoint(self.backbone.stages[2], c2, use_reentrant=False)
        c4 = checkpoint(self.backbone.stages[3], c3, use_reentrant=False)

        c1, c2, c3, c4 = self.ca1(c1), self.ca2(c2), self.ca3(c3), self.ca4(c4)
        feat = self.xattn([c1, c2, c3, c4])  # [B,C,H,W]

        head_pose_6d = self.head_pose(feat)              # [B,6]
        gaze_out     = self.gaze_vector(feat, head_pose_6d)  # {'mu':[B,3], 'kappa':[B,1]}
        pupil_out    = self.pupil_center(feat)           # {'ellipse':[B,2,6], 'delta_cm':[B,2,1], ...}

        # NOTE: we don't compute a 3D origin here; geometry is done in the loss.
        return {
            "head_pose_6d": head_pose_6d,
            "gaze": gaze_out,
            "pupil": pupil_out,
            "feat": feat,                # handy if you later add Stage-2
            "direction": gaze_out["mu"]  # convenience
        }
