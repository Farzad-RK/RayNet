# head_gaze/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from RayNet.coordatt import CoordAtt

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

class FiLM(nn.Module):
    """Optional FiLM conditioner from head-pose 6D -> (gamma,beta) for the hidden."""
    def __init__(self, cond_dim=6, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
    def forward(self, h, cond):
        gb = self.mlp(cond)                # [B,2H]
        gamma, beta = gb.chunk(2, dim=1)   # [B,H],[B,H]
        return gamma * h + beta

class GazeVectorRegressionHead(nn.Module):
    """
    vMF gaze head:
      - predicts mean direction μ ∈ S^2 and concentration κ ≥ 0
      - optionally emits a yaw–pitch heatmap (disabled by default)

    Returns by default: dict {"mu":[B,3], "kappa":[B,1]}
    If produce_heatmap=True: also {"heatmap":[B,S_y,S_x]}
    """
    def __init__(self,
                 in_channels: int = 256,
                 hidden_dim: int = 128,
                 reduction: int = 32,
                 dropout: float = 0.0,
                 use_bn: bool = False,
                 use_film: bool = False,
                 produce_heatmap: bool = False,
                 heatmap_size: tuple = (64, 128),  # (pitch bins, yaw bins)
                 heatmap_temperature: float = 1.0):
        super().__init__()
        self.use_bn = use_bn
        self.use_film = use_film
        self.produce_heatmap = produce_heatmap
        self.Sy, self.Sx = heatmap_size
        self.hm_temp = heatmap_temperature

        # CoordAtt + global pooling
        self.coord_att = CoordAtt(in_channels, in_channels, reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP trunk
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if use_film:
            self.film = FiLM(cond_dim=6, hidden_dim=hidden_dim)

        # Outputs: μ (3) and κ (1)
        self.fc_mu = nn.Linear(hidden_dim, 3)
        self.fc_k  = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc_mu.weight, std=1e-3)
        nn.init.constant_(self.fc_mu.bias, 0.)
        nn.init.constant_(self.fc_k.weight, 0.)
        nn.init.constant_(self.fc_k.bias, 0.)

        # Precompute yaw–pitch unit-direction grid (if needed)
        if self.produce_heatmap:
            # yaw in [-pi, pi], pitch in [-pi/2, pi/2]
            yaw   = torch.linspace(-math.pi, math.pi,  self.Sx)
            pitch = torch.linspace(-math.pi/2, math.pi/2, self.Sy)
            Yaw, Pitch = torch.meshgrid(yaw, pitch, indexing='xy')  # [Sx,Sy]
            # Convert (yaw, pitch) -> unit direction (x,y,z)
            # x = cos(pitch) * cos(yaw)
            # y = sin(pitch)
            # z = cos(pitch) * sin(yaw)
            x = torch.cos(Pitch) * torch.cos(Yaw)
            y = torch.sin(Pitch)
            z = torch.cos(Pitch) * torch.sin(Yaw)
            dirs = torch.stack([x, y, z], dim=-1)  # [Sx,Sy,3]
            dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
            # store as buffer in [Sy*Sx, 3], row-major (pitch,yaw) for [Sy,Sx] reshape later
            dirs = dirs.permute(1, 0, 2).contiguous().view(-1, 3)  # [Sy*Sx,3]
            self.register_buffer("dir_grid", dirs, persistent=False)

    def forward(self, x: torch.Tensor, head_pose_6d: torch.Tensor = None) -> dict:
        """
        x:             [B,C,H,W]
        head_pose_6d:  [B,6] or None (FiLM conditioner)
        returns: dict {"mu":[B,3], "kappa":[B,1]} (+ optional "heatmap":[B,Sy,Sx])
        """
        x = self.coord_att(x)
        x = self.pool(x).flatten(1)             # [B,C]
        h = self.act(self.bn1(self.fc1(x)))     # [B,H]
        h = self.drop(h)
        if self.use_film and head_pose_6d is not None:
            h = self.film(h, head_pose_6d)

        mu_raw = self.fc_mu(h)                  # [B,3]
        k_raw  = self.fc_k(h)                   # [B,1]
        mu = _normalize(mu_raw)                 # unit direction
        kappa = F.softplus(k_raw)               # >= 0

        out = {"mu": mu, "kappa": kappa}

        if self.produce_heatmap:
            # compute logits = κ * (μ · d_grid) / τ   -> softmax to [B, Sy*Sx]
            # dir_grid: [Sy*Sx,3]
            dots = torch.matmul(mu, self.dir_grid.t())            # [B, Sy*Sx]
            logits = (kappa * dots.unsqueeze(1)).squeeze(1) / max(self.hm_temp, 1e-6)
            heat = F.softmax(logits, dim=-1).view(-1, self.Sy, self.Sx)  # [B,Sy,Sx]
            out["heatmap"] = heat

        return out
