"""
Causal dilated TCN for OpenEDS temporal smoothing.

Inputs are per-frame feature vectors from the OpenEDS path:

    e_t  =  [gaze_3d (3),
             ellipse_iris (5),  # cx, cy, a, b, theta
             ellipse_pupil (5),
             torsion_raw (1),   # raw per-frame torsion from IrisPolarTorsion
             pupil_area_norm (1),
             ...]

The TCN aggregates a window of frames (causal, so prediction at t
depends only on t-K..t) and emits per-frame outputs:

    smoothed_gaze    (3,)
    torsion_residual (1,)        added to torsion_raw to denoise
    blink_logit      (1,)
    movement_class   (4,)        {fixation, saccade, pursuit, blink}

Receptive field of the default 4-layer dilated stack with kernel 3
and dilations [1, 2, 4, 8] is ``1 + 2 * (k-1) * sum(d) = 61`` frames
(two causal convs per residual block). At 100 Hz this is ~610 ms —
covers a saccade (30-80 ms) plus the full post-saccade fixation
onset (~250 ms). Use :meth:`TCNTemporalBlock.receptive_field` to
query the value at runtime if you adjust dilations.

The block is ``training-time parallel`` (depthwise temporal conv has
no recurrent state) and ``streaming-deployable`` (causal padding +
constant memory). Ideal for real-time medical eye-tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCNOutputs:
    """Per-frame outputs from :class:`TCNTemporalBlock`."""
    smoothed_gaze: torch.Tensor       # (B, T, 3) unit
    torsion_residual: torch.Tensor    # (B, T, 1) degrees, added to raw torsion
    blink_logit: torch.Tensor         # (B, T, 1) BCE logit
    movement_class: torch.Tensor      # (B, T, 4) softmax logits


class _CausalDilatedConv1d(nn.Module):
    """Causal 1D conv with left-only zero padding so output[t] depends
    on input[<= t]."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int,
                 dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))   # (B, C, T+left_pad)
        return self.conv(x)                 # (B, C, T)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = _CausalDilatedConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = _CausalDilatedConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class TCNTemporalBlock(nn.Module):
    """4-stage dilated TCN with multi-head per-frame outputs.

    Args:
        in_dim: per-frame feature dimensionality (matches the OpenEDS
            front-end's ``e_t``). Default 15 covers the canonical
            (gaze 3 + ellipse_iris 5 + ellipse_pupil 5 + torsion 1 +
            pupil_area 1) packing.
        hidden_dim: internal channel width.
        kernel_size: temporal kernel size (3 by default).
        dilations: dilation per residual block. Default [1, 2, 4, 8]
            gives RF≈31 frames.
        dropout: spatial dropout inside each residual block.

    The receptive field is reported by :meth:`receptive_field`.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden_dim, kernel_size, d, dropout)
            for d in dilations
        ])
        # Multi-head per-frame outputs.
        self.head_gaze = nn.Conv1d(hidden_dim, 3, 1)
        self.head_torsion = nn.Conv1d(hidden_dim, 1, 1)
        self.head_blink = nn.Conv1d(hidden_dim, 1, 1)
        self.head_movement = nn.Conv1d(hidden_dim, 4, 1)
        self._kernel_size = kernel_size
        self._dilations = tuple(dilations)

    def receptive_field(self) -> int:
        """Causal receptive field in frames (each input frame
        affects how many frames into the future)."""
        # Each ResidualBlock has 2 convs; total RF = 1 + 2*(k-1)*sum(d).
        return 1 + 2 * (self._kernel_size - 1) * sum(self._dilations)

    def forward(self, e_seq: torch.Tensor) -> TCNOutputs:
        """
        Args:
            e_seq: (B, T, in_dim) per-frame features.

        Returns:
            :class:`TCNOutputs` with per-frame predictions.
        """
        # 1D conv expects (B, C, T)
        x = e_seq.transpose(1, 2)              # (B, in_dim, T)
        x = self.in_proj(x)                    # (B, hidden, T)
        for block in self.blocks:
            x = block(x)                       # (B, hidden, T)

        gaze = self.head_gaze(x)               # (B, 3, T)
        torsion = self.head_torsion(x)         # (B, 1, T)
        blink = self.head_blink(x)
        movement = self.head_movement(x)

        # Normalise the gaze residual so the output is a unit vector.
        gaze = gaze.transpose(1, 2)            # (B, T, 3)
        gaze = F.normalize(gaze, dim=-1)

        return TCNOutputs(
            smoothed_gaze=gaze,
            torsion_residual=torsion.transpose(1, 2),
            blink_logit=blink.transpose(1, 2),
            movement_class=movement.transpose(1, 2),
        )


def derive_pseudo_blink_labels(
    pupil_mask_areas: torch.Tensor, ratio: float = 0.3,
) -> torch.Tensor:
    """Pseudo-label blink frames where pupil-mask area drops below a
    fraction of the per-sequence median.

    Args:
        pupil_mask_areas: (B, T) per-frame pixel counts of class==pupil.
        ratio: threshold relative to per-sequence median.

    Returns:
        (B, T) bool tensor — True for frames classified as blinks.
    """
    median = pupil_mask_areas.median(dim=1, keepdim=True).values
    return pupil_mask_areas < (ratio * median).clamp(min=1.0)


def derive_pseudo_saccade_labels(
    gaze_seq: torch.Tensor, threshold_deg_per_frame: float = 0.3,
) -> torch.Tensor:
    """Pseudo-label saccade frames where per-frame gaze velocity
    exceeds a threshold (degrees per frame).

    Saccade peak velocity is ~300-700°/s; at 100 Hz that's 3-7°/frame.
    The default 0.3°/frame is intentionally permissive — we want a
    high-recall signal that the supervised classifier head can refine.

    Args:
        gaze_seq: (B, T, 3) unit gaze vectors.
        threshold_deg_per_frame: detection threshold.

    Returns:
        (B, T) bool tensor — True for frames classified as saccade.
    """
    eps = 1e-6
    gaze_t = F.normalize(gaze_seq[:, :-1], dim=-1)
    gaze_tp1 = F.normalize(gaze_seq[:, 1:], dim=-1)
    cos = (gaze_t * gaze_tp1).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    angle_deg = torch.acos(cos) * 180.0 / 3.14159265358979
    # Pad first frame with the gradient of frame[1] so output length matches.
    angle_deg = F.pad(angle_deg, (1, 0), mode='replicate')
    return angle_deg > threshold_deg_per_frame


__all__ = [
    'TCNTemporalBlock',
    'TCNOutputs',
    'derive_pseudo_blink_labels',
    'derive_pseudo_saccade_labels',
]
