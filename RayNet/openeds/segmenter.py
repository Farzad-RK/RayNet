"""
RITnet-style semantic segmenter for the OpenEDS foveal stage.

RITnet (Chaudhary et al., 2019) is the canonical light-weight U-Net
variant for OpenEDS-class IR eye-segmentation with a parameter budget
around 250k. The implementation here is faithful to that recipe but
parametrises the input channel count and class count.

Key design choices:
- **1-channel input** for OpenEDS grayscale IR; ``in_channels=2``
  enables the v6.2 *geometric prior* channel that seeds the
  segmenter with a Gaussian ROI around the GazeGene-predicted 3D
  eyeball centre. The prior channel is in [0, 1] and is concatenated
  before the stem.
- **4-class softmax output** matching FovalNet labels (background /
  sclera / iris / pupil). The :func:`combined_loss` helper mixes
  weighted cross-entropy + soft Dice; cross-entropy alone collapses
  on the rare classes (pupil ~1% of pixels).
- Stride-32-friendly: receptive-field down/up steps are factor-of-2
  so 416×640 inputs (the recommended pad for native 400×640) flow
  through cleanly.

Two factory presets:
- :func:`build_ritnet_full` (default ``base_channels=32``,
  ``growth_rate=16``) — ~2.3M params, our higher-capacity variant.
- :func:`build_ritnet_tiny` (``base_channels=16``, ``growth_rate=8``)
  — ~0.5M params, faithful to the original RITnet budget. Use this
  when the ~1.5k labelled OpenEDS frames overfit the larger variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3) -> nn.Sequential:
    """Conv2d → BN → LeakyReLU (RITnet uses LeakyReLU(0.01))."""
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


class _DenseBlock(nn.Module):
    """4-layer dense block with concatenated skip connections.

    Each layer adds ``growth_rate`` channels; final output channels
    are ``in_c + 4 * growth_rate``. Lightweight version of the
    DenseNet block used in RITnet (Sec 3.2 of the paper).
    """

    def __init__(self, in_c: int, growth_rate: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(
                _conv_bn_relu(in_c + i * growth_rate, growth_rate)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(feats, dim=1))
            feats.append(new_feat)
        return torch.cat(feats, dim=1)


class RITnetStyleSegmenter(nn.Module):
    """4-class semantic segmenter for OpenEDS IR eye crops.

    Args:
        in_channels: number of input channels (1 for grayscale IR,
            3 if the caller pre-replicates to RGB).
        num_classes: number of output classes. Default 4 matches the
            FovalNet preprocessed mask convention.
        base_channels: width multiplier for the encoder's first
            stage. RITnet uses 32; smaller datasets benefit from 16
            to control overfitting on the 1.5k labelled subset.
        growth_rate: dense-block growth rate (channels per layer).

    Forward returns ``(B, num_classes, H, W)`` logits at the input
    resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        growth_rate: int = 16,
    ) -> None:
        super().__init__()

        c0 = base_channels
        c1 = c0 + 4 * growth_rate
        c2 = c1 + 4 * growth_rate
        c3 = c2 + 4 * growth_rate

        # ── Encoder ────────────────────────────────────────────────
        self.stem = _conv_bn_relu(in_channels, c0, k=3)
        self.enc1 = _DenseBlock(c0, growth_rate)            # → c1
        self.down1 = nn.MaxPool2d(2)                        # /2
        self.enc2 = _DenseBlock(c1, growth_rate)            # → c2
        self.down2 = nn.MaxPool2d(2)                        # /4
        self.enc3 = _DenseBlock(c2, growth_rate)            # → c3
        self.down3 = nn.MaxPool2d(2)                        # /8

        # ── Bottleneck ─────────────────────────────────────────────
        self.bottleneck = _DenseBlock(c3, growth_rate)      # → c3 + 4g

        # ── Decoder (transposed conv upsampling, RITnet style) ─────
        c4 = c3 + 4 * growth_rate
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _conv_bn_relu(c3 + c3, c3)              # skip from enc3
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_bn_relu(c2 + c2, c2)              # skip from enc2
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_bn_relu(c1 + c1, c1)              # skip from enc1

        # ── Classifier ─────────────────────────────────────────────
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)               # (B, c0, H, W)
        x1 = self.enc1(x0)              # (B, c1, H, W)
        x2 = self.enc2(self.down1(x1))  # (B, c2, H/2, W/2)
        x3 = self.enc3(self.down2(x2))  # (B, c3, H/4, W/4)
        b = self.bottleneck(self.down3(x3))  # (B, c4, H/8, W/8)

        u3 = self.up3(b)
        u3 = self.dec3(torch.cat([u3, x3], dim=1))
        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, x1], dim=1))
        return self.classifier(u1)


def _soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int,
    eps: float = 1.0,
) -> torch.Tensor:
    """Mean soft-Dice loss across non-background classes.

    Background (class 0) is excluded from the Dice score because in
    OpenEDS roughly 80% of pixels are background; including it would
    dominate the gradient and starve pupil/iris.
    """
    probs = logits.softmax(dim=1)                         # (B, K, H, W)
    B, K, H, W = probs.shape
    target_one_hot = F.one_hot(target.clamp(0, num_classes - 1),
                               num_classes=num_classes)    # (B, H, W, K)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)  # average over batch + spatial → per-class score
    intersection = (probs * target_one_hot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    # Drop background, average over remaining K-1 classes.
    return 1.0 - dice_per_class[1:].mean()


def build_ritnet_full(num_classes: int = 4, in_channels: int = 1
                      ) -> 'RITnetStyleSegmenter':
    """Higher-capacity variant (~2.3M params) — default v6 segmenter."""
    return RITnetStyleSegmenter(in_channels=in_channels,
                                num_classes=num_classes,
                                base_channels=32, growth_rate=16)


def build_ritnet_tiny(num_classes: int = 4, in_channels: int = 1
                      ) -> 'RITnetStyleSegmenter':
    """Tiny variant (~0.5M params) — faithful to the RITnet 250k-budget
    spirit and recommended when the ~1.5k labelled OpenEDS frames
    overfit the full variant. Use ``in_channels=2`` to also accept a
    geometric prior channel from the GazeGene Macro-Locator."""
    return RITnetStyleSegmenter(in_channels=in_channels,
                                num_classes=num_classes,
                                base_channels=16, growth_rate=8)


# ── Geometric prior channel ─────────────────────────────────────────

def make_geometric_prior_channel(
    eyeball_center_2d_px: torch.Tensor,
    img_size: tuple[int, int],
    sigma_px: float = 60.0,
) -> torch.Tensor:
    """Build a 2D Gaussian prior channel centred at the projected 3D
    eyeball centre.

    Used by the v6.2 geometric bridge: the GazeGene Macro-Locator
    predicts the 3D eyeball centre, projects it through ``K`` to 2D,
    and this helper turns those pixel coordinates into a soft ROI
    that seeds the OpenEDS segmenter (concatenated as channel 2 of a
    2-channel input).

    Args:
        eyeball_center_2d_px: ``(B, 2)`` predicted eyeball centre in
            pixel coordinates of the eye-patch frame.
        img_size: ``(H, W)`` of the segmenter input.
        sigma_px: Gaussian radius in pixels. Default 60 covers a
            generous iris-region ROI without hard-cropping the sclera.

    Returns:
        ``(B, 1, H, W)`` float tensor in [0, 1].
    """
    H, W = img_size
    device = eyeball_center_2d_px.device
    ys = torch.arange(H, device=device, dtype=eyeball_center_2d_px.dtype)
    xs = torch.arange(W, device=device, dtype=eyeball_center_2d_px.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    cx = eyeball_center_2d_px[:, 0:1].unsqueeze(-1)        # (B, 1, 1)
    cy = eyeball_center_2d_px[:, 1:2].unsqueeze(-1)        # (B, 1, 1)
    sq = (grid_x[None] - cx) ** 2 + (grid_y[None] - cy) ** 2
    prior = torch.exp(-sq / (2.0 * sigma_px ** 2))
    return prior.unsqueeze(1)                              # (B, 1, H, W)


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    dice_weight: float = 1.0,
    num_classes: int = 4,
) -> torch.Tensor:
    """Cross-entropy + soft-Dice for OpenEDS 4-class segmentation.

    OpenEDS class frequencies (per FovalNet stats): bg ≈ 80%,
    sclera ≈ 12%, iris ≈ 6%, pupil ≈ 2%. Without class weights
    cross-entropy underfits pupil, the smallest yet most clinically
    important class. Default ``class_weights`` of None falls back to
    uniform; pass a (4,) tensor like ``[0.5, 1.0, 2.0, 4.0]`` to
    invert the frequency.
    """
    ce = F.cross_entropy(logits, target, weight=class_weights)
    if dice_weight <= 0:
        return ce
    dice = _soft_dice_loss(logits, target, num_classes)
    return ce + dice_weight * dice


__all__ = [
    'RITnetStyleSegmenter',
    'build_ritnet_full',
    'build_ritnet_tiny',
    'make_geometric_prior_channel',
    'combined_loss',
]
