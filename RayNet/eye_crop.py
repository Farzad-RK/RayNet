"""
Differentiable landmark-guided eye crop.

Extracts a square eye patch from a face image using the 14 iris/pupil
landmarks as the crop anchor. Implemented with F.affine_grid +
F.grid_sample so gradients can flow back through the crop coordinates
(we always detach in practice, but the differentiable path keeps the
compute graph clean and lets future experiments re-enable landmark
gradient flow through gaze loss).

Motivation:
    Gaze at the face-crop level hits a spatial-resolution ceiling:
    a 224x224 face through stride-16 backbones gives a 14x14 feature
    map where the iris occupies 2-3 cells. Landmark-guided cropping
    zooms in on the eye so a dedicated encoder sees the iris at
    subpixel resolution.

Geometry:
    Given 14 landmarks in pixel space (all eye-region), compute a
    square bbox centered on their centroid with half-size
        s = max(x_range, y_range) / 2 * (1 + pad_frac)
    clamped to `min_half_size` pixels. Sample a square crop into an
    `out_size x out_size` output via affine_grid + grid_sample with
    `padding_mode='zeros'` so off-image regions are black rather than
    edge-replicated (MAGE-style, prevents edge pixel contamination).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeCropModule(nn.Module):
    """
    Landmark-guided differentiable eye crop.

    Args:
        out_size: output spatial size (default 112, matches MAC-Gaze)
        pad_frac: padding fraction beyond tight landmark bbox (default 0.25)
        min_half_size: minimum half-side in input pixels (default 24)
    """

    def __init__(self, out_size=112, pad_frac=0.25, min_half_size=24.0):
        super().__init__()
        self.out_size = out_size
        self.pad_frac = pad_frac
        self.min_half_size = float(min_half_size)

    def forward(self, image, landmarks_px):
        """
        Args:
            image: (B, 3, H, W) float image (already normalized to [0,1])
            landmarks_px: (B, N, 2) landmark coordinates in pixel space
                          (x, y) in the frame of `image`

        Returns:
            crop: (B, 3, out_size, out_size) eye patch
        """
        B, _, H, W = image.shape

        cx = landmarks_px[:, :, 0].mean(dim=1)                       # (B,)
        cy = landmarks_px[:, :, 1].mean(dim=1)                       # (B,)

        x_range = landmarks_px[:, :, 0].amax(dim=1) - landmarks_px[:, :, 0].amin(dim=1)
        y_range = landmarks_px[:, :, 1].amax(dim=1) - landmarks_px[:, :, 1].amin(dim=1)
        half = 0.5 * torch.maximum(x_range, y_range) * (1.0 + self.pad_frac)
        half = half.clamp(min=self.min_half_size)                    # (B,)

        # align_corners=True convention: pixel 0 → -1, pixel (W-1) → +1.
        # Output normalized pos u∈[-1,+1] should sample input at
        #     x_in_norm = theta[0,0]*u + theta[0,2]
        # so:
        #     scale_x = 2 * half / (W - 1)
        #     tx      = 2 * cx   / (W - 1) - 1
        scale_x = 2.0 * half / (W - 1)
        scale_y = 2.0 * half / (H - 1)
        tx = 2.0 * cx / (W - 1) - 1.0
        ty = 2.0 * cy / (H - 1) - 1.0

        zero = torch.zeros_like(scale_x)
        theta = torch.stack([
            torch.stack([scale_x, zero,    tx], dim=-1),
            torch.stack([zero,    scale_y, ty], dim=-1),
        ], dim=-2)  # (B, 2, 3)

        grid = F.affine_grid(
            theta,
            size=(B, image.shape[1], self.out_size, self.out_size),
            align_corners=True,
        )
        crop = F.grid_sample(
            image, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return crop
