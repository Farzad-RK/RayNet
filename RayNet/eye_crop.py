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

    Default `out_size` is 224 (matches the face-crop input size, so the
    EyeBackbone produces a 14x14 stride-16 token map dedicated to the
    eye region). The earlier 112x112 default doubled feature-map
    capacity vs. the face path but capped Stage 2 P3 val_angular at
    ~17deg; cropping at 224 quadruples token capacity over 112 with
    no upstream pixel-information change.

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
        out_size: output spatial size (default 224, full face-crop scale —
            doubles feature-map capacity vs. the previous 112 default).
        pad_frac: padding fraction beyond tight landmark bbox (default 0.30).
            Slightly wider than the 0.25 used at 112x112 to keep eyelid
            and brow context that the higher-capacity feature map can
            actually use.
        min_half_size: minimum half-side in input pixels (default 32).
            At 224 output, a 24px source half-side meant ~9.3x bilinear
            upsample, which adds interpolation noise without information.
            32 keeps the upsample under 7x.
    """

    def __init__(self, out_size=224, pad_frac=0.30, min_half_size=32.0):
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
            affine: dict with per-sample crop geometry, used by the
                    refinement landmark head to project face-frame GT
                    coordinates into eye-patch space and to project
                    refined predictions back. Keys:
                      cx, cy: (B,) crop centroid in face-frame pixels
                      half:   (B,) crop half-side in face-frame pixels
                      out_size: int, eye-patch size (square)
                      H, W:   ints, source face-frame dims
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

        affine = {
            'cx': cx, 'cy': cy, 'half': half,
            'out_size': self.out_size, 'H': H, 'W': W,
        }
        return crop, affine

    @staticmethod
    def face_to_eye_coords(coords_face, affine, eye_feat_size=None):
        """
        Project face-frame pixel coordinates into eye-patch coordinates.

        Use this to compute GT for the refinement landmark head: the
        face-frame GT is mapped through the same affine that produced
        the eye crop, into the coordinate system the refinement head
        outputs in.

        Args:
            coords_face: (B, N, 2) (x, y) in face-frame pixels.
            affine: dict from EyeCropModule.forward.
            eye_feat_size: optional output spatial size of the
                refinement decoder (e.g. 56 for a 56x56 heatmap). When
                supplied, the result is scaled from `out_size` pixels
                into `eye_feat_size` cells, matching the heatmap's
                coordinate frame so soft-argmax outputs are directly
                comparable. When None, returns coords in eye-patch
                pixel space (i.e. in 0..out_size-1).

        Returns:
            coords_eye: (B, N, 2) in eye-patch (or feature-map) space.
        """
        cx = affine['cx'][:, None]   # (B, 1) — broadcast across N landmarks
        cy = affine['cy'][:, None]
        half = affine['half'][:, None]
        out_size = affine['out_size']

        # In eye-patch pixel space (align_corners=True convention):
        #     coord_eye_px = (coord_face - (cx - half)) * (out_size - 1) / (2 * half)
        # which collapses to:
        #     coord_eye_px = (coord_face - cx) * (out_size - 1) / (2 * half) + (out_size - 1) / 2
        scale = (out_size - 1) / (2.0 * half)
        x_eye_px = (coords_face[..., 0] - cx) * scale + (out_size - 1) / 2.0
        y_eye_px = (coords_face[..., 1] - cy) * scale + (out_size - 1) / 2.0

        if eye_feat_size is not None and eye_feat_size != out_size:
            ratio = (eye_feat_size - 1) / (out_size - 1)
            x_eye_px = x_eye_px * ratio
            y_eye_px = y_eye_px * ratio

        return torch.stack([x_eye_px, y_eye_px], dim=-1)

    @staticmethod
    def eye_to_face_coords(coords_eye, affine, eye_feat_size=None):
        """
        Inverse of `face_to_eye_coords`: lift refined eye-patch
        predictions back to face-frame pixels for downstream consumers
        (pupillometry, iris contour rendering, second-pass crops).

        Args:
            coords_eye: (B, N, 2) coords in eye-patch pixel space, OR
                in `eye_feat_size`-cell space when `eye_feat_size` is
                given (in which case they are first rescaled to
                eye-patch pixel space).
            affine: dict from EyeCropModule.forward.
            eye_feat_size: see `face_to_eye_coords`.

        Returns:
            coords_face: (B, N, 2) in face-frame pixels.
        """
        cx = affine['cx'][:, None]
        cy = affine['cy'][:, None]
        half = affine['half'][:, None]
        out_size = affine['out_size']

        x = coords_eye[..., 0]
        y = coords_eye[..., 1]
        if eye_feat_size is not None and eye_feat_size != out_size:
            ratio = (out_size - 1) / (eye_feat_size - 1)
            x = x * ratio
            y = y * ratio

        scale = (2.0 * half) / (out_size - 1)
        x_face = (x - (out_size - 1) / 2.0) * scale + cx
        y_face = (y - (out_size - 1) / 2.0) * scale + cy
        return torch.stack([x_face, y_face], dim=-1)
