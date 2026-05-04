"""
Classical 3DeepVOG-style cyclotorsion estimator.

The pipeline (no learned parameters initially):

    iris/pupil masks  →  ellipse fit  →  iris-annulus polar warp
                      →  N mini-patches  →  NCC against rolling reference
                      →  weighted-median Δφ  →  torsion angle (degrees)

Torsion is intrinsically a *relative* measurement (how far the iris
texture has rotated about the visual axis since some reference). The
estimator maintains a rolling reference template per session: the
first ``ref_frames`` non-blink frames seed the reference, and any
subsequent frame's torsion is measured relative to that reference.

Why patches and not full polar correlation: a single polar phase
correlation collapses under partial eyelid occlusion (the upper
eyelid removes ~20° of arc at most blink phases). Splitting the
annulus into N independent patches lets us robustly aggregate via
weighted median while throwing away occluded patches.

This module is a pure function of the segmentation output and previous
frames; it adds no trainable parameters by default. A learned variant
(``IrisPolarTorsion(learn_features=True)``) replaces the raw patch
pixels with a small CNN feature map for higher-precision NCC, but
that wrapper is left as a stub for the first integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class TorsionEstimate:
    """One frame's torsion estimate."""
    angle_deg: float        # cyclotorsion in degrees (positive = CCW)
    confidence: float       # mean of accepted patch correlation peaks
    n_patches_used: int     # accepted patches (out of N total)


def _fit_ellipse_from_mask(mask: np.ndarray) -> Optional[tuple]:
    """Fit a 5-DOF ellipse to a binary mask via OpenCV's least squares.

    Returns ``(cx, cy, a, b, theta_deg)`` or ``None`` if the mask is
    too small to fit an ellipse (< 5 contour points).
    """
    import cv2
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    (cx, cy), (a, b), theta = cv2.fitEllipse(contour)
    return float(cx), float(cy), float(a) / 2.0, float(b) / 2.0, float(theta)


def _polar_unwrap(
    image: np.ndarray, cx: float, cy: float,
    r_inner: float, r_outer: float,
    n_phi: int = 360, n_r: int = 64,
) -> np.ndarray:
    """Sample the iris annulus into a (n_r, n_phi) polar map.

    Bilinear sampling via ``cv2.remap``. φ wraps to [0, 360) so the
    output is periodic along axis 1.
    """
    import cv2
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    rs = np.linspace(r_inner, r_outer, n_r)
    rr, pp = np.meshgrid(rs, phis, indexing='ij')           # (n_r, n_phi)
    map_x = (cx + rr * np.cos(pp)).astype(np.float32)
    map_y = (cy + rr * np.sin(pp)).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _ncc_shift(
    template: np.ndarray, frame: np.ndarray, max_shift_px: int,
) -> tuple[int, float]:
    """Best integer φ-shift that maximises NCC between two patches.

    Both patches are ``(n_r, patch_w)`` slices of polar maps. Shift
    is along axis 1 (φ). Returns ``(shift_px, peak_ncc)``.
    """
    n_r, w = template.shape
    if w != frame.shape[1] or n_r != frame.shape[0]:
        raise ValueError(
            f'NCC shape mismatch: template={template.shape} frame={frame.shape}')

    # Subtract per-patch mean so NCC is centred.
    tpl = template - template.mean()
    frm = frame - frame.mean()
    tpl_norm = np.sqrt((tpl * tpl).sum()) + 1e-8

    best_shift = 0
    best_score = -np.inf
    for s in range(-max_shift_px, max_shift_px + 1):
        rolled = np.roll(frm, shift=s, axis=1)
        rolled_norm = np.sqrt((rolled * rolled).sum()) + 1e-8
        score = float((tpl * rolled).sum() / (tpl_norm * rolled_norm))
        if score > best_score:
            best_score = score
            best_shift = s
    return best_shift, best_score


class IrisPolarTorsion:
    """Stateful per-session cyclotorsion estimator.

    Usage::

        torsion_est = IrisPolarTorsion()
        for frame_idx in range(num_frames):
            img = ...                # (H, W) uint8 grayscale
            iris_mask = ...          # (H, W) bool — class 2 from segmenter
            pupil_mask = ...         # (H, W) bool — class 3
            est = torsion_est.update(img, iris_mask, pupil_mask)
            print(est.angle_deg, est.confidence)

    Args:
        n_patches: number of mini-patches sampled around the annulus.
        patch_arc_deg: angular width of each patch in degrees.
        ref_frames: number of consecutive non-blink frames used to
            seed the rolling reference. After warmup the reference is
            held fixed for the rest of the session.
        max_shift_deg: search radius around 0° torsion for NCC peak.
        n_phi: angular resolution of the polar unwrap (pixels).
        n_r: radial resolution of the polar unwrap.
        ncc_accept_threshold: minimum per-patch NCC peak below which
            the patch is treated as occluded and dropped from the
            weighted-median aggregation.
    """

    def __init__(
        self,
        n_patches: int = 8,
        patch_arc_deg: float = 30.0,
        ref_frames: int = 30,
        max_shift_deg: float = 15.0,
        n_phi: int = 360,
        n_r: int = 64,
        ncc_accept_threshold: float = 0.4,
    ) -> None:
        self.n_patches = int(n_patches)
        self.patch_arc_deg = float(patch_arc_deg)
        self.ref_frames = int(ref_frames)
        self.max_shift_deg = float(max_shift_deg)
        self.n_phi = int(n_phi)
        self.n_r = int(n_r)
        self.ncc_accept_threshold = float(ncc_accept_threshold)

        # Reference state — populated after the first ref_frames calls
        # to update().
        self._reference_patches: Optional[List[np.ndarray]] = None
        self._reference_buffer: List[np.ndarray] = []   # accumulator during warmup
        self._patch_centers_deg: np.ndarray = np.linspace(
            0.0, 360.0, self.n_patches, endpoint=False)
        self._patch_arc_px = max(2, int(round(
            self.patch_arc_deg / 360.0 * self.n_phi)))
        self._max_shift_px = max(1, int(round(
            self.max_shift_deg / 360.0 * self.n_phi)))

    # --- helpers ----------------------------------------------------

    def _slice_patches(self, polar: np.ndarray) -> List[np.ndarray]:
        """Split a (n_r, n_phi) polar map into N equal patches with
        circular wrap. Returns N (n_r, patch_arc_px) arrays."""
        patches = []
        half = self._patch_arc_px // 2
        n_phi = polar.shape[1]
        for centre_deg in self._patch_centers_deg:
            centre_px = int(round(centre_deg / 360.0 * n_phi)) % n_phi
            # Use np.roll to handle wraparound, then take the central slice.
            rolled = np.roll(polar, shift=-centre_px + half, axis=1)
            patches.append(rolled[:, :self._patch_arc_px].copy())
        return patches

    def _polar_from_masks(
        self, image: np.ndarray, iris_mask: np.ndarray,
        pupil_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        iris_fit = _fit_ellipse_from_mask(iris_mask)
        pupil_fit = _fit_ellipse_from_mask(pupil_mask)
        if iris_fit is None or pupil_fit is None:
            return None
        iris_cx, iris_cy, iris_a, iris_b, _ = iris_fit
        _, _, pupil_a, pupil_b, _ = pupil_fit
        # Use the iris centre for both inner and outer radii — pupil
        # centre is slightly offset from iris centre (~0.5 mm) but
        # for the ANNULUS sampling we want concentric circles.
        r_inner = 0.5 * (pupil_a + pupil_b) * 1.05   # 5% margin off pupil edge
        r_outer = 0.5 * (iris_a + iris_b) * 0.95     # 5% margin inside limbus
        if r_outer <= r_inner + 2:
            return None
        return _polar_unwrap(image, iris_cx, iris_cy, r_inner, r_outer,
                             n_phi=self.n_phi, n_r=self.n_r)

    # --- public API -------------------------------------------------

    def update(
        self,
        image: np.ndarray,
        iris_mask: np.ndarray,
        pupil_mask: np.ndarray,
    ) -> Optional[TorsionEstimate]:
        """Process one frame; return torsion estimate or ``None``.

        Returns ``None`` when the masks are too small for ellipse
        fitting (typical of blink frames) or while the reference
        template is still being accumulated.
        """
        polar = self._polar_from_masks(image, iris_mask, pupil_mask)
        if polar is None:
            return None

        if self._reference_patches is None:
            # Still in warmup — accumulate average polar map.
            self._reference_buffer.append(polar)
            if len(self._reference_buffer) >= self.ref_frames:
                avg = np.mean(np.stack(self._reference_buffer), axis=0)
                self._reference_patches = self._slice_patches(avg)
                self._reference_buffer = []
            return None

        frame_patches = self._slice_patches(polar)
        shifts: List[float] = []
        scores: List[float] = []
        for ref, frm in zip(self._reference_patches, frame_patches):
            shift_px, peak = _ncc_shift(ref, frm, self._max_shift_px)
            if peak < self.ncc_accept_threshold:
                continue
            shifts.append(shift_px / self.n_phi * 360.0)
            scores.append(peak)

        if not shifts:
            return None

        shifts_a = np.asarray(shifts)
        scores_a = np.asarray(scores)
        # Weighted median (per scores) is robust to a few mis-matched
        # patches and gives a more stable estimate than the mean under
        # partial occlusion.
        order = np.argsort(shifts_a)
        sorted_shifts = shifts_a[order]
        cum = np.cumsum(scores_a[order])
        target = 0.5 * cum[-1]
        idx = int(np.searchsorted(cum, target))
        idx = min(idx, len(sorted_shifts) - 1)
        weighted_median = float(sorted_shifts[idx])
        return TorsionEstimate(
            angle_deg=weighted_median,
            confidence=float(scores_a.mean()),
            n_patches_used=len(shifts),
        )

    def reset(self) -> None:
        """Drop the rolling reference. Subsequent calls re-warm-up."""
        self._reference_patches = None
        self._reference_buffer = []


def torsion_self_supervised_pretext(
    polar_map: torch.Tensor,
    max_shift_deg: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (rotated_polar, target_shift_rad) pairs for SSL.

    Useful for pre-training a learnable variant of the torsion head:
    take a clean polar rubbersheet, roll it by a random Δφ, and ask
    the model to predict the inverse Δφ. Pure rotation pretext, no
    OpenEDS labels needed.

    Args:
        polar_map: (B, n_r, n_phi) float tensor.
        max_shift_deg: half-range of the uniform rotation distribution.

    Returns:
        rotated: (B, n_r, n_phi) — input rolled by random Δφ.
        target: (B,) — the *negative* Δφ in radians (the inverse).
    """
    B, n_r, n_phi = polar_map.shape
    max_shift_rad = float(max_shift_deg) / 180.0 * float(np.pi)
    delta = (torch.rand(B, device=polar_map.device) * 2.0 - 1.0) * max_shift_rad
    shifts_px = (delta / (2.0 * float(np.pi)) * n_phi).round().long()
    rotated = torch.stack([
        torch.roll(polar_map[i], shifts=int(shifts_px[i].item()), dims=-1)
        for i in range(B)
    ])
    return rotated, -delta


__all__ = [
    'IrisPolarTorsion',
    'TorsionEstimate',
    'torsion_self_supervised_pretext',
]
