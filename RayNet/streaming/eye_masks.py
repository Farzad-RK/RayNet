"""
Anatomical Eye Region Isolation (AERI) mask generation.

Given GazeGene per-sample ground truth + per-subject anatomical
attributes, render two binary masks aligned with the 224x224 face crop
and downsampled to 56x56 for shard storage and segmentation-head
supervision:

  - iris_mask    : closed polygon from the 100-point iris_mesh_2D.
  - eyeball_mask : tangent-cone silhouette of the 3D eyeball sphere
                   UNION the cornea-sphere silhouette, both projected
                   through K. Theoretical (no eyelid clip) — occluded
                   sclera IS included, which is deliberate: the model
                   must learn to look through eyelid/nose occlusion.

Pupil is supervised through the 3D pupil_center_3d L1 head; at 56x56 a
2-4 mm pupil projects to ~1 px of mask which carries no useful
segmentation signal, so no separate pupil mask is rendered.

Subject-specific anatomy (from GazeGene subject_label.pkl) replaces the
old hard-coded 12 mm eyeball — the dataset provides per-subject
eyeball_radius / cornea_radius / cornea2center in centimetres, and the
cornea bulge is the reason the iris silhouette poked outside the plain
eyeball silhouette in the first debug run.

All masks are rasterized at 224 and then area-downsampled to 56, which
gives smoother edges than drawing directly at 56 (iris is ~4 px across
at 56, so a single-pixel rasterization error is a 25% IoU hit).

Coordinate conventions:
  - iris_mesh_2D as shipped in GazeGene is in the native 448x448 crop
    frame. Callers pass `native_size` so masks come out in the
    requested rendering frame.
  - K is the intrinsic matrix already rescaled to `face_size` (see
    dataset.py line 272-275).
  - eyeball_center_3d and pupil_center_3d are in CCS (camera
    coordinate system), centimetres, per the dataset README:
    "any physically meaningful labels in the dataset are measured in
    centimeters".
"""

from __future__ import annotations

import numpy as np
import cv2


# Fallback radius if subject_label.pkl is missing (e.g. streaming shard
# that doesn't carry attrs). Typical adult eyeball ≈ 1.2 cm.
DEFAULT_EYEBALL_RADIUS_CM = 1.2
DEFAULT_CORNEA_RADIUS_CM = 0.8
DEFAULT_CORNEA_OFFSET_CM = 0.55


# ---------------------------------------------------------------------
# Iris polygon mask
# ---------------------------------------------------------------------

def render_iris_mask(iris_mesh_2d, face_size=224, native_size=448,
                     out_size=56):
    """
    Rasterize the 100-point iris contour as a filled polygon.

    Args:
        iris_mesh_2d : (100, 2) iris contour points in `native_size` px.
        face_size    : pixel resolution of the face crop the model sees.
        native_size  : pixel resolution the landmarks were labelled in.
        out_size     : final mask resolution (typically 56).

    Returns:
        (out_size, out_size) uint8 mask in {0, 255}.
    """
    pts = np.asarray(iris_mesh_2d, dtype=np.float64) * (face_size / native_size)
    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return _downsample(mask, out_size)


# ---------------------------------------------------------------------
# Eyeball silhouette mask (two-sphere: eyeball ∪ cornea)
# ---------------------------------------------------------------------

def render_eyeball_mask(eyeball_center_3d, K,
                        pupil_center_3d=None,
                        eyeball_radius_cm=DEFAULT_EYEBALL_RADIUS_CM,
                        cornea_radius_cm=DEFAULT_CORNEA_RADIUS_CM,
                        cornea_offset_cm=DEFAULT_CORNEA_OFFSET_CM,
                        face_size=224, out_size=56,
                        n_samples=96):
    """
    Project the eyeball + cornea silhouettes and return their UNION.

    Anatomy:
        The human eyeball isn't a single sphere. A ~12 mm main sphere
        is centred at `eyeball_center_3d`; a smaller ~8 mm cornea
        sphere sits forward along the optical axis, offset by
        `cornea2center` (~5.5 mm) from the eyeball center. The iris
        contour lives on the cornea surface, which is why a single-
        sphere silhouette with radius 12 mm clipped the iris at extreme
        angles. The two-sphere union matches the true silhouette.

    The optical axis direction is derived from
        axis_hat = normalize(pupil_center_3d - eyeball_center_3d)
    which is the same geometric definition used by GeometricGazeHead.
    If `pupil_center_3d` is None, the cornea sphere is skipped and this
    reduces to a plain single-sphere silhouette.

    Args:
        eyeball_center_3d : (3,) CCS, centimetres.
        K                 : (3, 3) intrinsic matrix rescaled to
                             `face_size` pixel space.
        pupil_center_3d   : (3,) CCS, centimetres. Used to orient the
                             cornea offset direction. Pass None to
                             disable cornea rendering.
        eyeball_radius_cm : subject-specific eyeball radius.
        cornea_radius_cm  : subject-specific cornea radius.
        cornea_offset_cm  : distance from eyeball center to cornea
                             center along the optical axis.
        face_size         : render resolution before downsampling.
        out_size          : final mask resolution.
        n_samples         : silhouette-circle sample count per sphere.

    Returns:
        (out_size, out_size) uint8 mask in {0, 255}.
    """
    eyeball_center_3d = np.asarray(eyeball_center_3d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    _draw_sphere_silhouette(
        mask, eyeball_center_3d, eyeball_radius_cm, K, n_samples)

    if pupil_center_3d is not None and cornea_radius_cm > 0:
        pupil_center_3d = np.asarray(pupil_center_3d, dtype=np.float64)
        axis = pupil_center_3d - eyeball_center_3d
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-6:
            axis_hat = axis / axis_norm
            cornea_center = eyeball_center_3d + cornea_offset_cm * axis_hat
            _draw_sphere_silhouette(
                mask, cornea_center, cornea_radius_cm, K, n_samples)

    return _downsample(mask, out_size)


# ---------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------

def render_all_masks(iris_mesh_2d, eyeball_center_3d, pupil_center_3d, K,
                     subject_attrs=None,
                     face_size=224, native_size=448, out_size=56):
    """
    Render (iris, eyeball) for one sample.

    If `subject_attrs` is a dict with keys 'eyeball_radius',
    'cornea_radius', 'cornea2center' (as per GazeGene subject_label.pkl),
    those override the defaults. Missing keys fall back to the
    DEFAULT_* constants defined at module top.

    Returns:
        (iris_mask, eyeball_mask) as uint8 {0, 255} at out_size.
    """
    a = subject_attrs or {}
    iris = render_iris_mask(
        iris_mesh_2d,
        face_size=face_size, native_size=native_size, out_size=out_size)
    eyeball = render_eyeball_mask(
        eyeball_center_3d, K,
        pupil_center_3d=pupil_center_3d,
        eyeball_radius_cm=float(a.get('eyeball_radius',
                                      DEFAULT_EYEBALL_RADIUS_CM)),
        cornea_radius_cm=float(a.get('cornea_radius',
                                     DEFAULT_CORNEA_RADIUS_CM)),
        cornea_offset_cm=float(a.get('cornea2center',
                                     DEFAULT_CORNEA_OFFSET_CM)),
        face_size=face_size, out_size=out_size)
    return iris, eyeball


# ---------------------------------------------------------------------
# Containment / consistency metrics for the debug script
# ---------------------------------------------------------------------

def extract_subject_attrs(attr_entry, eye_idx):
    """Pull per-eye (eyeball_radius, cornea_radius, cornea2center) from
    a GazeGene subject_label.pkl entry.

    Each attribute may be scalar, (L, R) pair, or missing. Values land
    in the dict shape expected by `render_all_masks` (same keys, all in
    centimetres). Missing keys are simply absent — callers fall back to
    the DEFAULT_* constants.
    """
    if not attr_entry:
        return {}
    out = {}
    for key in ('eyeball_radius', 'cornea_radius', 'cornea2center'):
        v = attr_entry.get(key)
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size >= 2:
            out[key] = float(arr[eye_idx])
        elif arr.size == 1:
            out[key] = float(arr[0])
    return out


def mask_stats(iris, eyeball):
    """Per-sample statistics on an (iris, eyeball) pair."""
    iris_b = iris > 0
    eyeball_b = eyeball > 0
    n = iris.size
    iris_n = int(iris_b.sum())
    eyeball_n = int(eyeball_b.sum())
    return {
        'iris_area_frac': iris_n / n,
        'eyeball_area_frac': eyeball_n / n,
        'iris_in_eyeball_frac': (
            float((iris_b & eyeball_b).sum()) / max(iris_n, 1)),
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _draw_sphere_silhouette(mask, center_3d, radius_cm, K, n_samples):
    """In-place: fill the silhouette polygon of one sphere onto `mask`.

    Geometry:
        Tangent cone from camera origin to a sphere of radius r at
        center C (|C| = d) has half-angle α with sin α = r/d. The
        tangent points lie on a circle perpendicular to C at distance
        (d^2 - r^2)/d, with radius r * sqrt(1 - (r/d)^2). Project that
        circle through K, fill the polygon.
    """
    d = float(np.linalg.norm(center_3d))
    if d <= radius_cm:
        return
    ratio_sq = (radius_cm / d) ** 2
    c_sil = center_3d * (1.0 - ratio_sq)
    r_sil = radius_cm * float(np.sqrt(1.0 - ratio_sq))

    c_hat = center_3d / d
    helper = (np.array([1.0, 0.0, 0.0]) if abs(c_hat[0]) < 0.9
              else np.array([0.0, 1.0, 0.0]))
    u = np.cross(c_hat, helper)
    u /= np.linalg.norm(u)
    v = np.cross(c_hat, u)

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    circle_3d = (c_sil[None, :]
                 + r_sil * (np.cos(theta)[:, None] * u[None, :]
                            + np.sin(theta)[:, None] * v[None, :]))
    proj = circle_3d @ K.T
    pts_2d = proj[:, :2] / proj[:, 2:3]
    cv2.fillPoly(mask, [np.round(pts_2d).astype(np.int32)], 255)


def _downsample(mask224, out_size):
    """Area-interp downsample then rebinarize — smoother than drawing
    at out_size directly (a 1-px rasterization error at 56 is a 25%
    IoU hit on a 4-pixel iris)."""
    if out_size == mask224.shape[0]:
        return mask224
    small = cv2.resize(mask224, (out_size, out_size),
                       interpolation=cv2.INTER_AREA)
    return (small >= 128).astype(np.uint8) * 255
