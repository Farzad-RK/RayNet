"""
Anatomical Eye Region Isolation (AERI) mask generation.

Given GazeGene per-sample ground truth, render three binary masks
aligned with the 224x224 face crop, downsampled to 56x56 for shard
storage and segmentation-head supervision:

  - iris_mask    : closed polygon from the 100-point iris_mesh_2D.
  - pupil_mask   : disk at pupil_center_2D with radius inferred from
                   the 4 iris landmarks closest to the pupil center.
  - eyeball_mask : tangent-cone silhouette of the 3D eyeball sphere
                   projected through K. Theoretical (no eyelid clip),
                   so occluded sclera IS included — this is deliberate
                   (the model must learn to look *through* occlusion).

All masks are rasterized at the full 224 pixel grid and then area-
downsampled to 56, which gives smoother edges than drawing directly at
56 (iris is ~4 px across at 56, so a single-pixel rasterization error
is a 25% IoU hit).

Coordinate conventions:
  - iris_mesh_2D and pupil_center_2D as shipped in GazeGene are in the
    native 448x448 crop frame. Callers pass `native_size` so masks come
    out in the requested rendering frame.
  - K is the intrinsic matrix already rescaled to `face_size` (224 by
    default in this repo — see dataset.py line 272-275).
  - eyeball_center_3d is in CCS (camera coordinate system), centimeters,
    which is the GazeGene convention.

Anatomical constants:
  - EYEBALL_RADIUS_CM = 1.2  (adult eyeball radius ≈ 12 mm)
"""

from __future__ import annotations

import numpy as np
import cv2


EYEBALL_RADIUS_CM = 1.2


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
# Pupil disk mask
# ---------------------------------------------------------------------

def render_pupil_mask(pupil_center_2d, iris_mesh_2d,
                      face_size=224, native_size=448, out_size=56):
    """
    Rasterize a pupil disk centred on `pupil_center_2d`.

    Radius is estimated as the mean distance from the pupil center to
    its 4 nearest iris-mesh points — matching the 4-pupil-boundary
    selection already done by dataset.py so the disk boundary is
    consistent with the 4 pupil landmarks the model also learns.

    Args:
        pupil_center_2d : (2,)   pupil center in `native_size` px.
        iris_mesh_2d    : (100, 2) iris contour points in `native_size` px
                           (used to infer the pupil radius).
        face_size       : see render_iris_mask.
        native_size     : see render_iris_mask.
        out_size        : see render_iris_mask.

    Returns:
        (out_size, out_size) uint8 mask in {0, 255}.
    """
    scale = face_size / native_size
    c_native = np.asarray(pupil_center_2d, dtype=np.float64)
    iris_native = np.asarray(iris_mesh_2d, dtype=np.float64)
    dists = np.linalg.norm(iris_native - c_native[None, :], axis=1)
    r_native = float(np.mean(np.sort(dists)[:4]))

    c_px = c_native * scale
    r_px = r_native * scale

    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.circle(mask, (int(round(c_px[0])), int(round(c_px[1]))),
               int(round(r_px)), 255, thickness=-1)
    return _downsample(mask, out_size)


# ---------------------------------------------------------------------
# Eyeball sphere silhouette mask
# ---------------------------------------------------------------------

def render_eyeball_mask(eyeball_center_3d, K,
                        face_size=224, out_size=56,
                        radius_cm=EYEBALL_RADIUS_CM,
                        n_samples=96):
    """
    Project the 3D eyeball sphere's silhouette circle through K.

    Geometry (view from camera origin looking down +Z):
      Let C be the sphere center in CCS, d = |C|, r = radius. The set of
      points on the sphere that are tangent to rays from the origin form
      a circle (the "limb" of the sphere as seen from that viewpoint).
      Call it the silhouette circle; it lies in the plane perpendicular
      to C, centered at

          C_sil = C * (d^2 - r^2) / d^2

      with radius

          r_sil = r * sqrt(1 - (r/d)^2).

      Projecting this circle through K yields (in general) an ellipse in
      the image plane. We sample the circle at `n_samples` angles and
      rasterize the resulting projected polygon with cv2.fillPoly, which
      is exact to ≈1 pixel for n_samples ≥ 48.

    Args:
        eyeball_center_3d : (3,) CCS center in centimetres.
        K                 : (3, 3) intrinsic matrix already rescaled to
                             `face_size` pixel space.
        face_size         : render resolution before downsampling.
        out_size          : final mask resolution.
        radius_cm         : eyeball radius (default 1.2 cm).
        n_samples         : silhouette-circle sample count.

    Returns:
        (out_size, out_size) uint8 mask in {0, 255}. Returns an all-zero
        mask if the sphere contains the camera (d <= r, degenerate).
    """
    C = np.asarray(eyeball_center_3d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    d = float(np.linalg.norm(C))
    if d <= radius_cm:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    # Silhouette circle in 3D
    ratio_sq = (radius_cm / d) ** 2
    C_sil = C * (1.0 - ratio_sq)
    r_sil = radius_cm * float(np.sqrt(1.0 - ratio_sq))

    # Orthonormal basis for the silhouette plane (perpendicular to C_hat)
    C_hat = C / d
    helper = (np.array([1.0, 0.0, 0.0]) if abs(C_hat[0]) < 0.9
              else np.array([0.0, 1.0, 0.0]))
    u = np.cross(C_hat, helper)
    u /= np.linalg.norm(u)
    v = np.cross(C_hat, u)

    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    circle_3d = (C_sil[None, :]
                 + r_sil * (np.cos(theta)[:, None] * u[None, :]
                            + np.sin(theta)[:, None] * v[None, :]))

    # Project each point: K @ P_ccs, then divide by z.
    proj = circle_3d @ K.T        # (n_samples, 3)
    pts_2d = proj[:, :2] / proj[:, 2:3]

    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts_2d).astype(np.int32)], 255)
    return _downsample(mask, out_size)


# ---------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------

def render_all_masks(iris_mesh_2d, pupil_center_2d, eyeball_center_3d, K,
                     face_size=224, native_size=448, out_size=56):
    """Convenience wrapper — returns (iris, pupil, eyeball) at out_size."""
    iris = render_iris_mask(iris_mesh_2d,
                            face_size=face_size,
                            native_size=native_size,
                            out_size=out_size)
    pupil = render_pupil_mask(pupil_center_2d, iris_mesh_2d,
                              face_size=face_size,
                              native_size=native_size,
                              out_size=out_size)
    eyeball = render_eyeball_mask(eyeball_center_3d, K,
                                  face_size=face_size,
                                  out_size=out_size)
    return iris, pupil, eyeball


# ---------------------------------------------------------------------
# Containment / consistency metrics for the debug script
# ---------------------------------------------------------------------

def mask_stats(iris, pupil, eyeball):
    """
    Per-sample statistics on a (iris, pupil, eyeball) triple.

    Returns a dict with:
      - iris_area_frac, pupil_area_frac, eyeball_area_frac :
            fraction of mask pixels that are foreground.
      - pupil_in_iris_frac    : fraction of pupil pixels inside iris.
      - iris_in_eyeball_frac  : fraction of iris pixels inside eyeball.
      - pupil_over_iris_ratio : pupil_area / iris_area (anatomical
            expected range 0.05 – 0.30).
    """
    iris_b = iris > 0
    pupil_b = pupil > 0
    eyeball_b = eyeball > 0
    n = iris.size

    iris_n = int(iris_b.sum())
    pupil_n = int(pupil_b.sum())
    eyeball_n = int(eyeball_b.sum())

    return {
        'iris_area_frac': iris_n / n,
        'pupil_area_frac': pupil_n / n,
        'eyeball_area_frac': eyeball_n / n,
        'pupil_in_iris_frac': (
            float((pupil_b & iris_b).sum()) / max(pupil_n, 1)),
        'iris_in_eyeball_frac': (
            float((iris_b & eyeball_b).sum()) / max(iris_n, 1)),
        'pupil_over_iris_ratio': pupil_n / max(iris_n, 1),
    }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _downsample(mask224, out_size):
    """Area-interp downsample then rebinarize — smoother than drawing
    at out_size directly (a 1-px rasterization error at 56 is a 25%
    IoU hit on a 4-pixel iris)."""
    if out_size == mask224.shape[0]:
        return mask224
    small = cv2.resize(mask224, (out_size, out_size),
                       interpolation=cv2.INTER_AREA)
    return (small >= 128).astype(np.uint8) * 255
