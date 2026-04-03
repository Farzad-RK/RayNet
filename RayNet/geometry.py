"""
Geometric post-processing for RayNet.

Computes metric pupil diameter and depth from iris/pupil landmarks
using projective geometry. No learning required -- purely geometric.

The iris acts as a known-size ruler in the scene:
  - Population mean iris radius: 5.9mm (Goldsmith et al.)
  - GazeGene provides per-subject iris_radius when available
"""

import numpy as np
from numpy.linalg import svd


def fit_ellipse_algebraic(points):
    """
    Fit an ellipse to 2D points using Fitzgibbon's direct algebraic method.
    Guaranteed to return an ellipse (not hyperbola/parabola).

    Args:
        points: (N, 2) array of 2D points, N >= 5

    Returns:
        coeffs: (6,) array [A, B, C, D, E, F] for Ax^2+Bxy+Cy^2+Dx+Ey+F=0
    """
    x = points[:, 0]
    y = points[:, 1]

    # Design matrix
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])

    # Scatter matrix
    S = D.T @ D

    # Constraint matrix for ellipse (B^2 - 4AC < 0)
    C1 = np.zeros((6, 6))
    C1[0, 2] = 2
    C1[2, 0] = 2
    C1[1, 1] = -1

    # Solve generalized eigenvalue problem
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C1)
        # Find the positive eigenvalue
        idx = np.argmax(np.real(eigvals) > 0)
        coeffs = np.real(eigvecs[:, idx])
    except np.linalg.LinAlgError:
        # Fallback: SVD-based fit (not guaranteed ellipse)
        _, _, Vt = svd(D)
        coeffs = Vt[-1]

    return coeffs


def ellipse_to_axes(coeffs):
    """
    Convert algebraic ellipse coefficients to geometric parameters.

    Args:
        coeffs: (6,) [A, B, C, D, E, F]

    Returns:
        center: (2,) ellipse center (cx, cy)
        semi_a: semi-major axis length
        semi_b: semi-minor axis length
    """
    A, B, C, D, E, F = coeffs

    # Center
    denom = 4 * A * C - B**2
    if abs(denom) < 1e-10:
        # Degenerate case
        return np.array([0.0, 0.0]), 1.0, 1.0

    cx = (B * E - 2 * C * D) / denom
    cy = (B * D - 2 * A * E) / denom

    # Semi-axes
    M = np.array([[A, B/2], [B/2, C]])
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.abs(eigvals)

    num = A * cx**2 + B * cx * cy + C * cy**2 - F
    if num < 0:
        num = abs(num)

    if eigvals[0] < 1e-10 or eigvals[1] < 1e-10:
        return np.array([cx, cy]), 1.0, 1.0

    semi_a = np.sqrt(num / eigvals[0])
    semi_b = np.sqrt(num / eigvals[1])

    # Ensure a >= b
    if semi_a < semi_b:
        semi_a, semi_b = semi_b, semi_a

    return np.array([cx, cy]), semi_a, semi_b


def metric_pupil_diameter(iris_pts_2d, pupil_pts_2d, K, iris_radius_mm=5.9):
    """
    Compute metric pupil diameter from 2D landmarks and camera intrinsics.

    Uses the iris as a known-size ruler to estimate depth, then computes
    pupil size in mm from the apparent size ratio.

    Args:
        iris_pts_2d: (N_iris, 2) iris landmark pixel coordinates (N >= 5)
        pupil_pts_2d: (N_pupil, 2) pupil landmark pixel coordinates (N >= 4)
        K: (3, 3) camera intrinsic matrix
        iris_radius_mm: known iris radius in mm (default 5.9, population mean)

    Returns:
        Z_mm: estimated depth of iris plane from camera (mm)
        pupil_diam_mm: metric pupil diameter (mm)
    """
    # Fit ellipses
    iris_coeffs = fit_ellipse_algebraic(iris_pts_2d)
    pupil_coeffs = fit_ellipse_algebraic(pupil_pts_2d)

    _, a_iris, b_iris = ellipse_to_axes(iris_coeffs)
    _, a_pupil, b_pupil = ellipse_to_axes(pupil_coeffs)

    # Focal length
    f = K[0, 0]

    # Apparent iris radius (geometric mean of semi-axes)
    apparent_iris_r = np.sqrt(a_iris * b_iris)

    # Depth from known iris size: Z = f * R_iris / apparent_radius
    Z_mm = (f * iris_radius_mm) / apparent_iris_r

    # Metric pupil radius from apparent size and depth
    apparent_pupil_r = np.sqrt(a_pupil * b_pupil)
    pupil_diam_mm = (2 * apparent_pupil_r * Z_mm) / f

    return Z_mm, pupil_diam_mm


def gaze_to_screen_point(gaze_origin, gaze_direction, screen_normal, screen_point, screen_axes):
    """
    Intersect a 3D gaze ray with the screen plane to get screen coordinates.

    Args:
        gaze_origin: (3,) eye position in camera coords (mm)
        gaze_direction: (3,) gaze direction unit vector
        screen_normal: (3,) screen plane normal (pointing toward camera)
        screen_point: (3,) a point on the screen plane (e.g., center)
        screen_axes: (2, 3) screen horizontal and vertical unit vectors

    Returns:
        screen_uv: (2,) screen coordinates (u, v) in mm from screen origin
        hit_point: (3,) intersection point in camera coords
    """
    # Ray-plane intersection: t = dot(screen_point - origin, normal) / dot(direction, normal)
    denom = np.dot(gaze_direction, screen_normal)
    if abs(denom) < 1e-8:
        return None, None  # Ray parallel to screen

    t = np.dot(screen_point - gaze_origin, screen_normal) / denom
    if t < 0:
        return None, None  # Intersection behind the eye

    hit_point = gaze_origin + t * gaze_direction

    # Project onto screen axes
    offset = hit_point - screen_point
    u = np.dot(offset, screen_axes[0])
    v = np.dot(offset, screen_axes[1])

    return np.array([u, v]), hit_point
