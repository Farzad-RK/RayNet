"""
Image normalization for gaze estimation (Zhang et al. 2018).

Constructs a virtual camera that views the eye center from a canonical
distance and focal length, removing depth ambiguity from the learning task.
"""

import numpy as np
import cv2


def normalize_sample(image, K, R_head, t_eye, d_norm=600, f_norm=960, size=224):
    """
    Normalize an eye image so the eye center is at canonical depth.

    Args:
        image: (H, W, 3) BGR or RGB image
        K: (3, 3) camera intrinsic matrix
        R_head: (3, 3) head rotation matrix in camera coords
        t_eye: (3,) eye center position in camera coords (mm)
        d_norm: canonical distance in mm (default 600)
        f_norm: canonical focal length in px (default 960)
        size: output image size (default 224)

    Returns:
        img_norm: (size, size, 3) normalized image
        R_norm: (3, 3) normalization rotation matrix
    """
    # Distance from camera to eye center
    z_actual = np.linalg.norm(t_eye)

    # Scale matrix: move eye to canonical distance
    S = np.diag([1.0, 1.0, d_norm / z_actual])

    # Rotation: virtual camera faces eye center directly
    z_axis = t_eye / z_actual
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        # Degenerate case: eye is directly above/below camera
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)

    # Canonical intrinsics
    K_norm = np.array([
        [f_norm, 0, size / 2.0],
        [0, f_norm, size / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Perspective warp matrix
    M = K_norm @ R_norm @ S @ np.linalg.inv(K)
    img_norm = cv2.warpPerspective(image, M, (size, size))

    return img_norm, R_norm


def denormalize_gaze(gaze_norm, R_norm):
    """
    Convert gaze direction from normalized space back to camera space.

    Args:
        gaze_norm: (3,) gaze direction in normalized space
        R_norm: (3, 3) normalization rotation matrix

    Returns:
        gaze_cam: (3,) gaze direction in camera space (unit vector)
    """
    g = R_norm.T @ gaze_norm
    return g / np.linalg.norm(g)


def normalize_gaze(gaze_cam, R_norm, R_head=None):
    """
    Transform gaze direction from camera space to normalized space.

    Args:
        gaze_cam: (3,) gaze direction in camera coordinate system
        R_norm: (3, 3) normalization rotation matrix
        R_head: (3, 3) head rotation matrix (unused, gaze already in CCS)

    Returns:
        gaze_norm: (3,) gaze direction in normalized space (unit vector)
    """
    g = R_norm @ gaze_cam
    return g / np.linalg.norm(g)


def warp_points_2d(points_2d, M):
    """
    Warp 2D points using the normalization homography.

    Args:
        points_2d: (N, 2) pixel coordinates in original image
        M: (3, 3) perspective warp matrix from normalize_sample

    Returns:
        warped: (N, 2) pixel coordinates in normalized image
    """
    N = points_2d.shape[0]
    # Convert to homogeneous
    ones = np.ones((N, 1), dtype=points_2d.dtype)
    pts_h = np.concatenate([points_2d, ones], axis=1)  # (N, 3)
    # Warp
    warped_h = (M @ pts_h.T).T  # (N, 3)
    # Dehomogenize
    warped = warped_h[:, :2] / (warped_h[:, 2:3] + 1e-8)
    return warped


def compute_normalization_matrix(K, t_eye, d_norm=600, f_norm=960, size=224):
    """
    Compute the normalization warp matrix M without applying it.
    Useful for warping landmarks separately.

    Args:
        K: (3, 3) camera intrinsic matrix
        t_eye: (3,) eye center in camera coords (mm)
        d_norm: canonical distance (mm)
        f_norm: canonical focal length (px)
        size: output image size

    Returns:
        M: (3, 3) perspective warp matrix
        R_norm: (3, 3) normalization rotation
    """
    z_actual = np.linalg.norm(t_eye)
    S = np.diag([1.0, 1.0, d_norm / z_actual])

    z_axis = t_eye / z_actual
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)

    K_norm = np.array([
        [f_norm, 0, size / 2.0],
        [0, f_norm, size / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    M = K_norm @ R_norm @ S @ np.linalg.inv(K)
    return M, R_norm
