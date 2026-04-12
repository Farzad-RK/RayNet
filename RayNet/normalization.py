"""
Easy-Norm: MAGE-style image normalization for gaze estimation.

Simplified normalization that requires only a face center (from bounding box
or landmarks) — no head pose needed.

Reference: Bao et al., "MAGE: Multi-task Architecture for Gaze Estimation",
           CVPR 2025.

Split pipeline support:
  - normalize_for_gaze():      Full normalization (rotation + scaling)
  - normalize_for_landmarks():  Partial normalization (scaling only, no roll)
  - 3D geometry:                No normalization (stay in original CCS)

NOTE on GazeGene dataset:
  Easy-Norm CANNOT be applied to GazeGene training data because the dataset
  applies random translation and scaling during face cropping, which breaks
  the camera intrinsic → pixel coordinate mapping. The K_cropped intrinsics
  no longer correspond to the actual pixel positions after augmentation.
  For GazeGene, images go through as-is and the model learns to handle
  augmented crops directly.

  Easy-Norm IS intended for:
  - Inference on live camera feeds with known intrinsics
  - Other datasets with valid (un-augmented) intrinsics
  - Future datasets where normalization is applied BEFORE augmentation
"""

import numpy as np
import cv2


def build_K_norm(f_norm, img_size):
    """Build canonical intrinsic matrix with centered principal point."""
    cx = img_size / 2.0
    cy = img_size / 2.0
    return np.array([
        [f_norm, 0.0, cx],
        [0.0, f_norm, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _build_rotation_to_face(t_face):
    """
    Build rotation matrix that orients the virtual camera's z-axis
    toward the face center.

    Args:
        t_face: (3,) face center in camera coordinates

    Returns:
        R_norm: (3, 3) rotation matrix
    """
    z_axis = t_face / np.linalg.norm(t_face)

    # Handle degenerate case where face is directly above/below camera
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_norm = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)
    return R_norm


class EasyNorm:
    """
    MAGE Easy-Norm: bounding-box-only image normalization.

    Uses only the face center in 3D camera coordinates to build a
    normalization homography. No head pose required.

    The normalization:
      1. Rotates the virtual camera's z-axis toward the face center
      2. Scales depth to a canonical distance (d_norm)
      3. Applies canonical intrinsics (f_norm, centered principal point)
      4. Produces a homography M for cv2.warpPerspective

    Args:
        f_norm: canonical focal length in pixels (default 960)
        d_norm: canonical distance from camera to face in mm (default 600)
        img_size: output normalized image size (default 224)
    """

    def __init__(self, f_norm=960.0, d_norm=600.0, img_size=224):
        self.f_norm = f_norm
        self.d_norm = d_norm
        self.img_size = img_size
        self.K_norm = build_K_norm(f_norm, img_size)

    def compute(self, K, t_face, distance=None):
        """
        Compute normalization homography from camera intrinsics and face center.

        Args:
            K: (3, 3) original camera intrinsic matrix
            t_face: (3,) face center in camera coordinates (mm or cm)
            distance: optional override for face distance; if None, uses ||t_face||

        Returns:
            M: (3, 3) homography for cv2.warpPerspective
            M_inv: (3, 3) inverse homography
            R_norm: (3, 3) normalization rotation matrix
        """
        # Face distance
        z_actual = np.linalg.norm(t_face) if distance is None else distance
        if z_actual < 1e-6:
            z_actual = self.d_norm  # fallback to canonical distance

        # Rotation: z-axis → face center
        R_norm = _build_rotation_to_face(t_face)

        # Scaling matrix: move from actual distance to canonical
        S = np.diag([1.0, 1.0, self.d_norm / z_actual])

        # Homography: K_norm @ S @ R_norm @ K_inv
        K_inv = np.linalg.inv(K)
        M = self.K_norm @ S @ R_norm @ K_inv
        M_inv = np.linalg.inv(M)

        return M, M_inv, R_norm

    def normalize_for_gaze(self, image, K, t_face, distance=None):
        """
        Full normalization for gaze regression.

        Rotates z-axis toward face center and scales to canonical distance.
        The gaze backbone receives a pose-normalized, scale-normalized image.

        Args:
            image: (H, W, 3) BGR image
            K: (3, 3) camera intrinsics
            t_face: (3,) face center in camera coordinates

        Returns:
            img_norm: (img_size, img_size, 3) normalized image
            R_norm: (3, 3) normalization rotation (for gaze vector transform)
            M: (3, 3) homography
            M_inv: (3, 3) inverse homography
        """
        M, M_inv, R_norm = self.compute(K, t_face, distance)
        img_norm = cv2.warpPerspective(
            image, M, (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )
        return img_norm, R_norm, M, M_inv

    def normalize_for_landmarks(self, image, K, t_face, distance=None):
        """
        Partial normalization for landmark detection (no roll).

        Scales the image to canonical distance but does NOT apply full
        rotation. This preserves iris ellipse appearance and avoids
        artificial Ocular Counter-Rolling (OCR) artifacts.

        Uses only the depth-scaling component of the normalization.

        Args:
            image: (H, W, 3) BGR image
            K: (3, 3) camera intrinsics
            t_face: (3,) face center in camera coordinates

        Returns:
            img_scaled: (img_size, img_size, 3) scaled image
            scale_factor: float, the depth scaling ratio (d_norm / z_actual)
            M_scale: (3, 3) scale-only homography
            M_scale_inv: (3, 3) inverse
        """
        z_actual = np.linalg.norm(t_face) if distance is None else distance
        if z_actual < 1e-6:
            z_actual = self.d_norm

        scale_factor = self.d_norm / z_actual

        # Scale-only: no rotation, just depth scaling
        S = np.diag([scale_factor, scale_factor, 1.0])
        K_inv = np.linalg.inv(K)
        M_scale = self.K_norm @ S @ K_inv
        M_scale_inv = np.linalg.inv(M_scale)

        img_scaled = cv2.warpPerspective(
            image, M_scale, (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )
        return img_scaled, scale_factor, M_scale, M_scale_inv


# ---------------------------------------------------------------------------
# Gaze vector transformations
# ---------------------------------------------------------------------------

def normalize_gaze_vector(gaze_ccs, R_norm):
    """
    Transform gaze direction from camera coordinate space (CCS) to
    normalized space.

    Args:
        gaze_ccs: (..., 3) gaze unit vector in CCS
        R_norm: (3, 3) normalization rotation matrix

    Returns:
        gaze_norm: (..., 3) gaze unit vector in normalized space
    """
    gaze_norm = (R_norm @ gaze_ccs[..., None])[..., 0]
    gaze_norm = gaze_norm / (np.linalg.norm(gaze_norm, axis=-1, keepdims=True) + 1e-8)
    return gaze_norm


def denormalize_gaze_vector(gaze_norm, R_norm):
    """
    Transform gaze direction from normalized space back to CCS.

    Args:
        gaze_norm: (..., 3) gaze unit vector in normalized space
        R_norm: (3, 3) normalization rotation matrix

    Returns:
        gaze_ccs: (..., 3) gaze unit vector in CCS
    """
    gaze_ccs = (R_norm.T @ gaze_norm[..., None])[..., 0]
    gaze_ccs = gaze_ccs / (np.linalg.norm(gaze_ccs, axis=-1, keepdims=True) + 1e-8)
    return gaze_ccs


# ---------------------------------------------------------------------------
# 2D point warping
# ---------------------------------------------------------------------------

def warp_points_2d(points, M):
    """
    Warp 2D points using a homography matrix.

    Args:
        points: (N, 2) pixel coordinates
        M: (3, 3) homography matrix

    Returns:
        warped: (N, 2) warped pixel coordinates
    """
    N = points.shape[0]
    pts_h = np.column_stack([points, np.ones(N)])  # (N, 3)
    warped_h = (M @ pts_h.T).T  # (N, 3)
    warped = warped_h[:, :2] / (warped_h[:, 2:3] + 1e-8)  # dehomogenize
    return warped
