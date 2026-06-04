"""
Head-pose-normalized eye-patch extraction for 3DeepVOG.

3DeepVOG (https://github.com/DSGZ-MotionLab/3DeepVOG) segments the pupil/iris and
fits a 3D eye model from video where a single eye fills the frame, as if shot by
a fixed head-mounted eye camera. Its segmentation net takes 240x320 (HxW) RGB.

Here the eyes are seen by a *remote* webcam, so head pose contaminates the eye
appearance and the eye does not fill the frame. We remove the head-pose
component with the standard data-normalization warp (Zhang et al., "Revisiting
Data Normalization for Appearance-Based Gaze Estimation", 2018):

  1. Back-project the eye center pixel to a viewing ray `forward` (camera→eye).
  2. Build a normalized camera rotation R_norm that looks along `forward` and is
     rolled to match the head's x-axis (R_head[:,0], head→camera) — this cancels
     head yaw/pitch (re-pointing) and roll (leveling).
  3. H = K · R_norm · K⁻¹ is the rotation homography into that virtual view.
  4. A similarity transform then scales/centers the eye to fill the 320x240 patch.

The eye is centered on the SOCKET (canthi midpoint), NOT the iris, so gaze stays
visible as pupil displacement within the frame — which is what 3DeepVOG measures.

Monocular note: only the *direction* to the eye is needed (no metric depth), so
no camera calibration is required beyond an approximate focal length (defaults to
the image width, ~60° HFOV). Roll is taken from the head-pose model, so its error
maps directly to apparent ocular torsion — see the README/caveats.
"""

import numpy as np
import cv2

from filters import OneEuroFilter

# MediaPipe FaceMesh canthi (eye corners). "right"/"left" are the subject's.
_R_OUT, _R_IN = 33, 133     # right eye: outer (lateral), inner (medial)
_L_IN, _L_OUT = 362, 263    # left eye:  inner (medial), outer (lateral)

# The four canthi above drive the patch translation (their midpoints) and scale
# (their separations). They are the points we temporally smooth.
_ANCHORS = (_R_OUT, _R_IN, _L_IN, _L_OUT)


class _PointSmoother:
    """One Euro filter on a 2D pixel point (x and y filtered independently)."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, p, t):
        return np.array([self._fx(float(p[0]), t), self._fy(float(p[1]), t)])


def intrinsics(w, h, focal=None):
    """Approximate pinhole camera matrix; focal defaults to image width."""
    f = float(focal) if focal else float(w)
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


class EyeNormalizer:
    """
    Produces head-pose-normalized eye patches from FaceMesh landmarks + head R.

    Args:
        out_w, out_h: patch size (3DeepVOG wants 320x240, W x H).
        fill:    fraction of the patch width the eye (canthus-to-canthus) spans.
        focal:   assumed focal length in px (None -> image width).
        flip_right: horizontally flip the right-eye patch so both eyes share the
                 same handedness (some per-eye models expect this).
    """

    def __init__(self, out_w=320, out_h=240, fill=0.8, focal=None,
                 flip_right=False, smooth=True, smooth_min_cutoff=1.0,
                 smooth_beta=0.3):
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.fill = float(fill)
        self.focal = focal
        self.flip_right = bool(flip_right)
        # Temporal smoothing of the canthi anchors (translation + scale jitter).
        # Orientation is already smoothed upstream via R_head; this filters the
        # raw per-frame FaceMesh landmark wobble that the warp amplifies.
        self.smooth = bool(smooth)
        self.smooth_min_cutoff = float(smooth_min_cutoff)
        self.smooth_beta = float(smooth_beta)
        self._anchor_sm = None  # lazy: dict {landmark idx -> _PointSmoother}

    def reset(self):
        """Drop anchor-smoothing state (call after a sustained face loss)."""
        self._anchor_sm = None

    def _norm_one(self, frame, K, Kinv, R_head, center_uv, corner_a, corner_b):
        # 1. Viewing ray to the eye center (direction only — no metric depth).
        ray = Kinv @ np.array([center_uv[0], center_uv[1], 1.0])
        n = np.linalg.norm(ray)
        if n < 1e-9:
            return None
        forward = ray / n

        # 2. Normalized-camera rotation: look along `forward`, roll-leveled to the
        #    head x-axis (R_head[:,0] is the head's rightward axis in camera frame).
        hRx = R_head[:, 0]
        down = np.cross(forward, hRx)
        nd = np.linalg.norm(down)
        if nd < 1e-6:               # forward ∥ head-x (degenerate) — skip frame
            return None
        down /= nd
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R_norm = np.array([right, down, forward])  # rows: camera→normalized

        # 3. Rotation homography into the virtual frontal/leveled view.
        H = K @ R_norm @ Kinv

        def warp_pt(p):
            v = H @ np.array([p[0], p[1], 1.0])
            return v[:2] / v[2]

        c = warp_pt(center_uv)
        a, b = warp_pt(corner_a), warp_pt(corner_b)
        eye_w = float(np.linalg.norm(a - b))
        if eye_w < 1e-3:
            return None

        # 4. Similarity: scale eye to `fill`·out_w and center it in the patch.
        scale = (self.fill * self.out_w) / eye_w
        S = np.array([[scale, 0,     self.out_w / 2.0 - scale * c[0]],
                      [0,     scale, self.out_h / 2.0 - scale * c[1]],
                      [0,     0,     1.0]])
        W = S @ H
        patch = cv2.warpPerspective(frame, W, (self.out_w, self.out_h))
        return patch, W

    def process(self, frame, landmarks, R_head, t=None):
        """
        frame:     BGR image (any size).
        landmarks: (N, 3) FaceMesh landmarks in pixel units (from FaceCrop).
        R_head:    (3, 3) head→camera rotation (smoothed pose recommended).
        t:         frame timestamp (s). Required for anchor smoothing; if None,
                   smoothing is skipped (raw landmarks used).

        Returns a dict possibly containing 'left' and 'right' -> (out_h, out_w, 3)
        patches. An eye is omitted if its warp is degenerate.
        """
        if landmarks is None:
            return {}
        h, w = frame.shape[:2]
        K = intrinsics(w, h, self.focal)
        Kinv = np.linalg.inv(K)
        lm = np.asarray(landmarks)[:, :2].astype(np.float64)
        R_head = np.asarray(R_head, dtype=np.float64)

        # Temporally smooth the canthi anchors so the patch stops shivering. We
        # smooth the landmark points (not the derived center/scale) so both the
        # translation and the scale settle, and the warp geometry stays exact.
        if self.smooth and t is not None:
            if self._anchor_sm is None:
                self._anchor_sm = {
                    i: _PointSmoother(self.smooth_min_cutoff, self.smooth_beta)
                    for i in _ANCHORS
                }
            lm = lm.copy()
            for i, sm in self._anchor_sm.items():
                lm[i] = sm(lm[i], t)

        out = {}
        # Right eye (subject's right; image-left).
        c_r = 0.5 * (lm[_R_OUT] + lm[_R_IN])
        res = self._norm_one(frame, K, Kinv, R_head, c_r, lm[_R_OUT], lm[_R_IN])
        if res is not None:
            patch = res[0]
            if self.flip_right:
                patch = cv2.flip(patch, 1)
            out['right'] = patch
        # Left eye (subject's left; image-right).
        c_l = 0.5 * (lm[_L_OUT] + lm[_L_IN])
        res = self._norm_one(frame, K, Kinv, R_head, c_l, lm[_L_OUT], lm[_L_IN])
        if res is not None:
            out['left'] = res[0]
        return out
