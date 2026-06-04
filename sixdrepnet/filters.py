"""
Temporal smoothing for head-pose predictions.

The per-frame model predicts each frame independently, so the live signal is
jittery. This module smooths the *rotation* over time with a One Euro filter
(Casiez et al., CHI 2012) — an adaptive low-pass filter that trades lag against
jitter based on how fast the signal is moving (heavy smoothing when still, low
lag during fast head motion).

Smoothing is done in quaternion space, NOT on Euler angles: Euler angles wrap
around and hit gimbal lock near +/-90 deg yaw, which would inject artifacts into
the filtered signal. Quaternions are continuous and renormalize cleanly.
"""

import math
from collections import deque

import numpy as np


# -----------------------------------------------------------------------------
# One Euro filter (scalar)
# -----------------------------------------------------------------------------
class _LowPass:
    """First-order exponential low-pass with externally supplied alpha."""

    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """
    Scalar One Euro filter.

    Args:
        min_cutoff: baseline cutoff frequency (Hz). Lower -> smoother but more lag
                    when the signal is (nearly) still.
        beta:       speed coefficient. Higher -> less lag during fast motion.
        d_cutoff:   cutoff for the derivative low-pass (Hz).
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()
        self._x_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x

        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3  # guard against duplicate/non-monotonic timestamps

        dx = (x - self._x_prev) / dt
        dx_hat = self._dx_lp(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_lp(x, self._alpha(cutoff, dt))

        self._x_prev = x
        self._t_prev = t
        return x_hat


# -----------------------------------------------------------------------------
# Quaternion smoother
# -----------------------------------------------------------------------------
class OneEuroQuaternion:
    """
    One Euro filter applied to a unit quaternion (w, x, y, z).

    Each component is filtered independently with shared parameters; the input
    is hemisphere-aligned to the previous sample (quaternion double-cover: q and
    -q are the same rotation) and the output is renormalized to stay valid.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(4)
        ]
        self._q_prev = None

    def reset(self):
        for f in self._filters:
            f._x_lp = _LowPass()
            f._dx_lp = _LowPass()
            f._x_prev = None
            f._t_prev = None
        self._q_prev = None

    def __call__(self, q, t):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return q
        q = q / n

        # Resolve double cover so successive samples live in the same hemisphere.
        if self._q_prev is not None and float(np.dot(q, self._q_prev)) < 0.0:
            q = -q
        self._q_prev = q

        q_hat = np.array([f(q[i], t) for i, f in enumerate(self._filters)])
        nh = np.linalg.norm(q_hat)
        if nh < 1e-12:
            return q
        return q_hat / nh


# -----------------------------------------------------------------------------
# Rotation matrix <-> quaternion (numpy, no torch dependency)
# -----------------------------------------------------------------------------
def rotmat_to_quat(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# -----------------------------------------------------------------------------
# Weighted sliding-window quaternion pre-smoother (FIR)
# -----------------------------------------------------------------------------
class WeightedWindowQuaternion:
    """
    Weighted sliding-window pre-smoother for unit quaternions.

    Keeps the last `size` (quaternion, timestamp) samples and returns their
    weighted average (hemisphere-aligned to the newest sample, renormalized).
    The per-sample weight combines two factors:

      * recency  — newer samples weigh more (exponential decay in time), and
      * speed    — when the head moves fast, the effective window collapses
                   toward the newest sample to cut lag; when the head is still,
                   the full window is used for maximum jitter rejection.

    This is an FIR stage meant to sit *in front of* the One Euro (IIR) filter:
    the window rejects high-frequency per-frame noise, One Euro then provides the
    adaptive lag/jitter trade-off. Speed-adaptive weighting keeps the extra
    smoothing from adding visible lag during real head motion.

    Args:
        size: number of samples in the window (>= 2 to have any effect).
        tau:  recency time constant (s) when the head is still. Larger -> flatter
              (more uniform) weights -> stronger smoothing.
        beta: speed sensitivity. Higher -> window collapses sooner as the head
              speeds up (less lag, less smoothing during motion).
        robust: if True, additionally down-weight samples that sit far (geodesic)
              from the window consensus, turning the weighted mean into a soft
              median. This is what rejects the rare single-frame spikes that make
              the residual distribution heavy-tailed (leptokurtic).
        robust_c: tuning constant of the Welsch redescending weight, in units of
              the robust scale (MAD). Smaller -> more aggressive outlier
              rejection.
    """

    def __init__(self, size=7, tau=0.15, beta=1.5, robust=True, robust_c=2.5):
        self.size = int(size)
        self.tau = float(tau)
        self.beta = float(beta)
        self.robust = bool(robust)
        self.robust_c = float(robust_c)
        self._buf = deque(maxlen=self.size)  # entries: (q (w,x,y,z), t)

    def reset(self):
        self._buf.clear()

    def __call__(self, q, t):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return q
        q = q / n

        # Hemisphere-align the incoming sample to the most recent stored one
        # (quaternion double cover: q and -q are the same rotation).
        if self._buf and float(np.dot(q, self._buf[-1][0])) < 0.0:
            q = -q
        self._buf.append((q, t))

        if len(self._buf) == 1:
            return q

        qs = np.stack([qb for qb, _ in self._buf], axis=0)
        ts = np.array([tb for _, tb in self._buf], dtype=np.float64)

        # Hemisphere-align every buffered sample to the newest before averaging.
        signs = np.sign(qs @ qs[-1])
        signs[signs == 0] = 1.0
        qs = qs * signs[:, None]

        # --- Robust (soft-median) inlier weights -----------------------------
        # The residual distribution is heavy-tailed: most frames are tight but
        # there are rare single-frame spikes. A plain weighted mean lets one
        # spike smear across the whole window. We down-weight samples that sit
        # far (geodesic) from the window consensus with a Welsch redescending
        # weight, exp(-(d / (c*scale))^2), scale = MAD spread.
        #
        # The consensus is found by IRLS toward a geometric median, starting
        # from the UNIFORM mean (recency-independent). This is essential: if the
        # center were recency-weighted it would be biased toward the newest
        # sample, so a spike that just arrived (the common case) could never be
        # flagged as an outlier.
        rob = np.ones(len(qs))
        if self.robust and len(self._buf) >= 3:
            center = qs.sum(axis=0)
            ncen = np.linalg.norm(center)
            if ncen > 1e-12:
                center = center / ncen
                for _ in range(3):  # IRLS; converges in a couple of steps
                    d = 2.0 * np.arccos(np.clip(np.abs(qs @ center), -1.0, 1.0))
                    scale = 1.4826 * float(np.median(d)) + 1e-6
                    rob = np.exp(-(d / (self.robust_c * scale)) ** 2)
                    center = (rob[:, None] * qs).sum(axis=0)
                    ncen = np.linalg.norm(center)
                    if ncen < 1e-12:
                        break
                    center = center / ncen

        # Angular speed from the two most recent samples (geodesic, rad/s),
        # DISCOUNTED by the newest sample's inlier weight. A single-frame spike
        # inflates the raw speed spuriously; if we trusted it, the window would
        # collapse onto the spike (low recency lag) and the robust rejection
        # would be undone. Gating by rob[-1] keeps the window wide on spikes
        # (so they average out) while still collapsing on genuine fast motion.
        dt = max(ts[-1] - ts[-2], 1e-3)
        dot = abs(float(np.clip(np.dot(qs[-1], qs[-2]), -1.0, 1.0)))
        speed = (2.0 * math.acos(dot)) / dt * float(rob[-1])

        # Speed factor in (0, 1]: ~1 when still, -> 0 when fast.
        s = 1.0 / (1.0 + self.beta * speed)

        # Recency weights (sharpened the faster the head moves) times robustness.
        eff_tau = self.tau * (s + 1e-3)
        w = np.exp(-(ts[-1] - ts) / eff_tau) * rob

        wsum = float(w.sum())
        if wsum < 1e-12:
            return qs[-1]
        q_avg = (w[:, None] * qs).sum(axis=0)
        nrm = np.linalg.norm(q_avg)
        if nrm < 1e-12:
            return qs[-1]
        return q_avg / nrm


class HeadPoseSmoother:
    """
    Convenience wrapper: smooths a 3x3 rotation matrix over time and returns a
    smoothed 3x3 rotation matrix. Pass the wall-clock timestamp of each frame.

    Pipeline (quaternion space):
        R -> quat -> [weighted window FIR] -> [One Euro IIR] -> quat -> R

    The weighted window can be disabled (use_window=False) to recover the plain
    One Euro behaviour.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0,
                 window=7, window_tau=0.15, window_beta=1.5, use_window=True,
                 robust=True, robust_c=2.5):
        self._window = (
            WeightedWindowQuaternion(window, window_tau, window_beta,
                                     robust=robust, robust_c=robust_c)
            if use_window and window > 1 else None
        )
        self._q_filter = OneEuroQuaternion(min_cutoff, beta, d_cutoff)

    def reset(self):
        if self._window is not None:
            self._window.reset()
        self._q_filter.reset()

    def __call__(self, R, t):
        q = rotmat_to_quat(R)
        if self._window is not None:
            q = self._window(q, t)
        q_hat = self._q_filter(q, t)
        return quat_to_rotmat(q_hat)
