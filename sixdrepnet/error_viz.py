"""
Real-time visualization of the head-pose smoothing error distribution.

A live webcam stream has no ground truth, so the "error" we can actually measure
is the *smoothing residual*: the raw per-frame prediction minus the smoothed
output (i.e. the jitter the temporal filter removes each frame). Watching the
distribution of this residual tells us what kind of noise the model produces —
near-Gaussian, heavy-tailed (occasional large jumps), or skewed.

Two views are provided:
  * RealtimeErrorViz  — yaw/pitch/roll pooled into one distribution.
  * PerAxisErrorViz   — pitch / yaw / roll as three stacked distributions, so a
                        per-axis pathology (e.g. yaw spikes near profile) is
                        visible separately.

Both draw a histogram with a fitted Gaussian overlay plus summary stats (mean,
std, skew, excess kurtosis). Rendering uses plain OpenCV/numpy (no matplotlib),
so it is cheap and safe to call from inside the capture loop.
"""

from collections import deque

import numpy as np
import cv2


def wrap_deg(a):
    """Wrap angle(s) to (-180, 180] degrees."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


class _Hist:
    """One rolling histogram with a Gaussian overlay + stats, drawn into a rect."""

    def __init__(self, maxlen, range_deg, bins, title, color):
        self.buf = deque(maxlen=maxlen)
        self.range_deg = float(range_deg)
        self.bins = int(bins)
        self.title = title
        self.color = color

    def add(self, vals):
        for v in np.ravel(vals):
            self.buf.append(float(v))

    def reset(self):
        self.buf.clear()

    @staticmethod
    def _stats(a):
        mean = float(a.mean())
        std = float(a.std())
        if std < 1e-9 or a.size < 3:
            return mean, std, 0.0, 0.0
        z = (a - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)  # excess kurtosis (0 = Gaussian)
        return mean, std, skew, kurt

    def render_into(self, img, x0, y0, x1, y1):
        """Draw this histogram into the rectangle (x0,y0)-(x1,y1) of `img`."""
        # Inner plot area (leave room for title, ticks, stats line).
        px0, px1 = x0 + 42, x1 - 10
        py0, py1 = y0 + 20, y1 - 26
        pw, ph = px1 - px0, py1 - py0

        cv2.rectangle(img, (px0, py0), (px1, py1), (70, 70, 70), 1)
        cv2.putText(img, self.title, (x0 + 2, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        if len(self.buf) < 5:
            cv2.putText(img, 'collecting...', (px0 + 8, (py0 + py1) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            return

        a = np.asarray(self.buf, dtype=np.float64)
        rng = self.range_deg
        edges = np.linspace(-rng, rng, self.bins + 1)
        hist, _ = np.histogram(np.clip(a, -rng, rng), bins=edges)
        hmax = max(int(hist.max()), 1)

        bin_w_px = pw / self.bins
        for i, c in enumerate(hist):
            bx0 = int(px0 + i * bin_w_px)
            bx1 = int(px0 + (i + 1) * bin_w_px) - 1
            bh = int((c / hmax) * ph)
            cv2.rectangle(img, (bx0, py1 - bh), (max(bx1, bx0), py1),
                          self.color, -1)

        mean, std, skew, kurt = self._stats(a)

        # Gaussian overlay fitted to data mean/std, scaled to the histogram peak.
        if std > 1e-6:
            xs = np.linspace(-rng, rng, 160)
            pdf = np.exp(-0.5 * ((xs - mean) / std) ** 2)
            pdf /= max(pdf.max(), 1e-9)
            pts = np.empty((xs.size, 2), np.int32)
            pts[:, 0] = (px0 + (xs + rng) / (2 * rng) * pw).astype(np.int32)
            pts[:, 1] = (py1 - pdf * ph).astype(np.int32)
            cv2.polylines(img, [pts], False, (80, 90, 240), 2)

        # Zero reference line + x ticks.
        zx = int(px0 + 0.5 * pw)
        cv2.line(img, (zx, py0), (zx, py1), (120, 120, 120), 1)
        for tv in (-rng, 0, rng):
            tx = int(px0 + (tv + rng) / (2 * rng) * pw)
            cv2.putText(img, f'{tv:.0f}', (tx - 6, py1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

        if kurt > 1.0:
            shape = 'heavy-tailed'
        elif kurt < -0.6:
            shape = 'light-tailed'
        else:
            shape = 'near-Gaussian'
        txt = (f'n={a.size} mu={mean:+.2f} sd={std:.2f} '
               f'sk={skew:+.2f} exk={kurt:+.1f} [{shape}]')
        cv2.putText(img, txt, (x0 + 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 220, 180), 1)


class RealtimeErrorViz:
    """Single pooled histogram of all per-axis residuals (deg)."""

    def __init__(self, maxlen=900, width=480, height=320,
                 range_deg=6.0, bins=61):
        self.width = int(width)
        self.height = int(height)
        self._h = _Hist(maxlen, range_deg, bins,
                        'Smoothing residual (raw - smoothed), deg',
                        (90, 160, 230))

    def add(self, residuals):
        self._h.add(residuals)

    def reset(self):
        self._h.reset()

    def render(self):
        img = np.full((self.height, self.width, 3), 30, np.uint8)
        self._h.render_into(img, 0, 0, self.width, self.height)
        return img


class PerAxisErrorViz:
    """Three stacked histograms: pitch, yaw, roll residuals (deg)."""

    # Order matches euler[:, 0]=pitch, euler[:, 1]=yaw, euler[:, 2]=roll.
    _AXES = (
        ('pitch', (90, 160, 230)),   # blue
        ('yaw',   (120, 200, 120)),  # green
        ('roll',  (200, 160, 90)),   # orange
    )

    def __init__(self, maxlen=400, width=440, sub_height=180,
                 range_deg=6.0, bins=51):
        self.width = int(width)
        self.sub_height = int(sub_height)
        self.height = self.sub_height * len(self._AXES)
        self._hists = [
            _Hist(maxlen, range_deg, bins, name, color)
            for name, color in self._AXES
        ]

    def add(self, residual_per_axis):
        """residual_per_axis: [pitch, yaw, roll] signed residuals (deg)."""
        r = np.ravel(residual_per_axis)
        for h, v in zip(self._hists, r):
            h.add([v])

    def reset(self):
        for h in self._hists:
            h.reset()

    def render(self):
        img = np.full((self.height, self.width, 3), 30, np.uint8)
        for i, h in enumerate(self._hists):
            y0 = i * self.sub_height
            h.render_into(img, 0, y0, self.width, y0 + self.sub_height)
        return img
