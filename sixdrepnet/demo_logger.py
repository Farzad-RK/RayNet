"""
CSV logger for the head-pose demo.

Writes one row per frame so a run can be analyzed offline:

    frame, t_wall, dt, face, infer_ms,
    raw_pitch, raw_yaw, raw_roll,      # model output (pre-smoothing)
    sm_pitch,  sm_yaw,  sm_roll        # what is drawn (post-smoothing)

From these three derived signals tell the whole noise story:
  * residual   = raw - smoothed         (what the filter removed)
  * raw jitter = frame-to-frame d(raw)  (model + crop noise)
  * sm  jitter = frame-to-frame d(sm)   (noise the user actually sees)

A running summary (per-axis std + excess kurtosis of each signal) is printed
every `summary_every` frames and once more on close, so the terminal alone gives
a quick read even without parsing the CSV.
"""

import csv
import os
from collections import deque

import numpy as np

from error_viz import wrap_deg

_AXES = ('pitch', 'yaw', 'roll')


def _stats(buf):
    a = np.asarray(buf, dtype=np.float64)
    if a.size < 3:
        return dict(n=int(a.size), mean=float('nan'), std=float('nan'),
                    skew=float('nan'), exkurt=float('nan'))
    m = float(a.mean())
    s = float(a.std())
    if s < 1e-9:
        return dict(n=int(a.size), mean=m, std=0.0, skew=0.0, exkurt=0.0)
    z = (a - m) / s
    return dict(n=int(a.size), mean=m, std=s,
                skew=float(np.mean(z ** 3)),
                exkurt=float(np.mean(z ** 4) - 3.0))


class FrameLogger:
    def __init__(self, path, detector='', meta=None, summary_every=150,
                 buf=4000):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        self._f = open(path, 'w', newline='')
        self._f.write('# head-pose demo log\n')
        self._f.write('# detector=%s\n' % detector)
        for k, v in (meta or {}).items():
            self._f.write('# %s=%s\n' % (k, v))
        self._w = csv.writer(self._f)
        self._w.writerow(['frame', 't_wall', 'dt', 'face', 'infer_ms',
                          'raw_pitch', 'raw_yaw', 'raw_roll',
                          'sm_pitch', 'sm_yaw', 'sm_roll'])

        self.frame = 0
        self.n_face = 0
        self.n_miss = 0
        self.summary_every = int(summary_every)

        self._t_prev = None
        self._raw_prev = None
        self._sm_prev = None
        # Rolling buffers, one deque per axis, for the live summary.
        self._res = [deque(maxlen=buf) for _ in _AXES]
        self._raw_jit = [deque(maxlen=buf) for _ in _AXES]
        self._sm_jit = [deque(maxlen=buf) for _ in _AXES]

    def log_frame(self, t_wall, face=True, infer_ms=None, raw=None, sm=None):
        dt = 0.0 if self._t_prev is None else max(t_wall - self._t_prev, 0.0)
        self._t_prev = t_wall
        self.frame += 1
        ms = '' if infer_ms is None else '%.2f' % infer_ms

        if face and raw is not None and sm is not None:
            self.n_face += 1
            raw = np.asarray(raw, dtype=np.float64)
            sm = np.asarray(sm, dtype=np.float64)

            res = wrap_deg(raw - sm)
            for i in range(3):
                self._res[i].append(float(res[i]))
            if self._raw_prev is not None:
                draw = wrap_deg(raw - self._raw_prev)
                for i in range(3):
                    self._raw_jit[i].append(float(draw[i]))
            if self._sm_prev is not None:
                dsm = wrap_deg(sm - self._sm_prev)
                for i in range(3):
                    self._sm_jit[i].append(float(dsm[i]))
            self._raw_prev = raw
            self._sm_prev = sm

            row = [self.frame, '%.4f' % t_wall, '%.4f' % dt, 1, ms,
                   '%.3f' % raw[0], '%.3f' % raw[1], '%.3f' % raw[2],
                   '%.3f' % sm[0], '%.3f' % sm[1], '%.3f' % sm[2]]
        else:
            self.n_miss += 1
            # Break frame-to-frame continuity across a face gap.
            self._raw_prev = None
            self._sm_prev = None
            row = [self.frame, '%.4f' % t_wall, '%.4f' % dt, 0, ms,
                   '', '', '', '', '', '']

        self._w.writerow(row)
        if self.frame % 50 == 0:
            self._f.flush()
        if self.summary_every and self.frame % self.summary_every == 0:
            self.print_summary()

    def summary(self):
        out = {'frames': self.frame, 'faces': self.n_face, 'misses': self.n_miss}
        for i, ax in enumerate(_AXES):
            out['residual_' + ax] = _stats(self._res[i])
            out['raw_jitter_' + ax] = _stats(self._raw_jit[i])
            out['sm_jitter_' + ax] = _stats(self._sm_jit[i])
        return out

    def print_summary(self):
        s = self.summary()
        print('--- log summary @ frame %d (faces=%d misses=%d) ---'
              % (s['frames'], s['faces'], s['misses']))
        for ax in _AXES:
            r = s['residual_' + ax]
            rj = s['raw_jitter_' + ax]
            sj = s['sm_jitter_' + ax]
            print('  %-5s | residual sd=%.3f exk=%+5.1f | rawd sd=%.3f | '
                  'smd sd=%.3f' % (ax, r['std'], r['exkurt'], rj['std'],
                                   sj['std']))

    def close(self):
        try:
            if self.frame:
                self.print_summary()
        finally:
            self._f.flush()
            self._f.close()
