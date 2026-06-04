"""
Offline analysis of a head-pose demo CSV log (see demo_logger.py).

Usage:
    python analyze_log.py [path-to-log.csv]

With no argument it picks the most recent demo_log_*.csv under
experimental_samples/. Reports, per axis (pitch/yaw/roll):

  * residual    raw - smoothed         — what the filter removed
  * raw jitter  frame-to-frame d(raw)  — model + crop noise
  * sm  jitter  frame-to-frame d(sm)   — noise actually seen on screen

For each it prints std, skew, excess kurtosis; the raw→sm jitter reduction
factor; the lag-1 autocorrelation of the smoothed jitter (≈0 = white noise the
filter handles well; strongly positive = slow drift/low-freq residual a lag
filter can't remove); and the rate of residual spikes (|·| > 3·std).
"""

import sys
import os
import csv
import glob

import numpy as np

_AXES = ('pitch', 'yaw', 'roll')


def _latest_log():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = glob.glob(os.path.join(here, 'experimental_samples', 'demo_log_*.csv'))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _read(path):
    meta = {}
    with open(path) as f:
        rows = []
        for line in f:
            if line.startswith('#'):
                if '=' in line:
                    k, v = line[1:].split('=', 1)
                    meta[k.strip()] = v.strip()
                continue
            rows.append(line)
    rd = list(csv.reader(rows))
    hdr, body = rd[0], rd[1:]
    col = {name: i for i, name in enumerate(hdr)}
    return meta, col, body


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return dict(n=a.size, std=float('nan'), skew=float('nan'),
                    exkurt=float('nan'))
    m, s = a.mean(), a.std()
    if s < 1e-9:
        return dict(n=a.size, std=0.0, skew=0.0, exkurt=0.0)
    z = (a - m) / s
    return dict(n=a.size, std=float(s), skew=float(np.mean(z ** 3)),
                exkurt=float(np.mean(z ** 4) - 3.0))


def _autocorr1(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return float('nan')
    a = a - a.mean()
    denom = float(np.dot(a, a))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a[:-1], a[1:]) / denom)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _latest_log()
    if not path or not os.path.exists(path):
        print('No log file found. Pass a path or run the demo first.')
        return
    meta, col, body = _read(path)
    print('Log: %s' % path)
    if meta:
        print('Meta: ' + '  '.join('%s=%s' % kv for kv in meta.items()))

    face = np.array([r[col['face']] == '1' for r in body])
    n, nf = len(body), int(face.sum())
    print('Frames: %d  faces: %d  misses: %d (%.1f%% miss)'
          % (n, nf, n - nf, 100.0 * (n - nf) / max(n, 1)))

    dt = np.array([float(r[col['dt']]) for r in body])
    good_dt = dt[(dt > 1e-4) & (dt < 1.0)]
    if good_dt.size:
        print('Mean FPS: %.1f  (dt %.1f ms)'
              % (1.0 / good_dt.mean(), 1000.0 * good_dt.mean()))
    infer = np.array([float(r[col['infer_ms']]) for r in body
                      if r[col['infer_ms']]])
    if infer.size:
        print('Inference: mean %.2f ms  p95 %.2f ms'
              % (infer.mean(), np.percentile(infer, 95)))

    # Per-axis signals over contiguous face runs (so frame-to-frame deltas
    # never straddle a face gap).
    print('\n%-6s | %-22s | %-12s | %-12s | %-7s | %-7s' %
          ('axis', 'residual (filter removed)', 'raw jitter', 'sm jitter',
           'reduce', 'sm ac1'))
    print('-' * 86)
    for ai, ax in enumerate(_AXES):
        raw_c, sm_c = 'raw_' + ax, 'sm_' + ax
        res_all, rawj_all, smj_all = [], [], []
        run_raw, run_sm = [], []
        for r, isface in zip(body, face):
            if isface:
                rv = float(r[col[raw_c]]); sv = float(r[col[sm_c]])
                res_all.append(rv - sv)
                run_raw.append(rv); run_sm.append(sv)
            else:
                if len(run_raw) > 1:
                    rawj_all.extend(np.diff(run_raw))
                    smj_all.extend(np.diff(run_sm))
                run_raw, run_sm = [], []
        if len(run_raw) > 1:
            rawj_all.extend(np.diff(run_raw))
            smj_all.extend(np.diff(run_sm))

        res, rawj, smj = _stats(res_all), _stats(rawj_all), _stats(smj_all)
        reduce = (rawj['std'] / smj['std']) if smj['std'] > 1e-9 else float('inf')
        ac1 = _autocorr1(smj_all)
        print('%-6s | sd%6.3f sk%+5.2f exk%+6.1f | sd%9.3f | sd%9.3f | %5.1fx | %+6.2f'
              % (ax, res['std'], res['skew'], res['exkurt'],
                 rawj['std'], smj['std'], reduce, ac1))

    print('\nReading guide:')
    print('  sm jitter sd  = what you see; lower is smoother.')
    print('  reduce        = raw/sm jitter; higher = filter doing more work.')
    print('  sm ac1 ~ 0    = residual noise is white (filter handling it well);')
    print('                  >> 0 = slow drift/low-freq wobble a lag filter cannot remove.')
    print('  residual exk high = heavy-tailed spikes -> robust window territory.')


if __name__ == '__main__':
    main()
