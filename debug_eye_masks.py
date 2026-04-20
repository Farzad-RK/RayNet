"""
Validate AERI mask generation on raw GazeGene data.

Run BEFORE re-sharding with masks baked in. Loads N samples from a
GazeGeneDataset, generates the (iris, eyeball) masks via
RayNet.streaming.eye_masks, and writes:

  - per-sample PNG overlays    (image | iris | eyeball | combined)
  - aggregate stats to stdout  (area fractions, containment, failures)

Checks:
  1. iris ⊂ eyeball                    (containment must be ≥ 0.98)
  2. 10 iris landmarks lie on iris polygon boundary   (≤ 1 px at 224)
  3. Multi-view consistency: the same 3D eyeball projected through each
     camera's K must yield a 2D silhouette whose pixel area is stable
     across cameras (rel_std < 30%).

Usage:
    python debug_eye_masks.py <GazeGene_FaceCrops_dir>
                              [--subject N] [--n 6] [--out debug_mask_out]
                              [--eye L]
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import cv2

from RayNet.dataset import GazeGeneDataset, IRIS_SUBSAMPLE_IDX
from RayNet.streaming.eye_masks import (
    render_eyeball_mask, render_all_masks, mask_stats,
    DEFAULT_EYEBALL_RADIUS_CM, DEFAULT_CORNEA_RADIUS_CM,
    DEFAULT_CORNEA_OFFSET_CM,
)


MASK_SIZE = 56
FACE_SIZE = 224
NATIVE_SIZE = 448


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------

def _overlay(image_rgb, mask56, color_bgr, alpha=0.45):
    """Upsample a 56x56 mask to the image resolution and alpha-blend."""
    H, W = image_rgb.shape[:2]
    up = cv2.resize(mask56, (W, H), interpolation=cv2.INTER_NEAREST)
    sel = up > 0
    out = image_rgb.copy()
    out[sel] = (
        (1 - alpha) * out[sel] + alpha * np.array(color_bgr, dtype=np.float32)
    ).astype(np.uint8)
    return out


def _label(img, text, y=18):
    out = img.copy()
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_points(img, pts_224, color=(0, 255, 255), r=2):
    out = img.copy()
    for (x, y) in pts_224:
        cv2.circle(out, (int(round(x)), int(round(y))), r, color, -1)
    return out


def _subject_attrs(ds, subj_num, eye_idx):
    """Extract per-subject anatomy for the requested eye.

    GazeGene's subject_label.pkl stores eyeball_radius / cornea_radius /
    cornea2center as (L, R) tuples (or arrays). If the attribute is
    scalar or missing, the mask renderer falls back to defaults.
    """
    attr = ds.attr_dict.get(subj_num, {}) if hasattr(ds, 'attr_dict') else {}
    out = {}
    for key in ('eyeball_radius', 'cornea_radius', 'cornea2center'):
        v = attr.get(key)
        if v is None:
            continue
        try:
            v_arr = np.asarray(v, dtype=np.float64).ravel()
            out[key] = float(v_arr[eye_idx]) if v_arr.size >= 2 else float(v_arr[0])
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------
# Per-sample check
# ---------------------------------------------------------------------

def process_sample(ds, idx, eye_idx, out_dir):
    """Render masks for one sample, save overlay PNG, return metric dict."""
    raw = ds.samples[idx]
    sample = ds[idx]

    iris_2d = np.asarray(raw['iris_mesh_2D'][eye_idx], dtype=np.float64)
    eyeball_3d = np.asarray(raw['eyeball_center_3D'][eye_idx], dtype=np.float64)
    pupil_3d = np.asarray(raw['pupil_center_3D'][eye_idx], dtype=np.float64)

    K = sample['K'].numpy().astype(np.float64)
    subj_attrs = _subject_attrs(ds, int(raw['subject']), eye_idx)

    iris, eyeball = render_all_masks(
        iris_2d, eyeball_3d, pupil_3d, K,
        subject_attrs=subj_attrs,
        face_size=FACE_SIZE, native_size=NATIVE_SIZE, out_size=MASK_SIZE)

    stats = mask_stats(iris, eyeball)

    # --- Visualize -----------------------------------------------------
    img_rgb = sample['image'].numpy().transpose(1, 2, 0)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    panel_iris = _overlay(img_bgr, iris, (255, 255, 0))
    panel_eyeball = _overlay(img_bgr, eyeball, (0, 255, 255))

    combined = img_bgr.copy()
    combined = _overlay(combined, eyeball, (0, 255, 255), alpha=0.25)
    combined = _overlay(combined, iris, (255, 255, 0), alpha=0.45)

    landmarks_px = sample['landmark_coords_px'].numpy()
    combined = _draw_points(combined, landmarks_px, (0, 0, 255), r=2)

    panels = [
        _label(img_bgr, 'face'),
        _label(panel_iris, 'iris'),
        _label(panel_eyeball, 'eyeball'),
        _label(combined, 'all + 14 lm'),
    ]
    row = np.concatenate(panels, axis=1)

    name = (f"subj{raw['subject']:02d}_cam{raw['cam_id']:02d}"
            f"_frame{raw['frame_idx']:04d}.png")
    cv2.imwrite(os.path.join(out_dir, name), row)

    # --- Landmark-on-polygon check ------------------------------------
    iris_10_native = iris_2d[IRIS_SUBSAMPLE_IDX]
    iris_10_224 = iris_10_native * (FACE_SIZE / NATIVE_SIZE)
    lm_iris_224 = landmarks_px[:10]
    lm_mismatch_px = float(np.mean(
        np.linalg.norm(iris_10_224 - lm_iris_224, axis=1)))

    stats.update({
        'subject': int(raw['subject']),
        'cam_id': int(raw['cam_id']),
        'frame_idx': int(raw['frame_idx']),
        'lm_iris10_scale_err_px': lm_mismatch_px,
        'eyeball_radius_cm': subj_attrs.get(
            'eyeball_radius', DEFAULT_EYEBALL_RADIUS_CM),
        'cornea_radius_cm': subj_attrs.get(
            'cornea_radius', DEFAULT_CORNEA_RADIUS_CM),
        'cornea_offset_cm': subj_attrs.get(
            'cornea2center', DEFAULT_CORNEA_OFFSET_CM),
    })
    return stats


# ---------------------------------------------------------------------
# Multi-view consistency (per subject+frame across cameras)
# ---------------------------------------------------------------------

def multiview_consistency(ds, eye_idx):
    """Reproject the same 3D eyeball through every camera's K and check
    that the rendered silhouette pixel area is stable (rel_std < 30%).
    Big discrepancies mean K is in the wrong pixel frame for some views.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for idx in range(len(ds)):
        s = ds.samples[idx]
        groups[(s['subject'], s['frame_idx'])].append(idx)

    full_groups = [v for v in groups.values() if len(v) == 9][:8]
    rows = []
    for group in full_groups:
        subj = int(ds.samples[group[0]]['subject'])
        subj_attrs = _subject_attrs(ds, subj, eye_idx)
        areas = []
        for idx in group:
            raw = ds.samples[idx]
            sample = ds[idx]
            K = sample['K'].numpy().astype(np.float64)
            eyeball = render_eyeball_mask(
                np.asarray(raw['eyeball_center_3D'][eye_idx],
                           dtype=np.float64),
                K,
                pupil_center_3d=np.asarray(raw['pupil_center_3D'][eye_idx],
                                           dtype=np.float64),
                eyeball_radius_cm=float(subj_attrs.get(
                    'eyeball_radius', DEFAULT_EYEBALL_RADIUS_CM)),
                cornea_radius_cm=float(subj_attrs.get(
                    'cornea_radius', DEFAULT_CORNEA_RADIUS_CM)),
                cornea_offset_cm=float(subj_attrs.get(
                    'cornea2center', DEFAULT_CORNEA_OFFSET_CM)),
                face_size=FACE_SIZE, out_size=MASK_SIZE)
            areas.append(int((eyeball > 0).sum()))
        areas = np.asarray(areas, dtype=np.float64)
        rows.append({
            'subject': subj,
            'frame_idx': int(ds.samples[group[0]]['frame_idx']),
            'area_mean': float(areas.mean()),
            'area_std': float(areas.std()),
            'area_rel_std': float(areas.std() / max(areas.mean(), 1e-6)),
        })
    return rows


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=str)
    ap.add_argument('--subject', type=int, default=None,
                    help='Restrict to one subject (else scans all loaded).')
    ap.add_argument('--eye', type=str, default='L', choices=['L', 'R'])
    ap.add_argument('--n', type=int, default=6,
                    help='Number of samples to visualize.')
    ap.add_argument('--out', type=str, default='debug_mask_out')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    subject_ids = [args.subject] if args.subject else list(range(1, 5))
    ds = GazeGeneDataset(
        base_dir=args.data_dir,
        subject_ids=subject_ids,
        samples_per_subject=30,
        eye=args.eye,
        augment=False,
    )
    print(f"Loaded {len(ds)} samples.")
    eye_idx = 0 if args.eye == 'L' else 1

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    all_stats = []
    fails = 0
    for idx in pick:
        st = process_sample(ds, int(idx), eye_idx, args.out)
        all_stats.append(st)

        cont_ok = st['iris_in_eyeball_frac'] >= 0.98
        scale_ok = st['lm_iris10_scale_err_px'] <= 1.0
        ok = cont_ok and scale_ok
        tag = 'OK ' if ok else 'FAIL'
        if not ok:
            fails += 1
        print(f"[{tag}] s{st['subject']:02d} c{st['cam_id']:02d} "
              f"f{st['frame_idx']:04d} | "
              f"iris={st['iris_area_frac']*100:.2f}% "
              f"eye={st['eyeball_area_frac']*100:.2f}% | "
              f"i⊂e={st['iris_in_eyeball_frac']:.3f} | "
              f"r_eye={st['eyeball_radius_cm']:.2f} "
              f"r_cor={st['cornea_radius_cm']:.2f} "
              f"off={st['cornea_offset_cm']:.2f}cm | "
              f"lm_scale_err={st['lm_iris10_scale_err_px']:.2f}px")

    # --- Aggregate ----------------------------------------------------
    def _col(key):
        return np.asarray([s[key] for s in all_stats], dtype=np.float64)

    print("\n--- aggregate ---")
    for key in ('iris_area_frac', 'eyeball_area_frac',
                'iris_in_eyeball_frac', 'lm_iris10_scale_err_px'):
        v = _col(key)
        print(f"  {key:26s} mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print(f"  failures (containment/scale): {fails}/{len(all_stats)}")

    # --- Multi-view ---------------------------------------------------
    print("\n--- multi-view eyeball-area stability ---")
    mv = multiview_consistency(ds, eye_idx)
    if not mv:
        print("  (no 9-camera groups found)")
    else:
        for row in mv:
            print(f"  s{row['subject']:02d} f{row['frame_idx']:04d} | "
                  f"mean_area={row['area_mean']:.1f}px "
                  f"rel_std={row['area_rel_std']*100:.1f}%")
        rel = np.asarray([r['area_rel_std'] for r in mv])
        print(f"  groups: {len(mv)}  rel_std mean={rel.mean()*100:.1f}% "
              f"max={rel.max()*100:.1f}%")

    print(f"\nOverlays written to {args.out}/")
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
