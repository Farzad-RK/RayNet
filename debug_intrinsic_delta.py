"""
Verify the Intrinsic Delta method for RayNet v5 BoxEncoder supervision.

Pipeline:
  1. Load a raw GazeGene sample (cropped 224x224 face + K_orig + K_crop).
  2. Derive the crop box [x1, y1, x2, y2] in original-image coordinates
     from (K_orig, K_crop, crop_size).
  3. Reconstruct a black-padded "original-resolution" canvas by pasting
     the 224x224 crop back into its derived box (inverse scale).
  4. Re-apply the derived crop box to that canvas and resize to 224x224.
     If the math is correct, the result matches the input crop exactly
     (modulo resampling).
  5. Reproject eyeball_center_3D with K_crop onto the crop; the dot must
     land on the eye. This is the independent sanity check required by
     the Visual Verification Protocol.
  6. Report the normalized BoxEncoder input (x_p, y_p, L_x) used for
     supervision.

Usage:
    python debug_intrinsic_delta.py <GazeGene_FaceCrops_dir> [--subject N]
                                     [--n 4] [--out debug_out]
"""
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from RayNet.dataset import GazeGeneDataset


def derive_crop_box(K_orig, K_crop, crop_size=224):
    """
    Given (K_orig, K_crop), return (x1, y1, x2, y2) in original-image pixels
    such that cropping the original image to this box and resizing to
    (crop_size, crop_size) reproduces K_crop from K_orig.

    K_crop = S * [I | -t] * K_orig semantics:
        f_cx = s_x * f_ox,   cx_c = s_x * (cx_o - x1)
        f_cy = s_y * f_oy,   cy_c = s_y * (cy_o - y1)
        s_x = crop_size / (x2 - x1)
        s_y = crop_size / (y2 - y1)
    """
    f_ox, f_oy = K_orig[0, 0], K_orig[1, 1]
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
    f_cx, f_cy = K_crop[0, 0], K_crop[1, 1]
    cx_c, cy_c = K_crop[0, 2], K_crop[1, 2]

    s_x = f_cx / f_ox
    s_y = f_cy / f_oy

    x1 = cx_o - cx_c / s_x
    y1 = cy_o - cy_c / s_y
    x2 = x1 + crop_size / s_x
    y2 = y1 + crop_size / s_y
    return float(x1), float(y1), float(x2), float(y2), float(s_x), float(s_y)


def box_encoder_params(K_orig, K_crop, crop_size=224):
    """
    Derive (x_p, y_p, L_x) normalized for BoxEncoder.

    x_p, y_p in [-1, 1]: face center in original image normalized by
        (W_o/2, H_o/2), where (W_o, H_o) ≈ (2 * cx_o, 2 * cy_o).
    L_x > 0: face width in original pixels normalized by W_o — a
        focal-ratio proxy for depth (s = f_crop / f_orig, L_x = 1/s).
    """
    x1, y1, x2, y2, s_x, s_y = derive_crop_box(K_orig, K_crop, crop_size)
    cx_o, cy_o = K_orig[0, 2], K_orig[1, 2]
    W_o, H_o = 2.0 * cx_o, 2.0 * cy_o

    face_cx = 0.5 * (x1 + x2)
    face_cy = 0.5 * (y1 + y2)
    face_w = x2 - x1

    x_p = (face_cx - W_o / 2.0) / (W_o / 2.0)
    y_p = (face_cy - H_o / 2.0) / (H_o / 2.0)
    L_x = face_w / W_o
    return float(x_p), float(y_p), float(L_x), (x1, y1, x2, y2, W_o, H_o)


def reconstruct_original_canvas(crop_bgr, box, orig_size):
    """Paste the 224x224 crop back into a black canvas at original resolution."""
    x1, y1, x2, y2 = box
    W_o, H_o = orig_size
    canvas = np.zeros((int(round(H_o)), int(round(W_o)), 3), dtype=np.uint8)

    # Target box rounded to int; clip to canvas.
    tx1 = int(round(x1)); ty1 = int(round(y1))
    tx2 = int(round(x2)); ty2 = int(round(y2))
    tw = tx2 - tx1; th = ty2 - ty1
    if tw <= 0 or th <= 0:
        return canvas

    resized = cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)

    # Intersect with canvas.
    sx0 = max(0, -tx1); sy0 = max(0, -ty1)
    dx0 = max(0, tx1);  dy0 = max(0, ty1)
    dx1 = min(canvas.shape[1], tx2)
    dy1 = min(canvas.shape[0], ty2)
    if dx1 <= dx0 or dy1 <= dy0:
        return canvas
    sw = dx1 - dx0; sh = dy1 - dy0
    canvas[dy0:dy0 + sh, dx0:dx0 + sw] = resized[sy0:sy0 + sh, sx0:sx0 + sw]
    return canvas


def recrop_from_canvas(canvas, box, crop_size=224):
    """Apply the derived box to the reconstructed canvas → recover 224 crop."""
    x1, y1, x2, y2 = box
    H, W = canvas.shape[:2]
    # Use cv2.warpAffine for sub-pixel accurate crop+resize.
    s = crop_size / (x2 - x1)
    M = np.array([[s, 0, -x1 * s],
                  [0, crop_size / (y2 - y1), -y1 * crop_size / (y2 - y1)]],
                 dtype=np.float64)
    return cv2.warpAffine(canvas, M, (crop_size, crop_size),
                          flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def project_point(K, P_ccs):
    """Project a 3D CCS point with K (no extrinsics — already in CCS)."""
    P = np.asarray(P_ccs, dtype=np.float64).reshape(3)
    u = K @ P
    return u[0] / u[2], u[1] / u[2]


def project_candidates(K_crop, P, R_cam, T_cam, head_R, head_t):
    """
    Try several frame conventions and return {name: (u, v, Z)}.
    Whichever one lands on the eye reveals the ground-truth frame for P.
    """
    P = np.asarray(P, dtype=np.float64).reshape(3)
    R_cam = np.asarray(R_cam, dtype=np.float64)
    T_cam = np.asarray(T_cam, dtype=np.float64).reshape(3)
    head_R = np.asarray(head_R, dtype=np.float64)
    head_t = np.asarray(head_t, dtype=np.float64).reshape(3)

    variants = {
        'ccs_raw':              P,
        'ccs_y_flip':           P * np.array([1., -1., 1.]),
        'ccs_yz_flip':          P * np.array([1., -1., -1.]),
        'wcs_to_ccs(R,T)':      R_cam @ P + T_cam,
        'wcs_to_ccs(Rt,T)':     R_cam.T @ P + T_cam,
        'wcs_to_ccs(R,-RT)':    R_cam @ (P - T_cam),
        'wcs_to_ccs(Rt,-RtT)':  R_cam.T @ (P - T_cam),
        'hcs_to_ccs(hR,ht)':    head_R @ P + head_t,
        'hcs_to_ccs(hRt,ht)':   head_R.T @ P + head_t,
    }
    out = {}
    for name, Q in variants.items():
        Z = Q[2]
        if abs(Z) < 1e-6:
            out[name] = (float('nan'), float('nan'), Z)
            continue
        u = K_crop @ Q
        out[name] = (u[0] / u[2], u[1] / u[2], Z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base_dir', help='Path to GazeGene_FaceCrops root')
    ap.add_argument('--subject', type=int, default=None)
    ap.add_argument('--n', type=int, default=4, help='Number of samples to visualise')
    ap.add_argument('--out', default='debug_intrinsic_delta_out')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    subj_ids = [args.subject] if args.subject is not None else None
    ds = GazeGeneDataset(
        base_dir=args.base_dir,
        subject_ids=subj_ids,
        samples_per_subject=args.n,
        eye='L',
        augment=False,
    )
    print(f"Loaded {len(ds)} samples from {args.base_dir}")

    n = min(args.n, len(ds))
    for i in range(n):
        sample = ds[i]
        K_orig = sample['intrinsic_original'].numpy().astype(np.float64)
        K_crop = sample['K'].numpy().astype(np.float64)
        eye_3d = sample['eyeball_center_3d'].numpy().astype(np.float64)

        # Tensor -> BGR uint8 image.
        img_t = sample['image']
        if img_t.dtype != np.uint8 and hasattr(img_t, 'dtype'):
            arr = img_t.numpy()
        else:
            arr = img_t.numpy()
        if arr.dtype != np.uint8:
            arr = (arr.astype(np.float32)).clip(0, 255).astype(np.uint8)
        crop_rgb = np.transpose(arr, (1, 2, 0))
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        x_p, y_p, L_x, extras = box_encoder_params(K_orig, K_crop, 224)
        x1, y1, x2, y2, W_o, H_o = extras

        canvas = reconstruct_original_canvas(crop_bgr, (x1, y1, x2, y2),
                                             (W_o, H_o))
        recrop = recrop_from_canvas(canvas, (x1, y1, x2, y2), 224)

        diff = cv2.absdiff(crop_bgr, recrop)
        mae = float(diff.mean())

        # Reprojection test: project 3D eyeball center with K_crop.
        u_c, v_c = project_point(K_crop, eye_3d)

        # Try candidate frames to find the one that puts the eye in bounds.
        R_cam = sample['R_cam'].numpy().astype(np.float64)
        T_cam = sample['T_cam'].numpy().astype(np.float64)
        head_R = sample['head_R'].numpy().astype(np.float64)
        head_t = sample['head_t'].numpy().astype(np.float64)
        cands = project_candidates(K_crop, eye_3d, R_cam, T_cam, head_R, head_t)

        print(f"\n--- sample {i} subj={sample['subject']} cam={sample['cam_id']} "
              f"frame={sample['frame_idx']} ---")
        print(f"  W_o x H_o = {W_o:.1f} x {H_o:.1f}")
        print(f"  crop box  = ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})"
              f"   [w={x2 - x1:.2f}, h={y2 - y1:.2f}]")
        print(f"  BoxEncoder GT: x_p={x_p:+.4f}  y_p={y_p:+.4f}  L_x={L_x:.4f}")
        print(f"  recrop vs crop MAE (0-255): {mae:.2f}")
        print(f"  eyeball_3d projected to crop: ({u_c:.1f}, {v_c:.1f})  "
              f"[in-bounds: {0 <= u_c < 224 and 0 <= v_c < 224}]")
        print(f"  candidate frames (→ crop pixels, Z):")
        for name, (uu, vv, Z) in cands.items():
            inb = (0 <= uu < 224 and 0 <= vv < 224)
            mark = '  <-- IN BOUNDS' if inb else ''
            print(f"    {name:22s}: ({uu:7.1f}, {vv:7.1f})  Z={Z:+.3f}{mark}")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        # Overlay every candidate; the one landing on the eye is the right frame.
        colors = plt.cm.tab10(np.linspace(0, 1, len(cands)))
        for (name, (uu, vv, _)), c in zip(cands.items(), colors):
            if not (np.isfinite(uu) and np.isfinite(vv)):
                continue
            axes[0].scatter([uu], [vv], s=60, marker='x', color=c, label=name)
        axes[0].set_xlim(-20, 244); axes[0].set_ylim(244, -20)
        axes[0].legend(loc='upper right', fontsize=6, framealpha=0.7)
        axes[0].set_title('crop + candidate reprojections')

        axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].set_title(f'reconstructed {int(W_o)}x{int(H_o)} (black pad)')

        axes[2].imshow(cv2.cvtColor(recrop, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f're-crop (MAE={mae:.2f})')

        axes[3].imshow(diff.mean(axis=2), cmap='hot', vmin=0, vmax=20)
        axes[3].set_title('|input - recrop|')

        for a in axes:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(
            f"subj={sample['subject']} cam={sample['cam_id']} "
            f"frame={sample['frame_idx']}  |  "
            f"BoxEncoder GT = ({x_p:+.3f}, {y_p:+.3f}, {L_x:.3f})")
        out_path = os.path.join(args.out, f"sample_{i:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  wrote {out_path}")


if __name__ == '__main__':
    main()
