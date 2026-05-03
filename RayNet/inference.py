"""
RayNet v5 inference and visualization tool.

Triple-M1 + AERI architecture (raynet_v5). Embedded face detection: the
caller passes a full frame, the module detects the face internally
(MediaPipe with Haar cascade fallback), crops to 224x224, and runs the
v5 model. Visualizes:

  - 14 iris/pupil landmarks (10 iris + 4 pupil) projected back to frame
  - AERI iris + eyeball masks at 56x56, upsampled and overlaid
  - Optical axis drawn from the 2D pupil centroid (GazeGene §4.2.2):
    origin = mean(pupil_landmarks), direction = gaze_vector / gz
  - Head pose axes (RGB = XYZ) from the 6D rotation prediction
  - The face bounding box used by the model, drawn on the frame
  - Gaze pitch / yaw overlay; translation requires calibrated intrinsics

Face crop: always square, zero-padded when the box extends outside the
frame boundary so the aspect ratio is never distorted (prevents the
eyeball mask from appearing elliptical).

Camera intrinsics for webcam / video: pass --fx/--fy/--cx/--cy on the
command line. When omitted, a heuristic K (f = max(W, H), principal point
at centre) is used for the BoxEncoder bbox encoding.

Temporal smoothing: EMA (--ema_alpha, default 0.35) is applied to the
face bounding box, gaze predictions, and mask logits to reduce
frame-to-frame jitter. Set --ema_alpha 1.0 to disable smoothing.

Usage:
    python -m RayNet.inference --checkpoint best_model.pt --webcam
    python -m RayNet.inference --checkpoint best_model.pt --input face.jpg
    python -m RayNet.inference --checkpoint best_model.pt --input clip.mp4

    # With calibrated intrinsics (reduces jitter, improves arrow direction)
    python -m RayNet.inference --checkpoint best_model.pt --webcam \\
        --fx 800 --fy 800 --cx 320 --cy 240

    # MinIO checkpoint loading
    python -m RayNet.inference \\
        --ckpt_bucket raynet-checkpoints \\
        --minio_endpoint http://204.168.238.119:9000 \\
        --run_id triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115 \\
        --ckpt_file best_model.pt --webcam
"""

import argparse
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# 14 landmarks: 0-9 iris contour, 10-13 pupil
IRIS_IDX = list(range(10))
PUPIL_IDX = list(range(10, 14))

COLOR_IRIS = (0, 255, 0)
COLOR_PUPIL = (0, 200, 255)
COLOR_GAZE = (255, 0, 255)
COLOR_POSE_X = (0, 0, 255)
COLOR_POSE_Y = (0, 255, 0)
COLOR_POSE_Z = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (40, 40, 40)
COLOR_FACEBOX = (0, 220, 220)
COLOR_IRIS_MASK = (0, 255, 0)
COLOR_EYE_MASK = (0, 255, 255)


# ─── Temporal smoothing ─────────────────────────────────────────────

class EmaState:
    """Exponential moving average for reducing frame-to-frame jitter.

    alpha=1.0 → no memory (raw values); alpha→0 → very heavy smoothing.
    Default alpha=0.35 gives ~65% weight to history.
    """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self._state = {}

    def smooth(self, key, value):
        if key not in self._state:
            self._state[key] = value.copy()
        else:
            self._state[key] = (self.alpha * value
                                + (1.0 - self.alpha) * self._state[key])
        return self._state[key]


# ─── Camera intrinsics helpers ───────────────────────────────────────

def default_frame_K(img_w, img_h):
    """Heuristic full-frame K: f = max(W, H), principal point at centre."""
    f = float(max(img_w, img_h))
    return np.array([
        [f,   0.0, img_w / 2.0],
        [0.0, f,   img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _build_K_frame(args, img_w, img_h):
    """Build full-frame K from CLI args or fall back to heuristic."""
    if args.fx is not None:
        fx = args.fx
        fy = args.fy if args.fy is not None else args.fx
        cx = args.cx if args.cx is not None else img_w / 2.0
        cy = args.cy if args.cy is not None else img_h / 2.0
        return np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    return default_frame_K(img_w, img_h)


def load_model(args):
    """
    Build RayNet v5 (Triple-M1) and load the trained state dict from the
    checkpoint. Backbone .pth weights are NOT required at inference —
    every parameter is in the checkpoint.
    """
    from RayNet.raynet_v5 import create_raynet_v5, device

    model = create_raynet_v5(
        backbone_weight_path=None,
        n_landmarks=14,
    )

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device,
                           weights_only=False)
    elif args.run_id:
        from RayNet.streaming.checkpoint import CheckpointManager
        mgr = CheckpointManager(
            bucket=args.ckpt_bucket,
            run_id=args.run_id,
            endpoint=args.minio_endpoint,
        )
        state = mgr.load(args.ckpt_file, map_location=device)
    else:
        raise ValueError(
            "Provide --checkpoint (local) or --run_id + --ckpt_bucket (MinIO)")

    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    if 'model_state_dict' in state:
        missing, unexpected = target.load_state_dict(
            state['model_state_dict'], strict=False)
        epoch = state.get('epoch', '?')
        cfg = state.get('config', {}) or {}
        phase = cfg.get('phase', cfg.get('stage', '?'))
        print(f"Loaded v5 checkpoint: phase {phase}, epoch {epoch}")
        if missing:
            print(f"  missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} "
                  f"(first: {unexpected[:3]})")
    else:
        target.load_state_dict(state, strict=False)
        print("Loaded raw state dict")

    model.eval()
    return model, device


def preprocess_crop(crop_bgr, img_size=224):
    """BGR uint8 face crop → (1, 3, 224, 224) tensor in [-1, 1]."""
    img = cv2.resize(crop_bgr, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def rotation_6d_to_matrix(r6d):
    """6D rotation → 3x3 (numpy, single sample), Gram-Schmidt."""
    a1, a2 = r6d[:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


# ─── Face detection ─────────────────────────────────────────────────

def detect_faces_mediapipe(image_bgr):
    import mediapipe as mp
    h, w = image_bgr.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5) as fd:
        results = fd.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []
        boxes = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(w - x, int(bb.width * w))
            bh = min(h - y, int(bb.height * h))
            boxes.append((x, y, bw, bh))
        return boxes


def detect_faces_opencv(image_bgr):
    cascade_path = (cv2.data.haarcascades
                    + 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return [(x, y, w, h) for (x, y, w, h) in faces]


def detect_faces(image_bgr):
    try:
        boxes = detect_faces_mediapipe(image_bgr)
        if boxes:
            return boxes
    except ImportError:
        pass
    return detect_faces_opencv(image_bgr)


def expand_to_square(x, y, w, h, factor=1.3):
    """Square-ify a detection box.  Box may extend outside the frame —
    use `padded_crop` to sample it safely."""
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * factor
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    return x1, y1, x2, y2


def padded_crop(frame, x1, y1, x2, y2):
    """Extract [y1:y2, x1:x2] with zero-padding for out-of-frame regions.

    This guarantees the returned crop is exactly (y2-y1) × (x2-x1) and
    square when the box was produced by expand_to_square, regardless of
    whether the box extends outside the frame boundary.  Clipping the box
    instead would produce a non-square crop that cv2.resize stretches to
    224×224, distorting circles into ellipses (eyeball mask artefact).
    """
    h, w = frame.shape[:2]
    crop_h, crop_w = y2 - y1, x2 - x1
    result = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)

    # Source rectangle clamped to frame bounds
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)

    # Destination offsets in the padded canvas
    dx1, dy1 = sx1 - x1, sy1 - y1
    result[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = \
        frame[sy1:sy2, sx1:sx2]
    return result


def mage_bbox_from_pixels(x1, y1, x2, y2, img_w, img_h, K_frame=None):
    """
    Synthesise (x_p, y_p, L_x) for the MAGE BoxEncoder from a pixel-space
    bounding box, mirroring `dataset._intrinsic_delta_bbox`.

    When K_frame is provided its principal point is used; otherwise the
    principal point is assumed at the image centre.
    """
    if K_frame is not None:
        cx_o = K_frame[0, 2]
        cy_o = K_frame[1, 2]
    else:
        cx_o = img_w * 0.5
        cy_o = img_h * 0.5
    W_o, H_o = 2.0 * cx_o, 2.0 * cy_o
    cx_box = 0.5 * (x1 + x2)
    cy_box = 0.5 * (y1 + y2)
    x_p = (cx_box - W_o * 0.5) / (W_o * 0.5)
    y_p = (cy_box - H_o * 0.5) / (H_o * 0.5)
    L_x = (x2 - x1) / W_o
    return np.array([x_p, y_p, L_x], dtype=np.float32)


# ─── Inference ──────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, image_tensor, device, face_bbox=None):
    """
    Single-view forward pass through RayNet v5.

    Args:
        face_bbox: optional (3,) numpy or tensor [x_p, y_p, L_x]; passed
            to PoseBranch's BoxEncoder. If None, BoxEncoder zeros out
            (its residual is zero-init so the model still works, just
            without the bbox prior in pose features).
    """
    image_tensor = image_tensor.to(device)
    bbox_t = None
    if face_bbox is not None:
        bbox_t = torch.from_numpy(face_bbox).unsqueeze(0).to(device) \
            if isinstance(face_bbox, np.ndarray) else face_bbox.to(device)

    out = model(image_tensor, n_views=1, face_bbox=bbox_t,
                aeri_alpha=0.9)

    landmarks = out['landmark_coords'][0].cpu().numpy()
    iris_logits = out['iris_mask_logits'][0].cpu().numpy()
    eyeball_logits = out['eyeball_mask_logits'][0].cpu().numpy()
    eyeball_center = out['eyeball_center'][0].cpu().numpy()
    pupil_center = out['pupil_center'][0].cpu().numpy()
    gaze_vector = out['gaze_vector'][0].cpu().numpy()
    gaze_angles = out['gaze_angles'][0].cpu().numpy()
    pose_6d = out['pred_pose_6d'][0].cpu().numpy()
    pose_t = out['pred_pose_t'][0].cpu().numpy()

    return {
        'landmarks': landmarks,
        'iris_mask': iris_logits,
        'eyeball_mask': eyeball_logits,
        'eyeball_center': eyeball_center,   # (3,) cm, CCS
        'pupil_center': pupil_center,        # (3,) cm, CCS
        'gaze_vector': gaze_vector,
        'gaze_angles': gaze_angles,
        'pose_6d': pose_6d,
        'pose_t': pose_t,
    }


# ─── Drawing ────────────────────────────────────────────────────────

def draw_landmarks(canvas, lm_px, radius=2):
    for i in IRIS_IDX:
        cv2.circle(canvas, tuple(lm_px[i].astype(int)), radius,
                   COLOR_IRIS, -1, cv2.LINE_AA)
    for i in PUPIL_IDX:
        cv2.circle(canvas, tuple(lm_px[i].astype(int)), radius,
                   COLOR_PUPIL, -1, cv2.LINE_AA)
    for i in range(len(IRIS_IDX)):
        p1 = tuple(lm_px[IRIS_IDX[i]].astype(int))
        p2 = tuple(lm_px[IRIS_IDX[(i + 1) % len(IRIS_IDX)]].astype(int))
        cv2.line(canvas, p1, p2, COLOR_IRIS, 1, cv2.LINE_AA)
    for i in range(len(PUPIL_IDX)):
        p1 = tuple(lm_px[PUPIL_IDX[i]].astype(int))
        p2 = tuple(lm_px[PUPIL_IDX[(i + 1) % len(PUPIL_IDX)]].astype(int))
        cv2.line(canvas, p1, p2, COLOR_PUPIL, 1, cv2.LINE_AA)


def overlay_mask(canvas, mask_logits, x1, y1, x2, y2, color, alpha=0.35):
    """Overlay a sigmoid-thresholded mask within a bounding box on canvas.

    The box (x1,y1,x2,y2) may extend outside the canvas (e.g. after
    padded_crop).  We resize the mask to the full box, then crop both the
    mask and the canvas slice to the intersection before blending.
    """
    ch, cw = canvas.shape[:2]
    # Intersection of box with canvas
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(cw, x2), min(ch, y2)
    if rx1 >= rx2 or ry1 >= ry2:
        return

    prob = 1.0 / (1.0 + np.exp(-mask_logits))
    # Resize mask to full box size first …
    full_w, full_h = x2 - x1, y2 - y1
    prob_full = cv2.resize(prob, (full_w, full_h))
    # … then crop to the visible region (handles out-of-frame boxes)
    ox, oy = rx1 - x1, ry1 - y1
    prob_vis = prob_full[oy:oy + (ry2 - ry1), ox:ox + (rx2 - rx1)]

    bin_mask = (prob_vis > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return
    layer = canvas[ry1:ry2, rx1:rx2].copy()
    color_arr = np.array(color, dtype=np.uint8)
    layer[bin_mask == 1] = (
        alpha * color_arr + (1 - alpha) * layer[bin_mask == 1]
    ).astype(np.uint8)
    canvas[ry1:ry2, rx1:rx2] = layer


def draw_gaze_arrow(canvas, origin_2d, gaze_vec, length=80, thickness=2):
    """Optical axis arrow anchored to the 2D pupil centroid.

    The gaze_vector [gx, gy, gz] is in CCS (x right, y down, z forward).
    We project it onto the image plane via perspective division (gx/gz,
    gy/gz), which gives the correct on-screen direction for any
    forward-facing gaze without needing calibrated intrinsics.

    Using the 2D pupil landmark centroid as origin is geometrically
    equivalent to projecting pupil_center_3D with K — the landmarks ARE
    the model's 2D view of the pupil — but it works correctly regardless
    of whether the 3D depth estimate is accurate.
    """
    dx, dy, dz = gaze_vec[0], gaze_vec[1], gaze_vec[2]
    scale = length / max(abs(dz), 0.1)
    end = (int(origin_2d[0] + dx * scale), int(origin_2d[1] + dy * scale))
    cv2.arrowedLine(canvas, (int(origin_2d[0]), int(origin_2d[1])), end,
                    COLOR_GAZE, thickness, cv2.LINE_AA, tipLength=0.3)


def draw_pose_axes(canvas, center, R_mat, axis_length=40, thickness=2):
    origin = np.array(center, dtype=float)
    for i, color in enumerate([COLOR_POSE_X, COLOR_POSE_Y, COLOR_POSE_Z]):
        axis = R_mat[:, i]
        end = origin + axis_length * np.array([axis[0], axis[1]])
        cv2.arrowedLine(canvas,
                        (int(origin[0]), int(origin[1])),
                        (int(end[0]), int(end[1])),
                        color, thickness, cv2.LINE_AA, tipLength=0.25)


def draw_facebox(canvas, x1, y1, x2, y2, label='face'):
    ch, cw = canvas.shape[:2]
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(cw, x2), min(ch, y2)
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), COLOR_FACEBOX, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lx1, ly1 = rx1, max(0, ry1 - th - 6)
    cv2.rectangle(canvas, (lx1, ly1), (lx1 + tw + 6, ly1 + th + 6),
                  COLOR_FACEBOX, -1)
    cv2.putText(canvas, label, (lx1 + 3, ly1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_info_overlay(canvas, gaze_angles, pose_t, fps=None,
                      has_calibration=False):
    pitch_deg = math.degrees(gaze_angles[0])
    yaw_deg = math.degrees(gaze_angles[1])
    lines = [f"Gaze: pitch={pitch_deg:+.1f} yaw={yaw_deg:+.1f}"]
    if pose_t is not None:
        cal = "" if has_calibration else " (uncal)"
        lines.append(f"Trans{cal}: x={pose_t[0]:+.2f} y={pose_t[1]:+.2f} "
                     f"z={pose_t[2]:+.2f} m")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    y = 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)
        cv2.rectangle(canvas, (8, y - th - 4), (12 + tw, y + 4),
                      COLOR_BG, -1)
        cv2.putText(canvas, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                    cv2.LINE_AA)
        y += th + 12


# ─── Visualisation pipeline ─────────────────────────────────────────

def visualize(canvas, preds, x1, y1, x2, y2, feat_size=56, show_masks=True):
    """Annotate `canvas` with predictions for one face crop."""
    crop_w, crop_h = x2 - x1, y2 - y1

    if show_masks:
        overlay_mask(canvas, preds['eyeball_mask'], x1, y1, x2, y2,
                     COLOR_EYE_MASK, alpha=0.25)
        overlay_mask(canvas, preds['iris_mask'], x1, y1, x2, y2,
                     COLOR_IRIS_MASK, alpha=0.45)

    # 14 landmarks: feature space (56) → crop space → frame space
    sx = crop_w / feat_size
    sy = crop_h / feat_size
    lm_px = preds['landmarks'].copy()
    lm_px[:, 0] = lm_px[:, 0] * sx + x1
    lm_px[:, 1] = lm_px[:, 1] * sy + y1
    draw_landmarks(canvas, lm_px, radius=max(2, int(crop_w / 80)))

    # Optical axis: origin = 2D pupil centroid, direction = gaze_vector.
    # The 2D centroid of the pupil landmarks is the correct on-screen
    # location of the pupil regardless of whether the 3D depth estimate
    # (which requires calibrated intrinsics) is accurate.
    pupil_center_2d = lm_px[PUPIL_IDX].mean(axis=0)
    arrow_len = max(40, int(crop_w / 3))
    draw_gaze_arrow(canvas, pupil_center_2d, preds['gaze_vector'],
                    length=arrow_len, thickness=2)

    R_mat = rotation_6d_to_matrix(preds['pose_6d'])
    pose_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    draw_pose_axes(canvas, pose_center, R_mat,
                   axis_length=max(30, int(crop_w / 4)))

    draw_facebox(canvas, x1, y1, x2, y2, label='face (input)')


# ─── Drivers ────────────────────────────────────────────────────────

def process_frame(model, device, frame, args, K_frame=None, ema=None):
    canvas = frame.copy()
    h, w = frame.shape[:2]

    if K_frame is None:
        K_frame = default_frame_K(w, h)

    boxes = detect_faces(frame)
    if not boxes:
        cv2.putText(canvas, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return canvas, None

    has_calibration = args.fx is not None

    last_preds = None
    for (fx, fy, fw, fh) in boxes:
        x1, y1, x2, y2 = expand_to_square(fx, fy, fw, fh,
                                           factor=args.crop_factor)

        # EMA on bbox centre+size to damp detection jitter.
        # Keep box square: smooth cx, cy, side independently.
        if ema is not None:
            side = float(x2 - x1)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            s = ema.smooth('bbox', np.array([cx, cy, side]))
            half = s[2] / 2.0
            x1 = int(round(s[0] - half))
            y1 = int(round(s[1] - half))
            x2 = int(round(s[0] + half))
            y2 = int(round(s[1] + half))

        # padded_crop keeps the box square even when it extends outside
        # the frame, preventing aspect-ratio distortion of the masks.
        crop = padded_crop(frame, x1, y1, x2, y2)
        if crop.size == 0:
            continue

        tensor = preprocess_crop(crop)
        face_bbox = mage_bbox_from_pixels(x1, y1, x2, y2, w, h, K_frame)
        preds = run_inference(model, tensor, device, face_bbox=face_bbox)

        # EMA on gaze and masks to damp model-output jitter
        if ema is not None:
            preds['gaze_vector'] = ema.smooth('gaze_vec',
                                               preds['gaze_vector'])
            preds['gaze_angles'] = ema.smooth('gaze_ang',
                                               preds['gaze_angles'])
            preds['iris_mask'] = ema.smooth('iris_mask',
                                             preds['iris_mask'])
            preds['eyeball_mask'] = ema.smooth('eye_mask',
                                                preds['eyeball_mask'])

        visualize(canvas, preds, x1, y1, x2, y2,
                  show_masks=args.show_masks)
        last_preds = preds

    if last_preds is not None:
        draw_info_overlay(canvas, last_preds['gaze_angles'],
                          last_preds['pose_t'],
                          has_calibration=has_calibration)
    return canvas, last_preds


def process_single_image(model, device, image_path, args):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: cannot read {image_path}")
        return
    h, w = image.shape[:2]
    K_frame = _build_K_frame(args, w, h)
    canvas, _ = process_frame(model, device, image, args,
                               K_frame=K_frame, ema=None)
    cv2.imshow('RayNet v5 Inference', canvas)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.output:
        cv2.imwrite(str(args.output), canvas)
        print(f"Saved to {args.output}")


def process_video(model, device, source, args):
    if source == 'webcam':
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    writer = None
    if args.output:
        wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (wv, hv))

    print("Keys: q=quit  m=toggle masks")
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    K_frame = None  # built on first frame
    ema = EmaState(alpha=args.ema_alpha)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if K_frame is None:
            h, w = frame.shape[:2]
            K_frame = _build_K_frame(args, w, h)

        canvas, _ = process_frame(model, device, frame, args,
                                   K_frame=K_frame, ema=ema)

        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        mode = (f"FPS: {fps_display:.1f} | Masks: "
                f"{'ON' if args.show_masks else 'OFF'}")
        cv2.putText(canvas, mode, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1,
                    cv2.LINE_AA)

        cv2.imshow('RayNet v5 Inference', canvas)
        if writer:
            writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            args.show_masks = not args.show_masks
            print(f"Masks: {'ON' if args.show_masks else 'OFF'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='RayNet v5 inference and visualization '
                    '(Triple-M1 + AERI, embedded face detection)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Local checkpoint path')
    parser.add_argument('--run_id', type=str, default=None,
                        help='MinIO run ID')
    parser.add_argument('--ckpt_file', type=str, default='best_model.pt',
                        help='Checkpoint filename within run')
    parser.add_argument('--ckpt_bucket', type=str,
                        default='raynet-checkpoints',
                        help='MinIO bucket')
    parser.add_argument('--minio_endpoint', type=str, default=None,
                        help='MinIO endpoint URL')

    parser.add_argument('--input', type=str, default=None,
                        help='Image or video path')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam')
    parser.add_argument('--camera_id', type=int, default=0)

    parser.add_argument('--output', type=str, default=None,
                        help='Save annotated image / video here')

    parser.add_argument('--show_masks', action='store_true', default=True,
                        help='Overlay AERI iris+eyeball masks (default on)')
    parser.add_argument('--no_masks', dest='show_masks',
                        action='store_false',
                        help='Disable AERI mask overlay')
    parser.add_argument('--crop_factor', type=float, default=1.3,
                        help='Square-crop expansion factor on the '
                             'detected face box (default 1.3 — matches '
                             'GazeGene face crop)')

    # Camera intrinsics — if omitted a heuristic default is used
    parser.add_argument('--fx', type=float, default=None,
                        help='Camera focal length x (pixels)')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera focal length y (pixels, defaults to fx)')
    parser.add_argument('--cx', type=float, default=None,
                        help='Principal point x (pixels, defaults to W/2)')
    parser.add_argument('--cy', type=float, default=None,
                        help='Principal point y (pixels, defaults to H/2)')

    # Temporal smoothing
    parser.add_argument('--ema_alpha', type=float, default=0.35,
                        help='EMA smoothing weight for new frame '
                             '(0=frozen, 1=no smoothing; default 0.35)')

    args = parser.parse_args()

    if not args.input and not args.webcam:
        parser.error("Provide --input (image/video) or --webcam")

    print("Loading model...")
    model, device = load_model(args)
    print(f"Model loaded on {device}")

    if args.webcam:
        process_video(model, device, 'webcam', args)
    elif args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"Error: {path} does not exist")
            sys.exit(1)
        ext = path.suffix.lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
            process_single_image(model, device, path, args)
        elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            process_video(model, device, path, args)
        else:
            print(f"Unknown file type: {ext}")
            sys.exit(1)


if __name__ == '__main__':
    main()
