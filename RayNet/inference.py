"""
RayNet inference and visualization tool.

Runs RayNet on images, video files, or webcam feed and visualizes:
  - 14 iris/pupil landmarks (10 iris contour + 4 pupil boundary)
  - 3D gaze vector projected as an arrow from eye center
  - Head pose axes (RGB = XYZ) from 6D rotation prediction
  - Pitch/yaw angles as text overlay

Usage:
    # Single image
    python -m RayNet.inference --checkpoint best_model.pt --input face.jpg

    # Video file
    python -m RayNet.inference --checkpoint best_model.pt --input video.mp4

    # Webcam (default camera 0)
    python -m RayNet.inference --checkpoint best_model.pt --webcam

    # With MinIO checkpoint loading
    python -m RayNet.inference \
        --ckpt_bucket raynet-checkpoints \
        --minio_endpoint http://204.168.238.119:9000 \
        --run_id run_20260412_123641 \
        --ckpt_file best_model.pt \
        --webcam

    # With face detection (requires mediapipe)
    python -m RayNet.inference --checkpoint best_model.pt --webcam --face_detect
"""

import argparse
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ─── Landmark layout ────────────────────────────────────────────────
# 14 landmarks: indices 0-9 = iris contour, 10-13 = pupil boundary
IRIS_IDX = list(range(10))
PUPIL_IDX = list(range(10, 14))

# Colors (BGR)
COLOR_IRIS = (0, 255, 0)        # green
COLOR_PUPIL = (0, 200, 255)     # orange
COLOR_GAZE = (255, 0, 255)      # magenta
COLOR_POSE_X = (0, 0, 255)     # red
COLOR_POSE_Y = (0, 255, 0)     # green
COLOR_POSE_Z = (255, 0, 0)     # blue
COLOR_TEXT = (255, 255, 255)    # white
COLOR_BG = (40, 40, 40)        # dark gray


def load_model(args):
    """Load RayNet model with trained weights."""
    from RayNet.raynet import create_raynet, device

    model = create_raynet(
        core_backbone_name=args.core_backbone,
        core_backbone_weight_path=args.core_backbone_weight_path,
        pose_backbone_name=args.pose_backbone,
        pose_backbone_weight_path=args.pose_backbone_weight_path,
    )

    # Load checkpoint
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    elif args.run_id:
        from RayNet.streaming.checkpoint import CheckpointManager
        mgr = CheckpointManager(
            bucket=args.ckpt_bucket,
            run_id=args.run_id,
            endpoint=args.minio_endpoint,
        )
        state = mgr.load(args.ckpt_file, map_location=device)
    else:
        raise ValueError("Provide --checkpoint (local) or --run_id + --ckpt_bucket (MinIO)")

    # Handle torch.compile wrapper
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    if 'model_state_dict' in state:
        target.load_state_dict(state['model_state_dict'], strict=False)
        epoch = state.get('epoch', '?')
        stage = state.get('config', {}).get('stage', '?')
        print(f"Loaded checkpoint: stage {stage}, epoch {epoch}")
    else:
        target.load_state_dict(state, strict=False)
        print("Loaded raw state dict")

    model.eval()
    return model, device


def preprocess_image(image_bgr, img_size=224):
    """
    Preprocess a BGR image for RayNet inference.

    Args:
        image_bgr: (H, W, 3) BGR uint8 image (face crop)
        img_size: model input size

    Returns:
        tensor: (1, 3, img_size, img_size) normalized tensor
    """
    img = cv2.resize(image_bgr, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # Normalize to [-1, 1] (mean=0.5, std=0.5)
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor


def rotation_6d_to_matrix(r6d):
    """Convert 6D rotation to 3x3 matrix (numpy, single sample)."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def detect_faces_mediapipe(image_bgr):
    """Detect faces using MediaPipe. Returns list of (x, y, w, h) bounding boxes."""
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    h, w = image_bgr.shape[:2]
    with mp_face.FaceDetection(min_detection_confidence=0.5) as fd:
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
    """Fallback face detection using OpenCV's DNN or Haar cascade."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return [(x, y, w, h) for (x, y, w, h) in faces]


def expand_bbox(x, y, w, h, img_w, img_h, factor=1.3):
    """Expand bounding box by a factor and make it square."""
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * factor
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(img_w, int(cx + side / 2))
    y2 = min(img_h, int(cy + side / 2))
    return x1, y1, x2, y2


@torch.no_grad()
def run_inference(model, image_tensor, device, use_bridge=True):
    """
    Run single-view inference.

    Returns:
        landmarks: (14, 2) numpy array in feature map coords
        gaze_vector: (3,) numpy unit vector
        gaze_angles: (2,) numpy [pitch, yaw] in radians
        pose_6d: (6,) numpy 6D rotation
        pose_t: (3,) numpy translation in meters
    """
    image_tensor = image_tensor.to(device)
    out = model(image_tensor, n_views=1, use_bridge=use_bridge)

    landmarks = out['landmark_coords'][0].cpu().numpy()       # (14, 2)
    gaze_vector = out['gaze_vector'][0].cpu().numpy()          # (3,)
    gaze_angles = out['gaze_angles'][0].cpu().numpy()          # (2,)
    pose_6d = out['pred_pose_6d'][0].cpu().numpy() if out['pred_pose_6d'] is not None else None
    pose_t = out['pred_pose_t'][0].cpu().numpy() if out['pred_pose_t'] is not None else None

    return landmarks, gaze_vector, gaze_angles, pose_6d, pose_t


def draw_landmarks(canvas, landmarks_px, radius=2):
    """Draw 14 landmarks on the image. Iris=green, pupil=orange."""
    for i in IRIS_IDX:
        pt = tuple(landmarks_px[i].astype(int))
        cv2.circle(canvas, pt, radius, COLOR_IRIS, -1, cv2.LINE_AA)

    for i in PUPIL_IDX:
        pt = tuple(landmarks_px[i].astype(int))
        cv2.circle(canvas, pt, radius, COLOR_PUPIL, -1, cv2.LINE_AA)

    # Connect iris contour
    for i in range(len(IRIS_IDX)):
        p1 = tuple(landmarks_px[IRIS_IDX[i]].astype(int))
        p2 = tuple(landmarks_px[IRIS_IDX[(i + 1) % len(IRIS_IDX)]].astype(int))
        cv2.line(canvas, p1, p2, COLOR_IRIS, 1, cv2.LINE_AA)

    # Connect pupil points
    for i in range(len(PUPIL_IDX)):
        p1 = tuple(landmarks_px[PUPIL_IDX[i]].astype(int))
        p2 = tuple(landmarks_px[PUPIL_IDX[(i + 1) % len(PUPIL_IDX)]].astype(int))
        cv2.line(canvas, p1, p2, COLOR_PUPIL, 1, cv2.LINE_AA)


def draw_gaze_arrow(canvas, origin, gaze_vector, length=80, thickness=2):
    """
    Draw gaze direction as an arrow projected onto the image.

    gaze_vector is in CCS: x=right, y=down, z=forward.
    We project the xy components to get the 2D arrow direction and
    use the z component to scale (closer = longer arrow).
    """
    # Project 3D gaze to 2D image plane (perspective-like)
    dx = gaze_vector[0]  # right
    dy = gaze_vector[1]  # down
    dz = gaze_vector[2]  # forward (into screen)

    # Scale arrow length by how much the gaze goes into the screen
    scale = length / max(abs(dz), 0.1)
    end_x = int(origin[0] + dx * scale)
    end_y = int(origin[1] + dy * scale)

    origin_pt = (int(origin[0]), int(origin[1]))
    end_pt = (end_x, end_y)

    cv2.arrowedLine(canvas, origin_pt, end_pt, COLOR_GAZE, thickness,
                    cv2.LINE_AA, tipLength=0.3)


def draw_pose_axes(canvas, center, R_mat, axis_length=40, thickness=2):
    """Draw head pose as RGB axes (X=red, Y=green, Z=blue)."""
    origin = np.array(center, dtype=float)

    for i, color in enumerate([COLOR_POSE_X, COLOR_POSE_Y, COLOR_POSE_Z]):
        axis = R_mat[:, i]  # column i of rotation matrix
        # Project to 2D: x component → right, y component → down
        end = origin + axis_length * np.array([axis[0], axis[1]])
        cv2.arrowedLine(canvas,
                        (int(origin[0]), int(origin[1])),
                        (int(end[0]), int(end[1])),
                        color, thickness, cv2.LINE_AA, tipLength=0.25)


def draw_info_overlay(canvas, gaze_angles, pose_t, fps=None):
    """Draw text overlay with gaze angles and pose info."""
    pitch_deg = math.degrees(gaze_angles[0])
    yaw_deg = math.degrees(gaze_angles[1])

    lines = [
        f"Gaze: pitch={pitch_deg:+.1f} yaw={yaw_deg:+.1f}",
    ]
    if pose_t is not None:
        lines.append(f"Trans: x={pose_t[0]:.2f} y={pose_t[1]:.2f} z={pose_t[2]:.2f} m")
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    y_offset = 20
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (8, y_offset - th - 4), (12 + tw, y_offset + 4), COLOR_BG, -1)
        cv2.putText(canvas, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        y_offset += th + 12


def visualize_predictions(image_bgr, landmarks_feat, gaze_vector, gaze_angles,
                          pose_6d, pose_t, feat_size=56, img_size=224,
                          crop_bbox=None):
    """
    Draw all predictions on the image.

    Args:
        image_bgr: original image (or crop)
        landmarks_feat: (14, 2) landmark coords in feature map space (56x56)
        gaze_vector: (3,) unit vector in CCS
        gaze_angles: (2,) pitch, yaw in radians
        pose_6d: (6,) 6D rotation or None
        pose_t: (3,) translation or None
        feat_size: feature map spatial dim (P2 at stride 4 = 56 for 224 input)
        img_size: model input size
        crop_bbox: (x1, y1, x2, y2) if drawing on full frame

    Returns:
        canvas: annotated image
    """
    if crop_bbox is not None:
        canvas = image_bgr.copy()
        x1, y1, x2, y2 = crop_bbox
        crop_w, crop_h = x2 - x1, y2 - y1

        # Scale landmarks from feature space to crop space, then to frame space
        scale_x = crop_w / feat_size
        scale_y = crop_h / feat_size
        landmarks_px = landmarks_feat.copy()
        landmarks_px[:, 0] = landmarks_feat[:, 0] * scale_x + x1
        landmarks_px[:, 1] = landmarks_feat[:, 1] * scale_y + y1

        # Eye center for gaze arrow (average of all landmarks)
        eye_center = landmarks_px.mean(axis=0)

        # Pose axes center
        pose_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Draw crop box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (128, 128, 128), 1)

        lm_radius = max(2, int(crop_w / 80))
        arrow_len = max(40, int(crop_w / 3))
        axis_len = max(30, int(crop_w / 4))
    else:
        h, w = image_bgr.shape[:2]
        canvas = image_bgr.copy()

        # Scale from feature space to display image
        scale_x = w / feat_size
        scale_y = h / feat_size
        landmarks_px = landmarks_feat.copy()
        landmarks_px[:, 0] *= scale_x
        landmarks_px[:, 1] *= scale_y

        eye_center = landmarks_px.mean(axis=0)
        pose_center = (w / 2, h / 2)

        lm_radius = max(2, int(w / 80))
        arrow_len = max(40, int(w / 3))
        axis_len = max(30, int(w / 4))

    # Draw landmarks
    draw_landmarks(canvas, landmarks_px, radius=lm_radius)

    # Draw gaze arrow from eye center
    draw_gaze_arrow(canvas, eye_center, gaze_vector, length=arrow_len, thickness=2)

    # Draw head pose axes
    if pose_6d is not None:
        R_mat = rotation_6d_to_matrix(pose_6d)
        draw_pose_axes(canvas, pose_center, R_mat, axis_length=axis_len)

    # Draw text overlay
    draw_info_overlay(canvas, gaze_angles, pose_t)

    return canvas


def process_single_image(model, device, image_path, args):
    """Process a single image file."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: cannot read {image_path}")
        return

    if args.face_detect:
        faces = detect_faces(image)
        if not faces:
            print("No face detected")
            return
        h, w = image.shape[:2]
        for (fx, fy, fw, fh) in faces:
            x1, y1, x2, y2 = expand_bbox(fx, fy, fw, fh, w, h)
            crop = image[y1:y2, x1:x2]
            tensor = preprocess_image(crop)
            lm, gaze, angles, p6d, pt = run_inference(model, tensor, device,
                                                       use_bridge=args.use_bridge)
            image = visualize_predictions(image, lm, gaze, angles, p6d, pt,
                                          crop_bbox=(x1, y1, x2, y2))
    else:
        # Assume the input is already a face crop
        tensor = preprocess_image(image)
        lm, gaze, angles, p6d, pt = run_inference(model, tensor, device,
                                                   use_bridge=args.use_bridge)
        image = visualize_predictions(image, lm, gaze, angles, p6d, pt)

    # Display
    cv2.imshow('RayNet Inference', image)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save if requested
    if args.output:
        cv2.imwrite(str(args.output), image)
        print(f"Saved to {args.output}")


def detect_faces(image_bgr):
    """Try MediaPipe first, fall back to OpenCV Haar cascade."""
    try:
        boxes = detect_faces_mediapipe(image_bgr)
        if boxes:
            return boxes
    except ImportError:
        pass
    return detect_faces_opencv(image_bgr)


def process_video(model, device, source, args):
    """Process video file or webcam stream."""
    if source == 'webcam':
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    # Video writer
    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

    print("Press 'q' to quit, 'b' to toggle bridge, 'f' to toggle face detection")
    use_bridge = args.use_bridge
    use_face_det = args.face_detect
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = frame.copy()

        if use_face_det:
            faces = detect_faces(frame)
            h, w = frame.shape[:2]
            for (fx, fy, fw, fh) in faces:
                x1, y1, x2, y2 = expand_bbox(fx, fy, fw, fh, w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                tensor = preprocess_image(crop)
                lm, gaze, angles, p6d, pt = run_inference(
                    model, tensor, device, use_bridge=use_bridge)
                canvas = visualize_predictions(
                    canvas, lm, gaze, angles, p6d, pt,
                    crop_bbox=(x1, y1, x2, y2))
        else:
            tensor = preprocess_image(frame)
            lm, gaze, angles, p6d, pt = run_inference(
                model, tensor, device, use_bridge=use_bridge)
            canvas = visualize_predictions(
                canvas, lm, gaze, angles, p6d, pt)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_time = time.time()

        # FPS + mode overlay
        mode_text = f"FPS: {fps_display:.1f} | Bridge: {'ON' if use_bridge else 'OFF'} | FaceDet: {'ON' if use_face_det else 'OFF'}"
        cv2.putText(canvas, mode_text, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.imshow('RayNet Inference', canvas)

        if writer:
            writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            use_bridge = not use_bridge
            print(f"Bridge: {'ON' if use_bridge else 'OFF'}")
        elif key == ord('f'):
            use_face_det = not use_face_det
            print(f"Face detection: {'ON' if use_face_det else 'OFF'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='RayNet inference and visualization (OpenFace-style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument('--core_backbone', default='repnext_m3',
                        help='Core backbone name (default: repnext_m3)')
    parser.add_argument('--core_backbone_weight_path', default=None,
                        help='Path to pretrained core backbone weights')
    parser.add_argument('--pose_backbone', default='repnext_m1',
                        help='Pose backbone name (default: repnext_m1)')
    parser.add_argument('--pose_backbone_weight_path', default=None,
                        help='Path to pretrained pose backbone weights')

    # Checkpoint (local or MinIO)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to local checkpoint file')
    parser.add_argument('--run_id', type=str, default=None,
                        help='MinIO run ID for checkpoint loading')
    parser.add_argument('--ckpt_file', type=str, default='best_model.pt',
                        help='Checkpoint filename within the run (default: best_model.pt)')
    parser.add_argument('--ckpt_bucket', type=str, default='raynet-checkpoints',
                        help='MinIO bucket name')
    parser.add_argument('--minio_endpoint', type=str, default=None,
                        help='MinIO endpoint URL')

    # Input
    parser.add_argument('--input', type=str, default=None,
                        help='Path to image or video file')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam as input')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID (default: 0)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output image or video')

    # Options
    parser.add_argument('--face_detect', action='store_true',
                        help='Enable face detection (requires mediapipe or uses Haar)')
    parser.add_argument('--use_bridge', action='store_true', default=True,
                        help='Use LandmarkGazeBridge (default: True)')
    parser.add_argument('--no_bridge', action='store_true',
                        help='Disable LandmarkGazeBridge')

    args = parser.parse_args()

    if args.no_bridge:
        args.use_bridge = False

    if not args.input and not args.webcam:
        parser.error("Provide --input (image/video) or --webcam")

    # Load model
    print("Loading model...")
    model, device = load_model(args)
    print(f"Model loaded on {device}")

    # Dispatch
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
