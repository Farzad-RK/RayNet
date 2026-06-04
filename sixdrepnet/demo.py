import time
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import cv2
import torch
from torch.backends import cudnn
from torchvision import transforms
from PIL import Image

from model import SixDRepNet, SixDRepNet_RepNeXt
import utils
from filters import HeadPoseSmoother
from error_viz import RealtimeErrorViz, PerAxisErrorViz, wrap_deg
from face_crop import IrisFaceCropper, BoxFaceCropper
from demo_logger import FrameLogger
from eye_norm import EyeNormalizer


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=-1, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--video', type=str, default='',
                        help='Process a video file instead of the live camera. '
                             'When set, the camera is ignored and (with '
                             '--eye_patches) a composite canvas video is written '
                             'next to the input.')
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--backbone', type=str, default='repnext_m4')
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    # --- Face cropper ---
    parser.add_argument('--detector', type=str, default='mesh',
                        choices=['mesh', 'box'],
                        help="Face cropper: 'mesh' = iris-anchored FaceMesh crop "
                             "(stable), 'box' = old FaceDetection bounding box.")
    parser.add_argument('--crop_scale', type=float, default=4.0,
                        help='Iris crop side as a multiple of inter-ocular distance.')
    parser.add_argument('--crop_voff', type=float, default=0.5,
                        help='Iris crop vertical offset below eye line (in IOD units).')
    parser.add_argument('--draw_iris', action='store_true',
                        help='Draw the iris anchor points on the demo view.')
    # --- Temporal smoothing (One Euro filter) ---
    parser.add_argument('--no_filter', action='store_true',
                        help='Disable temporal smoothing (show raw per-frame pose).')
    parser.add_argument('--min_cutoff', type=float, default=1.0,
                        help='One Euro min cutoff (Hz). Lower = smoother but more lag.')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='One Euro speed coefficient. Higher = less lag on fast motion.')
    # --- Weighted sliding-window pre-smoother (FIR, before One Euro) ---
    parser.add_argument('--no_window', action='store_true',
                        help='Disable the weighted-window pre-smoother (One Euro only).')
    parser.add_argument('--window', type=int, default=7,
                        help='Window size (frames) for the weighted pre-smoother.')
    parser.add_argument('--window_tau', type=float, default=0.15,
                        help='Window recency time constant (s) when still. Larger = flatter weights.')
    parser.add_argument('--window_beta', type=float, default=1.5,
                        help='Window speed sensitivity. Higher = window collapses sooner on fast motion.')
    # --- Robust (soft-median) window weighting ---
    parser.add_argument('--no_robust', action='store_true',
                        help='Disable robust soft-median weighting (plain weighted mean window).')
    parser.add_argument('--robust_c', type=float, default=2.5,
                        help='Welsch robustness constant (in MAD units). Smaller = more aggressive spike rejection.')
    # --- Real-time error (residual) distribution ---
    parser.add_argument('--no_viz', action='store_true',
                        help='Disable the live residual-distribution histogram windows.')
    parser.add_argument('--no_axis_viz', action='store_true',
                        help='Disable the per-axis (pitch/yaw/roll) residual histograms.')
    parser.add_argument('--viz_range', type=float, default=6.0,
                        help='Histogram x-axis half-range (deg) for the residual plots.')
    # --- CSV logging ---
    parser.add_argument('--no_log', action='store_true',
                        help='Disable per-frame CSV logging.')
    parser.add_argument('--log', type=str, default='',
                        help='CSV log path. Empty = auto path under experimental_samples/.')
    parser.add_argument('--log_every', type=int, default=150,
                        help='Print a running summary every N frames (0 = never).')
    # --- Head-pose-normalized eye patches (for 3DeepVOG) ---
    parser.add_argument('--eye_patches', action='store_true',
                        help='Extract head-pose-normalized 320x240 eye patches (needs --detector mesh).')
    parser.add_argument('--eye_fill', type=float, default=0.8,
                        help='Fraction of patch width the eye spans (canthus-to-canthus).')
    parser.add_argument('--focal', type=float, default=0.0,
                        help='Assumed camera focal length (px); 0 = image width (~60 deg HFOV).')
    parser.add_argument('--flip_right', action='store_true',
                        help='Horizontally flip the right-eye patch to match left-eye handedness.')
    parser.add_argument('--no_eye_smooth', action='store_true',
                        help='Disable temporal smoothing of the eye anchors '
                             '(show the raw per-frame canthi jitter).')
    parser.add_argument('--eye_smooth_cutoff', type=float, default=1.0,
                        help='One Euro min cutoff (Hz) for the eye anchors. '
                             'Lower = steadier patch but more lag.')
    parser.add_argument('--eye_smooth_beta', type=float, default=0.3,
                        help='One Euro speed coefficient for the eye anchors. '
                             'Higher = less lag when the eye translates fast.')
    parser.add_argument('--save_eyes', type=str, default='',
                        help='Directory to write left/right normalized eye MP4s for 3DeepVOG.')
    parser.add_argument('--eye_fps', type=float, default=30.0,
                        help='FPS metadata stamped into the saved eye MP4s.')

    args = parser.parse_args()
    return args


transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Order matches the face as seen in the image: subject's right eye is on the
# image-left, so it goes in the left slot of the patch row.
_EYE_SLOTS = ('right', 'left')


def compose_canvas(frame, patches, patch_wh):
    """Stack the annotated face frame on top of a row of eye patches.

    The patches are placed strictly BELOW the face on a black canvas, so they
    never overlap it. Two fixed slots (subject right | left) are always
    reserved — a missing eye shows an empty placeholder — so the canvas size is
    constant frame-to-frame (required for the video writer).

    frame:    BGR face frame (with pose cube / iris dots already drawn).
    patches:  dict possibly containing 'left'/'right' -> (h, w, 3) BGR patch.
    patch_wh: (out_w, out_h) the normalizer's patch size, for slot reservation.
    """
    fh, fw = frame.shape[:2]
    pw, ph = int(patch_wh[0]), int(patch_wh[1])
    pad, label_h = 12, 22

    row_w = len(_EYE_SLOTS) * pw + (len(_EYE_SLOTS) + 1) * pad
    cw = max(fw, row_w)
    ch = fh + pad + label_h + ph + pad
    canvas = np.zeros((ch, cw, 3), np.uint8)

    # Face centered horizontally along the top.
    fx = (cw - fw) // 2
    canvas[0:fh, fx:fx + fw] = frame

    x = (cw - row_w) // 2 + pad
    y = fh + pad + label_h
    for name in _EYE_SLOTS:
        cv2.putText(canvas, name, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (210, 210, 210), 1)
        patch = patches.get(name)
        if patch is not None:
            canvas[y:y + ph, x:x + pw] = patch
        else:
            cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (60, 60, 60), 1)
            cv2.putText(canvas, 'no eye', (x + pw // 2 - 28, y + ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        x += pw + pad
    return canvas


def fit_to(img, max_w=1280, max_h=900):
    """Downscale (never upscale) so the image fits within max_w x max_h."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True

    # --- Device setup ---
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    cam = args.cam_id

    # --- Model setup ---
    model_snapshot_path = "/home/leviathan/PycharmProjects/GazeToolKit/RayNet/sixdrepnet/pretrained_models/repnext_m4/myexp_epoch_80.tar"

    # The checkpoint was saved in deploy mode (reparameterized convs), so build
    # the model with deploy=True to match the fused structure exactly.
    model = SixDRepNet_RepNeXt(
        backbone_fn=args.backbone,
        pretrained=False,
        deploy=True
    )

    # Load checkpoint. The saved 'model_state_dict' is the FULL model state dict:
    # 'backbone.*' for the backbone AND 'linear_reg.*' for the pose regression head.
    # It must be loaded into `model` (not `model.backbone`), otherwise the prefixes
    # don't match and the regression head stays randomly initialized -> garbage pose.
    checkpoint = torch.load(model_snapshot_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The only expected 'missing' keys are the unused RepNeXt classifier head
    # ('backbone.head.*'); we use forward_features + our own linear_reg instead.
    if unexpected:
        print('WARNING: unexpected keys in checkpoint:', unexpected)
    print('Model weights loaded (missing keys: %s).' % missing)

    model = model.to(device)
    model.eval()

    # --- Face cropper ---
    # The crop fed to the model is a major noise source: the FaceDetection box
    # ('box') wobbles frame to frame. The iris-anchored FaceMesh cropper ('mesh')
    # centers the crop on the temporally-tracked iris midpoint and scales it by
    # the (yaw-invariant) 3D inter-ocular distance, removing that wobble.
    if args.detector == 'mesh':
        cropper = IrisFaceCropper(
            size_factor=args.crop_scale, vert_offset=args.crop_voff
        )
    else:
        cropper = BoxFaceCropper(min_detection_confidence=0.9)
    print('Face cropper: %s' % args.detector)

    # --- Temporal smoother ---
    # Per-frame predictions are independent, so the raw signal is jittery.
    # The One Euro filter smooths the rotation over time in quaternion space.
    smoother = None if args.no_filter else HeadPoseSmoother(
        min_cutoff=args.min_cutoff, beta=args.beta,
        window=args.window, window_tau=args.window_tau,
        window_beta=args.window_beta, use_window=not args.no_window,
        robust=not args.no_robust, robust_c=args.robust_c,
    )
    # Reset the filter after this many consecutive frames without a face, so a
    # re-acquired face doesn't get smoothed against a stale pose.
    miss_count = 0
    MAX_MISS_BEFORE_RESET = 5

    # --- Live residual-distribution visualizer ---
    # Tracks raw-minus-smoothed pose residuals (the jitter the filter removes)
    # and renders their distribution. Only meaningful when smoothing is on.
    _viz_on = not args.no_viz and smoother is not None
    error_viz = RealtimeErrorViz(range_deg=args.viz_range) if _viz_on else None
    axis_viz = (PerAxisErrorViz(range_deg=args.viz_range)
                if _viz_on and not args.no_axis_viz else None)

    # --- CSV logger ---
    logger = None
    if not args.no_log:
        log_path = args.log
        if not log_path:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'experimental_samples', 'demo_log_%s_%s.csv' % (args.detector, stamp)
            )
        logger = FrameLogger(
            log_path, detector=args.detector, summary_every=args.log_every,
            meta={
                'backbone': args.backbone,
                'filter': not args.no_filter,
                'min_cutoff': args.min_cutoff, 'beta': args.beta,
                'window': args.window, 'robust': not args.no_robust,
                'robust_c': args.robust_c,
                'crop_scale': args.crop_scale, 'crop_voff': args.crop_voff,
            },
        )
        print('Logging frames to %s' % logger.path)

    # --- Head-pose-normalized eye patches (for 3DeepVOG) ---
    eye_normalizer = None
    eye_writers = {}
    if args.eye_patches:
        if args.detector != 'mesh':
            print('WARNING: --eye_patches requires --detector mesh (needs '
                  'FaceMesh landmarks); eye patches disabled.')
        else:
            eye_normalizer = EyeNormalizer(
                fill=args.eye_fill, focal=(args.focal or None),
                flip_right=args.flip_right,
                smooth=not args.no_eye_smooth,
                smooth_min_cutoff=args.eye_smooth_cutoff,
                smooth_beta=args.eye_smooth_beta,
            )
            print('Eye-patch extraction ON (320x240, head-pose normalized).')
            if args.save_eyes:
                print('Saving eye MP4s to %s/' % args.save_eyes)

    # --- Capture setup (video file or live camera) ---
    is_video_file = bool(args.video)
    if is_video_file:
        cap = cv2.VideoCapture(args.video)
        # Honor phone rotation metadata so portrait clips aren't read sideways.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not cap.isOpened():
            raise IOError("Cannot open video file: %s" % args.video)
        print('Processing video file: %s (%.2f fps)' % (args.video, src_fps))
    else:
        cap = cv2.VideoCapture(cam)
        # Request the camera's maximum resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        src_fps = 30.0
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

    # Composite-canvas video writer (face + eye patches), only for file input.
    canvas_writer = None
    canvas_path = None
    if is_video_file and eye_normalizer is not None:
        base, _ = os.path.splitext(args.video)
        canvas_path = base + '_eye_canvas.mp4'

    print('Starting demo. Press ESC to quit.')

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of file is normal; a camera dropout is not.
                print("End of stream." if is_video_file
                      else "Failed to read frame from camera.")
                break

            # Timebase for the smoother/logger: derive from the video's frame
            # index so smoothing matches the clip's real time, not how fast we
            # happen to process it. Live camera stays on the wall clock.
            now = (frame_idx / src_fps) if is_video_file else time.time()
            frame_idx += 1

            # Eye patches for this frame (filled below if extraction is on).
            patches = {}

            h, w = frame.shape[:2]

            # Pristine copy of the captured frame. All model/eye-patch sampling
            # must read from this, NOT from `frame`, because we draw the pose
            # cube / iris dots onto `frame` for display and those annotations
            # would otherwise bleed into the extracted eye patches.
            frame_clean = frame.copy()

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = cropper.process(rgb)

            if face is not None:
                miss_count = 0

                cx, cy = face.center

                # Iris-anchored (or padded-box) face crop (from the clean frame).
                face_crop = face.crop(frame_clean)
                if face_crop.size == 0:
                    # Skip degenerate crops (still show the frame/canvas).
                    disp = (compose_canvas(frame, {}, (eye_normalizer.out_w,
                                                        eye_normalizer.out_h))
                            if eye_normalizer is not None else frame)
                    cv2.imshow("Demo", fit_to(disp))
                    if cv2.waitKey(5) == 27:
                        break
                    continue

                img = Image.fromarray(face_crop)
                img = img.convert('RGB')
                img = transformations(img)
                img = torch.Tensor(img[None, :]).to(device)

                # --- Head pose inference ---
                start = time.time()
                R_pred = model(img)
                end = time.time()
                infer_ms = (end - start) * 1000.
                print('Head pose estimation: %.2f ms' % infer_ms)

                # --- Temporal smoothing (in rotation/quaternion space) ---
                # Smooth the rotation matrix BEFORE converting to Euler, so the
                # filter never sees Euler wraparound / gimbal lock.
                R_mat = R_pred[0].cpu().numpy()
                if smoother is not None:
                    R_mat = smoother(R_mat, now)
                    R_used = torch.from_numpy(R_mat.astype(np.float32))[None, :]
                else:
                    R_used = R_pred

                # Rotation matrix → Euler angles (degrees)
                euler = utils.compute_euler_angles_from_rotation_matrices(R_used) * 180 / np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # --- Smoothing residual = raw pose - smoothed pose (per axis) ---
                # This is the jitter the filter removed this frame; its
                # distribution tells us what kind of noise the model produces.
                euler_raw = euler_smooth = None
                if error_viz is not None or axis_viz is not None or logger is not None:
                    euler_raw = (utils.compute_euler_angles_from_rotation_matrices(R_pred)
                                 * 180 / np.pi)[0].cpu().numpy()
                    euler_smooth = euler[0].cpu().numpy()
                    residual = wrap_deg(euler_raw - euler_smooth)  # [pitch, yaw, roll]
                    if error_viz is not None:
                        error_viz.add(residual)
                    if axis_viz is not None:
                        axis_viz.add(residual)

                if logger is not None:
                    logger.log_frame(now, face=True, infer_ms=infer_ms,
                                     raw=euler_raw, sm=euler_smooth)

                utils.plot_pose_cube(
                    frame,
                    y_pred_deg, p_pred_deg, r_pred_deg,
                    cx, cy,
                    size=face.cube
                )

                # Optionally draw the iris anchor points (mesh cropper only).
                if args.draw_iris and face.iris_l is not None:
                    cv2.circle(frame, face.iris_l, 2, (0, 255, 0), -1)
                    cv2.circle(frame, face.iris_r, 2, (0, 255, 0), -1)

                # --- Head-pose-normalized eye patches (for 3DeepVOG) ---
                # Use the SMOOTHED head rotation (R_mat) so the normalization
                # frame is as stable as the drawn pose.
                if eye_normalizer is not None and face.landmarks is not None:
                    patches = eye_normalizer.process(
                        frame_clean, face.landmarks, R_mat, now)
                    if args.save_eyes:
                        for side, patch in patches.items():
                            wri = eye_writers.get(side)
                            if wri is None:
                                os.makedirs(args.save_eyes, exist_ok=True)
                                wpath = os.path.join(args.save_eyes,
                                                     'eye_%s.mp4' % side)
                                # Stamp the source fps for files so 3DeepVOG sees
                                # the clip's real timeline, not a 30fps guess.
                                wri = cv2.VideoWriter(
                                    wpath,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    src_fps if is_video_file else args.eye_fps,
                                    (patch.shape[1], patch.shape[0]),
                                )
                                eye_writers[side] = wri
                                print('Writing %s eye -> %s' % (side, wpath))
                            wri.write(patch)
            else:
                # No face this frame; reset the filter after a sustained gap.
                miss_count += 1
                if miss_count >= MAX_MISS_BEFORE_RESET:
                    if smoother is not None:
                        smoother.reset()
                    if eye_normalizer is not None:
                        eye_normalizer.reset()
                if logger is not None:
                    logger.log_frame(now, face=False)

            # Composite: face on top, eye patches in a row below (no overlap).
            if eye_normalizer is not None:
                display = compose_canvas(
                    frame, patches, (eye_normalizer.out_w, eye_normalizer.out_h))
                # Write the full-resolution composite to the output video.
                if canvas_path is not None:
                    if canvas_writer is None:
                        canvas_writer = cv2.VideoWriter(
                            canvas_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            src_fps,
                            (display.shape[1], display.shape[0]),
                        )
                        print('Writing composite canvas -> %s' % canvas_path)
                    canvas_writer.write(display)
            else:
                display = frame

            cv2.imshow("Demo", fit_to(display))
            if error_viz is not None:
                cv2.imshow("Error Distribution", error_viz.render())
            if axis_viz is not None:
                cv2.imshow("Error per Axis", axis_viz.render())
            if cv2.waitKey(5) == 27:   # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()
    if canvas_writer is not None:
        canvas_writer.release()
        print('Composite canvas written to %s' % canvas_path)
    for wri in eye_writers.values():
        wri.release()
    if logger is not None:
        logger.close()
        print('Log written to %s' % logger.path)
    print('Demo finished.')