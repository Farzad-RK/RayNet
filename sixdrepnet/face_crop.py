"""
Face cropping front-ends for the head-pose demo.

The head-pose model is fed a face crop. Where that crop comes from matters: a
jittery crop makes the per-frame pose jitter even when the model is stable.

Two croppers share one interface (`process(rgb) -> FaceCrop | None`):

  * BoxFaceCropper  — the original path: MediaPipe FaceDetection bounding box,
                      padded. The detection box wobbles frame to frame, which is
                      a real source of pose noise.

  * IrisFaceCropper — MediaPipe FaceMesh with iris refinement. The two iris
                      centers are a sub-pixel-stable, temporally-tracked
                      geometric anchor. The crop is centered on the iris midpoint
                      and sized from the *3D* inter-ocular distance (which is
                      nearly yaw-invariant, unlike the 2D distance that
                      foreshortens as the head turns). This removes the detector
                      wobble from the crop.

The crop is kept square and fully inside the frame (the center is shifted inward
rather than the box shrunk) so the downstream Resize/CenterCrop never distorts
the aspect ratio or changes framing near the image edges.
"""

import numpy as np
import cv2
import mediapipe as mp


# MediaPipe FaceMesh iris landmarks (present only with refine_landmarks=True).
# Each iris is a center point followed by a 4-point ring; averaging all five is
# steadier than the single center index.
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]


class FaceCrop:
    """Result of a cropper: a square crop box plus drawing anchors."""

    __slots__ = ('box', 'center', 'cube', 'iris_l', 'iris_r', 'landmarks')

    def __init__(self, box, center, cube, iris_l=None, iris_r=None,
                 landmarks=None):
        self.box = box          # (x0, y0, x1, y1) int, clamped to frame
        self.center = center    # (cx, cy) int — where to draw the pose cube
        self.cube = int(cube)   # pose-cube size in px
        self.iris_l = iris_l    # (x, y) int or None
        self.iris_r = iris_r    # (x, y) int or None
        # Full FaceMesh landmarks as an (N, 3) float array in PIXEL units
        # (x*w, y*h, z*w); None for the box cropper. Used by eye normalization.
        self.landmarks = landmarks

    def crop(self, frame):
        x0, y0, x1, y1 = self.box
        return frame[y0:y1, x0:x1]


def _square_box_in_frame(cx, cy, half, w, h):
    """Largest square of given half-size centered near (cx, cy), kept in frame.

    The center is clamped inward (not the size shrunk) whenever possible, so the
    crop stays square and the framing is preserved at the image edges.
    """
    half = min(half, w / 2.0, h / 2.0)
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    x0 = int(round(cx - half)); x1 = int(round(cx + half))
    y0 = int(round(cy - half)); y1 = int(round(cy + half))
    return x0, y0, x1, y1


class IrisFaceCropper:
    """
    Iris-anchored face cropper built on MediaPipe FaceMesh (refine_landmarks).

    Args:
        size_factor:  crop side length as a multiple of the inter-ocular
                      distance (IOD). ~4.0 frames a typical face (forehead→chin).
        vert_offset:  how far below the iris midpoint to place the crop center,
                      in IOD units (the eyes sit above the face center).
        min_detection_confidence / min_tracking_confidence: FaceMesh thresholds.
        static_image_mode: False enables FaceMesh's internal frame-to-frame
                      tracking — keep it False for video, it is much steadier.
    """

    def __init__(self, size_factor=4.0, vert_offset=0.5,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 static_image_mode=False):
        self.size_factor = float(size_factor)
        self.vert_offset = float(vert_offset)
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,   # required for the iris landmarks (468-477)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb):
        h, w = rgb.shape[:2]
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        # 2D (pixel) and 3D iris centers. MediaPipe's z is in roughly the same
        # units as x, so scale it by image width to get comparable pixels.
        def center2(idx):
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
            return pts.mean(axis=0)

        def center3(idx):
            pts = np.array([[lm[i].x * w, lm[i].y * h, lm[i].z * w] for i in idx])
            return pts.mean(axis=0)

        l2, r2 = center2(_LEFT_IRIS), center2(_RIGHT_IRIS)
        l3, r3 = center3(_LEFT_IRIS), center3(_RIGHT_IRIS)

        # Full landmark set in pixel units, for downstream eye normalization.
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm],
                       dtype=np.float64)

        # 3D inter-ocular distance: nearly constant under head rotation, so the
        # crop scale does not zoom in as the head turns (unlike 2D IOD, which
        # foreshortens). Floor it to avoid division blow-ups on a lost face.
        iod = max(float(np.linalg.norm(l3 - r3)), 1e-3)

        mid = 0.5 * (l2 + r2)
        cx = float(mid[0])
        cy = float(mid[1]) + self.vert_offset * iod  # +y is down → face center
        half = 0.5 * self.size_factor * iod

        box = _square_box_in_frame(cx, cy, half, w, h)
        # Cube ≈ face width ≈ 2·IOD; 0.9·(2·IOD) sits snugly on the head.
        cube = 1.8 * iod
        return FaceCrop(
            box,
            (int(round(cx)), int(round(cy))),
            cube,
            iris_l=(int(round(l2[0])), int(round(l2[1]))),
            iris_r=(int(round(r2[0])), int(round(r2[1]))),
            landmarks=pts,
        )


class BoxFaceCropper:
    """Original path: padded MediaPipe FaceDetection bounding box."""

    def __init__(self, min_detection_confidence=0.9, model_selection=0,
                 pad=0.2):
        self.pad = float(pad)
        self._det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def process(self, rgb):
        h, w = rgb.shape[:2]
        res = self._det.process(rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.score[0])
        b = best.location_data.relative_bounding_box

        x0 = int(b.xmin * w); y0 = int(b.ymin * h)
        x1 = int((b.xmin + b.width) * w); y1 = int((b.ymin + b.height) * h)
        bw, bh = x1 - x0, y1 - y0

        cx = x0 + bw // 2
        cy = y0 + int(bh * 0.35)  # eye level, not geometric center

        x0 = max(0, x0 - int(self.pad * bw)); y0 = max(0, y0 - int(self.pad * bh))
        x1 = min(w, x1 + int(self.pad * bw)); y1 = min(h, y1 + int(self.pad * bh))
        return FaceCrop((x0, y0, x1, y1), (cx, cy), int(bw * 0.6))
