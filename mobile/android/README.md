# RayNet Eye-Patch — Android demo (ONNX Runtime Mobile)

The simplest working on-device port of the desktop eye-patch pipeline
(`sixdrepnet/demo.py`). Front camera → face landmarks → head pose (ONNX) →
temporal smoothing → head-pose-normalized eye patches → on-screen composite.

> **Status:** this is a reference scaffold written to be faithful to the Python
> pipeline. It was **not compiled in this repo** (no Android toolchain here) —
> open it in Android Studio, drop in the two model files (below), and build. The
> correctness-critical logic (the warp, the One Euro filter, the crop geometry,
> the preprocessing) is a line-by-line port and is the part that matters; the
> CameraX/MediaPipe glue is standard and may need minor version tweaks.

> **Integration spec:** for the end-to-end pipeline, the 3DeepVOG handoff
> contract, and the remaining inference-mode work (on-device patch recorder),
> see [`INTEGRATION.md`](./INTEGRATION.md). This README is the build/run guide.

---

## 1. Architecture

Three stages — only the **head-pose net** is ONNX. (We chose ONNX Runtime Mobile
over TFLite because the graph's `ScatterND`/dynamic-shape/reparam-conv ops are
exactly what ONNX→TFLite converters break on, and one `.ort` ships to Android
*and* iOS.)

```
 CameraX frame (upright Bitmap)
        │
        ▼
 ┌──────────────────────┐   MediaPipe Face Landmarker (TFLite, native)
 │ FaceLandmarkerHelper │   → 478 landmarks incl. iris (468–477)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   iris-anchored square crop, side = 4·IOD(3D)
 │      FaceCropper      │   (port of face_crop.py IrisFaceCropper)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   Resize224 + ImageNet-norm → ONNX Runtime Mobile
 │   HeadPoseEstimator   │   → 3×3 rotation matrix   (head_pose_*.ort)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   One Euro in quaternion space
 │   HeadPoseSmoother    │   (port of filters.py)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   Zhang data-norm homography warp, socket-centered,
 │     EyeNormalizer     │   anchors One-Euro-smoothed (port of eye_norm.py)
 └──────────┬───────────┘   warp via android.graphics.Matrix + Canvas (no OpenCV)
            ▼
   Composite: face on top, "right | left" patches below  → ImageView
```

## 2. File map

| File | Role | Ported from |
|------|------|-------------|
| `LinAlg.kt` | Vec3 / Mat3 (Double, to match NumPy) | — |
| `OneEuro.kt` | One Euro scalar + quaternion + rotmat↔quat | `filters.py` |
| `FaceCropper.kt` | iris-anchored square crop geometry | `face_crop.py` |
| `HeadPoseEstimator.kt` | ORT `.ort` session + preprocessing | `demo.py` transforms |
| `EyeNormalizer.kt` | data-norm warp + anchor smoothing | `eye_norm.py` |
| `FaceLandmarkerHelper.kt` | MediaPipe Face Landmarker wrapper | `face_crop.py` (MediaPipe FaceMesh) |
| `EyePatchPipeline.kt` | glue + composite canvas | `demo.py` main loop + `compose_canvas` |
| `MainActivity.kt` | CameraX + permission + display | `demo.py` capture loop |

## 3. Get the two model files

Both go in **`app/src/main/assets/`** (create the folder).

**(a) Head-pose model — `head_pose_repnext_m4_mobile.ort`**
Generate it from the repo root:

```bash
pip install onnx onnxruntime onnxscript onnxsim   # one-time
python export_onnx.py --sixdrepnet --mobile \
  --weights sixdrepnet/pretrained_models/repnext_m4/myexp_epoch_80.tar \
  --backbone repnext_m4 \
  --output sixdrepnet/pretrained_models/repnext_m4/head_pose_repnext_m4_mobile.onnx
# → writes head_pose_repnext_m4_mobile.ort  (copy it into app/src/main/assets/)
```

This runs a PyTorch↔ONNX parity check before emitting the `.ort` (expect
max|diff| ≈ 1e-6). See `../../sixdrepnet/README.md` → "For ONNX Export".

**(b) Face landmarks — `face_landmarker.task`**
Download MediaPipe's bundle (includes the iris model):

```bash
curl -L -o app/src/main/assets/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

> If you rename either asset, update the `modelAsset` default in
> `HeadPoseEstimator.kt` / `FaceLandmarkerHelper.kt`.

## 4. Build & run

1. **Android Studio** (Hedgehog or newer) → *Open* → select `mobile/android/`.
   It will sync Gradle and offer to set up the Gradle wrapper (accept). No
   `gradlew` is committed; let Studio generate it, or run `gradle wrapper`.
2. Confirm both assets are in `app/src/main/assets/`.
3. Plug in a device (API 24+), press **Run**. Grant the camera permission.
4. You should see your face with the green crop box, and the two normalized eye
   patches in a `right | left` row beneath it.

## 5. Faithfulness & known simplifications

Kept identical to desktop (these drive correctness):
- **Preprocessing**: Resize→224, ImageNet mean/std, NCHW — matches `demo.py`.
- **Crop**: iris midpoint, side = `4·IOD(3D)`, `+0.5·IOD` vertical offset.
- **Warp**: same `W = S · K · R_norm · K⁻¹` homography; `android.graphics.Matrix`
  applies it as a source→dest forward map, exactly like `cv2.warpPerspective`.
- **Smoothing**: One Euro in quaternion space (pose) and on the canthi (patches),
  `min_cutoff=1.0, beta=0.3`.

Deliberately simplified for "simplest demo":
- **Robust soft-median window** (`WeightedWindowQuaternion`) is **omitted** — One
  Euro alone removes most jitter. Port it next if you see residual spikes.
- **MediaPipe runs in synchronous `IMAGE` mode** inside the analyzer. For higher
  FPS switch to `LIVE_STREAM` (async callback).
- **No eye-video recording / 3DeepVOG export** — display only. Add a
  `MediaMuxer`/`ImageWriter` on the per-eye patches to reproduce `--save_eyes`.
- **Front camera** frames may be mirrored vs. the desktop webcam; pose/warp still
  work, but flip in `toUprightBitmap()` if you want the desktop's handedness.

## 6. Things to verify in Android Studio (couldn't be run here)

- **ORT `.ort` load**: `HeadPoseEstimator` sets
  `addConfigEntry("session.load_model_format", "ORT")`. Recent ORT auto-detects
  the format from the bytes; if the config key is rejected on your ORT version,
  delete that line.
- **Output name/shape**: the model has a single output `rotation_matrix`
  `[1,3,3]`; we read `result[0]`. Confirm with `session.outputNames`.
- **`ImageProxy.toBitmap()`** requires `camera-core ≥ 1.3.0` (pinned 1.3.4).
- **Asset compression**: `.ort`/`.task` are excluded from compression in
  `build.gradle.kts` (`noCompress`) — needed for MediaPipe's mmap.

## 7. Optional: hardware acceleration

Add the NNAPI execution provider in `HeadPoseEstimator` for a speed-up on
supported devices:

```kotlin
val opts = OrtSession.SessionOptions().apply {
    addConfigEntry("session.load_model_format", "ORT")
    addNnapi()   // falls back to CPU for unsupported ops
}
```
Measure both — NNAPI isn't always faster for small conv nets.
