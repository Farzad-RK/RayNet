# CV & Portfolio Descriptions

Two related projects from the **GazeToolKit** work: a mobile head-pose model + on-device
eye-tracking Android app, and **RayNet**, the multi-task gaze/landmark/pupillometry network.
Each has a concise **CV** version and a longer **personal-website** version.

---

## Project 1 — On-Device Eye Tracking (Android)

### CV version

**On-Device Gaze & Pupillometry — Android App (Kotlin, ONNX/TFLite, MediaPipe)**

- Re-engineered 6DRepNet head-pose estimation on a **RepNeXt-M4 backbone**, reaching
  **3.91° MAE on AFLW2000** — better than both 6DRepNet baselines (RepVGG-B1 4.05°,
  ResNet50 3.97°) at **13.8M params, ~32% smaller than RepVGG-B1 and ~half the ResNet50 variant**.
- Built a real-time on-device pipeline: face landmarks → head-pose-normalized eye patches →
  ported **3DeepVOG** (segmentation → ellipse fit → 3D eyeball model) for gaze and metric
  pupil diameter.
- Reimplemented the full 3D eye geometry in Kotlin and exported the neural components to
  **GPU-accelerated TFLite**, reaching real-time tracking on a commodity phone.
- Built a synthetic ground-truth parity harness that caught a geometry sign bug, cutting gaze
  error from **73° to ~1e-4°**.

### Website version

**Real-time eye tracking on a phone.** I built an Android app that estimates head pose, gaze
direction, and pupil size directly on-device from the front camera — no server, no
head-mounted hardware. At its front end is a head-pose model I re-engineered by rebuilding
6DRepNet on a mobile-optimized RepNeXt-M4 backbone; it reaches **3.91° mean error on AFLW2000
— more accurate than both original 6DRepNet backbones (RepVGG-B1 and ResNet50) while using
13.8M parameters, roughly half the size of the ResNet50 variant**. The head pose is used to
frontalize and crop each eye, after which a fully on-device port of the 3DeepVOG pipeline
(segmentation, ellipse fitting, and a 3D eyeball model) recovers gaze and millimetric pupil
diameter. Getting classical 3D eye geometry to run accurately in real time on a phone meant
reimplementing the math in Kotlin, exporting the neural components to GPU-accelerated TFLite,
and building a parity test harness that caught a subtle geometric sign error — collapsing gaze
error from 73° to effectively zero.

---

## Project 2 — RayNet

### CV version

**RayNet — Multi-Task Gaze, Landmark & Pupillometry Network (PyTorch)**

- Designed a lightweight multi-task model for **3D gaze, iris/pupil landmarks, and metric
  pupillometry** on the multi-view GazeGene dataset (RepNeXt backbone + PANet neck).
- Predicts the **optical axis** with calibratable kappa handling; regresses landmarks via
  **heatmaps + soft-argmax** for sub-pixel accuracy.
- Recovers **pupil diameter (~0.03 mm)** and screen gaze analytically through projective
  geometry — no extra learned heads.
- Resolved adversarial multi-task training with a **gradient-isolated head-pose backbone**
  (6D rotation + 3D translation).

### Website version

**RayNet — a compact network for gaze and pupillometry.** RayNet is a multi-task vision model
that, from a single normalized eye image, predicts where a person is looking, locates the iris
and pupil to sub-pixel precision, and measures pupil diameter in millimeters — accurate enough
for cognitive-load analysis. Rather than learning everything end-to-end, it predicts the eye's
optical axis and precise landmarks, then recovers depth, true gaze (with calibratable kappa
correction), and screen position through projective geometry. The design favors efficiency — a
mobile-friendly RepNeXt backbone with multi-scale feature fusion — so it can run toward
real-time on edge devices, and its later revisions tackle the harder multi-task training
dynamics with a dedicated, gradient-isolated head-pose branch.

---

## Supporting facts & sources

### Head-pose benchmark — AFLW2000 (from `sixdrepnet/README.md` §8.1)

| Model | MAE (°) | Params (M) |
|-------|---------|------------|
| **RepNeXt-M4 (this work)** | **3.91** | **13.8** |
| 6DRepNet (ResNet50) | 3.97 | 25.6 |
| 6DRepNet (RepVGG-B1) | 4.05 | 20.3 |

- Param reduction: 13.8M vs RepVGG-B1 20.3M = **~32% fewer**; vs ResNet50 25.6M = **~46% fewer (≈ half)**.
- "Roughly half the parameters" is accurate **only against the ResNet50 variant** — do not
  claim it against RepVGG-B1.

### Claims that are accurate as stated

- RepNeXt-M4 has **lower MAE than both** 6DRepNet baselines (3.91 < 3.97 < 4.05).
- Gaze sign bug fix: **73° → ~1e-4°** (validated via synthetic ground-truth parity harness).

### Claims to verify before quoting as *achieved* results

- RayNet's "~0.03 mm" pupil precision and the README's accuracy/latency figures are **design
  targets**, not confirmed measurements — confirm against final training results before
  presenting them as results.
- "Real-time" on the Android app was ~4 fps tracking / ~6–7 fps idle on the test device;
  phrase as "real-time on a commodity phone" rather than a specific high frame rate.

### Key technologies

- **Mobile/Android:** Kotlin, ONNX Runtime, TFLite (GPU delegate), MediaPipe FaceLandmarker, CameraX, IMU sensor fusion.
- **Models / methods:** 6DRepNet (6D rotation + geodesic loss), RepNeXt-M4, SegResNet, 3DeepVOG, Zhang 2018 data normalization, Safaee-Rad conic unprojection.
- **RayNet:** PyTorch, RepNeXt + PANet, coordinate attention, heatmap + soft-argmax landmarks, GazeGene dataset.
