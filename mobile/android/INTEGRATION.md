# RayNet → 3DeepVOG — Android Integration Document

**Scope.** This is the authoritative end-to-end specification for shipping the
RayNet eye-patch pipeline on Android *in inference mode* and handing its output
to **3DeepVOG** for 3D gaze. It documents where the project stands today, the
target data flow, the contract between each stage, and the concrete work that
remains. For *build/run* mechanics see [`README.md`](./README.md); for the
head-pose model and ONNX export see [`../../sixdrepnet/README.md`](../../sixdrepnet/README.md).

- **Audience:** the Android engineer wiring the on-device pipeline + recorder.
- **Date:** 2026-06-17
- **Branch:** `feature/head_pose_fusion`

---

## 0. TL;DR

We already have a validated *desktop* pipeline (`sixdrepnet/demo.py
--eye_patches --save_eyes`) that turns a remote-webcam frame into two
**head-pose-normalized 320×240 eye patches** and feeds them to 3DeepVOG
(`3DeepVOG-main/run_cpu.py`). The Android port reproduces stages 1–4 of that
pipeline line-for-line (face landmarks → iris crop → ONNX head pose → One-Euro
smoothing → data-normalization warp) and **already produces the patches on
device**. What is left to "ship inference mode" is the **handoff**: record the
two patch streams to disk on device, then run 3DeepVOG off-device (Phase 1), and
optionally port the gaze stage on-device later (Phase 2).

```
 ┌────────────────────────── ON DEVICE (Android, done) ──────────────────────────┐
 │ CameraX → FaceLandmarker → IrisCrop → ONNX head-pose → OneEuro → EyeNormalizer │
 │                                                              → 320×240 patches  │
 └────────────────────────────────────┬───────────────────────────────────────────┘
                                       │  (RECORDER — the remaining glue)
                                       ▼
 ┌──────────────────── OFF DEVICE (desktop, validated) ───────────────────────────┐
 │ eye_left.mp4 + eye_right.mp4  →  3DeepVOG (SegResNet/SegFormer + pye3d)         │
 │                                  →  pupil segmentation + 3D eyeball + gaze       │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Progress so far

### 1.1 Head-pose net (sixdrepnet)
- **Architecture:** 6DRepNet regression head on a **RepNeXt-M4** backbone
  (replacing RepVGG) — first known RepNeXt×6DRepNet integration. 13.8 M params,
  AFLW2000 MAE ≈ 3.91° (yaw 3.68 / pitch 4.75 / roll 3.31).
- **Checkpoint:** `sixdrepnet/pretrained_models/repnext_m4/myexp_epoch_80.tar`,
  saved in **deploy (reparameterized / fused-conv) mode**.
- **Output:** continuous SO(3) rotation via Gram–Schmidt on a 6D vector.
- **ONNX mobile export:** `export_onnx.py --sixdrepnet --mobile` →
  static batch=1, **rotation-matrix-only** output `[1,3,3]` (drops the
  `euler_angles` head, removing mobile-awkward `ScatterND`/`Atan`), `onnxsim`
  1878 → ~803 nodes, emitted as a `.ort` flatbuffer with auto PyTorch↔ORT
  parity check (observed max|diff| ≈ 1e-6). **ONNX Runtime Mobile over TFLite**
  was a deliberate choice — the reparam-conv / dynamic-shape ops break common
  ONNX→TFLite converters, and one `.ort` ships to Android *and* iOS.

### 1.2 Eye-patch normalization (the part that was hard, now solved)
- The eyes are seen by a **remote** webcam, so head pose contaminates eye
  appearance and the eye does not fill the frame. We remove the head-pose
  component with the **Zhang et al. (2018) data-normalization warp**:
  `W = S · K · R_norm · K⁻¹`, where `R_norm` re-points the virtual camera along
  the eye-center viewing ray and rolls it to level with the head x-axis.
- Patches are **socket-centered** (canthi midpoint), *not* iris-centered, so
  gaze remains visible as pupil displacement inside the frame — which is exactly
  what 3DeepVOG measures.
- Output is **320×240 RGB** (W×H), `fill=0.8` (canthus-to-canthus spans 80 % of
  patch width), matching 3DeepVOG's expected input geometry.
- **Verified:** desktop `demo.py --eye_patches` produces clean, stable patches;
  jitter was traced to (a) FaceDetection box wobble → fixed with the
  iris-anchored FaceMesh crop, and (b) heavy-tailed per-frame residual → tamed
  with One-Euro (+ optional robust soft-median window) in quaternion space.

### 1.3 3DeepVOG bring-up (off-device, validated)
- Isolated venv at `sixdrepnet/3DeepVOG-main/.venv`; CPU runner `run_cpu.py`
  (SegResNet default, `--segformer` for accuracy). Validated end-to-end on the
  bundled `ES_gaze` sample **and** on our normalized patches.

### 1.4 Android scaffold (this directory)
- A faithful, **not-yet-compiled** port of stages 1–4. Every correctness-critical
  block is a line-by-line port of the Python; the CameraX/MediaPipe glue is
  standard. See §3 for the file-by-file contract and §4 for what each stage
  guarantees.

---

## 2. The target pipeline (end to end)

| # | Stage | Desktop (`demo.py`) | Android | Output contract |
|---|-------|---------------------|---------|-----------------|
| 1 | Face landmarks | MediaPipe FaceMesh | `FaceLandmarkerHelper` (`face_landmarker.task`) | 478 landmarks incl. iris (468–477), **pixel** units `[x, y, z·w]` |
| 2 | Face crop | `IrisFaceCropper` | `FaceCropper` | square crop, side = `4·IOD₃D`, center `+0.5·IOD` below eye line |
| 3 | Head pose | `SixDRepNet_RepNeXt` (PyTorch) | `HeadPoseEstimator` (ONNX RT Mobile) | 3×3 head→camera rotation `R` |
| 4a | Pose smoothing | `HeadPoseSmoother` | `HeadPoseSmoother` (`OneEuro.kt`) | smoothed `R` (quaternion-space One-Euro) |
| 4b | Patch warp + anchor smoothing | `EyeNormalizer` | `EyeNormalizer` | dict `{right, left}` → 320×240 RGB patch |
| 5 | **Sink** | `--save_eyes` → `eye_{side}.mp4` | **TODO recorder** (§5) | per-eye MP4 + timing metadata |
| 6 | Gaze | `run_cpu.py` (off-device) | off-device (Phase 1) / on-device (Phase 2) | pupil seg + 3D eyeball + gaze |

Only **stage 3** crosses ONNX. Stages 1, 4, 5 are native Android (MediaPipe,
linear algebra, MediaCodec). Stage 6 is, for now, desktop Python.

---

## 3. Android file-by-file contract

| File | Role | Ported from | Parity notes |
|------|------|-------------|--------------|
| `LinAlg.kt` | `Vec3`/`Mat3` in **Double** (matches NumPy) | — | keep Double; Float drifts the warp |
| `OneEuro.kt` | scalar + quaternion One-Euro, rotmat↔quat | `filters.py` | `minCutoff=1.0, beta=0.3`; smooths in quaternion space (no Euler wraparound) |
| `FaceLandmarkerHelper.kt` | MediaPipe Face Landmarker, IMAGE mode | desktop FaceMesh | returns `[x·w, y·h, z·w]` — `z` scaled by **width**, matching `face_crop.py` |
| `FaceCropper.kt` | iris-anchored square crop | `face_crop.py` IrisFaceCropper | side = `sizeFactor·IOD₃D` (4.0), `vertOffset·IOD` (0.5); IOD from **3D** iris means → yaw-invariant |
| `HeadPoseEstimator.kt` | ORT `.ort` session + preprocessing | `demo.py` transforms | Resize(224)→CenterCrop(224)→ToTensor→ImageNet norm, NCHW; CenterCrop is a no-op because the crop is already square |
| `EyeNormalizer.kt` | data-norm warp + anchor smoothing | `eye_norm.py` | `W = S·K·R_norm·K⁻¹`; **Android `Matrix` applies W as the same source→dest forward map as `cv2.warpPerspective`** — no transpose/inverse juggling |
| `EyePatchPipeline.kt` | glue + composite canvas | `demo.py` main loop + `compose_canvas` | reset filter + normalizer after `maxMissBeforeReset=5` faceless frames |
| `MainActivity.kt` | CameraX + permission + display | `demo.py` capture loop | `STRATEGY_KEEP_ONLY_LATEST`, single bg thread, `elapsedRealtimeNanos·1e-9` timebase |

### Critical invariants (do not "optimize" away)
1. **All sampling reads the clean frame.** The composite draws the crop box on a
   *copy*; never warp from an annotated frame or the box bleeds into patches
   (desktop uses `frame_clean`).
2. **`R` used for the warp is the *smoothed* rotation**, so the normalization
   frame is as stable as the drawn pose (`demo.py` passes `R_mat`, not `R_pred`).
3. **Anchors are smoothed, geometry is exact.** One-Euro is applied to the four
   canthi *points* (33/133/362/263), then center/scale are recomputed — so both
   translation and scale settle while the warp stays geometrically correct.
4. **Eye-slot order is `right | left`** = subject's right/left. The subject's
   right eye is on the image-left.

---

## 4. The 3DeepVOG handoff (the contract that matters)

3DeepVOG segments the pupil/iris and fits a 3D eyeball from video **where a
single eye fills the frame, as if shot by a fixed head-mounted eye camera**. Our
normalizer manufactures exactly that virtual view from a remote camera.

### 4.1 What 3DeepVOG consumes
- **Input:** a video where each frame is one eye, **240×320 (H×W) RGB**. Our
  patches are 320×240 (W×H) → same pixels, just stated W×H. (Confirm the
  segmentation net's axis order on first integration; the desktop `ES_gaze`
  sample is `320x240_pad`.)
- **One stream per eye:** `eye_left.mp4`, `eye_right.mp4` (separate fits).
- **Timeline:** fps stamped into the file so pye3d sees real time, not a 30 fps
  guess. On device, stamp the camera's actual frame timestamps.
- **Runner args (`run_cpu.py`):** `SegResNet_3in3out` (fast) or
  `SegFormerB0_3in3out` (`--segformer`, accurate); `gaze_tracking_flag=True`;
  `mode='auto'` (fit eyeball, then predict); writes `seg_overlay.mp4` +
  fit video (headless-safe, no vispy/display needed).

### 4.2 The calibration caveat (read before trusting absolute 3D)
- pye3d's `focal_length=16.0` and `sensor_size=(4.8, 3.6)` are **physical
  eye-camera intrinsics**. Our patches come from a **normalized *virtual*
  camera**, so these are placeholders. Consequences:
  - **2D pupil segmentation and *relative* gaze direction are valid.**
  - **Absolute 3D eyeball radius / metric scale / refraction correction are
    NOT physically calibrated.** Fine for a first look; do not report absolute
    metric gaze without deriving the virtual camera's effective intrinsics from
    the normalization (`K`, `fill`, patch size) and feeding those instead.
- **Roll → torsion coupling.** Roll is taken from the head-pose model, so head
  roll *error* maps directly to apparent ocular **torsion** in the patch. If
  torsion tracking (`--torsion`) looks biased, suspect roll, not the eye.

### 4.3 Two integration phases

**Phase 1 — record on device, gaze off device (ship this first).**
Android writes the two patch streams; 3DeepVOG runs on a workstation. This is the
exact desktop `--save_eyes` → `run_cpu.py` path, just with the patches produced
on the phone. Lowest risk, unblocks data collection immediately. **This is the
"ship inference mode" milestone.**

**Phase 2 — on-device gaze (later).** Port 3DeepVOG's two stages to mobile:
(a) the segmentation net (SegResNet → ONNX RT Mobile, 240×320×3 in / 3-channel
mask out — **already exported**, see below), and (b) the pye3d 3D eyeball fit +
gaze + pupil radius (the hard part). The full design — both eyes, annotation
rendering incl. pupil radius, and the tier-3 (3D eye model) port-strategy
decision — is in **[`THREEDEEPVOG_DEPLOYMENT.md`](./THREEDEEPVOG_DEPLOYMENT.md)**.
Tier 1 (segmentation `.ort`) is done and parity-checked.

---

## 5. The remaining work — on-device recorder (Phase 1)

The Android demo currently **only displays** the composite. To feed 3DeepVOG it
must persist the two raw 320×240 patches plus timing. This is the single missing
piece for inference-mode shipping.

### 5.1 Design
- Add an `EyePatchRecorder` driven by `EyePatchPipeline.process()` — it already
  has `patches["right"] / patches["left"]` before they go into the composite.
- **Two `MediaCodec` H.264 encoders + two `MediaMuxer`s** (or one muxer with two
  video tracks), each fed the per-eye `Bitmap` → `Surface`/`Image`. 320×240 is
  tiny; CBR at a high bitrate keeps pupil edges crisp for segmentation.
- **Stamp real timestamps.** Use the same monotonic clock the pipeline uses
  (`SystemClock.elapsedRealtimeNanos`) as the per-frame presentation time, and
  record the nominal fps in a sidecar so 3DeepVOG's timeline is honest.
- **Handle dropped eyes.** When a warp is degenerate an eye is omitted that
  frame. Either (a) repeat the previous patch to keep both streams frame-aligned
  (simplest for 3DeepVOG), or (b) write a per-frame validity sidecar. Prefer (a)
  for the first cut; document the choice.
- **Lifecycle:** start on a "record" toggle, flush/stop on pause/destroy, name
  files `eye_left.mp4` / `eye_right.mp4` under app-scoped storage; expose via
  share/`adb pull`.

### 5.2 Recommended encoding params
| Param | Value | Why |
|-------|-------|-----|
| Resolution | 320×240 | 3DeepVOG input; do not rescale |
| Codec | H.264 (AVC), baseline/main | universal `MediaCodec` support |
| Rate control | CBR, ~4–6 Mbps | tiny frame; over-allocate to preserve pupil edges |
| Color | full-range RGB→YUV | avoid limited-range crush on the iris |
| Keyframe interval | 1 s | seekability for analysis |

> ⚠️ **Lossy compression vs. segmentation.** If SegResNet/SegFormer accuracy
> degrades on H.264 patches, fall back to **lossless PNG frame dumps** (or a
> lossless codec) for capture sessions that feed the eyeball *fit*; use H.264
> only for long prediction runs. Validate on the first real recording.

### 5.3 Acceptance test for Phase 1
1. Record ~20 s on a device, both eyes, subject sweeping gaze L/R/U/D.
2. `adb pull` the two MP4s to the workstation.
3. `cd sixdrepnet/3DeepVOG-main && .venv/bin/python run_cpu.py /path/eye_left.mp4`
   (then `eye_right.mp4`); add `--segformer` for the accuracy check.
4. **Pass = ** `seg_overlay.mp4` shows the pupil tracked through the sweep and the
   fit produces a stable eyeball; gaze follows the sweep direction. Compare
   against the same subject captured with desktop `demo.py --save_eyes` as the
   reference.

---

## 6. Open items / risks

- [ ] **Recorder** (§5) — the gating task for inference-mode shipping.
- [ ] **Compile the scaffold** — never built in this repo; expect minor
  MediaPipe/CameraX/ORT version tweaks (see README §6).
- [ ] **Front-camera mirroring** — front frames may be mirrored vs. the desktop
  webcam; pose/warp still work, but flip in `toUprightBitmap()` if you want the
  desktop's handedness, and keep `flipRight` consistent with whatever 3DeepVOG
  expects.
- [ ] **Robust soft-median window** (`WeightedWindowQuaternion`) is omitted on
  device — One-Euro alone removes most jitter; port it if residual spikes appear
  on real hardware.
- [ ] **Virtual-camera intrinsics** (§4.2) — derive real `focal_length`/
  `sensor_size` equivalents from `K`/`fill`/patch size before any absolute 3D
  claim.
- [ ] **Phase 2 on-device gaze** — only after Phase 1 confirms patch quality.

---

## 7. References
- Zhang et al., *Revisiting Data Normalization for Appearance-Based Gaze
  Estimation*, 2018 — the warp in `eye_norm.py`.
- 3DeepVOG — <https://github.com/DSGZ-MotionLab/3DeepVOG> (segmentation + pye3d).
- 6DRepNet — <https://github.com/thohemp/6DRepNet>; RepNeXt —
  <https://github.com/suous/RepNeXt>.
- In-repo: `sixdrepnet/eye_norm.py`, `sixdrepnet/demo.py`,
  `sixdrepnet/3DeepVOG-main/run_cpu.py`, `export_onnx.py`.