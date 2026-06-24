# On-Device 3DeepVOG (Phase 2) — Android Deployment Design

**Goal.** Run the 3DeepVOG gaze stage *on device* — segmentation → ellipse fit →
3D eyeball model → **gaze + pupil radius** — for **both eyes**, and render the
annotations (pupil/iris ellipses, gaze vector, eyeball center, **pupil radius**)
live on the Android screen. This is the Phase 2 referenced in
[`INTEGRATION.md`](./INTEGRATION.md) §4.3.

> **TL;DR of the hard truth.** 3DeepVOG is *not* a single network like the head
> pose. It is **one CNN + a chunk of classical geometry + a 3D eye-model fit**.
> The CNN (segmentation) ports the same way the head pose did and is **already
> exported** (§1). The 3D eye model is **pye3d** (Pupil Labs, Cython/C++), and
> how we get it onto the device is a real decision with weeks of difference
> between options — see §5. **That decision is open and needs your call.**

---

## 1. Status — what is already done

**Tier 1 (segmentation) is shipped, the same way as the head pose.** A new
`--segvog` path in `export_onnx.py` exports the SegResNet backbone to a
mobile `.ort`, with the automatic PyTorch↔ORT parity check:

```bash
# run from repo root (needs torch + monai + onnx + onnxsim + onnxruntime)
python export_onnx.py --segvog \
  --weights sixdrepnet/3DeepVOG-main/threedeepvog/models/SegResNet_weights.pth \
  --output  sixdrepnet/pretrained_models/segvog/seg_segresnet_240x320_mobile.onnx
```

Produced & verified (2026-06-17):

| Artifact | Detail |
|----------|--------|
| `seg_segresnet_240x320_mobile.ort` | 6.1 MB, 89 nodes (onnxsim 95→89) |
| Parity | max\|torch−onnx\| = **8.5e-6** (atol 1e-4) ✅ |
| Input `input` | `(1,3,240,320)` float32 |
| Output `seg` | `(1,3,240,320)` float32 **post-sigmoid** [0,1]; ch 0=pupil, 1=iris, 2=sclera |

> **Why SegResNet, not SegFormer.** SegResNet is conv-only (MONAI ResNet, ~6 MB)
> and exports cleanly with mobile-safe ops. SegFormer-B0 is a transformer
> (15 MB, attention/LayerNorm ops that are heavier and more converter-fragile).
> Ship SegResNet; only revisit SegFormer if the seg accuracy gap proves to
> matter on real device patches.

**Input contract the Kotlin side must reproduce** (it mirrors
`Model_3DeepVOG.predict` in `deepvog3d_model.py`): take the 320×240 eye patch →
**grayscale** → **min-max scale to [0,1]** (MONAI `ScaleIntensity`, *per-image*,
not /255) → **repeat the 1 channel to 3** (`Gray2Rgb`) → NCHW. The sigmoid is
already baked into the graph, so on device you just **threshold at 0.5**.

**Tier-3 strategy is decided: Option A (classic geometry, on-device)** — see §5.
Kotlin scaffolding now in the repo (`app/src/main/java/com/raynet/eyepatch/`):

| File | State | What |
|------|-------|------|
| `EyeSegmenter.kt` | ✅ written | `.ort` session + exact preprocessing (gray → per-image min-max → 3ch → NCHW); returns pupil/iris/sclera planes |
| `EyeGeometry.kt` | ✅ written | direct port of Safaee-Rad unprojection + LSQ line-intersection + line–sphere; NaN-degeneracy guarded |
| `EyeModel3D.kt` | ✅ written | per-eye fit (eyeball center+radius) + predict (gaze + **pupil radius**); disambiguation reconstructed from DeepVOG |

A **golden-value parity harness** exists at
`sixdrepnet/3DeepVOG-main/tools/geom_parity.py` (+ `geom_golden.json`): it runs
the Python reference unprojection on canonical ellipses so `EyeGeometry.unproject`
can be checked input-for-input. (It also confirmed the centered/axis-aligned
pupil is a genuine reference degeneracy → NaN → skip that frame.)

The remaining tiers (ellipse fit, both-eyes orchestration, rendering, wiring) are
**design + remaining work** below.

---

## 2. The real pipeline (per eye)

```
 320x240 normalized eye patch  (from the Phase-1 EyeNormalizer — already on device)
        │
        ▼  gray → min-max [0,1] → repeat to 3ch → NCHW
 ┌────────────────────┐  ONNX Runtime Mobile (.ort, DONE)
 │  Seg (SegResNet)   │  → (3,240,320) probs: pupil / iris / sclera
 └─────────┬──────────┘
           ▼  threshold 0.5 + morphological open (erosion→dilation, 3x3)
 ┌────────────────────┐  classical CV  (port of PostProcessing.py)
 │  Post-processing   │
 └─────────┬──────────┘
           ▼  perimeter → fitEllipse(pupil), fitEllipse(iris) + blink heuristic
 ┌────────────────────┐  OpenCV-Android Imgproc.fitEllipse (port of EllipseFitting.py)
 │   Ellipse fit      │  → pupil ellipse (cx,cy,a,b,θ,conf), iris ellipse, blink
 └─────────┬──────────┘
           ▼  unproject ellipse → 3D pupil circle on the fitted eyeball
 ┌────────────────────┐  ★ pye3d / classic geometry — THE OPEN DECISION (§5)
 │  3D eye model +    │  → gaze direction, eyeball center,
 │  gaze + pupil r    │     ★ PUPIL RADIUS (circle_3d.radius, mm) + diameter (px)
 └─────────┬──────────┘
           ▼
 ┌────────────────────┐  Canvas overlay (port of ParamsRender.py / realtime_gaze.draw_gaze)
 │  Annotation render │  ellipses + gaze arrow + eye center + pupil-radius text
 └────────────────────┘
```

**Where pupil radius actually comes from.** It is an *output of the 3D eye
model*, not the ellipse: pye3d returns `circle_3d.radius` (the unprojected pupil
circle radius in **mm**) and `diameter_3d = 2·radius`; the 2D `diameter` is the
projected ellipse major axis in **px**. So **rendering pupil radius requires
tier 3** — the 2D ellipse alone gives an apparent (foreshortened) size, not the
true circle radius. (We can render the 2D apparent radius cheaply from the
ellipse as an interim, clearly labeled as uncalibrated.)

---

## 3. Both eyes

The desktop runs **one eye per video**. On device we run two independent
instances of the whole tier-2/3 stack:

- **Segmentation:** batch the two patches as `(2,3,240,320)` in a single
  `.ort` run (re-export with batch=2, or loop the batch-1 graph twice — measure
  both; one conv pass over 2 tiny images is cheap).
- **Per-eye state is separate:** each eye has its own ellipse history, its own
  3D eyeball model (`model_params` — different center/radius), and its own
  gaze/pupil-radius output. Keep a `EyeTracker` object per side (`left`/`right`).
- **The Phase-1 `EyeNormalizer` already emits both** `patches["left"]` and
  `patches["right"]`; tier 2/3 just consumes each.
- **Render:** reuse the existing `right | left` composite row; overlay each
  eye's annotations on its own patch.

---

## 4. Annotation rendering (incl. pupil radius)

Port of `ParamsRender.overlay_fit_model_cv` + `realtime_gaze.draw_gaze`, drawn
with `Canvas`/`Paint` on each 320×240 patch (no OpenCV needed for drawing):

| Annotation | Source | Notes |
|------------|--------|-------|
| Pupil ellipse | tier-2 ellipse | green outline |
| Iris ellipse | tier-2 ellipse | magenta outline |
| Gaze vector | tier-3 `gaze` (x,y proj) | arrow from pupil center, len ∝ angle |
| Eyeball center | tier-3 `c_eye2d` | dot |
| Pupil center | tier-2 ellipse center | dot |
| **Pupil radius** | tier-3 `circle_3d.radius` (mm) | **text e.g. `r=1.92 mm`** + optional `Ø px` |
| Gaze angles | tier-3 `hor/ver` (deg) | top-right text box |
| Blink/invalid | tier-2 blink flag | show "blink/invalid" instead of arrow |

The `eyeball/cornea mesh` overlay (the projected wireframe sphere) is optional
eye-candy — defer it; it needs the full `rend_params` projection and adds little
for a first device build.

---

## 5. ★ The open decision — how to get tier 3 (3D eye model) on device

Tier 3 is the crux. The active code uses **pye3d** (`from pye3d.detector_3d
import Detector3D`). The repo's own `simple`/`LeGrand` numpy single-sphere models
are **stubs (`pass`)** — not usable. The classic conic-unprojection geometry
(Safaee-Rad 1992) *is* present in `threedeepvog/utils/unprojection.py`,
`intersection.py`, `gaze_process.py` (~1.4k lines of numpy), which is the
pye3d-free reference. Three ways forward:

### Option A — Classic geometry on device (pye3d-free)
Port `unprojection`/`intersection`/`gaze_process` to Kotlin/C++: per frame,
unproject the pupil ellipse to two candidate 3D circles; across the fit window,
solve for the single eyeball-sphere center consistent with all observations;
then per frame pick the disambiguated circle → **gaze normal + pupil radius**.
- ➕ Fully on-device (fit **and** predict); no external lib; self-contained.
- ➖ **No refraction correction** → pupil radius/gaze biased a few % vs pye3d.
  Conic unprojection needs `np.roots` (quartic) → port a small polynomial solver.

### Option B — Full pye3d on device
Bring pye3d's two-model + refraction pipeline to mobile via C++/NDK (it is
Cython over C++ with baked polynomial refraction models).
- ➕ Highest accuracy; matches desktop exactly.
- ➖ **By far the most work** (Cython/C++ port, refraction model assets, temporal
  Kalman models). Weeks. Risky to schedule first.

### Option C — Hybrid: fit off-device, predict on-device (recommended first)
Do the **eyeball fit once per user off-device** (desktop `run_cpu.py` fit mode →
`model_params.json` with eye center + radius), ship that small JSON to the app,
and port **only the lighter per-frame predict geometry** on device (unproject
pupil on the *frozen* sphere → gaze + pupil radius).
- ➕ Smallest on-device port; live gaze + pupil radius now; fit is a one-time
  calibration we already run on desktop.
- ➖ Needs a per-user calibration step (look-around) done off-device or in a
  companion flow; not "zero-setup".
- Note: C is essentially A's *predict* half with the fit precomputed — so C now,
  then add A's on-device fit later, is a clean incremental path.

**DECISION (2026-06-17): Option A — classic geometry, fully on-device.** We port
the Safaee-Rad conic unprojection + single-sphere consistency fit AND per-frame
predict to Kotlin/C++, so both calibration (look-around) and live gaze run on the
phone with zero off-device setup. No refraction correction (pupil radius/gaze
biased a few % vs pye3d — acceptable). Reference numpy:
`threedeepvog/utils/{unprojection,intersection,gaze_process}.py`. A quartic root
solver is required for the conic unprojection (port a small companion-matrix /
Ferrari solver). Option B (refraction-correct pye3d) remains a later upgrade only
if clinical-grade pupil radius is needed.

---

## 6. Kotlin file plan (tiers 2–3 + render)

| File | State | Role | Ported from |
|------|-------|------|-------------|
| `EyeSegmenter.kt` | ✅ | `.ort` session, gray→[0,1]→3ch→NCHW, threshold | `deepvog3d_model.py` + tier-1 |
| `EyeGeometry.kt` | ✅ | unproject + LSQ intersect + line–sphere | `utils/unprojection,intersection` |
| `EyeModel3D.kt` | ✅ | per-eye fit + predict → gaze + **pupil radius** | `GazeTracker.py` + DeepVOG method |
| `SegPostProcess.kt` | ⬜ | threshold + morphological open per channel | `PostProcessing.py` |
| `EllipseFitter.kt` | ⬜ | perimeter + `Imgproc.fitEllipse` + blink + confidence | `EllipseFitting.py` |
| `EyeTracker.kt` | ⬜ | per-eye orchestration (fit→predict) + state | `realtime_gaze.VOGEngine` |
| `GazeOverlay.kt` | ⬜ | Canvas annotations incl. pupil-radius text | `ParamsRender.py` + `realtime_gaze.draw_gaze` |

These slot **after** `EyeNormalizer` in `EyePatchPipeline.process()`: feed each
of `patches["left"/"right"]` through a per-side `EyeTracker`, then composite with
overlays. **`focalPx` for `EyeModel3D` must be derived from the EyeNormalizer
warp** (the patch's effective focal), not guessed — see §7.

Dependencies to add to `app/build.gradle.kts`: **OpenCV-Android** (for
`Imgproc.fitEllipse` + morphology) — or a small hand-rolled Fitzgibbon ellipse
fit + 3×3 morphology to avoid the ~40 MB OpenCV AAR if size matters.

---

## 7. Camera-intrinsics caveat (carries over from Phase 1)
Patches come from a **normalized virtual camera**; pye3d's
`focal_length=16.0`/`sensor_size=(4.8,3.6)` are eye-camera placeholders. So **2D
segmentation + gaze direction are valid; absolute metric pupil radius (mm) is not
physically calibrated** until we derive the virtual camera's effective intrinsics
from the normalization (`K`, `fill`, patch size) and feed those to tier 3. Until
then, render pupil radius in **px** or label the mm value "uncalibrated". See
[`INTEGRATION.md`](./INTEGRATION.md) §4.2.

---

## 8. Sequenced next steps
1. ✅ **Tier 1**: SegResNet `.ort` exported + parity-checked.
2. ✅ **Tier-3 strategy decided**: Option A (classic geometry on-device).
3. ✅ `EyeSegmenter.kt`, `EyeGeometry.kt`, `EyeModel3D.kt` written + Python
   golden-value harness (`tools/geom_parity.py`).
4. ✅ **`EyeGeometry` validated on the JVM (2026-06-17).** Harness
   `tools/geom_parity/GeomParityCheck.kt` (+ `run.sh`) compiles the real
   `EyeGeometry.kt` with a standalone `kotlinc` and runs it against the Python
   reference — no Android toolchain needed. Results:
   - `ellipseToGeneral` vs `convert_ell_to_general`: exact.
   - `unproject` vs `unprojectGazePositions`: normals match to ~1e-13, centers
     to ~1e-10; the centered/axis-aligned degeneracy correctly returns `null`.
     (The golden's `np.roots` happened to return descending order for all
     cases, matching `EyeGeometry`'s explicit descending sort.)
   - `intersectLines` vs `intersect`, `lineSphereIntersect` vs
     `line_sphere_intersect`: match to ~1e-13.
   - Reprojection frame sanity: `unproject`'s disk center reprojects to within
     <2px of the ellipse center (NOT exact — perspective foreshortening offsets
     the projected disk-center from the ellipse centroid, and the two-fold
     pos/neg candidates bracket it). Confirms `EyeModel3D.project()`'s
     camera-at-origin `focal·X/Z` frame matches `unproject`'s output frame.
   - Synthetic eyeball-fit round-trip: `intersectLines` + `eyeRadius` recover a
     known eyeball center + pupil offset to ~1e-13 — validates the fit math
     `EyeModel3D.finishFit()` relies on.

   **Still unvalidated in `EyeModel3D` (needs real/desktop data, not a JVM
   golden):** the two-fold *disambiguation* branch choice in `unprojectChoose`
   and the pupil-radius depth-scaling formula in `predict`. Validate these
   against desktop output on a real patch when wiring `EyeTracker` (step 7).
5. ✅ `SegPostProcess.kt` + `EllipseFitter.kt` written & **JVM-validated**
   (`tools/tier2_parity/Tier2Check.kt`). DEVIATION: ellipse fit uses **image
   moments of the filled mask** (closed-form 2x2 eigendecomp), not
   `cv2.fitEllipse` on the perimeter — recovers known ellipses to <0.1px on the
   axes / exact center+angle, needs no OpenCV AAR, and is more robust to ragged
   seg boundaries. `SegPostProcess` = threshold → morphological open (3x3) →
   largest-connected-component. Both confirmed on real eyes on-device.
6. ⬜ **Derive `focalPx`/`pxPerMm`** for the patch from the EyeNormalizer warp
   (§7) — still placeholders; gaze scale + pupil-radius-mm uncalibrated.
7. ✅ `EyeTracker.kt` (per-eye state machine IDLE/CALIBRATING/TRACKING) +
   `GazeOverlay.kt` (ellipses, seg tint, gaze arrow, pupil Ø) written and wired
   into `EyePatchPipeline`; `MainActivity` has a calibrate/reset/mask UI with a
   "look around" flow + progress. **Runs on device** (built with CLI `gradlew`,
   installed via adb on this workstation — see §9).
8. ⬑ On-device acceptance: **PARTIAL.** Tier-2 is correct on real eyes (pupil +
   iris ellipses track, seg tint clean, calibration accumulates and the eyeball
   fit completes). **Tier-3 gaze is geometrically OFF**: the 3D-projected pupil
   center lands far from the measured 2D pupil (gaze-arrow origin was at the
   patch corner), and the two eyes' 3D pupil diameters disagree ~2x. Root cause
   is the unvalidated `EyeModel3D.predict`/`unprojectChoose` disambiguation (the
   reconstructed part — see step 4's caveat). **Interim:** `GazeOverlay` now
   anchors the gaze arrow at the measured 2D pupil center and shows the 2D
   apparent diameter, using tier-3 only for arrow direction. **Fix = the desktop
   parity below.**

### ⬜ Tier-3 parity (the next real validation — do before trusting gaze)
The geometry PRIMITIVES are proven (step 4); the eye-MODEL orchestration is not.
Capture one device eye-patch (or reuse a desktop `--save_eyes` patch), run it
through both `run_cpu.py` (pye3d) and our `EyeModel3D`, and compare gaze normal +
eyeball center frame-by-frame. The disambiguation branch sign and the
pupil-center placement are the prime suspects.

### Input-preprocessing parity (do early)
Feed one desktop-saved patch through `EyeSegmenter` and compare the three masks
to `run_cpu.py` on the same patch — this catches grayscale/min-max/channel-order
mistakes before they masquerade as geometry bugs.

## 9. Build & run on-device (established workflow)
The app **builds and runs** — the original on-device crash was simply the model
assets missing from `app/src/main/assets/` (MediaPipe `face_landmarker.task`
resolver failed with a RET_CHECK "no slash" in `onCreate`). Fixed by bundling the
three assets and moving model init off the main thread.
- Assets (in `app/src/main/assets/`): `head_pose_repnext_m4_mobile.ort`,
  `seg_segresnet_240x320_mobile.tflite` (TFLite, see §10), and
  `face_landmarker.task` (download:
  `storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`).
- Build: `ANDROID_HOME=<sdk> ./gradlew :app:assembleDebug` (Gradle 9.3, SDK 34).
- Install/run: `adb install -r app/build/outputs/apk/debug/app-debug.apk` then
  `adb shell am start -n com.raynet.eyepatch/.MainActivity`.

## 10. Real-time performance
Per-stage timing is logged as `RayNetPerf` (averaged every 30 frames in
`EyePatchPipeline.logTiming`). Baseline ORT-CPU was **~1 fps** (seg ×2 = 705 ms,
head pose = 235 ms dominate at 480×640 input).

**On-device acceleration findings (Snapdragon/MIUI):** ORT XNNPACK(4 threads) was
*slower*; ORT NNAPI fell back to `nnapi-reference` (CPU, no GPU/NPU). So the seg net
was moved to **TFLite on the GPU delegate**:
- `seg ONNX → TFLite` via `onnx2tf` (`pretrained_models/segvog/convert_tflite.py`,
  isolated venv `/media/leviathan/Game/tflite_venv`). Parity TFLite vs onnxruntime
  = **1.37e-7**. NHWC `(1,240,320,3)`, sigmoid baked in.
- `EyeSegmenter` uses `org.tensorflow:tensorflow-lite{,-gpu,-gpu-api}:2.16.1`
  `GpuDelegate` (CPU fallback). The GPU runs the **whole graph (80/80 nodes)**;
  seg = **~82 ms/eye** (vs ~350 ms ORT-CPU). fp16 (`setPrecisionLossAllowed`) did
  not help — bandwidth/op-bound.
- Structural throttling: head pose every 5th frame (cache smoothed R), segment
  **one eye per frame** (alternating), skip seg while IDLE, largest-CC only (no
  morph-open).

**Result:** tracking **~1 → ~4 fps**, idle preview **~7 fps**. Stages are now
balanced — landmark ~80, seg ~82, pose ~57 (amortized) ms — so no single stage
dominates. **To reach real-time (~10 fps+):** pipeline the stages across threads
(the analyzer is single-threaded today) and/or run MediaPipe in async `LIVE_STREAM`
mode, so landmark / pose / seg overlap instead of running serially. That is the
next lever and a deliberate architecture change.

## 11. Eye-patch quality — high-res backprojection
The patches were blurry (→ poor segmentation) because the WHOLE pipeline ran on the
small ~480-px analysis frame: the eye region was only ~40–50 px before the warp
upsampled it to 320×240. Fixed by **detecting on a small frame but slicing from a
high-res one**:
- `ImageAnalysis` requests a high resolution (`ResolutionSelector`, target
  1280×960; device delivered 960×1280).
- MediaPipe runs on a downscaled 480-px copy (cheap). Its landmarks are normalized,
  so `FaceLandmarkerHelper.detect(bmp, targetW, targetH)` returns them in **full-res**
  pixel coords.
- The head-pose face crop and the `EyeNormalizer` warp therefore sample the
  **full-res frame** → sharp 320×240 patches (eyelashes / iris striations visible).
- The on-screen composite/overlay is drawn on the small frame (cheap); the crop box
  is scaled hi→lo.
- `MainActivity.maybeFocus()` drives CameraX AF+AE metering onto the central ROI
  every ~2.5 s (front cameras are often fixed-focus → AF a no-op, but AE/AWB still
  helps eye exposure). Best-effort.

Patch quality is now good; the remaining gaze-*direction* inconsistency is tier-3
(§8 step 8 / the tier-3 parity item), not patch quality.
