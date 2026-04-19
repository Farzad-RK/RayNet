# Architecture

RayNet v5 — **Quad-M1** multi-task architecture with a **two-stage landmark cascade**. Three branches (coarse landmark, pose, and a gaze branch that owns a full private RepNeXt-M1) sit on a shared low-level encoder. The gaze branch reads a **224×224** landmark-guided eye crop rather than sharing the face's 14×14 feature map, and the same eye-crop features feed a `LandmarkRefinementHead` that re-predicts the 14 iris/pupil landmarks at subpixel precision for pupillometry. Source: `RayNet/raynet_v5.py`, `RayNet/eye_crop.py`.

## Why Quad-M1 (and why the pivot from Triple-M1)

The original Triple-M1 design (v5.0) had the gaze branch consume the same stride-16 feature tensor produced from the face crop — a `(B, 384, 7, 7)` map where the iris occupies only 2-3 cells. The Stage 2 fork experiment (`docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/`) confirmed that this is a spatial-resolution ceiling, not a loss-weighting problem: `val_angular_deg` stayed between 42° and 45° across eight epochs despite three different loss schedules. See [[Home]] for the log.

Quad-M1 breaks the ceiling by giving gaze its own full RepNeXt-M1 fed by a landmark-guided eye crop. At 224×224 the iris now occupies a native **56×56 stem map** (4× the token capacity of the earlier 112 variant), which matches the face path's token budget and gives the landmark refinement decoder enough spatial support to recover subpixel iris positions. Expected post-training eye-crop gain is documented in MAC-Gaze.

## Why a two-stage landmark cascade (subpixel pupillometry)

The face-frame landmark branch decodes to a 56×56 heatmap over the 224×224 face — **4 input pixels per cell**. Soft-argmax + offset recovers ~0.2 px at that scale, which is enough to anchor the eye crop but **not** enough for pupillometry: in GazeGene the pupil radius is typically 5-15 face-frame pixels, so a 0.5 px landmark error already biases diameter measurements by 5-10 %.

The `LandmarkRefinementHead` re-predicts the same 14 landmarks in **eye-patch coordinates** from the gaze branch's intermediates. Because the eye crop covers only ~50-90 face-frame pixels, a 56-cell heatmap over that region gives **~1.0-1.6 face-frame px per cell**; soft-argmax + offset then drops residual error to ~0.2 face-frame px — roughly 4-8× tighter than the coarse head. This is what feeds the pupil-diameter / iris-contour downstream consumers.

Gradient design:

- Coarse landmarks drive the eye-crop affine (detached, as before). They do not need subpixel precision.
- Refined landmarks are supervised by projecting the face-frame GT through the same eye-crop affine and comparing in eye-patch pixel space; the loss flows back only into the gaze branch's own backbone and the refinement decoder.
- The refined head does **not** feed back into the eye crop in the current design. A second-pass crop conditioned on refined landmarks is an obvious future experiment but is out of scope for this run.

## Diagram

```
Input: (3, 224, 224) face crop + (3,) face bbox (x_p, y_p, L_x)
                         │
                ┌────────▼─────────┐
                │   SharedStem     │  RepNeXt-M1 stem + stages[0..1]
                │ 3→48→96ch, 28×28 │  ~0.21M params
                └────────┬─────────┘
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  Landmark Branch    Pose Branch   (eye branch is separate — see below)
  (coarse)           M1 s2+s3 +
  M1 s2+s3 +         6D rot +
  U-Net decoder      3D trans head
  + attention gates  ↑ detached shared features
  → 14 landmarks
     @ 56×56 (face-frame, drives eye crop)

                    coarse landmarks (×4, detached)
                                  │
  Input image  ────────────►  EyeCropModule
  (224×224)                   F.affine_grid + F.grid_sample
                              224×224, 30% pad, 32px min half-side
                                  │
                                  ▼
                            EyeBackbone  (full RepNeXt-M1, private to gaze)
                            stem + s0 + s1 + s2 + s3
                            returns bottleneck + (s0, s1, s2) skips
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
           CoordAtt + pool + proj    LandmarkRefinementHead
                  eye_feat            (U-Net decoder 7→14→28→56
                       │              on s2/s1 skips)
                       ▼              → 14 refined landmarks
                 GazeFusionBlock         @ 56×56 eye-patch space
                 (zero-init residual)    → unprojected to face-frame
                   eye_feat +              via EyeCropModule affine
                   MLP([eye,pose,box])
                       │
                       ▼
                 GeometricGazeHead
                   eyeball_center, pupil_center
                   → optical_axis = normalize(pupil − eyeball)
                       ▲
              ┌────────┴─────────┐
              │  MAGE BoxEncoder │   (x_p, y_p, L_x) → 256d
              └──────────────────┘
```

## Components

### SharedStem

RepNeXt-M1 stem (3→48ch, stride 4, 56×56) + stage[0] (48→48ch, 56×56) + stage[1] (48→96ch, 28×28). Intermediate maps are exposed as skip connections for the landmark U-Net decoder. ~0.21M params.

### Task Branches

- **Coarse landmark branch**: RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Upsamples from 7×7 back to 56×56 with skips from the shared-stem intermediates. Predicts 14 heatmaps → soft-argmax + learned offset refinement. Operates in face-frame; drives the eye-crop affine. ~6.2M params.
- **Pose branch**: RepNeXt-M1 stages[2..3] on a `.detach()`-ed copy of the shared stem output. Predicts 6D rotation (Gram-Schmidt) + 3D translation. ~4.5M params.
- **Gaze branch**: full private RepNeXt-M1 (`EyeBackbone`) that reads a **224×224** landmark-guided eye crop. The backbone feeds two consumers: the `GazeFusionBlock` (pool → `d_model=256` → geometric gaze head) and the `LandmarkRefinementHead` (U-Net decoder → 14 subpixel landmarks in eye-patch space). ~6.9M params. No high-level weight sharing with the face path.

### EyeCropModule (`RayNet/eye_crop.py`)

Differentiable landmark-guided crop built on `F.affine_grid` + `F.grid_sample` with `padding_mode='zeros'` (MAGE-style, avoids edge replication).

- Centroid `(cx, cy)` and axis ranges are computed from the 14 coarse landmarks (detached).
- Half-side: `0.5 * max(x_range, y_range) * (1 + pad_frac)`, clamped to `min_half_size` pixels. Defaults: `pad_frac=0.30`, `min_half_size=32`.
- Affine parameters (align_corners=True convention):

```
scale_x = 2 * half / (W - 1)
tx      = 2 * cx   / (W - 1) − 1
```

- Output: `(B, 3, 224, 224)`.
- Also exposes the per-sample affine dict `{cx, cy, half, out_size, H, W}` plus two static projection helpers:
  - `EyeCropModule.face_to_eye_coords(coords_face, affine, eye_feat_size=56)` — project face-frame GT landmarks into the refinement head's 56-cell frame (used by the loss).
  - `EyeCropModule.eye_to_face_coords(coords_eye, affine, eye_feat_size=56)` — lift refined predictions back to face-frame pixels (used by pupillometry / visualisation).

The crop is differentiable (so gaze loss can flow into landmark positions in future experiments), but we always pass `coarse_coords.detach()` in practice — during Stage 2 the face path is frozen, so the predicted-landmark distribution matches what the model sees at inference.

### EyeBackbone

Wraps a full RepNeXt-M1 (`stem + stages[0..3]`) initialised from the pretrained weights. Its `forward` returns the `stages[3]` bottleneck **and** the `(s0, s1, s2)` intermediates so both the gaze head and the `LandmarkRefinementHead` can consume them. Private to the gaze branch.

### LandmarkRefinementHead

A second U-Net-style decoder (mirroring the face-path landmark head) that runs on the EyeBackbone's intermediates and predicts the same 14 iris/pupil landmarks at 56×56 in eye-patch coordinates.

- Decoder: `dec3` (UNetDecoderBlock 384→192 with s2 skip, 7→14) → `dec2` (192→96 with s1 skip, 14→28) → bilinear upsample + double conv (96→48, 28→56). The s0 skip (already at 56×56) is deliberately skipped — it's the same resolution as the upsampled feature and adds params with no spatial gain.
- Output heads: 14-channel heatmap + 28-channel (=14×2) offset head, both 32-ch intermediate. Soft-argmax + per-landmark offset, identical to the coarse head.
- Returns `(coords_eye, coords_face, heatmaps)`. `coords_face` comes from projecting `coords_eye` through the eye-crop affine, so downstream consumers receive predictions directly in the face image frame.
- Loss: supervised by projecting face-frame GT through `face_to_eye_coords(...)` and comparing in eye-patch space (same heatmap-MSE + coord-L1 as the coarse head). Weight is `lam_lm_refine` in the phase config.
- Precision budget: the eye crop typically covers 50–90 face-frame px, so 56 cells → ~1.0–1.6 face-frame px/cell; soft-argmax + offset brings residual face-frame error below ~0.3 px in practice — ~4–8× the coarse head.
- ~0.45M additional params over the existing face-path landmark decoder.

### GazeFusionBlock

Zero-init residual fusion with eye features as the anchor:

```
fused = MLP([eye_feat, pose_feat, box_feat])   # weights zero-init on last layer
out   = eye_feat + fused
```

This is a strict generalisation of "eye-only" gaze: at init the block returns the eye embedding unchanged, and pose/box refinements are learned from gradient signal rather than baked in by a pretrained bridge.

### BoxEncoder

`(x_p, y_p, L_x) → 64 → 128 → 256` MLP with GELU. Provides gaze origin information derived from the face bounding box (via the Intrinsic Delta method) without requiring a 468-point MediaPipe detector at inference.

### CrossViewAttention

9-camera attention conditioned on `R_cam` and `T_cam`. Exercised when batches are delivered as 9-grouped multi-view tuples (see `streaming.create_multiview_streaming_dataloaders`).

## Forward Pass

```python
out = model(
    images,                 # (B, 3, 224, 224), B = mv_groups * 9
    n_views=9,
    R_cam=R_cam,            # (B, 3, 3)
    T_cam=T_cam,            # (B, 3)
    face_bbox=face_bbox,    # (B, 3) (x_p, y_p, L_x)
    use_landmark_bridge=True,   # no-op in Quad-M1 (kept for back-compat)
    use_pose_bridge=True,       # if False, zeros pose_feat into GazeFusionBlock
)
```

Internally:

1. `s0, s1 = SharedStem(images)`.
2. `pose_feat, pose_6d, pose_t = PoseBranch(s1.detach())`.
3. `coarse_coords, coarse_heatmaps = LandmarkBranch(s0, s1)` (coords in 56×56 face space).
4. `coarse_px = coarse_coords.detach() * (224 / 56)`.
5. `eye_patch, affine = EyeCropModule(images, coarse_px)` → `(B, 3, 224, 224)`.
6. `s3, (s0_eye, s1_eye, s2_eye) = EyeBackbone(eye_patch)`.
7. `eye_feat = proj(pool(CoordAtt(s3)))` → `(B, 256)`.
8. `refine_eye, refine_face, refine_hm = LandmarkRefinementHead(s3, (s0_eye, s1_eye, s2_eye), affine)`.
9. `gaze_feat = GazeFusionBlock(eye_feat, pose_feat or 0, box_feat)`.
10. `eyeball_center, pupil_center, optical_axis = GeometricGazeHead(gaze_feat)`.

## Outputs

| Key | Shape | Description |
|-----|-------|-------------|
| `landmark_heatmaps` | (B, 14, 56, 56) | coarse per-landmark heatmaps (face-frame) |
| `landmark_coords` | (B, 14, 2) | coarse subpixel `(x, y)` in 56×56 face space |
| `landmark_refine_heatmaps` | (B, 14, 56, 56) | refined heatmaps in eye-patch space |
| `landmark_refine_coords_eye` | (B, 14, 2) | refined subpixel `(x, y)` in 56-cell eye-patch space |
| `landmark_refine_coords_face` | (B, 14, 2) | refined coords unprojected to face-frame px (224×224) |
| `eye_crop_affine` | dict | `{cx, cy, half, out_size=224, H, W}` per-sample crop params |
| `eyeball_center` | (B, 3) | camera-space, cm |
| `pupil_center` | (B, 3) | camera-space, cm |
| `gaze_vector` | (B, 3) | unit optical axis = normalize(pupil − eyeball) |
| `gaze_angles` | (B, 2) | pitch, yaw |
| `pred_pose_6d` | (B, 6) | Gram-Schmidt rotation |
| `pred_pose_t` | (B, 3) | translation (meters, direct linear) |

The pupillometry pipeline should consume `landmark_refine_coords_face`. The coarse `landmark_coords` is retained for the eye-crop affine and for visualisation/backwards compatibility, not for downstream measurement.

## Tensor Shapes

```
Shared stem        -> (B, 96,  28, 28)
Branch stages[2]   -> (B, 192, 14, 14)
Branch stages[3]   -> (B, 384,  7,  7)
Landmark decoder   -> (B, 14,  56, 56)
Eye crop           -> (B, 3,  224, 224)
Eye backbone skips -> s0:(B,48,56,56), s1:(B,96,28,28), s2:(B,192,14,14)
Eye backbone out   -> s3:(B,384,7,7) → pooled (B, 256)
Refine decoder     -> (B, 14, 56, 56) eye-patch heatmaps
Gaze head          -> (B, 3), (B, 3), (B, 3)
Pose head          -> (B, 6), (B, 3)
```

## Parameter Budget

| Module | Params |
|--------|--------|
| SharedStem | 0.21M |
| LandmarkBranch (coarse; M1 s2+s3 + U-Net + heads) | 6.18M |
| PoseBranch (M1 s2+s3 + pose head) | 4.45M |
| GazeBranch (full M1 EyeBackbone + CoordAtt + fusion + gaze head + LandmarkRefinementHead) | 6.86M |
| CrossViewAttention + CameraEmbedding | 1.07M |
| **Total** | **~18.8M** |

## Gradient Flow

- Task branches do not share high-level weights — each owns stages[2..3] (or a full M1 for gaze).
- The pose branch reads shared-stem features through `.detach()` so pose-only gradients cannot perturb the shared encoder.
- Coarse landmarks passed to the eye crop are `.detach()`-ed, so gaze and refinement losses never flow back through the coarse landmark branch. This is the lever that makes it safe to freeze the face path in Stage 2 without the gaze branch seeing a moving target.
- The `LandmarkRefinementHead`'s gradients flow into the `EyeBackbone` (which is also driven by the gaze loss), so the eye encoder is jointly shaped by "gaze angle" and "subpixel iris location" signals. In practice these objectives are well aligned — the iris/pupil landmarks are the same features that determine gaze direction — and there is no need to detach either branch from the eye features.
- `GazeFusionBlock` is zero-init on its last linear layer: at initialisation the fused output equals the eye embedding, so pose/box contributions are learned rather than baked in.

## Freeze-Face Discipline (Stage 2, permanent)

`set_face_frozen(model, frozen=True)` (in `train.py`) does both `requires_grad_(False)` **and** `.eval()` on `shared_stem`, `landmark_branch` (the coarse one), and `pose_branch`. The `.eval()` call is load-bearing: otherwise BatchNorm's running stats would drift under the gaze-only training distribution and silently shift the coarse-landmark distribution that the eye crop depends on.

In the current Stage 2 curriculum (P1 → P3) the face path stays frozen for the full stage. The LandmarkRefinementHead is unaffected by this freeze because it lives inside the gaze branch; its loss (`lam_lm_refine`, on by default from P1) is what keeps subpixel landmark precision improving even though the coarse face head is inert.

Because DDP's default reducer expects every parameter to receive a gradient every step, `hardware_profiles.build_accelerator()` constructs the `Accelerator` with `DistributedDataParallelKwargs(find_unused_parameters=True)`. The runtime cost is a single extra graph traversal per step, which is negligible next to a full M1 forward pass.

## Validation EMA

Validation runs through a `torch.optim.swa_utils.AveragedModel` (weight EMA, decay `--ema_decay`, default 0.999). Only parameters are EMA'd; BN running stats are mirrored from the live model after every optimiser step so the shadow model uses consistent statistics. This smooths out per-step weight noise in the reported metrics (a material effect in Stage 2 P3 where the gaze loss is the only driver of updates and `val_angular_deg` oscillates across steps on the live weights).

## Key Differences from v4.1 and v5.0 (Triple-M1) and earlier Quad-M1

- No PANet — each branch owns its high-level encoder path (unchanged from v5.0).
- U-Net landmark decoder with attention gates (unchanged from v5.0).
- Explicit 3D eyeball geometry (unchanged from v5.0).
- **Gaze branch now owns a full private RepNeXt-M1 operating on a 224×224 eye crop** (new eye-crop design, upgraded from the earlier 112×112 Quad-M1). The 224×224 crop matches the face path's 56×56 token grid so the refinement decoder has sufficient spatial support.
- **`LandmarkRefinementHead` added** to the gaze branch — subpixel landmarks in eye-patch coordinates for pupillometry, with an inverse-affine lift to face-frame px.
- **Deleted from v5.0**: `FusionBlock` (2-input pose/box), `PoseGazeModulation`, `LandmarkGazeAttention`. Replaced by `GazeFusionBlock` (3-input eye/pose/box, zero-init residual with eye as anchor).
- BoxEncoder retained; bridge toggles `use_landmark_bridge` / `use_pose_bridge` are kept only for config back-compat — coarse landmarks are always used to anchor the eye crop, and `use_pose_bridge=False` zeros the pose stream into the fusion block.
- Stage 2 face-freeze is now permanent across P1–P3 (no unfreeze in P3). The gaze branch + refinement head have enough capacity to refine on their own; unfreezing risks moving the coarse-landmark distribution out from under the stable eye crop.
