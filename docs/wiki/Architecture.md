# Architecture

RayNet v5 — **Quad-M1** multi-task architecture. Three branches (landmark, pose, plus a gaze branch that owns a full private RepNeXt-M1) sit on a shared low-level encoder, and the gaze branch reads a 112×112 landmark-guided eye crop rather than sharing the face's 14×14 feature map. Source: `RayNet/raynet_v5.py`, `RayNet/eye_crop.py`.

## Why Quad-M1 (and why the pivot from Triple-M1)

The original Triple-M1 design (v5.0) had the gaze branch consume the same stride-16 feature tensor produced from the face crop — a `(B, 384, 7, 7)` map where the iris occupies only 2-3 cells. The Stage 2 fork experiment (`docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/`) confirmed that this is a spatial-resolution ceiling, not a loss-weighting problem: `val_angular_deg` stayed between 42° and 45° across eight epochs despite three different loss schedules. See [[Home]] for the log.

Quad-M1 breaks the ceiling by giving gaze its own full RepNeXt-M1 fed by a landmark-guided eye crop. The iris now occupies a native 28×28 stem map. Expected post-training eye-crop gain is documented in MAC-Gaze (112×112 patch, sub-pixel iris fit).

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
  M1 s2+s3 +         M1 s2+s3 +
  U-Net decoder      6D rot +
  + attention gates  3D trans head
  → 14 landmarks     ↑ detached shared features
     @ 56×56

                    predicted landmarks (×4, detached)
                                  │
  Input image  ────────────►  EyeCropModule
  (224×224)                   F.affine_grid + F.grid_sample
                              112×112, 25% pad, 24px min half-size
                                  │
                                  ▼
                            EyeBackbone  (full RepNeXt-M1, private to gaze)
                            stem + s0 + s1 + s2 + s3
                                  │
                                  ▼
                            GazeFusionBlock (zero-init residual)
                              eye_feat + MLP([eye, pose, box])
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

- **Landmark branch**: RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Upsamples from 7×7 back to 56×56 with skips from the shared-stem intermediates. Predicts 14 heatmaps → soft-argmax + learned offset refinement. ~6.2M params.
- **Pose branch**: RepNeXt-M1 stages[2..3] on a `.detach()`-ed copy of the shared stem output. Predicts 6D rotation (Gram-Schmidt) + 3D translation. ~4.5M params.
- **Gaze branch**: full private RepNeXt-M1 (`EyeBackbone`) that reads a 112×112 landmark-guided eye crop. ~5.0M params. No longer shares high-level features with landmark/pose.

### EyeCropModule (`RayNet/eye_crop.py`)

Differentiable landmark-guided crop built on `F.affine_grid` + `F.grid_sample` with `padding_mode='zeros'` (MAGE-style, avoids edge replication).

- Centroid `(cx, cy)` and axis ranges are computed from the 14 predicted landmarks (detached).
- Half-side: `0.5 * max(x_range, y_range) * (1 + pad_frac)`, clamped to `min_half_size` pixels. Defaults: `pad_frac=0.25`, `min_half_size=24`.
- Affine parameters (align_corners=True convention):

```
scale_x = 2 * half / (W - 1)
tx      = 2 * cx   / (W - 1) − 1
```

- Output: `(B, 3, 112, 112)`.

The crop is differentiable for future experiments that want gradient flow from gaze loss into landmark positions, but in practice we always pass `landmark_coords.detach()` — during Stage 2 P1/P2 the landmark branch is frozen, so the predicted-landmark distribution matches what the model sees at inference.

### EyeBackbone

Wraps a full RepNeXt-M1 (`stem + stages[0..3]`) initialised from the pretrained weights, then global-pools to a `(B, 256)` embedding. Private to the gaze branch.

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
3. `landmark_coords, landmark_heatmaps = LandmarkBranch(s0, s1)` (coords in 56×56 space).
4. `landmarks_px = landmark_coords.detach() * (224 / 56)`.
5. `eye_patch = EyeCropModule(images, landmarks_px)` → `(B, 3, 112, 112)`.
6. `eye_feat = EyeBackbone(eye_patch)` → `(B, 256)`.
7. `gaze_feat = GazeFusionBlock(eye_feat, pose_feat or 0, box_feat)`.
8. `eyeball_center, pupil_center, optical_axis = GeometricGazeHead(gaze_feat)`.

## Outputs

| Key | Shape | Description |
|-----|-------|-------------|
| `landmark_heatmaps` | (B, 14, 56, 56) | per-landmark heatmaps |
| `landmark_coords` | (B, 14, 2) | subpixel `(x, y)` in 56×56 space |
| `eyeball_center` | (B, 3) | camera-space, cm |
| `pupil_center` | (B, 3) | camera-space, cm |
| `gaze_vector` | (B, 3) | unit optical axis = normalize(pupil − eyeball) |
| `gaze_angles` | (B, 2) | pitch, yaw |
| `pred_pose_6d` | (B, 6) | Gram-Schmidt rotation |
| `pred_pose_t` | (B, 3) | translation (tanh + exp) |

## Tensor Shapes

```
Shared stem       -> (B, 96,  28, 28)
Branch stages[2]  -> (B, 192, 14, 14)
Branch stages[3]  -> (B, 384,  7,  7)
Landmark decoder  -> (B, 14,  56, 56)
Eye crop          -> (B, 3,  112, 112)
Eye backbone out  -> (B, 256)
Gaze head         -> (B, 3), (B, 3), (B, 3)
Pose head         -> (B, 6), (B, 3)
```

## Parameter Budget

| Module | Params |
|--------|--------|
| SharedStem | 0.21M |
| LandmarkBranch (M1 s2+s3 + U-Net + heads) | 6.18M |
| PoseBranch (M1 s2+s3 + pose head) | 4.45M |
| GazeBranch (full M1 EyeBackbone + fusion + heads) | 5.04M |
| CrossViewAttention + CameraEmbedding | 1.07M |
| **Total** | **~17M** |

## Gradient Flow

- Task branches do not share high-level weights — each owns stages[2..3] (or a full M1 for gaze).
- The pose branch reads shared-stem features through `.detach()` so pose-only gradients cannot perturb the shared encoder.
- Landmarks passed to the eye crop are `.detach()`-ed, so gaze loss never flows back through the landmark branch. This is the lever that makes it safe to freeze the face path in Stage 2 P1/P2 without the gaze branch seeing a moving target.
- `GazeFusionBlock` is zero-init on its last linear layer: at initialisation the fused output equals the eye embedding, so pose/box contributions are learned rather than baked in.

## Freeze-Face Discipline (Stage 2 P1/P2)

`set_face_frozen(model, frozen=True)` (in `train.py`) does both `requires_grad_(False)` **and** `.eval()` on `shared_stem`, `landmark_branch`, and `pose_branch`. The `.eval()` call is load-bearing: otherwise BatchNorm's running stats would drift under the gaze-only training distribution and silently shift the predicted-landmark distribution that the eye crop depends on.

Because DDP's default reducer expects every parameter to receive a gradient every step, `hardware_profiles.build_accelerator()` constructs the `Accelerator` with `DistributedDataParallelKwargs(find_unused_parameters=True)`. The runtime cost is a single extra graph traversal per step, which is negligible next to a full M1 forward pass.

## Key Differences from v4.1 and v5.0 (Triple-M1)

- No PANet — each branch owns its high-level encoder path (unchanged from v5.0).
- U-Net landmark decoder with attention gates (unchanged from v5.0).
- Explicit 3D eyeball geometry (unchanged from v5.0).
- **Gaze branch now owns a full private RepNeXt-M1 operating on a 112×112 eye crop (new in v5 Quad-M1).**
- **Deleted from v5.0**: `FusionBlock` (2-input pose/box), `PoseGazeModulation`, `LandmarkGazeAttention`. Replaced by `GazeFusionBlock` (3-input eye/pose/box, zero-init residual with eye as anchor).
- BoxEncoder retained; bridge toggles `use_landmark_bridge` / `use_pose_bridge` are kept only for config back-compat — landmarks are always used to anchor the eye crop, and `use_pose_bridge=False` zeros the pose stream into the fusion block.
