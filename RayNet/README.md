# RayNet v5 — Model Package

**Quad-M1** architecture: a shared low-level encoder, dedicated landmark and pose branches, and a **private full RepNeXt-M1 for gaze** that reads a 112×112 landmark-guided eye crop. See `raynet_v5.py` and `eye_crop.py` for the implementation.

## Components

- **SharedStem** — RepNeXt-M1 stem + stages[0..1]. 3→48→96ch, 28×28. ~0.21M params. Intermediate maps are exposed as U-Net skip connections for the landmark branch.
- **Landmark Branch** — RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Predicts 14 iris/pupil heatmaps at 56×56 with soft-argmax + offset refinement. ~6.2M params.
- **Pose Branch** — RepNeXt-M1 stages[2..3] on a gradient-detached copy of the shared stem. Predicts 6D rotation + 3D translation. ~4.5M params.
- **Gaze Branch** — `EyeCropModule` (differentiable 112×112 landmark-guided crop) → `EyeBackbone` (full private RepNeXt-M1) → `GazeFusionBlock` (zero-init residual, eye anchor + pose + box) → geometry head predicting `eyeball_center` and `pupil_center` in camera coordinates; optical axis is `normalize(pupil − eyeball)`. ~5.0M params.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → 256d via 3→64→128→256 MLP with GELU. Provides gaze origin without a 468-point face detector at inference.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`. ~1.1M params including the camera embedding.

## Forward Signature

```python
from RayNet.raynet_v5 import create_raynet_v5

model = create_raynet_v5(
    backbone_weight_path='./ptrained_models/repnext_m1_distill_300e.pth',
    n_landmarks=14,
)

out = model(
    images,              # (B, 3, 224, 224), B = n_views * mv_groups
    n_views=9,
    R_cam=R_cam,         # (B, 3, 3) camera rotation
    T_cam=T_cam,         # (B, 3)    camera translation
    face_bbox=face_bbox, # (B, 3)    (x_p, y_p, L_x) from Intrinsic Delta
    use_landmark_bridge=True,   # no-op in Quad-M1 (back-compat)
    use_pose_bridge=True,       # False → zero pose stream into GazeFusionBlock
)
```

Outputs: `landmark_coords`, `landmark_heatmaps`, `eyeball_center`, `pupil_center`, `gaze_vector`, `gaze_angles`, `pred_pose_6d`, `pred_pose_t`.

Internally, predicted landmarks in 56-space are rescaled to 224-space (×4), detached, and passed to the gaze branch as the eye-crop anchor — so the eye crop is deterministic w.r.t. the face path and sees the same distribution at train and inference time.

## Intrinsic Delta for BoxEncoder GT

`face_bbox` is derived from the two camera intrinsics (original and cropped) that GazeGene already stores — no manual labels. `K_cropped` is calibrated for the native 448×448 JPG, so the dataset rescales it to the 224-space tensor before computing `(x_p, y_p, L_x)`. See `dataset.py::__getitem__` and `docs/wiki/Geometry-and-Kappa.md`.

## Losses

`losses.total_loss` combines, with phase-dependent weights:

- Landmark heatmap + coordinate L1
- Angular error on optical axis
- GazeGene 3D structure: `lam_eyeball`, `lam_pupil`, `lam_geom_angular`
- Pose: 6D rotation + translation
- Optional ray-to-target reprojection

Multi-view consistency is applied by `multiview_loss.multiview_consistency_loss` on 9-camera batches.

## Staged Training

Three stages in `train.py::STAGE_CONFIGS`. Each stage is a sequence of phases with progressive loss activation and its own `CosineAnnealingLR`.

- **Stage 1** — landmark + pose baseline; `lam_gaze=0`. Recommended recipe: `--samples_per_subject 500`, 15 epochs, `kaggle_t4x2`. Reference run: `docs/experiments/raynet_v5_500_samples_per_subject/` (`val_landmark_px` = 2.64 at best epoch 14).
- **Stage 2** — eye-crop gaze curriculum. P1/P2 run with `freeze_face=True` (face path held at `.eval()` + `requires_grad=False` via `train.set_face_frozen`, so BN running stats also freeze); P3 unfreezes for gentle joint fine-tuning. This schedule was chosen after `docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/` showed the Triple-M1 shared-feature design flooring `val_angular` at ~42°.
- **Stage 3** — full pipeline with optional bridges + BoxEncoder fusion; retained for A/B comparisons.

AMP uses GradScaler only when dtype is fp16; bf16 skips the scaler. Cross-stage forks use `_filter_compatible_state` to drop shape-mismatched tensors (old Triple-M1 → Quad-M1) and `_optimizer_state_compatible` to guard `optimizer.load_state_dict` when the parameter groups changed.

## Data Loaders

- `dataset.create_dataloaders` — local disk, builds single-view and multi-view loaders.
- `streaming.create_multiview_streaming_dataloaders` — MosaicML MDS streaming with 9-grouped batches for both train and val.

## Parameters

~17M total: SharedStem 0.21M, LandmarkBranch 6.18M, PoseBranch 4.45M, GazeBranch 5.04M, CrossViewAttention + CameraEmbedding 1.07M.
