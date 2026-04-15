# RayNet v5 — Model Package

Triple-M1 architecture with a shared low-level encoder and three dedicated task branches (landmark, gaze, pose), each built on RepNeXt-M1. See `raynet_v5.py` for the implementation.

## Components

- **SharedStem** — RepNeXt-M1 stem + stages[0..1]. 3→48→96ch, 28×28. ~1.5M params. Intermediate maps are exposed as U-Net skip connections for the landmark branch.
- **Landmark Branch** — RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Predicts 14 iris/pupil heatmaps at 56×56 with soft-argmax + offset refinement.
- **Gaze Branch** — RepNeXt-M1 stages[2..3] + geometry head. Predicts `eyeball_center_3d` and `pupil_center_3d` in camera coordinates; optical axis is `normalize(pupil − eyeball)`. Pose-conditioned SHMA modulation + landmark cross-attention.
- **Pose Branch** — RepNeXt-M1 stages[2..3] on a gradient-detached copy of the shared stem. Predicts 6D rotation + 3D translation.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → 256d via 3→64→128→256 MLP with GELU. Zero-init FusionBlock combines with pose features to provide gaze origin.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`.

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
    use_landmark_bridge=True,
    use_pose_bridge=True,
)
```

Outputs include `landmark_heatmaps`, `landmark_coords`, `eyeball_center_3d`, `pupil_center_3d`, `optical_axis`, `pose_6d`, `pose_t`.

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

Three stages in `train.py::STAGE_CONFIGS`. Each stage is a sequence of phases with progressive loss activation and its own `CosineAnnealingLR` (optimizer state carries across phase boundaries to avoid "phase shock"). AMP uses GradScaler only when dtype is fp16; bf16 skips the scaler.

## Data Loaders

- `dataset.create_dataloaders` — local disk, builds single-view and multi-view loaders.
- `streaming.create_multiview_streaming_dataloaders` — MosaicML MDS streaming with 9-grouped batches for both train and val.

## Parameters

~17.1M total: ~1.5M shared stem, 3 × ~3.3M branch encoders, ~2.5M heads/bridges/box encoder.
