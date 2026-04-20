# RayNet v5 — Model Package

**Triple-M1 + AERI** architecture: a shared low-level encoder (landmark-owned) and three task-specific branches that all consume the full 224×224 face crop. The gaze branch carries its own mini U-Net (`AERIHead`) that produces iris + eyeball binary segmentation logits at 56×56; the eyeball mask soft-gates the gaze bottleneck. See `raynet_v5.py`.

## Components

- **SharedStem** — RepNeXt-M1 stem + stages[0..1]. 3→48→96ch, 28×28. ~0.21M params. Intermediate maps (`s0`, `s1`) are exposed as U-Net skip connections for the landmark branch. Pose and gaze branches receive `s1.detach()` so only landmark loss backprops here.
- **Landmark Branch** — RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Predicts 14 iris/pupil heatmaps at 56×56 with soft-argmax + offset refinement. ~6.18M params.
- **Pose Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()`, CoordAtt + pooled feature fused with `BoxEncoder(face_bbox)` via a zero-init residual. Predicts 6D rotation + 3D translation. ~4.69M params.
- **Gaze Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()` + **AERIHead** (mini U-Net, 2-class: iris + eyeball). The predicted eyeball mask is downsampled to 7×7 and applied as a soft attention gate `0.25 + 0.75·sigmoid(·)` on the gaze bottleneck before pooling. `GazeFusionBlock` folds in `pose_feat` via a zero-init residual. Geometry head predicts `eyeball_center` and `pupil_center`; optical axis is `normalize(pupil − eyeball)`. ~6.53M params.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → `d_model` via 3→64→128→256 MLP with GELU. Consumed by the **pose** branch (not gaze). Provides head-pose prior without a 468-point face detector at inference.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`. ~1.07M params including the camera embedding.

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
)
```

Outputs: `landmark_coords`, `landmark_heatmaps`, `iris_mask_logits`, `eyeball_mask_logits`, `eyeball_center`, `pupil_center`, `gaze_vector`, `gaze_angles`, `pred_pose_6d`, `pred_pose_t`.

No eye-crop anchor is passed: the entire forward path is pixel-crop-free. AERI provides the eye-region inductive bias via a soft mask instead of a hard crop.

## Intrinsic Delta for BoxEncoder GT

`face_bbox` is derived from the two camera intrinsics (original and cropped) that GazeGene already stores — no manual labels. `K_cropped` is calibrated for the native 448×448 JPG, so the dataset rescales it to the 224-space tensor before computing `(x_p, y_p, L_x)`. See `dataset.py::__getitem__` and `docs/wiki/Geometry-and-Kappa.md`.

## AERI Masks

Iris and eyeball binary masks are **baked into MDS shards** at conversion time (see `streaming/eye_masks.py` + `streaming/convert_to_mds.py`). Each sample carries `iris_mask` and `eyeball_mask` as `uint8` arrays at 56×56. The dataset loader passes them to the training loop; `losses.mask_seg_loss` applies BCE-with-logits against the AERI head outputs.

## Losses

`losses.total_loss` combines, with phase-dependent weights:

- Landmark heatmap + coordinate L1 (`lam_lm`, `lam_heatmap`)
- Optical-axis angular + gaze-vector L1 (`lam_gaze`)
- GazeGene 3D structure: `lam_eyeball`, `lam_pupil`, `lam_geom_angular`
- Pose: 6D geodesic (`lam_pose`) + translation (`lam_trans`)
- Optional ray-to-target reprojection (`lam_ray`, `lam_reproj`)
- **AERI segmentation**: `lam_iris_seg`, `lam_eyeball_seg` (BCE-with-logits at 56×56)

Multi-view consistency is applied by `multiview_loss.multiview_consistency_loss` on 9-camera batches in phases 2 and 3.

## Parallel MTL — Single Stage, Three Phases

One stage. All heads active from epoch 1. Phases only adjust loss weights and LR. See `train.py::PHASE_CONFIG`.

| Phase | Epochs | Purpose | LR | Multi-view |
|-------|--------|---------|----|------------|
| 1 | 1–8 | warmup — all losses on, moderate weights | 5e-4 | off |
| 2 | 9–16 | main — full weights + multi-view consistency | 3e-4 | on |
| 3 | 17–25 | fine-tune — lower LR, gaze emphasis | 1e-4 | on |

Phase transitions preserve optimizer momentum and rebuild only the `CosineAnnealingLR` for the new phase window. Gradient clipping is `max_norm=5.0` in phase 1 and `max_norm=2.0` afterwards. AMP uses GradScaler only when dtype is fp16; bf16 skips the scaler.

No `--stage` flag. No `freeze_face` / `set_face_frozen`. Fork/warmstart/resume machinery is preserved for cross-architecture migrations via `_filter_compatible_state` (drops shape-mismatched tensors) and `_optimizer_state_compatible` (guards `optimizer.load_state_dict` across parameter-group changes).

## Data Loaders

- `dataset.create_dataloaders` — local disk, renders AERI masks on the fly.
- `streaming.create_multiview_streaming_dataloaders` — MosaicML MDS streaming with 9-grouped batches; masks are read directly from the shard.

## Parameters

~18.7M total: SharedStem 0.21M, LandmarkBranch 6.18M, PoseBranch 4.69M, GazeBranch 6.53M (includes AERIHead), CrossViewAttention + CameraEmbedding 1.07M.
