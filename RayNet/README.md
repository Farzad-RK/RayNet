# RayNet v5 — Model Package

**Triple-M1 + AERI + HRFH-α** architecture. A shared low-level encoder (landmark-owned) and three task-specific branches that all consume the full 224×224 face crop. The gaze branch carries its own mini U-Net (`AERIHead`) producing **iris + eyeball** binary segmentation logits at 56×56. The two masks are combined into a saliency map (`0.8·iris + 0.2·eyeball`); the saliency is blended with a uniform field through an `α` schedule, the result both gates the gaze bottleneck (7×7) AND multiplies the high-resolution `d1` decoder tensor (56×56, 48ch) to produce a 48-d **foveal vector** that represents sub-pixel iris/pupil dynamics. See `raynet_v5.py`.

## Components

- **SharedStem** — RepNeXt-M1 stem + stages[0..1]. 3→48→96ch, 28×28. ~0.21M params. Intermediate maps (`s0`, `s1`) are exposed as U-Net skip connections for the landmark branch. Pose and gaze branches receive `s1.detach()` so only landmark loss backprops here.
- **Landmark Branch** — RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Predicts 14 iris/pupil heatmaps at 56×56 with soft-argmax + offset refinement. ~6.18M params.
- **Pose Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()`, CoordAtt + pooled feature fused with `BoxEncoder(face_bbox)` via a zero-init residual. Predicts 6D rotation + 3D translation. `face_bbox` is **optional at inference** — when omitted the BoxEncoder residual zeroes out and pose collapses to CNN features. ~4.69M params.
- **Gaze Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()` + **AERIHead** (mini U-Net, 2-class: iris + eyeball). Pipeline:
  1. AERI predicts iris/eyeball logits at 56×56 plus the decoder tensor `d1` (48ch, 56×56).
  2. `saliency = 0.8·sigmoid(iris) + 0.2·sigmoid(eyeball)`.
  3. `scheduled_mask = α·saliency + (1−α)·1` — α controlled by `get_scheduled_alpha(epoch)` in `train.py`.
  4. Stochastic mask dropout (10% chance during training) replaces the scheduled mask with the uniform field.
  5. **HRFH harvesting** — `scheduled_mask` is pooled to 7×7 and applied to the gaze bottleneck → 384-d global vector. Same mask gates `d1` at 56×56 → 48-d foveal vector.
  6. `[global ‖ foveal]` (432-d) → `LayerNorm` → `Linear → 256`.
  7. `GazeFusionBlock` folds in `pose_feat` via a zero-init residual.
  8. `CrossViewAttention` (when `n_views > 1`).
  9. `GeometricGazeHead` predicts `eyeball_center` and `pupil_center`; optical axis is `normalize(pupil − eyeball)`. ~6.53M params.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → `d_model` via 3→64→128→256 MLP with GELU. Consumed by the **pose** branch (not gaze). Provides head-pose prior; optional at inference.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`. Identity short-circuit when `n_views ≤ 1`. ~1.07M params including the camera embedding.

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
    face_bbox=face_bbox, # (B, 3)    (x_p, y_p, L_x) — optional at inference
    aeri_alpha=0.9,      # 0..1     AERI saliency vs. uniform-field blend
)
```

Outputs: `landmark_coords`, `landmark_heatmaps`, `iris_mask_logits`, `eyeball_mask_logits`, `eyeball_center`, `pupil_center`, `gaze_vector`, `gaze_vector_sv`, `gaze_angles`, `pred_pose_6d`, `pred_pose_t`.

No eye-crop anchor is passed: the entire forward path is pixel-crop-free. AERI provides the eye-region inductive bias via a soft mask instead of a hard crop.

### Inference without external face crops

The reference `inference.py` runs MediaPipe (Haar fallback) inside the module. The caller passes a full frame; the module detects the face, square-crops with a 1.3× expansion, runs the model, projects landmarks/masks/gaze back to frame coordinates, and draws the bounding box that was actually fed to the model. `face_bbox` is synthesised from the detected pixels (`mage_bbox_from_pixels`) assuming a centred principal point, or set to `None` to bypass the BoxEncoder entirely.

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

## Three-Phase Training Schedule

The schedule is now interpreted as a **branch-staged curriculum** rather than parallel MTL. All branches are wired into the same forward graph, but loss weights + selective freezing isolate one objective at a time. See `train.py::PHASE_CONFIG` and `train.py::get_scheduled_alpha`.

| Phase | Epochs | Active branches | Frozen | LR | Multi-view |
|-------|--------|-----------------|--------|----|------------|
| 1 | 1–8 | landmark + AERI iris/eyeball seg + headpose | gaze branch (encoder, fusion, head) | 5e-4 | off |
| 2 | 9–18 | gaze branch + AERI fine-tune | shared stem + landmark + pose | 3e-4 | off (monocular only) |
| 3 | 19–35 | all branches, multi-view fusion + ray consistency | none | 3e-5 — 5e-5 | on |

**Phase 1** establishes the landmark foundation (val_landmark_px ≤ 2.2 px) and crisp iris/eyeball masks (val seg loss ≤ 0.005 / 0.012). The gaze loss is OFF (`lam_gaze = lam_eyeball = lam_pupil = lam_geom_angular = 0`); seg weights are lifted to 1.0 to drive masks to convergence before HRFH-α consumes them.

**Phase 2** trains gaze monocularly. Hold `aeri_alpha = 0.7` constant — do NOT ramp during fine-tune. Mask dropout is disabled so train and val see the same gating function. `lam_iris_seg = lam_eyeball_seg = 0.5` keeps the masks stable while gaze adapts. CrossViewAttention is short-circuited (`n_views=1`); use a single-view dataloader.

**Phase 3** unfreezes everything at a much lower LR (5×–10× lower than P2) for 2 epochs of monocular settling, then turns on `multiview=True`, `lam_reproj=0.1`, `lam_mask=0.05`. Early-stop on `val_angular_deg` with patience=3. `aeri_alpha` stays constant at the P2 value — the ARI ramp belongs in P1→P2, not in fine-tune.

Phase transitions preserve optimizer momentum and rebuild only the `CosineAnnealingLR` for the new phase window. Gradient clipping is `max_norm=5.0` in phase 1 and `max_norm=2.0` afterwards. AMP uses GradScaler only when dtype is fp16; bf16 skips the scaler.

Fork/warmstart/resume machinery is preserved for cross-architecture migrations via `_filter_compatible_state` (drops shape-mismatched tensors) and `_optimizer_state_compatible` (guards `optimizer.load_state_dict` across parameter-group changes). `--warmstart_phase` lets a Phase-2 fork start at the right LR/loss weights when warmstarting from a Phase-1 checkpoint without replaying P1.

## AERI-α Schedule (`get_scheduled_alpha`)

α controls how much the saliency mask gates gaze features vs. the uniform field. `scheduled_mask = α·saliency + (1−α)·1`.

  - `α = 0.4` (Phases 1, P2 epochs 1-3) — moderate reliance, lets the gaze head see global features while masks are fresh.
  - `α = 0.7` (Phase 2 fine-tune) — held constant after the gaze head has committed to the saliency-conditioned features.
  - The previously-shipped `0.4 → 0.9` linear ramp during the cosine LR decay caused validation drift in `triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115` (val_angular climbs from 12.4° at epoch 28 to 15.3° at epoch 35 as α approaches 0.9). Hold α constant during fine-tune.

## Data Loaders

- `dataset.create_dataloaders` — local disk, renders AERI masks on the fly.
- `streaming.create_multiview_streaming_dataloaders` — MosaicML MDS streaming with 9-grouped batches; masks are read directly from the shard.

## Parameters

~18.7M total: SharedStem 0.21M, LandmarkBranch 6.18M, PoseBranch 4.69M, GazeBranch 6.53M (includes AERIHead), CrossViewAttention + CameraEmbedding 1.07M.

## Inference

```bash
python -m RayNet.inference --checkpoint best_model.pt --webcam
python -m RayNet.inference --checkpoint best_model.pt --input clip.mp4 --output annotated.mp4
python -m RayNet.inference \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://204.168.238.119:9000 \
    --run_id triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115 \
    --ckpt_file best_model.pt --webcam
```

The inference module embeds face detection (MediaPipe → Haar fallback). It draws: detected face box, 14 iris/pupil landmarks, AERI iris (green) and eyeball (yellow) masks overlaid at 56×56 → upsampled to crop, gaze arrow from the eye-center, RGB pose axes, pitch/yaw/translation overlay. Toggle the mask overlay with the `m` key. Use `--no_masks` to start with the overlay off.
